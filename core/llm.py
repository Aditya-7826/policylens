"""One LLM client for the whole project.

Every call goes through `LLM.chat()`, which:
  1. looks the request up in the on-disk cache (key = hash of model + messages + params);
  2. skips any provider/model whose daily token budget would be exceeded;
  3. calls Groq, retrying 429/5xx with exponential backoff (honouring Retry-After);
  4. falls back to NVIDIA NIM if Groq is unavailable;
  5. raises LLMUnavailable with a readable message if every provider fails.

`size` picks the model tier: "large" (answers) or "small" (extraction, classification, judging).
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Literal, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from core.config import Settings, get_settings
from core.store import LLMStore, cache_key

log = logging.getLogger(__name__)

Size = Literal["large", "small"]
T = TypeVar("T", bound=BaseModel)

# If a provider asks us to wait longer than this, fall back instead of blocking the user.
MAX_WAIT_SECONDS = 30.0


class LLMUnavailable(RuntimeError):
    """Raised when no provider could answer. The message says why, for the UI."""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    cached: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Provider:
    name: str
    api_key: str
    base_url: str
    models: dict[str, str]  # size -> model id
    budgets: dict[str, int]  # size -> daily token budget (free-tier caps are per model)
    client: object | None = field(default=None, repr=False)  # injectable for tests

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or self.client is not None

    def get_client(self):
        if self.client is None:
            # max_retries=0: we do our own backoff so we can fall back on long waits.
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url,
                                        max_retries=0, timeout=60.0)
        return self.client


def providers_from_settings(s: Settings) -> list[Provider]:
    return [
        Provider("groq", s.groq_api_key, s.groq_base_url,
                 {"large": s.groq_large_model, "small": s.groq_small_model},
                 {"large": s.groq_large_daily_tokens, "small": s.groq_small_daily_tokens}),
        Provider("nvidia", s.nvidia_api_key, s.nvidia_base_url,
                 {"large": s.nvidia_large_model, "small": s.nvidia_small_model},
                 {"large": s.nvidia_large_daily_tokens, "small": s.nvidia_small_daily_tokens}),
    ]


def _estimate_tokens(messages: list[dict], max_tokens: int) -> int:
    chars = sum(len(m.get("content", "")) for m in messages)
    return chars // 4 + max_tokens


def _retry_after(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class LLM:
    def __init__(self, providers: list[Provider] | None = None, store: LLMStore | None = None,
                 max_retries: int = 4, base_delay: float = 1.0, sleep=time.sleep):
        s = get_settings()
        self.providers = providers if providers is not None else providers_from_settings(s)
        self.store = store or LLMStore(s.cache_dir / "llm.sqlite")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._sleep = sleep

    # ------------------------------------------------------------------ public
    def chat(self, messages: list[dict], size: Size = "small", temperature: float = 0.0,
             max_tokens: int = 1024, use_cache: bool = True) -> LLMResult:
        primary = next((p for p in self.providers if p.configured), None)
        if primary is None:
            raise LLMUnavailable(
                "No LLM provider is configured. Put GROQ_API_KEY (and optionally NVIDIA_API_KEY) in .env."
            )
        # Key on the *primary* model id: changing models in .env invalidates the cache,
        # while a fallback answer is reused for the same request.
        key = cache_key(model=primary.models[size], size=size, messages=messages,
                        temperature=temperature, max_tokens=max_tokens)
        if use_cache and (hit := self.store.get(key)):
            return LLMResult(hit["text"], hit["provider"], hit["model"], True,
                             hit["prompt_tokens"], hit["completion_tokens"])

        errors: list[str] = []
        for provider in self.providers:
            if not provider.configured:
                continue
            est = _estimate_tokens(messages, max_tokens)
            ledger = f"{provider.name}:{provider.models[size]}"
            used, budget = self.store.used_today(ledger), provider.budgets[size]
            if used + est > budget:
                errors.append(f"{ledger}: daily token budget reached ({used}/{budget})")
                continue
            try:
                result = self._call_with_backoff(provider, size, messages, temperature, max_tokens)
            except _ProviderFailed as exc:
                errors.append(f"{provider.name}: {exc}")
                log.warning("LLM provider %s failed: %s", provider.name, exc)
                continue
            self.store.add_usage(ledger, result.prompt_tokens + result.completion_tokens)
            self.store.put(key, result.provider, result.model, result.text,
                           result.prompt_tokens, result.completion_tokens)
            return result

        raise LLMUnavailable("All LLM providers failed. " + "; ".join(errors))

    def chat_json(self, messages: list[dict], schema: type[T], size: Size = "small",
                  max_tokens: int = 1024) -> T:
        """Ask for JSON and validate it with Pydantic. One repair attempt on invalid output."""
        result = self.chat(messages, size=size, max_tokens=max_tokens)
        try:
            return schema.model_validate(extract_json(result.text))
        except (ValueError, ValidationError) as exc:
            repair = messages + [
                {"role": "assistant", "content": result.text},
                {"role": "user", "content": f"That was not valid JSON for the schema ({str(exc)[:300]}). "
                                            "Reply with only the corrected JSON object."},
            ]
            fixed = self.chat(repair, size=size, max_tokens=max_tokens)
            return schema.model_validate(extract_json(fixed.text))

    def usage(self) -> dict:
        report = self.store.usage_report()
        report["budgets"] = {f"{p.name}:{p.models[s]}": p.budgets[s] for p in self.providers for s in p.budgets}
        report["configured"] = [p.name for p in self.providers if p.configured]
        return report

    # ----------------------------------------------------------------- private
    def _call_with_backoff(self, provider: Provider, size: Size, messages, temperature, max_tokens):
        model = provider.models[size]
        for attempt in range(self.max_retries + 1):
            try:
                resp = provider.get_client().chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                )
                usage = getattr(resp, "usage", None)
                return LLMResult(
                    text=resp.choices[0].message.content or "",
                    provider=provider.name, model=model, cached=False,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )
            except (openai.RateLimitError, openai.InternalServerError,
                    openai.APIConnectionError, openai.APITimeoutError) as exc:
                wait = _retry_after(exc)
                if wait is None:
                    wait = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                if wait > MAX_WAIT_SECONDS:
                    raise _ProviderFailed(f"rate limited, asked to wait {wait:.0f}s") from exc
                if attempt == self.max_retries:
                    raise _ProviderFailed(f"{type(exc).__name__} after {attempt + 1} attempts") from exc
                log.info("%s %s, retrying in %.1fs", provider.name, type(exc).__name__, wait)
                self._sleep(wait)
            except openai.APIStatusError as exc:  # 400/401/404: retrying will not help
                raise _ProviderFailed(f"HTTP {exc.status_code}: {str(exc)[:200]}") from exc
        raise _ProviderFailed("unreachable")


class _ProviderFailed(Exception):
    pass


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str):
    """Pull the first JSON object/array out of a model reply (handles ``` fences and chatter)."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model output")


_default: LLM | None = None


def get_llm() -> LLM:
    global _default
    if _default is None:
        _default = LLM()
    return _default
