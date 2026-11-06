"""LLM client: caching, backoff, budgets, fallback. Uses fake OpenAI clients - no network."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

from core.llm import LLM, LLMUnavailable, Provider, extract_json
from core.store import LLMStore


def _resp(text, pt=10, ct=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )


def _rate_limit(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    req = httpx.Request("POST", "http://x")
    return openai.RateLimitError("429", response=httpx.Response(429, headers=headers, request=req), body=None)


class FakeClient:
    """Plays a script of responses/exceptions, records the calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _resp(item)


def make(tmp_path, groq_script, nvidia_script=None, groq_budget=10_000):
    groq = FakeClient(groq_script)
    nv = FakeClient(nvidia_script or [])
    providers = [
        Provider("groq", "", "", {"large": "g-large", "small": "g-small"},
                 {"large": groq_budget, "small": groq_budget}, client=groq),
        Provider("nvidia", "", "", {"large": "n-large", "small": "n-small"},
                 {"large": 10_000, "small": 10_000}, client=nv),
    ]
    llm = LLM(providers, LLMStore(tmp_path / "llm.sqlite"), max_retries=2, sleep=lambda s: None)
    return llm, groq, nv


MSG = [{"role": "user", "content": "hi"}]


def test_cache_hit_skips_provider(tmp_path):
    llm, groq, _ = make(tmp_path, ["hello"])
    first = llm.chat(MSG)
    second = llm.chat(MSG)
    assert first.text == second.text == "hello"
    assert not first.cached and second.cached
    assert len(groq.calls) == 1


def test_size_picks_model(tmp_path):
    llm, groq, _ = make(tmp_path, ["a", "b"])
    llm.chat(MSG, size="small")
    llm.chat(MSG, size="large")
    assert [c["model"] for c in groq.calls] == ["g-small", "g-large"]


def test_backoff_then_success(tmp_path):
    llm, groq, nv = make(tmp_path, [_rate_limit(), _rate_limit(1), "ok"])
    assert llm.chat(MSG).text == "ok"
    assert len(groq.calls) == 3 and not nv.calls


def test_long_retry_after_falls_back(tmp_path):
    llm, groq, nv = make(tmp_path, [_rate_limit(retry_after=600)], ["from nvidia"])
    r = llm.chat(MSG)
    assert r.provider == "nvidia" and r.text == "from nvidia"


def test_budget_exhausted_skips_groq(tmp_path):
    llm, groq, nv = make(tmp_path, ["unused"], ["nv"], groq_budget=5)
    assert llm.chat(MSG).provider == "nvidia"
    assert not groq.calls


def test_tokens_are_tracked(tmp_path):
    llm, _, _ = make(tmp_path, ["x"])
    llm.chat(MSG)
    assert llm.store.used_today("groq:g-small") == 15
    assert llm.store.used_today("groq:g-large") == 0


def test_all_fail_gives_clear_error(tmp_path):
    llm, _, _ = make(tmp_path, [_rate_limit()] * 3, [_rate_limit()] * 3)
    with pytest.raises(LLMUnavailable, match="All LLM providers failed"):
        llm.chat(MSG)


def test_no_provider_configured(tmp_path):
    providers = [Provider("groq", "", "", {"large": "a", "small": "b"}, {"large": 100, "small": 100})]
    llm = LLM(providers, LLMStore(tmp_path / "x.sqlite"))
    with pytest.raises(LLMUnavailable, match="GROQ_API_KEY"):
        llm.chat(MSG)


class Pair(BaseModel):
    a: int
    b: str


def test_chat_json_repairs_once(tmp_path):
    llm, groq, _ = make(tmp_path, ["not json at all", '```json\n{"a": 1, "b": "x"}\n```'])
    assert llm.chat_json(MSG, Pair) == Pair(a=1, b="x")
    assert len(groq.calls) == 2


def test_extract_json_with_chatter():
    assert extract_json('Sure! {"a": 2, "b": "y"} hope that helps') == {"a": 2, "b": "y"}
    with pytest.raises(ValueError):
        extract_json("nothing here")


def test_daily_request_cap_falls_back(tmp_path):
    llm, groq, nv = make(tmp_path, ["a", "b"], ["from nvidia"])
    llm.providers[0].daily_requests = 1
    llm.chat([{"role": "user", "content": "one"}])
    assert llm.chat([{"role": "user", "content": "two"}]).provider == "nvidia"
    assert len(groq.calls) == 1


def test_reasoning_effort_only_for_reasoning_models(tmp_path):
    llm, groq, _ = make(tmp_path, ["a", "b"])
    llm.providers[0].reasoning_effort = "low"
    llm.providers[0].models = {"large": "openai/gpt-oss-120b", "small": "qwen/qwen3.8-27b"}
    llm.chat(MSG, size="large")
    llm.chat(MSG, size="small")
    assert groq.calls[0]["extra_body"] == {"reasoning_effort": "low"}
    assert "extra_body" not in groq.calls[1]


def test_minute_window_paces_calls():
    from core.llm import MinuteWindow

    now = [0.0]
    w = MinuteWindow(limit=1000, clock=lambda: now[0])
    assert w.wait_time(600) == 0
    w.record(600)
    assert w.wait_time(300) == 0           # 900 fits
    w.record(300)
    wait = w.wait_time(300)                # 1200 would not fit: wait for the first call to age out
    assert 59.9 < wait < 60.2
    now[0] = 61.0
    assert w.wait_time(300) == 0
