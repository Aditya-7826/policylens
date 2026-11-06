"""Evaluation metrics. Pure functions over one pipeline result + its gold item.

Retrieval (no LLM):
  recall@5   share of the expected sections that appear among the top-5 evidence chunks
  MRR        1 / rank of the first evidence chunk from an expected section (0 if none)
Answer quality:
  key-fact accuracy   share of the gold key facts mentioned in the answer (deterministic string match)
  citation precision  share of cited sentences the judge says their citation supports (partial = 0.5)
  faithfulness        share of the answer's claims the judge says the shown evidence supports
Abstention:
  abstention accuracy  unanswerable -> abstained, answerable -> answered
Latency: p50 / p95 of end-to-end seconds.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field


def section_of(chunk: dict) -> str:
    return f"{chunk['chunk_id'].split('::', 1)[0]}::{chunk['section']}"


def recall_at_k(evidence: list[dict], expected: list[str], k: int = 5) -> float | None:
    if not expected:
        return None
    found = {section_of(c) for c in evidence[:k]}
    return sum(1 for e in expected if e in found) / len(expected)


def mrr(evidence: list[dict], expected: list[str]) -> float | None:
    if not expected:
        return None
    for rank, c in enumerate(evidence, start=1):
        if section_of(c) in expected:
            return 1.0 / rank
    return 0.0


_DASHES = re.compile(r"[‐-―−-]")


def _fold(text: str) -> str:
    """Lower-case, fold Unicode (NBSP, non-breaking hyphens), hyphens as spaces: 'supply‑chain' == 'supply chain'."""
    text = _DASHES.sub(" ", unicodedata.normalize("NFKC", text).lower())
    return " ".join(text.split())


def key_fact_accuracy(answer: str, key_facts: list[list[str]]) -> float | None:
    if not key_facts:
        return None
    a = _fold(answer)
    return sum(1 for alts in key_facts if any(_fold(x) in a for x in alts)) / len(key_facts)


def abstention_correct(abstained: bool, answerable: bool) -> bool:
    return abstained != answerable


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    idx = min(len(v) - 1, max(0, round(p / 100 * (len(v) - 1))))
    return round(v[idx], 2)


def mean(values: list[float | None]) -> float | None:
    xs = [x for x in values if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


# --------------------------------------------------------------------------- faithfulness judge
class Faithfulness(BaseModel):
    claims: int = Field(ge=0)
    supported: int = Field(ge=0)
    unsupported_examples: list[str] = Field(default_factory=list)
    verdict: Literal["faithful", "mostly", "unfaithful"] = "faithful"


FAITH_SYSTEM = """You grade whether an answer is faithful to the evidence it was given.
Count the factual claims in the answer, and how many of them the evidence states (ignore citation markers).
Outside knowledge does not count as support, even if true.
Return ONLY JSON: {"claims": <int>, "supported": <int>, "unsupported_examples": ["..."],
 "verdict": "faithful" | "mostly" | "unfaithful"}"""


def faithfulness_messages(answer: str, evidence: list[dict], max_chars: int = 1500) -> list[dict]:
    ev = "\n\n".join(f"[{c['doc_title']} {c['section']}]\n{c['text'][:max_chars]}" for c in evidence)
    return [{"role": "system", "content": FAITH_SYSTEM},
            {"role": "user", "content": f"Evidence:\n\n{ev}\n\nAnswer:\n{answer}"}]


def faithfulness_score(f: Faithfulness) -> float | None:
    if f.claims == 0:
        return None
    return min(1.0, f.supported / f.claims)
