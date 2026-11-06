"""Query analysis: classify the question and break it into sub-questions.

The small LLM does the work; a keyword fallback keeps the pipeline usable when the
LLM is unavailable. Regulation hints become per-sub-question document filters.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

QueryType = Literal["simple", "multi_hop", "comparison"]

# Words that point at a regulation in the corpus (used by the fallback and to sanity-check the LLM).
REG_HINTS = {
    "dpdp_act_2023": ["dpdp", "india", "indian", "data fiduciar", "data principal", "digital personal data"],
    "gdpr": ["gdpr", "general data protection regulation", "eu ", "european", "data subject", "supervisory authorit"],
    "nist_csf_2.0": ["csf", "cybersecurity framework"],
    "nist_sp_800-53r5": ["800-53", "sp 800", "security and privacy controls", "control family"],
}
COMPARE_WORDS = re.compile(r"\b(compare|comparison|versus|vs\.?|differ|difference|similar|both|contrast)\b", re.I)


class SubQuestion(BaseModel):
    question: str = Field(min_length=3)
    regulations: list[str] = Field(default_factory=list)


class Analysis(BaseModel):
    query_type: QueryType = "simple"
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    method: str = "llm"  # llm | heuristic | none


ANALYZE_SYSTEM = """You plan retrieval for a compliance question-answering system.
Available documents (use these ids): {docs}.
Return ONLY JSON: {{"query_type": "simple" | "multi_hop" | "comparison",
 "sub_questions": [{{"question": "...", "regulations": ["<doc id>", ...]}}]}}
- simple: one fact from one place -> exactly one sub-question (the question itself).
- multi_hop: needs facts from several provisions (e.g. a duty and its penalty) -> 2-3 sub-questions.
- comparison: compares documents -> one sub-question per document, each restricted to that document.
Set "regulations" only when the question names or clearly implies a document; else use [].
Keep sub-questions self-contained (repeat the subject, do not say "it")."""


def detect_regulations(text: str, available: set[str]) -> list[str]:
    t = f" {text.lower()} "
    return [d for d, hints in REG_HINTS.items() if d in available and any(h in t for h in hints)]


def heuristic_analysis(question: str, available: set[str] | dict[str, str]) -> Analysis:
    titles = available if isinstance(available, dict) else {d: d for d in available}
    regs = detect_regulations(question, set(titles))
    if COMPARE_WORDS.search(question) and len(regs) >= 2:
        subs = [SubQuestion(question=f"{question} (only the part about {titles[r]})", regulations=[r])
                for r in regs]
        return Analysis(query_type="comparison", sub_questions=subs, method="heuristic")
    return Analysis(query_type="simple", sub_questions=[SubQuestion(question=question, regulations=regs)],
                    method="heuristic")


def analyze(question: str, available_docs: dict[str, str], llm=None, decompose: bool = True) -> Analysis:
    available = set(available_docs)
    if not decompose:
        return Analysis(query_type="simple", sub_questions=[SubQuestion(question=question)], method="none")
    if llm is not None:
        docs = ", ".join(f"{d} ({t})" for d, t in available_docs.items())
        try:
            a = llm.chat_json(
                [{"role": "system", "content": ANALYZE_SYSTEM.format(docs=docs)},
                 {"role": "user", "content": question}],
                Analysis, size="small", max_tokens=400)
            subs = []
            for sq in a.sub_questions[:4]:
                regs = [r for r in sq.regulations if r in available]
                subs.append(SubQuestion(question=sq.question, regulations=regs))
            if subs:
                if a.query_type == "simple":  # keep the user's own wording for a single lookup
                    subs = [SubQuestion(question=question, regulations=subs[0].regulations)]
                return Analysis(query_type=a.query_type, sub_questions=subs, method="llm")
        except Exception:  # LLM unavailable or unusable output -> fall back
            pass
    return heuristic_analysis(question, available_docs)
