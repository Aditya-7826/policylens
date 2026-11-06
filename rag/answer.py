"""Answer generation, citation verification and confidence.

Generation: the large model sees numbered evidence blocks [E1]..[En] and must end every
factual sentence with the handles it relies on, e.g. "... within 72 hours [E2]."
Code then maps each handle to a full citation [doc, section, chunk_id]. The model never
types a chunk ID, so it cannot invent one.

Verification: each cited sentence is checked against its cited chunks two ways:
  1. embedding similarity (local, free): best match between the sentence and the chunk's sentences;
  2. the small LLM as a judge (one batched call): supported / partial / unsupported.
Unsupported sentences are removed from the final answer and listed as flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ingest.chunk import Chunk

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

ANSWER_SYSTEM = """You answer compliance questions using ONLY the evidence blocks provided.
Rules:
- Every sentence that states a fact must end with the evidence handle(s) it relies on, like
  "The controller must notify the supervisory authority within 72 hours [E2]." or "[E1][E4]".
- Use only facts stated in the evidence. Do not use outside knowledge. Do not guess numbers or deadlines.
- For comparisons, cover each document separately, then state the key differences (each cited).
- If the evidence does not answer the question, reply with exactly: INSUFFICIENT_EVIDENCE
- If it answers only part, answer that part and say plainly which part the evidence does not cover.
- Be concise: short paragraphs or bullet points, no preamble."""


def evidence_block(i: int, chunk: Chunk, max_chars: int = 2200) -> str:
    text = chunk.text if len(chunk.text) <= max_chars else chunk.text[:max_chars] + " ..."
    return f"[E{i}] {chunk.header}\n{text}"


def answer_messages(question: str, sub_questions: list[str], chunks: list[Chunk]) -> list[dict]:
    blocks = "\n\n".join(evidence_block(i + 1, c) for i, c in enumerate(chunks))
    plan = ""
    if len(sub_questions) > 1:
        plan = "\nSub-questions to cover:\n" + "\n".join(f"- {q}" for q in sub_questions) + "\n"
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Evidence:\n\n{blocks}\n\nQuestion: {question}{plan}"},
    ]


# ------------------------------------------------------------------------ sentence parsing
_HANDLE = re.compile(r"\[(E\d+(?:\s*[,;]\s*E?\d+)*)\]")


@dataclass
class Sentence:
    text: str  # without handles
    handles: list[int]  # evidence numbers cited (1-based)
    line: int  # which line of the answer it came from (keeps bullets/paragraphs)
    prefix: str = ""  # bullet marker etc.
    status: str = "pending"  # supported | partial | unsupported | uncited | connective
    similarity: float | None = None
    judge: str | None = None


def split_sentences(answer: str) -> list[Sentence]:
    out: list[Sentence] = []
    for li, raw in enumerate(answer.split("\n")):
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^([-*•]|\d+[.)])\s+", line)
        prefix = m.group(0) if m else ""
        body = line[len(prefix):]
        # split at sentence ends, keeping trailing handles with the sentence before them
        parts, start = [], 0
        for mm in re.finditer(r"[.!?](?:\s*\[E[\d,;\sE]+\])*(?=\s+[A-Z(\"“]|\s*$)", body):
            parts.append(body[start:mm.end()].strip())
            start = mm.end()
        if body[start:].strip():
            parts.append(body[start:].strip())
        for pi, part in enumerate(parts):
            handles = []
            for grp in _HANDLE.findall(part):
                handles += [int(x) for x in re.findall(r"\d+", grp)]
            clean = re.sub(r"\s*" + _HANDLE.pattern, "", part).strip()
            if clean:
                out.append(Sentence(clean, sorted(set(handles)), li, prefix if pi == 0 else ""))
    return out


def is_connective(s: Sentence) -> bool:
    """Short framing sentences ("Key differences:") carry no facts and need no citation."""
    words = s.text.split()
    return len(words) <= 10 and not re.search(r"\d", s.text) and not re.search(
        r"\b(must|shall|required?|within|penalt|fine|hours|days)\b", s.text, re.I)


# --------------------------------------------------------------------------- verification
class Verdict(BaseModel):
    claim: int
    verdict: Literal["supported", "partial", "unsupported"]


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


JUDGE_SYSTEM = """You check whether claims are supported by their cited evidence.
For each claim answer "supported" (the evidence states it), "partial" (some of it is stated, some is not)
or "unsupported" (the evidence does not state it, or contradicts it). Judge only against the given evidence.
Return ONLY JSON: {"verdicts": [{"claim": <number>, "verdict": "supported" | "partial" | "unsupported"}]}"""


def judge_messages(claims: list[tuple[int, Sentence]], chunks: list[Chunk]) -> list[dict]:
    used = sorted({h for _, s in claims for h in s.handles if 1 <= h <= len(chunks)})
    ev = "\n\n".join(evidence_block(h, chunks[h - 1], max_chars=1800) for h in used)
    cl = "\n".join(f"{i}. {s.text} (cites {', '.join(f'E{h}' for h in s.handles)})" for i, s in claims)
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Evidence:\n\n{ev}\n\nClaims:\n{cl}"}]


def _chunk_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.;:])\s+|\n", text) if len(p.strip()) > 20]
    return parts or [text]


def similarity_support(embedder, sentence: str, chunks: list[Chunk]) -> float:
    """Best cosine similarity between the claim and any sentence (or pair of sentences) of the cited chunks."""
    import numpy as np

    pieces: list[str] = []
    for c in chunks:
        sents = _chunk_sentences(c.text)
        pieces += sents + [f"{a} {b}" for a, b in zip(sents, sents[1:])]
    if not pieces:
        return 0.0
    vecs = np.array(embedder.embed_documents([sentence] + pieces[:200]))
    return float((vecs[1:] @ vecs[0]).max())


@dataclass
class VerifyResult:
    sentences: list[Sentence]
    judge_used: bool
    supported_rate: float  # over factual sentences
    flagged: list[Sentence] = field(default_factory=list)


SIM_SUPPORT, SIM_REJECT = 0.72, 0.55


def verify(sentences: list[Sentence], chunks: list[Chunk], embedder, llm=None,
           use_judge: bool = True) -> VerifyResult:
    claims: list[tuple[int, Sentence]] = []
    for s in sentences:
        valid = [h for h in s.handles if 1 <= h <= len(chunks)]
        if not s.handles:
            s.status = "connective" if is_connective(s) else "uncited"
            continue
        if not valid:
            s.status = "unsupported"  # cites evidence that does not exist
            continue
        s.handles = valid
        s.similarity = round(similarity_support(embedder, s.text, [chunks[h - 1] for h in valid]), 3)
        claims.append((len(claims) + 1, s))

    judge_used = False
    if claims and use_judge and llm is not None:
        try:
            verdicts = llm.chat_json(judge_messages(claims, chunks), Verdicts, size="small", max_tokens=600)
            by_id = {v.claim: v.verdict for v in verdicts.verdicts}
            for i, s in claims:
                s.judge = by_id.get(i)
            judge_used = True
        except Exception:
            judge_used = False

    for _, s in claims:
        sim = s.similarity or 0.0
        if s.judge == "supported":
            s.status = "supported" if sim >= SIM_REJECT - 0.1 else "partial"
        elif s.judge == "partial":
            s.status = "partial" if sim >= SIM_REJECT else "unsupported"
        elif s.judge == "unsupported":
            s.status = "partial" if sim >= 0.9 else "unsupported"
        else:  # no judge: similarity alone
            s.status = "supported" if sim >= SIM_SUPPORT else "partial" if sim >= SIM_REJECT else "unsupported"

    factual = [s for s in sentences if s.status not in ("connective",)]
    good = sum(1.0 if s.status == "supported" else 0.5 if s.status == "partial" else 0.0 for s in factual)
    rate = good / len(factual) if factual else 0.0
    flagged = [s for s in sentences if s.status in ("unsupported", "uncited")]
    return VerifyResult(sentences, judge_used, rate, flagged)


def render(sentences: list[Sentence], chunks: list[Chunk], drop_flagged: bool = True) -> str:
    """Rebuild the answer with full [doc, section, chunk_id] citations, dropping flagged sentences."""
    lines: dict[int, list[str]] = {}
    prefixes: dict[int, str] = {}
    for s in sentences:
        if drop_flagged and s.status in ("unsupported", "uncited"):
            continue
        cites = "".join(chunks[h - 1].citation() for h in s.handles if 1 <= h <= len(chunks))
        text = s.text
        if cites:
            end = text[-1] if text[-1] in ".!?" else ""
            text = f"{text[:-1] if end else text} {cites}{end}"
        lines.setdefault(s.line, []).append(text)
        if s.prefix:
            prefixes[s.line] = s.prefix
    return "\n".join(prefixes.get(li, "") + " ".join(parts) for li, parts in sorted(lines.items()))


def confidence(evidence_scores: list[list[float]], verify_rate: float | None) -> float:
    """Retrieval strength = for each sub-question, mean of its top-2 reranker scores; take the weakest
    sub-question (a comparison is only as good as its worst-covered side). Blend with verification."""
    per_sub = []
    for scores in evidence_scores:
        top = sorted(scores, reverse=True)[:2]
        per_sub.append(sum(top) / len(top) if top else 0.0)
    retrieval = min(per_sub) if per_sub else 0.0
    if verify_rate is None:
        return round(retrieval, 3)
    return round(0.5 * retrieval + 0.5 * verify_rate, 3)
