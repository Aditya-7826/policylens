"""The query pipeline as a LangGraph StateGraph.

    analyze -> retrieve -> (enough evidence?) -> generate -> verify -> finalize
                                  \\-- no ----------------------------> abstain

`PipelineConfig` switches parts off, which is how the evals compare
vector-only RAG, vector + BM25, and the full hybrid GraphRAG pipeline.
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterator, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core.llm import LLMUnavailable
from rag.analyze import analyze
from rag.answer import (
    INSUFFICIENT, answer_messages, confidence, render, split_sentences, verify,
)
from rag.resources import Resources, get_resources
from rag.retrieve import rerank, retrieve

# If even the best reranked chunk scores below this, don't spend tokens generating.
PRE_ABSTAIN_SCORE = 0.08


class PipelineConfig(BaseModel):
    name: str = "full"
    decompose: bool = True
    use_bm25: bool = True
    use_graph: bool = True
    use_exact: bool = True  # look up provisions named in the question ("CP-9", "Article 33")
    use_rerank: bool = True
    use_verify: bool = True
    abstain: bool = True  # confidence-based abstention (the model can always say INSUFFICIENT_EVIDENCE)


CONFIGS = {
    "vector": PipelineConfig(name="vector", decompose=False, use_bm25=False, use_graph=False, use_exact=False,
                             use_rerank=False, use_verify=False, abstain=False),
    "hybrid": PipelineConfig(name="hybrid", decompose=False, use_bm25=True, use_graph=False, use_exact=False,
                             use_rerank=False, use_verify=False, abstain=False),
    "full": PipelineConfig(name="full"),
}


class Citation(BaseModel):
    handle: int
    chunk_id: str
    doc_title: str
    section: str
    section_title: str
    page: int | None
    text: str
    rerank: float | None = None
    sources: list[str] = Field(default_factory=list)


class AskResult(BaseModel):
    question: str
    config: str
    answer: str
    abstained: bool
    confidence: float | None
    query_type: str
    sub_questions: list[dict]
    citations: list[Citation]          # evidence actually cited in the final answer
    evidence: list[Citation]           # everything shown to the model
    sentences: list[dict]              # per-sentence verification results
    flagged: list[dict]                # removed sentences and why
    retrieval: list[dict]              # per sub-question trace
    timings: dict[str, float]
    notes: list[str] = Field(default_factory=list)


class State(TypedDict, total=False):
    question: str
    config: PipelineConfig
    analysis: Any
    evidence_ids: list[str]
    evidence_scores: list[list[float]]
    candidates_meta: dict[str, dict]
    retrieval: list[dict]
    draft: str
    verify: Any
    result: AskResult
    timings: dict[str, float]
    notes: list[str]


def _timed(state: State, key: str, t0: float) -> dict[str, float]:
    t = dict(state.get("timings", {}))
    t[key] = round(time.time() - t0, 2)
    return t


def build_graph(res: Resources | None = None):
    res = res or get_resources()
    s = res.settings

    def analyze_node(state: State) -> dict:
        t0 = time.time()
        cfg = state["config"]
        a = analyze(state["question"], res.doc_titles, llm=res.llm, decompose=cfg.decompose)
        return {"analysis": a, "timings": _timed(state, "analyze", t0), "notes": []}

    def retrieve_node(state: State) -> dict:
        t0 = time.time()
        cfg, a = state["config"], state["analysis"]
        n_sub = len(a.sub_questions)
        per_sub = s.rerank_top_n if n_sub == 1 else max(3, math.ceil(s.rerank_top_n / n_sub) + 1)
        evidence_ids: list[str] = []
        evidence_scores: list[list[float]] = []
        meta: dict[str, dict] = {}
        traces = []
        picks: list[list] = []
        for sq in a.sub_questions:
            cands, trace = retrieve(res, sq.question, sq.regulations or None, s.retrieve_top_k,
                                    use_bm25=cfg.use_bm25, use_graph=cfg.use_graph, use_exact=cfg.use_exact)
            if cfg.use_rerank:
                cands = rerank(res, sq.question, cands)
            # A provision the user names explicitly ("CP-9", "Article 33") is always shown to the model.
            named = [c for c in cands if "exact" in c.ranks][:per_sub]
            chosen = (named + [c for c in cands if "exact" not in c.ranks])[:per_sub]
            evidence_scores.append([c.rerank for c in chosen if c.rerank is not None])
            picks.append(chosen)
            traces.append({
                "question": sq.question, "regulations": sq.regulations,
                "counts": trace.get("counts", {}), "graph": trace.get("graph"),
                "top": [{"chunk_id": c.chunk_id, "ranks": c.ranks, "rrf": round(c.rrf, 4),
                         "rerank": None if c.rerank is None else round(c.rerank, 3),
                         "graph_path": c.graph_path} for c in cands[:8]],
            })
        # Interleave sub-questions (1st of each, then 2nd of each, ...) so a comparison shows the
        # model both sides with equal prominence instead of all of side A before side B.
        for rank in range(max((len(p) for p in picks), default=0)):
            for p in picks:
                if rank < len(p) and p[rank].chunk_id not in meta:
                    c = p[rank]
                    evidence_ids.append(c.chunk_id)
                    meta[c.chunk_id] = {"rerank": c.rerank, "sources": c.sources, "graph_path": c.graph_path}
        return {"evidence_ids": evidence_ids[:10], "evidence_scores": evidence_scores, "candidates_meta": meta,
                "retrieval": traces, "timings": _timed(state, "retrieve", t0)}

    def route_after_retrieve(state: State) -> str:
        cfg = state["config"]
        if not state["evidence_ids"]:
            return "abstain"
        if cfg.abstain and cfg.use_rerank:
            best = max((x for sc in state["evidence_scores"] for x in sc), default=0.0)
            if best < PRE_ABSTAIN_SCORE:
                return "abstain"
        return "generate"

    def generate_node(state: State) -> dict:
        t0 = time.time()
        chunks = _chunks(state)
        a = state["analysis"]
        msgs = answer_messages(state["question"], [q.question for q in a.sub_questions], chunks)
        notes = list(state.get("notes", []))
        try:
            draft = res.llm.chat(msgs, size="large", max_tokens=1600).text.strip()  # includes hidden reasoning
        except LLMUnavailable as exc:
            draft = INSUFFICIENT
            notes.append(f"LLM unavailable: {exc}")
        return {"draft": draft, "notes": notes, "timings": _timed(state, "generate", t0)}

    def verify_node(state: State) -> dict:
        t0 = time.time()
        sentences = split_sentences(state["draft"])
        vr = verify(sentences, _chunks(state), res.embedder, llm=res.llm,
                    use_judge=state["config"].use_verify)
        return {"verify": vr, "timings": _timed(state, "verify", t0)}

    def finalize_node(state: State) -> dict:
        cfg = state["config"]
        chunks = _chunks(state)
        draft = state.get("draft", "")
        notes = list(state.get("notes", []))
        vr = state.get("verify")
        model_abstained = draft.strip().startswith(INSUFFICIENT) or not draft
        rate = vr.supported_rate if (vr and cfg.use_verify) else None
        conf = confidence(state["evidence_scores"], rate) if cfg.use_rerank else None
        sentences = vr.sentences if vr else []
        drop = cfg.use_verify
        answer = render(sentences, chunks, drop_flagged=drop) if sentences else ""
        abstained = model_abstained or not answer.strip()
        if not abstained and cfg.abstain and conf is not None and conf < s.abstain_threshold:
            abstained = True
            notes.append(f"confidence {conf:.2f} is below the threshold {s.abstain_threshold:.2f}")
        cited = sorted({h for st in sentences for h in st.handles
                        if not (drop and st.status in ("unsupported", "uncited"))})
        result = _result(state, answer if not abstained else abstain_text(chunks), abstained, conf,
                         sentences, cited, notes)
        return {"result": result}

    def abstain_node(state: State) -> dict:
        notes = list(state.get("notes", [])) + ["retrieved evidence scored too low to answer"]
        conf = confidence(state.get("evidence_scores", []), None) if state["config"].use_rerank else None
        return {"result": _result(state, abstain_text(_chunks(state)), True, conf, [], [], notes)}

    def _chunks(state: State):
        got = res.store.get(state["evidence_ids"])
        return [got[c] for c in state["evidence_ids"] if c in got]

    def _result(state, answer, abstained, conf, sentences, cited, notes) -> AskResult:
        chunks = _chunks(state)
        meta = state.get("candidates_meta", {})

        def cit(i, c):
            m = meta.get(c.chunk_id, {})
            return Citation(handle=i, chunk_id=c.chunk_id, doc_title=c.doc_title, section=c.section,
                            section_title=c.section_title, page=c.page, text=c.text,
                            rerank=None if m.get("rerank") is None else round(m["rerank"], 3),
                            sources=m.get("sources", []))
        evidence = [cit(i + 1, c) for i, c in enumerate(chunks)]
        a = state["analysis"]
        timings = dict(state.get("timings", {}))
        timings["total"] = round(sum(timings.values()), 2)
        return AskResult(
            question=state["question"], config=state["config"].name, answer=answer, abstained=abstained,
            confidence=conf, query_type=a.query_type,
            sub_questions=[{"question": q.question, "regulations": q.regulations} for q in a.sub_questions],
            citations=[e for e in evidence if e.handle in cited] if not abstained else [],
            evidence=evidence,
            sentences=[{"text": x.text, "handles": x.handles, "status": x.status, "similarity": x.similarity,
                        "judge": x.judge} for x in sentences],
            flagged=[{"text": x.text, "status": x.status, "similarity": x.similarity, "judge": x.judge}
                     for x in sentences if x.status in ("unsupported", "uncited")],
            retrieval=state.get("retrieval", []), timings=timings, notes=notes,
        )

    g = StateGraph(State)
    g.add_node("analyze", analyze_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    g.add_node("finalize", finalize_node)
    g.add_node("abstain", abstain_node)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"generate": "generate", "abstain": "abstain"})
    g.add_edge("generate", "verify")
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("abstain", END)
    return g.compile()


def abstain_text(chunks) -> str:
    found = "; ".join(f"{c.doc_title} {c.section}" for c in chunks[:5]) or "nothing relevant"
    return ("I don't have enough evidence in the indexed documents to answer this reliably. "
            f"Closest material found: {found}.")


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def ask(question: str, config: PipelineConfig | str = "full") -> AskResult:
    cfg = CONFIGS[config] if isinstance(config, str) else config
    out = get_graph().invoke({"question": question, "config": cfg})
    return out["result"]


STAGE_LABELS = {
    "analyze": "Analysed the question",
    "retrieve": "Retrieved and reranked evidence",
    "generate": "Drafted an answer",
    "verify": "Verified citations",
    "finalize": "Done",
    "abstain": "Not enough evidence",
}


def stream_ask(question: str, config: str = "full") -> Iterator[tuple[str, dict]]:
    """Yields (event, data) after each pipeline stage, then ("result", AskResult dict)."""
    cfg = CONFIGS.get(config, CONFIGS["full"])
    for update in get_graph().stream({"question": question, "config": cfg}, stream_mode="updates"):
        for node, delta in update.items():
            info: dict = {"stage": node, "label": STAGE_LABELS.get(node, node)}
            if node == "analyze" and delta.get("analysis"):
                a = delta["analysis"]
                info["query_type"] = a.query_type
                info["sub_questions"] = [q.question for q in a.sub_questions]
            if node == "retrieve":
                info["evidence"] = len(delta.get("evidence_ids", []))
            yield "stage", info
            if "result" in delta:
                yield "result", delta["result"].model_dump()
