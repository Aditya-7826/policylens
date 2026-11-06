"""Hybrid retrieval for one sub-question: vector + BM25 + graph, fused with RRF, then reranked.

Each retriever returns a ranked list of chunk IDs. Reciprocal Rank Fusion combines them
using ranks only (scores from different retrievers are not comparable):

    rrf(chunk) = sum over retrievers of 1 / (60 + rank)

The local cross-encoder then scores (question, chunk) pairs directly, which is slower but
much more accurate, so it only sees the fused top candidates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
RRF_K = 60


@dataclass
class Candidate:
    chunk_id: str
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank
    rrf: float = 0.0
    rerank: float | None = None
    graph_path: dict | None = None  # how the graph reached this chunk (for the UI)

    @property
    def sources(self) -> list[str]:
        return sorted(self.ranks)


def rrf_fuse(ranked_lists: dict[str, list[str]], k: int = RRF_K) -> list[Candidate]:
    cands: dict[str, Candidate] = {}
    for source, ids in ranked_lists.items():
        for rank, cid in enumerate(ids, start=1):
            c = cands.setdefault(cid, Candidate(cid))
            c.ranks[source] = rank
            c.rrf += 1.0 / (k + rank)
    return sorted(cands.values(), key=lambda c: c.rrf, reverse=True)


def _doc_of(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def graph_retrieve(graph, question: str, k: int, doc_ids: list[str] | None, hops: int = 2):
    """Find graph nodes matching the question (full-text), walk 1-2 hops, collect linked chunks.
    Chunk score = best (seed match score / (1 + hops)); closer and better-matching wins."""
    seeds = graph.search_nodes(question, k=8)
    if doc_ids:
        seeds = [s for s in seeds if s["key"].startswith("Entity:") or any(d in s["key"] for d in doc_ids)]
    if not seeds:
        return [], {"seeds": []}
    top = max(s["score"] for s in seeds) or 1.0
    seed_score = {s["key"]: s["score"] / top for s in seeds}
    best: dict[str, tuple[float, dict]] = {}
    for row in graph.expand(list(seed_score), hops=hops):
        cid = row["chunk_id"]
        if doc_ids and _doc_of(cid) not in doc_ids:
            continue
        score = seed_score.get(row["seed"], 0) / (1 + row["hops"])
        if cid not in best or score > best[cid][0]:
            best[cid] = (score, {"nodes": row["path_nodes"], "rels": row["path_rels"], "hops": row["hops"]})
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:k]
    return ranked, {"seeds": [{"name": s["name"], "label": s["labels"], "score": round(s["score"], 2)}
                              for s in seeds]}


_REF_PATTERNS = [
    (re.compile(r"\b([A-Z]{2}-\d{1,2})\b"), "{0}"),                 # 800-53 control: CP-9
    (re.compile(r"\b([A-Z]{2}\.[A-Z]{2})\b"), "{0}"),               # CSF category: GV.SC
    (re.compile(r"\bArticles?\s+(\d{1,3})\b", re.I), "Article {0}"),
    (re.compile(r"\b[Ss]ections?\s+(\d{1,3})\b"), "Section {0}"),
    (re.compile(r"\bRecitals?\s+(\d{1,3})\b", re.I), "Recital {0}"),
]


def named_provisions(question: str) -> list[str]:
    """Section labels the question names explicitly ("What does CP-9 require?" -> ["CP-9"])."""
    out: list[str] = []
    for pat, label in _REF_PATTERNS:
        for m in pat.finditer(question):
            lab = label.format(m.group(1))
            if lab not in out:
                out.append(lab)
    return out


def exact_retrieve(store, question: str, doc_ids: list[str] | None, per_section: int = 3) -> list[str]:
    """Chunks of provisions named in the question. Keyword and vector search are weak at this:
    "CP-9" appears in dozens of "Related Controls" lists, so BM25 ranks the control itself low."""
    ids: list[str] = []
    for label in named_provisions(question):
        ids += store.chunks_for_section(label, doc_ids)[:per_section]
    return ids[:6]


def retrieve(res, question: str, doc_ids: list[str] | None, k: int, use_bm25: bool = True,
             use_graph: bool = True, use_exact: bool = True) -> tuple[list[Candidate], dict]:
    """Run the enabled retrievers and fuse them. Returns (candidates, trace)."""
    lists: dict[str, list[str]] = {}
    trace: dict = {}

    if use_exact and (exact := exact_retrieve(res.store, question, doc_ids)):
        lists["exact"] = exact
        trace["exact"] = exact

    vec = res.vectors.search(res.embedder.embed_query(question), k, doc_ids)
    lists["vector"] = [cid for cid, _ in vec]

    if use_bm25:
        lists["bm25"] = [cid for cid, _ in res.bm25.search(question, k, doc_ids)]

    paths: dict[str, dict] = {}
    if use_graph:
        graph = res.graph
        if graph is None:
            trace["graph"] = "Neo4j unavailable"
        else:
            try:
                ranked, gtrace = graph_retrieve(graph, question, k, doc_ids)
                lists["graph"] = [cid for cid, _ in ranked]
                paths = {cid: info[1] for cid, info in ranked}
                trace["graph"] = gtrace
            except Exception as exc:  # graph problems must never break answering
                log.warning("graph retrieval failed: %s", exc)
                trace["graph"] = f"error: {exc}"

    cands = rrf_fuse(lists)
    for c in cands:
        c.graph_path = paths.get(c.chunk_id)
    trace["counts"] = {name: len(ids) for name, ids in lists.items()}
    return cands, trace


def rerank(res, question: str, cands: list[Candidate], max_candidates: int = 20) -> list[Candidate]:
    pool = cands[:max_candidates]
    pool += [c for c in cands[max_candidates:] if "exact" in c.ranks]  # named provisions always get scored
    chunks = res.store.get([c.chunk_id for c in pool])
    pool = [c for c in pool if c.chunk_id in chunks]
    scores = res.reranker.score(question, [chunks[c.chunk_id].context_text for c in pool])
    for c, s in zip(pool, scores):
        c.rerank = s
    return sorted(pool, key=lambda c: c.rerank, reverse=True)
