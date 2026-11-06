"""Search indexes over the chunk store.

VectorIndex - ChromaDB (persistent, cosine). Holds embeddings of chunk.context_text.
BM25Index   - rank_bm25 over the same text. Rebuilt in memory from the chunk store at
              startup (about a second for ~1k chunks), so it can never drift out of sync.
"""

from __future__ import annotations

import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ingest.chunk import Chunk

COLLECTION = "chunks"


class VectorIndex:
    def __init__(self, path: Path, collection: str = COLLECTION):
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]):
        for i in range(0, len(chunks), 500):
            batch, vecs = chunks[i:i + 500], embeddings[i:i + 500]
            self.col.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vecs,
                documents=[c.context_text for c in batch],
                metadatas=[{"doc_id": c.doc_id, "section": c.section, "page": c.page or -1} for c in batch],
            )

    def delete(self, chunk_ids: list[str]):
        if chunk_ids:
            self.col.delete(ids=chunk_ids)

    def search(self, embedding: list[float], k: int, doc_ids: list[str] | None = None) -> list[tuple[str, float]]:
        if self.col.count() == 0:
            return []
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        res = self.col.query(query_embeddings=[embedding], n_results=min(k, self.col.count()), where=where,
                             include=["distances"])
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def count(self) -> int:
        return self.col.count()


_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOPWORDS = set(
    "a an and are as at be by for from has have in into is it its of on or such that the their them "
    "then there these this those to was were which while who will with within without shall may any "
    "what how does do under after before about".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps identifiers like 'ac-2', 'gv.oc-01' and '72' intact."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.ids = [c.chunk_id for c in chunks]
        self.doc_of = {c.chunk_id: c.doc_id for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.context_text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if doc_ids and self.doc_of[self.ids[i]] not in doc_ids:
                continue
            out.append((self.ids[i], float(scores[i])))
            if len(out) == k:
                break
        return out
