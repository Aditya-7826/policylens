"""Everything the query pipeline needs, loaded once per process."""

from __future__ import annotations

import logging
import threading
from functools import cached_property

from core.config import get_settings

log = logging.getLogger(__name__)


class Resources:
    def __init__(self):
        self.settings = get_settings()
        self._graph_checked = False
        self._graph = None
        self._lock = threading.Lock()

    @cached_property
    def store(self):
        from ingest.pipeline import get_chunk_store

        return get_chunk_store()

    @cached_property
    def vectors(self):
        from ingest.index import VectorIndex

        return VectorIndex(self.settings.chroma_dir)

    @cached_property
    def bm25(self):
        from ingest.index import BM25Index

        return BM25Index(self.store.all())

    @cached_property
    def embedder(self):
        from core.models import get_embedder

        return get_embedder()

    @cached_property
    def reranker(self):
        from core.models import get_reranker

        return get_reranker()

    @cached_property
    def llm(self):
        from core.llm import get_llm

        return get_llm()

    @property
    def graph(self):
        """GraphStore, or None if Neo4j is not running (graph retrieval is then skipped)."""
        with self._lock:
            if not self._graph_checked:
                from ingest.graph import GraphStore

                g = GraphStore()
                if g.ping():
                    self._graph = g
                else:
                    log.warning("Neo4j not reachable at %s; graph retrieval disabled", self.settings.neo4j_uri)
                    g.close()
                self._graph_checked = True
            return self._graph

    @cached_property
    def doc_titles(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for c in self.store.all():
            out.setdefault(c.doc_id, c.doc_title)
        return out

    def refresh(self):
        """Drop cached indexes after ingestion so new chunks become searchable."""
        for name in ("bm25", "doc_titles", "vectors"):
            self.__dict__.pop(name, None)


_res: Resources | None = None


def get_resources() -> Resources:
    global _res
    if _res is None:
        _res = Resources()
    return _res
