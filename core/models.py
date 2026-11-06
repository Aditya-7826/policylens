"""Local CPU models: embedder (bi-encoder) and reranker (cross-encoder).

Loaded lazily and once per process, because each load takes seconds and hundreds of MB.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from core.config import get_settings

_lock = threading.Lock()

# BGE models were trained with this instruction on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 64)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.model.encode([a, b], normalize_embeddings=True)
        return float(va @ vb)


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device="cpu", max_length=512)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance in [0, 1] (sigmoid of the cross-encoder logit)."""
        if not passages:
            return []
        import torch

        logits = self.model.predict([(query, p) for p in passages], batch_size=16,
                                    activation_fn=torch.nn.Identity(), convert_to_numpy=True)
        return [float(1 / (1 + pow(2.718281828, -x))) for x in logits]


@lru_cache
def get_embedder() -> Embedder:
    with _lock:
        return Embedder(get_settings().embed_model)


@lru_cache
def get_reranker() -> Reranker:
    with _lock:
        return Reranker(get_settings().rerank_model)
