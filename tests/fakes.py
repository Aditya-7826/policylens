"""Small offline stand-ins for the models, so pipeline tests run in milliseconds."""

import hashlib
import re

import numpy as np

from ingest.chunk import Chunk


def mk_chunk(cid, text, doc="gdpr", title="GDPR", section="Article 33", section_title=""):
    return Chunk(chunk_id=cid, doc_id=doc, doc_title=title, section=section, section_title=section_title, page=1,
                 text=text, n_tokens=len(text.split()), content_hash=hashlib.sha1(text.encode()).hexdigest()[:16])


_STOP = {"the", "for", "and", "what", "must", "with", "under", "after", "shall", "that", "this", "does"}


def _words(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 2 and w not in _STOP]


class FakeEmbedder:
    """Bag-of-words hashed into 256 dims: similar wording -> high cosine."""

    def _vec(self, t):
        v = np.zeros(256)
        for w in _words(t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return [self._vec(t).tolist() for t in texts]

    def embed_query(self, t):
        return self._vec(t).tolist()


class FakeReranker:
    def score(self, query, passages):
        q = set(_words(query))
        return [len(q & set(_words(p))) / max(1, len(q)) for p in passages]


class FakeVectors:
    def __init__(self, chunks):
        self.chunks, self.emb = chunks, FakeEmbedder()

    def search(self, embedding, k, doc_ids=None):
        e = np.array(embedding)
        scored = [(c.chunk_id, float(np.array(self.emb.embed_query(c.context_text)) @ e))
                  for c in self.chunks if not doc_ids or c.doc_id in doc_ids]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


class ScriptedLLM:
    """Answers by matching the system prompt: analysis, answer, judge."""

    def __init__(self, answer, verdicts=None, analysis=None):
        self.answer, self.verdicts, self.analysis = answer, verdicts, analysis
        self.calls = []

    def chat(self, messages, size="small", **kw):
        from types import SimpleNamespace

        self.calls.append(("chat", size))
        return SimpleNamespace(text=self.answer)

    def chat_json(self, messages, schema, size="small", **kw):
        self.calls.append((schema.__name__, size))
        if schema.__name__ == "Analysis":
            if self.analysis is None:
                raise RuntimeError("no analysis scripted")
            return schema.model_validate(self.analysis)
        if schema.__name__ == "Verdicts":
            if self.verdicts is None:
                raise RuntimeError("judge down")
            return schema.model_validate({"verdicts": self.verdicts})
        raise AssertionError(schema)
