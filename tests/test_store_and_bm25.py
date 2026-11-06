from ingest.chunk import Chunk
from ingest.index import BM25Index, tokenize
from ingest.store import ChunkStore


def mk(cid, text, doc="d"):
    import hashlib

    return Chunk(chunk_id=cid, doc_id=doc, doc_title=doc.upper(), section="S", section_title="", page=1,
                 text=text, n_tokens=len(text.split()), content_hash=hashlib.sha1(text.encode()).hexdigest()[:16])


def test_sync_is_incremental(tmp_path):
    store = ChunkStore(tmp_path / "c.sqlite")
    first = store.sync_document("d", [mk("a", "one"), mk("b", "two")])
    assert sorted(first["added"]) == ["a", "b"]
    store.mark_embedded(store.pending_embedding())
    assert store.pending_embedding() == []

    # same content again: nothing to do
    again = store.sync_document("d", [mk("a", "one"), mk("b", "two")])
    assert again == {"added": [], "changed": [], "removed": []}

    # b changed, a removed, c added -> only b and c need embedding
    diff = store.sync_document("d", [mk("b", "two v2"), mk("c", "three")])
    assert diff == {"added": ["c"], "changed": ["b"], "removed": ["a"]}
    assert sorted(c.chunk_id for c in store.pending_embedding()) == ["b", "c"]


def test_kg_status_tracking(tmp_path):
    store = ChunkStore(tmp_path / "c.sqlite")
    store.sync_document("d", [mk("a", "one"), mk("b", "two")])
    a, b = store.pending_kg()
    store.mark_kg(a, "done")
    store.mark_kg(b, "error")
    assert [c.chunk_id for c in store.pending_kg()] == ["b"]  # errors are retried
    store.mark_kg(b, "skipped")
    assert store.pending_kg() == []


def test_file_level_skip(tmp_path):
    store = ChunkStore(tmp_path / "c.sqlite")
    assert not store.file_unchanged("d", "sha1", "v1")
    store.record_file("d", "d.pdf", "sha1", "v1")
    assert store.file_unchanged("d", "sha1", "v1")
    assert not store.file_unchanged("d", "sha1", "v2")  # chunker code changed


def test_tokenize_keeps_identifiers():
    assert tokenize("Under AC-2 and GV.OC-01, notify within 72 hours.") == \
        ["ac-2", "gv.oc-01", "notify", "72", "hours"]


def test_bm25_ranks_and_filters():
    chunks = [mk("x", "personal data breach notification 72 hours", "gdpr"),
              mk("y", "account management for information systems", "nist"),
              mk("z", "breach of the duties of a data principal", "dpdp")]
    idx = BM25Index(chunks)
    assert idx.search("data breach notification", 2)[0][0] == "x"
    assert [c for c, _ in idx.search("breach", 5, doc_ids=["dpdp"])] == ["z"]


def test_paged_iterators_and_batched_embedding(tmp_path):
    from ingest.pipeline import embed_pending

    store = ChunkStore(tmp_path / "c.sqlite")
    store.sync_document("d", [mk(f"c{i}", f"text number {i}") for i in range(10)])
    assert [len(p) for p in store.iter_all(batch=4)] == [4, 4, 2]
    assert store.doc_titles() == {"d": "D"} and store.sections_by_doc() == {"d": {"S"}}

    # pending kg pages: chunks left pending mid-run are not re-fetched (no endless loop)
    seen = [c.chunk_id for page in store.iter_pending_kg(batch=3) for c in page]
    assert len(seen) == 10 == len(set(seen))

    class FakeVindex:
        def __init__(self): self.batches = []
        def upsert(self, chunks, vecs): self.batches.append(len(chunks))

    import core.models as models

    class E:
        def embed_documents(self, texts): return [[0.0] for _ in texts]
    orig = models.get_embedder
    models.get_embedder = lambda: E()
    try:
        v = FakeVindex()
        assert embed_pending(store, v, batch_size=4, progress=lambda *_: None) == 10
        assert v.batches == [4, 4, 2] and store.count_pending_embedding() == 0
    finally:
        models.get_embedder = orig


def test_bm25_accepts_pages():
    # 3+ docs: with 2, BM25's IDF for a term in 1 of them is log(1.5/1.5) = 0
    pages = iter([[mk("x", "breach notification"), mk("z", "penalty schedule")], [mk("y", "account management")]])
    assert BM25Index(pages).search("account", 1)[0][0] == "y"
