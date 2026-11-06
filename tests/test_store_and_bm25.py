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
