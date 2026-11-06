import hashlib

from core.llm import LLMUnavailable
from ingest.chunk import Chunk
from ingest.graph import build_graph, lucene_query
from ingest.kg import Extraction, build_chunk_graph, find_references, node_key, normalize_name, worth_extracting
from ingest.store import ChunkStore


def mk(cid, text, doc="gdpr", section="Article 33", title="Notification"):
    return Chunk(chunk_id=cid, doc_id=doc, doc_title=doc.upper(), section=section, section_title=title, page=1,
                 text=text, n_tokens=len(text.split()), content_hash=hashlib.sha1(text.encode()).hexdigest()[:16])


BREACH = ("In the case of a personal data breach, the controller shall without undue delay and, where feasible, "
          "not later than 72 hours after having become aware of it, notify the personal data breach to the "
          "supervisory authority competent in accordance with Article 55, unless the personal data breach is "
          "unlikely to result in a risk to the rights and freedoms of natural persons. See also Articles 34 and 35.")


def test_normalize_name_merges_variants():
    assert normalize_name("The Data Fiduciaries") == normalize_name("data fiduciary") == "data fiduciary"
    assert normalize_name("“Controller”") == "controller"
    assert normalize_name("Data Protection Board of India") == "data protection board of india"


def test_entities_are_global_other_nodes_scoped():
    assert node_key("Entity", "Controller", "gdpr") == node_key("Entity", "the controllers", "dpdp")
    assert node_key("Right", "Right to erasure", "gdpr") != node_key("Right", "Right to erasure", "dpdp")


def test_find_references():
    c = mk("gdpr::article-33::0", BREACH)
    known = {"Article 33", "Article 34", "Article 35", "Article 55", "Article 99"}
    assert find_references(c, known) == {"Article 34", "Article 35", "Article 55"}
    nist = mk("n::ac-2::0", "Related Controls: AC-3, AU-6, ZZ-9.", doc="nist", section="AC-2")
    assert find_references(nist, {"AC-2", "AC-3", "AU-6"}) == {"AC-3", "AU-6"}


def test_build_chunk_graph_filters_bad_triples():
    raw = {
        "nodes": [
            {"name": "Notify supervisory authority of breach", "type": "obligation", "description": "within 72h"},
            {"name": "Controller", "type": "Entity"},
            {"name": "x", "type": "Entity"},                      # too short -> dropped
            {"name": "Something", "type": "Planet"},              # bad type -> dropped
        ],
        "relations": [
            {"source": "Notify supervisory authority of breach", "relation": "applies to", "target": "controller"},
            {"source": "Controller", "relation": "APPLIES_TO", "target": "Controller"},   # self loop
            {"source": "Controller", "relation": "PENALIZES", "target": "Notify supervisory authority of breach"},
        ],
    }
    ext = Extraction.model_validate(raw)
    assert [n.name for n in ext.nodes] == ["Notify supervisory authority of breach", "Controller"]
    g = build_chunk_graph(mk("gdpr::article-33::0", BREACH), ext, {"Article 33", "Article 55"})
    rels = {(e.source.split(":")[0], e.rel, e.target.split(":")[0]) for e in g.edges}
    assert ("Regulation", "CONTAINS", "gdpr") in rels                      # section key starts with doc id
    assert ("gdpr", "REQUIRES", "Obligation") in rels                       # section -> obligation
    assert ("gdpr", "REFERENCES", "gdpr") in rels                           # Article 33 -> Article 55
    assert ("Obligation", "APPLIES_TO", "Entity") in rels
    assert not any(e.rel == "PENALIZES" for e in g.edges)                  # Entity PENALIZES is not allowed


def test_prefilter():
    ok, _ = worth_extracting(mk("a", BREACH))
    assert ok
    short, why = worth_extracting(mk("b", "Article 99 Entry into force."))
    assert not short and "content words" in why
    withdrawn, _ = worth_extracting(mk("c", "[Withdrawn: Incorporated into AC-2.]"))
    assert not withdrawn


def test_lucene_query_escapes():
    q = lucene_query("What is AC-2 (account management) under GDPR?")
    assert "AC\\-2" in q and "account" in q and "(" not in q.replace("\\(", "")


class FakeGraph:
    def __init__(self):
        self.written, self.forgotten = [], []

    def ping(self): return True
    def setup(self): pass
    def forget_chunk(self, cid): self.forgotten.append(cid)
    def write(self, cg): self.written.append(cg)
    def prune_orphans(self): return 0
    def link_similar(self, progress=print): return 0
    def stats(self): return {}


class FakeLLM:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = 0, fail_after

    def chat_json(self, messages, schema, size, max_tokens):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise LLMUnavailable("budget reached")
        self.calls += 1
        return schema.model_validate({"nodes": [{"name": "Controller", "type": "Entity"}], "relations": []})


def _store(tmp_path, n=4):
    store = ChunkStore(tmp_path / "c.sqlite")
    chunks = [mk(f"gdpr::article-{i}::0", BREACH, section=f"Article {i}") for i in range(n)]
    chunks.append(mk("gdpr::article-99::0", "Entry into force.", section="Article 99"))
    store.sync_document("gdpr", chunks)
    return store


def test_build_graph_is_resumable_when_llm_runs_out(tmp_path):
    store, g = _store(tmp_path), FakeGraph()
    r1 = build_graph(store, graph=g, llm=FakeLLM(fail_after=2), progress=lambda *_: None, cooldown=0)
    assert r1["done"] == 2 and r1["skipped"] == 1 and r1["structure_only"] == 2
    assert len(g.written) == 5  # every chunk at least has its structure in the graph
    # second run: only the 2 structure-only chunks are left, and they get extracted
    r2 = build_graph(store, graph=g, llm=FakeLLM(), progress=lambda *_: None, cooldown=0)
    assert r2["done"] == 2 and r2["skipped"] == 0
    assert store.pending_kg() == []


def test_build_graph_respects_limit(tmp_path):
    store, g = _store(tmp_path), FakeGraph()
    llm = FakeLLM()
    r = build_graph(store, graph=g, llm=llm, limit=1, progress=lambda *_: None, cooldown=0)
    assert llm.calls == 1 and r["done"] == 1 and r["structure_only"] == 3


def test_structure_is_not_rewritten_when_llm_is_down(tmp_path):
    store, g = _store(tmp_path), FakeGraph()
    build_graph(store, graph=g, llm=FakeLLM(fail_after=0), progress=lambda *_: None, cooldown=0)
    first = len(g.written)
    assert first == 5  # every chunk gets its structure once
    r = build_graph(store, graph=g, llm=FakeLLM(fail_after=0), progress=lambda *_: None, cooldown=0)
    assert len(g.written) == first and r["structure_only"] == 4  # nothing rewritten the second time
    assert len(store.pending_kg()) == 4  # still queued for LLM extraction
