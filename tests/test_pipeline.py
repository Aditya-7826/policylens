"""The full LangGraph pipeline with fake models: routing, abstention, citations, configs."""

from types import SimpleNamespace

from ingest.index import BM25Index
from ingest.store import ChunkStore
from rag.pipeline import CONFIGS, build_graph
from tests.fakes import FakeEmbedder, FakeReranker, FakeVectors, ScriptedLLM, mk_chunk

CHUNKS = [
    mk_chunk("gdpr::article-33::0", "In the case of a personal data breach, the controller shall notify the "
             "supervisory authority not later than 72 hours after having become aware of it.",
             section_title="Notification of a personal data breach to the supervisory authority"),
    mk_chunk("dpdp_act_2023::section-8::0", "In the event of a personal data breach, the Data Fiduciary shall "
             "give the Board and each affected Data Principal intimation of such breach.",
             doc="dpdp_act_2023", title="DPDP Act 2023", section="Section 8"),
    mk_chunk("nist_sp_800-53r5::ac-2::0", "Define and document the types of accounts allowed for use within "
             "the system and assign account managers.", doc="nist_sp_800-53r5", title="NIST SP 800-53r5",
             section="AC-2"),
]


def make_res(tmp_path, llm):
    store = ChunkStore(tmp_path / "c.sqlite")
    for doc in {c.doc_id for c in CHUNKS}:
        store.sync_document(doc, [c for c in CHUNKS if c.doc_id == doc])
    settings = SimpleNamespace(rerank_top_n=4, retrieve_top_k=5, abstain_threshold=0.45)
    return SimpleNamespace(
        settings=settings, store=store, vectors=FakeVectors(CHUNKS), bm25=BM25Index(CHUNKS),
        embedder=FakeEmbedder(), reranker=FakeReranker(), llm=llm, graph=None,
        doc_titles={"gdpr": "GDPR", "dpdp_act_2023": "DPDP Act 2023", "nist_sp_800-53r5": "NIST SP 800-53r5"},
    )


def run(tmp_path, question, llm, config="full"):
    g = build_graph(make_res(tmp_path, llm))
    return g.invoke({"question": question, "config": CONFIGS[config]})["result"]


def test_comparison_answer_is_cited_and_verified(tmp_path):
    llm = ScriptedLLM(
        answer="Under GDPR the controller shall notify the supervisory authority within 72 hours of a personal "
               "data breach [E1]. Under the DPDP Act the Data Fiduciary shall give the Board and each affected "
               "Data Principal intimation of the breach [E2]. The DPDP Act also requires a fine of 5 euros [E2].",
        verdicts=[{"claim": 1, "verdict": "supported"}, {"claim": 2, "verdict": "supported"},
                  {"claim": 3, "verdict": "unsupported"}],
        analysis={"query_type": "comparison", "sub_questions": [
            {"question": "What must a data controller do after a personal data breach under GDPR?", "regulations": ["gdpr"]},
            {"question": "What must a Data Fiduciary do after a personal data breach under DPDP?", "regulations": ["dpdp_act_2023", "made_up"]},
        ]})
    r = run(tmp_path, "Compare breach duties under GDPR and the DPDP Act", llm)
    assert r.query_type == "comparison" and len(r.sub_questions) == 2
    assert r.sub_questions[1]["regulations"] == ["dpdp_act_2023"]  # unknown doc id filtered out
    assert not r.abstained
    assert "[GDPR, Article 33, gdpr::article-33::0]" in r.answer
    assert "[DPDP Act 2023, Section 8, dpdp_act_2023::section-8::0]" in r.answer
    assert "5 euros" not in r.answer and r.flagged[0]["status"] == "unsupported"
    assert {c.chunk_id for c in r.citations} == {"gdpr::article-33::0", "dpdp_act_2023::section-8::0"}
    assert set(r.timings) >= {"analyze", "retrieve", "generate", "verify", "total"}


def test_out_of_corpus_question_abstains_before_generating(tmp_path):
    llm = ScriptedLLM(answer="should not be called")
    r = run(tmp_path, "What is the Brazilian LGPD fine for late filing?", llm)
    assert r.abstained and "don't have enough evidence" in r.answer
    assert ("chat", "large") not in llm.calls  # no tokens spent on the big model


def test_model_saying_insufficient_abstains(tmp_path):
    llm = ScriptedLLM(answer="INSUFFICIENT_EVIDENCE")
    r = run(tmp_path, "What must the controller do after a personal data breach?", llm)
    assert r.abstained and r.citations == []


def test_vector_baseline_skips_judge_and_bm25(tmp_path):
    llm = ScriptedLLM(answer="The controller shall notify the supervisory authority within 72 hours [E1].")
    r = run(tmp_path, "What must the controller do after a personal data breach?", llm, config="vector")
    assert not r.abstained and r.confidence is None
    assert r.retrieval[0]["counts"] == {"vector": 5} or set(r.retrieval[0]["counts"]) == {"vector"}
    assert ("Verdicts", "small") not in llm.calls and ("Analysis", "small") not in llm.calls


def test_named_provision_is_pinned_and_sides_interleave(tmp_path):
    llm = ScriptedLLM(answer="INSUFFICIENT_EVIDENCE", analysis={"query_type": "comparison", "sub_questions": [
        {"question": "accounts for systems", "regulations": ["nist_sp_800-53r5"]},
        {"question": "Article 33 breach", "regulations": ["gdpr"]},
    ]})
    g = build_graph(make_res(tmp_path, llm))
    out = g.invoke({"question": "Compare AC-2 and Article 33", "config": CONFIGS["full"]})
    assert out["evidence_ids"][:2] == ["nist_sp_800-53r5::ac-2::0", "gdpr::article-33::0"]
    assert "exact" in out["retrieval"][1]["top"][0]["ranks"]
