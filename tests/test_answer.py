from rag.answer import confidence, render, split_sentences, verify
from rag.retrieve import rrf_fuse
from tests.fakes import FakeEmbedder, mk_chunk

C1 = mk_chunk("gdpr::article-33::0", "In the case of a personal data breach, the controller shall notify the "
              "supervisory authority not later than 72 hours after having become aware of it.",
              section_title="Notification of a personal data breach")
C2 = mk_chunk("dpdp_act_2023::section-8::0", "In the event of a personal data breach, the Data Fiduciary shall "
              "give the Board and each affected Data Principal intimation of such breach.",
              doc="dpdp_act_2023", title="DPDP Act 2023", section="Section 8")


def test_split_sentences_keeps_handles_and_bullets():
    ans = ("Under GDPR the controller must notify within 72 hours [E1]. The DPDP Act requires intimation "
           "to the Board [E2][E1].\n- Key difference: no fixed deadline in DPDP [E2].\nIn summary:")
    s = split_sentences(ans)
    assert [x.handles for x in s] == [[1], [1, 2], [2], []]
    assert s[0].text == "Under GDPR the controller must notify within 72 hours."
    assert s[2].prefix == "- " and s[2].line == 1


def test_rrf_prefers_agreement():
    fused = rrf_fuse({"vector": ["a", "b", "c"], "bm25": ["b", "d"], "graph": ["b", "a"]})
    assert [c.chunk_id for c in fused][:2] == ["b", "a"]
    assert fused[0].sources == ["bm25", "graph", "vector"]


def test_verify_with_judge_drops_unsupported():
    ans = ("The controller shall notify the supervisory authority within 72 hours [E1]. "
           "The Data Fiduciary must pay a fine of one million euros [E2]. Overview:")
    sents = split_sentences(ans)
    vr = verify(sents, [C1, C2], FakeEmbedder(),
                llm=_Judge([{"claim": 1, "verdict": "supported"}, {"claim": 2, "verdict": "unsupported"}]))
    assert vr.judge_used
    assert [s.status for s in sents] == ["supported", "unsupported", "connective"]
    out = render(sents, [C1, C2])
    assert "[GDPR, Article 33, gdpr::article-33::0]" in out and "million" not in out
    assert vr.supported_rate == 0.5


def test_verify_without_judge_uses_similarity():
    sents = split_sentences("The controller shall notify the supervisory authority not later than 72 hours [E1]. "
                            "Bananas are yellow and grow in tropical places [E1].")
    verify(sents, [C1], FakeEmbedder(), llm=None)
    assert sents[0].status == "supported" and sents[1].status == "unsupported"


def test_citing_missing_evidence_is_unsupported():
    sents = split_sentences("The controller must notify within 72 hours [E9].")
    verify(sents, [C1], FakeEmbedder(), llm=None)
    assert sents[0].status == "unsupported"


def test_uncited_fact_is_flagged():
    sents = split_sentences("Controllers must notify regulators within 72 hours of a breach.")
    vr = verify(sents, [C1], FakeEmbedder(), llm=None)
    assert sents[0].status == "uncited" and vr.flagged


def test_confidence_takes_weakest_subquestion():
    assert confidence([[0.9, 0.8], [0.2, 0.1]], None) == 0.15
    assert confidence([[0.9, 0.8]], 1.0) == round(0.5 * 0.85 + 0.5, 3)


class _Judge:
    def __init__(self, verdicts):
        self.verdicts = verdicts

    def chat_json(self, messages, schema, **kw):
        return schema.model_validate({"verdicts": self.verdicts})


def test_named_provisions():
    from rag.retrieve import named_provisions

    assert named_provisions("What does NIST SP 800-53 CP-9 require, and how is GV.SC related?") == ["CP-9", "GV.SC"]
    assert named_provisions("Compare Article 33 GDPR with section 8 of the DPDP Act") == ["Article 33", "Section 8"]
    assert named_provisions("What is a personal data breach?") == []
