import json

from evals.metrics import (
    Faithfulness, abstention_correct, faithfulness_score, key_fact_accuracy, mrr, percentile, recall_at_k,
)
from evals.report import summarize, to_markdown
from evals.run import load_gold

EV = [{"chunk_id": "gdpr::article-6::0", "section": "Article 6"},
      {"chunk_id": "gdpr::article-33::0", "section": "Article 33"},
      {"chunk_id": "gdpr::article-34::1", "section": "Article 34"}]


def test_retrieval_metrics():
    assert recall_at_k(EV, ["gdpr::Article 33", "gdpr::Article 83"]) == 0.5
    assert mrr(EV, ["gdpr::Article 33"]) == 0.5
    assert mrr(EV, ["gdpr::Article 99"]) == 0.0
    assert recall_at_k(EV, []) is None and mrr(EV, []) is None
    assert recall_at_k(EV, ["gdpr::Article 34"], k=2) == 0.0


def test_key_facts_and_abstention():
    assert key_fact_accuracy("Notify within 72 Hours of the breach", [["72 hours"], ["Board"]]) == 0.5
    assert key_fact_accuracy("up to 250 crore", [["two hundred and fifty crore", "250 crore"]]) == 1.0
    # LLM output uses non-breaking hyphens / NBSPs; they must still match plain gold strings
    assert key_fact_accuracy("a supply‑chain risk program", [["supply chain"]]) == 1.0
    assert key_fact_accuracy("EUR 10 000 000", [["10 000 000"]]) == 1.0
    assert abstention_correct(abstained=True, answerable=False)
    assert not abstention_correct(abstained=True, answerable=True)


def test_faithfulness_and_percentiles():
    assert faithfulness_score(Faithfulness(claims=4, supported=3)) == 0.75
    assert faithfulness_score(Faithfulness(claims=0, supported=0)) is None
    assert percentile([1, 2, 3, 4, 100], 50) == 3 and percentile([1, 2, 3, 4, 100], 95) == 100


def test_gold_set_shape():
    gold = load_gold()
    assert len(gold) >= 40
    types = {g["type"] for g in gold}
    assert types == {"simple", "multi_hop", "comparison", "unanswerable"}
    assert all(g["expected"] == [] for g in gold if g["type"] == "unanswerable")
    assert all(g["expected"] and g["key_facts"] for g in gold if g["type"] != "unanswerable")
    assert len({g["id"] for g in gold}) == len(gold)


def test_report_table():
    rows = [
        {"id": "s1", "type": "simple", "config": "full", "abstained": False, "abstention_correct": True,
         "recall@5": 1.0, "mrr": 1.0, "key_facts": 1.0, "citation_precision": 0.9, "faithfulness": 1.0,
         "latency": 3.0},
        {"id": "u1", "type": "unanswerable", "config": "full", "abstained": True, "abstention_correct": True,
         "recall@5": None, "mrr": None, "key_facts": None, "citation_precision": None, "faithfulness": None,
         "latency": 1.0},
    ]
    s = summarize(rows)
    assert s["full"]["abstention_accuracy"] == 1.0 and s["full"]["recall@5"] == 1.0
    md = to_markdown(s)
    assert "| Recall@5 | 1.00 |" in md and json.dumps(s)
