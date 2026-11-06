"""Aggregate an eval run into report.md + summary.json.

    uv run python -m evals.report evals/results/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.metrics import mean, percentile

METRICS = [
    ("recall@5", "Recall@5"), ("mrr", "MRR"), ("key_facts", "Key-fact acc."),
    ("citation_precision", "Citation precision"), ("faithfulness", "Faithfulness"),
]


def summarize(rows: list[dict]) -> dict:
    by_cfg: dict[str, list[dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    summary = {}
    for cfg, rs in by_cfg.items():
        answerable = [r for r in rs if r["type"] != "unanswerable"]
        unanswerable = [r for r in rs if r["type"] == "unanswerable"]
        lat = [r["latency"] for r in rs if r["latency"] is not None]
        s = {name: mean([r[name] for r in answerable]) for name, _ in METRICS}
        s.update({
            "n": len(rs),
            "abstain_unanswerable": mean([1.0 if r["abstained"] else 0.0 for r in unanswerable]),
            "answer_rate_answerable": mean([0.0 if r["abstained"] else 1.0 for r in answerable]),
            "abstention_accuracy": mean([1.0 if r["abstention_correct"] else 0.0 for r in rs]),
            "latency_p50": percentile(lat, 50), "latency_p95": percentile(lat, 95),
            "by_type": {t: {name: mean([r[name] for r in rs if r["type"] == t]) for name in ("recall@5", "mrr", "key_facts")}
                        for t in ("simple", "multi_hop", "comparison")},
        })
        summary[cfg] = s
    return summary


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.2f}" if isinstance(x, float) else str(x)


def to_markdown(summary: dict) -> str:
    cfgs = [c for c in ("vector", "hybrid", "full") if c in summary] + \
           [c for c in summary if c not in ("vector", "hybrid", "full")]
    head = "| Metric | " + " | ".join(cfgs) + " |\n|---|" + "---|" * len(cfgs) + "\n"
    rows = [(label, key) for key, label in METRICS] + [
        ("Abstains on unanswerable", "abstain_unanswerable"), ("Answers answerable", "answer_rate_answerable"),
        ("Abstention accuracy (all)", "abstention_accuracy"),
        ("Latency p50 (s)", "latency_p50"), ("Latency p95 (s)", "latency_p95"), ("Questions", "n"),
    ]
    body = "".join(f"| {label} | " + " | ".join(_fmt(summary[c].get(key)) for c in cfgs) + " |\n" for label, key in rows)
    by_type = "\n**Recall@5 by question type**\n\n| Type | " + " | ".join(cfgs) + " |\n|---|" + "---|" * len(cfgs) + "\n"
    for t in ("simple", "multi_hop", "comparison"):
        by_type += f"| {t} | " + " | ".join(_fmt(summary[c]["by_type"][t]["recall@5"]) for c in cfgs) + " |\n"
    return head + body + by_type


def write_report(out: Path) -> str:
    rows = [json.loads(line)["score"] for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = to_markdown(summary)
    (out / "report.md").write_text(md, encoding="utf-8")
    return md


def rescore_key_facts(out: Path) -> int:
    """Recompute the deterministic key-fact score from saved answers (no LLM calls), e.g. after a matcher fix."""
    from evals.metrics import key_fact_accuracy
    from evals.run import load_gold

    gold = {g["id"]: g for g in load_gold()}
    path = out / "results.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for rec in records:
        score, result = rec["score"], rec["result"]
        if score.get("key_facts") is None or result.get("abstained"):
            continue
        new = key_fact_accuracy(result["answer"], gold[score["id"]]["key_facts"])
        changed += new != score["key_facts"]
        score["key_facts"] = new
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return changed


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    if "--rescore-facts" in sys.argv:
        print(f"key facts rescored: {rescore_key_facts(run_dir)} changed")
    print(write_report(run_dir))
