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


if __name__ == "__main__":
    print(write_report(Path(sys.argv[1])))
