"""Run the gold set through one or more pipeline configurations.

    uv run python -m evals.run                         # all 3 configs, all questions
    uv run python -m evals.run --configs full --ids c01,s03
    uv run python -m evals.run --out evals/results/my-run   # resume an earlier run

Resumable: each (config, question) result is appended to <out>/results.jsonl as soon as it
finishes, and finished pairs are skipped on the next run. Together with the LLM cache this
means an eval that hits the free-tier daily limit simply continues the next day.
Writes <out>/report.md and <out>/summary.json at the end (see evals.report).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from core.config import ROOT
from evals.metrics import (
    Faithfulness, abstention_correct, faithfulness_messages, faithfulness_score, key_fact_accuracy, mrr,
    recall_at_k,
)

GOLD = ROOT / "evals" / "gold.jsonl"
RESULTS = ROOT / "evals" / "results"


def load_gold(ids: list[str] | None = None) -> list[dict]:
    items = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [g for g in items if not ids or g["id"] in ids]


def judge_baseline_citations(result: dict, llm) -> float | None:
    """Baselines keep every sentence and never judged them, so run the pipeline's own judge
    prompt over their cited sentences. (The full config already has verdicts: see score_one.)"""
    from ingest.chunk import Chunk
    from rag.answer import Sentence, Verdicts, judge_messages

    cited = [s for s in result["sentences"] if s["handles"]]
    if not cited:
        return None
    chunks = [Chunk(chunk_id=e["chunk_id"], doc_id=e["chunk_id"].split("::")[0], doc_title=e["doc_title"],
                    section=e["section"], section_title=e["section_title"], page=e["page"], text=e["text"],
                    n_tokens=0, content_hash="") for e in result["evidence"]]
    claims = [(i + 1, Sentence(s["text"], s["handles"], 0)) for i, s in enumerate(cited)]
    v = llm.chat_json(judge_messages(claims, chunks), Verdicts, size="small", max_tokens=600)
    score = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
    got = {x.claim: score[x.verdict] for x in v.verdicts}
    return round(sum(got.get(i, 0.0) for i, _ in claims) / len(claims), 3)


def score_one(gold: dict, result: dict, llm, cfg_name: str) -> dict:
    answerable = gold["type"] != "unanswerable"
    ev = result["evidence"]
    row = {
        "id": gold["id"], "type": gold["type"], "config": cfg_name,
        "abstained": result["abstained"], "confidence": result["confidence"],
        "abstention_correct": abstention_correct(result["abstained"], answerable),
        "recall@5": recall_at_k(ev, gold["expected"]), "mrr": mrr(ev, gold["expected"]),
        "latency": result["timings"].get("total"),
        "key_facts": None, "citation_precision": None, "faithfulness": None,
        "n_flagged": len(result["flagged"]), "notes": result.get("notes", []),
    }
    if answerable and not result["abstained"]:
        row["key_facts"] = key_fact_accuracy(result["answer"], gold["key_facts"])
        try:
            # Citation precision over the sentences the user actually sees.
            if cfg_name == "full" or any(s["judge"] for s in result["sentences"]):
                shown = [s for s in result["sentences"] if s["handles"] and s["status"] not in ("unsupported", "uncited")]
                val = {"supported": 1.0, "partial": 0.5}
                row["citation_precision"] = (round(sum(val.get(s["status"], 0) for s in shown) / len(shown), 3)
                                             if shown else None)
            else:
                row["citation_precision"] = judge_baseline_citations(result, llm)
            f = llm.chat_json(faithfulness_messages(result["answer"], ev), Faithfulness, size="small",
                              max_tokens=400)
            row["faithfulness"] = faithfulness_score(f)
        except Exception as exc:  # judge unavailable: keep the deterministic metrics
            row["notes"] = row["notes"] + [f"judge failed: {exc}"]
    return row


def main():
    # Batch job: sleep through per-minute rate limits (up to 2 min) instead of giving up.
    os.environ.setdefault("LLM_MAX_WAIT_SECONDS", "120")
    ap = argparse.ArgumentParser(description="Run the PolicyLens gold-set evaluation")
    ap.add_argument("--configs", default="vector,hybrid,full")
    ap.add_argument("--ids", help="comma-separated gold ids (default: all)")
    ap.add_argument("--out", help="results directory (reuse one to resume)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "chromadb", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    from rag.pipeline import CONFIGS, ask
    from rag.resources import get_resources

    out = Path(args.out) if args.out else RESULTS / datetime.now().strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results_file = out / "results.jsonl"
    done = set()
    if results_file.exists():
        for line in results_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["score"]["config"], r["score"]["id"]))

    gold = load_gold(args.ids.split(",") if args.ids else None)
    configs = args.configs.split(",")
    llm = get_resources().llm
    total = len(gold) * len(configs)
    print(f"{len(gold)} questions x {len(configs)} configs -> {out}  ({len(done)} already done)")
    n = 0
    for cfg in configs:
        for g in gold:
            n += 1
            if (cfg, g["id"]) in done:
                continue
            t0 = time.time()
            result = ask(g["question"], CONFIGS[cfg]).model_dump()
            result["timings"]["wall"] = round(time.time() - t0, 2)
            score = score_one(g, result, llm, cfg)
            with results_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"score": score, "result": result}, ensure_ascii=False) + "\n")
            print(f"[{n}/{total}] {cfg:6} {g['id']} abstained={result['abstained']!s:5} "
                  f"R@5={score['recall@5']} MRR={score['mrr']} facts={score['key_facts']} "
                  f"{result['timings']['wall']}s")

    from evals.report import write_report

    print(write_report(out))


if __name__ == "__main__":
    main()
