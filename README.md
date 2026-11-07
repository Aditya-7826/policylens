# PolicyLens

**A hybrid GraphRAG assistant for policy and compliance documents.** Ask a question such as *"Under India's DPDP Act, what must a company do after a personal data breach, and how does that compare with GDPR?"*. PolicyLens works through these steps:

1. splits the question into sub-questions;
2. retrieves evidence with a **knowledge graph, vector search and BM25**;
3. reranks the evidence with a local cross-encoder;
4. writes an answer where **every factual sentence cites a specific source chunk**;
5. **checks those citations** and removes sentences their sources don't support;
6. reports a confidence score, and says *"I don't have enough evidence"* instead of guessing.

The project runs entirely on free tiers: Groq for the LLM, NVIDIA NIM as a fallback, and local CPU models for embeddings and reranking. It runs on an 8 GB laptop with no GPU.

Corpus: India's **DPDP Act 2023**, **GDPR**, **NIST CSF 2.0** and **NIST SP 800-53 Rev. 5** (about 1,000 chunks). You can add any PDF, HTML or TXT file.

---

## Architecture

```mermaid
flowchart LR
  subgraph Ingestion [ingest/  incremental and idempotent]
    A[PDF / HTML / TXT] --> B[layout-aware parse<br/>pypdf, pdfplumber, BS4]
    B --> C[structure-aware chunks<br/>Article / Section / Control]
    C --> D[(chunk store<br/>SQLite)]
    D --> E[(ChromaDB<br/>bge-small embeddings)]
    D --> F[BM25]
    D --> G[spaCy pre-filter] --> H[small LLM extraction<br/>Pydantic-validated]
    H --> I[(Neo4j<br/>knowledge graph)]
  end
  subgraph Query [rag/  LangGraph StateGraph]
    Q[question] --> QA[analyze:<br/>classify + decompose]
    QA --> R[per sub-question:<br/>vector + BM25 + graph]
    R --> RRF[RRF fusion] --> RR[cross-encoder rerank]
    RR -->|weak evidence| AB[abstain]
    RR --> GEN[large LLM answer<br/>with E-handles]
    GEN --> V[verify citations:<br/>embedding sim + LLM judge]
    V --> FIN[confidence, render citations,<br/>abstain if low]
  end
  E & F & I --> R
```

| Layer | What it does | Key file |
|---|---|---|
| LLM client | Disk cache keyed by a hash of the prompt. Exponential backoff on 429, honouring `Retry-After`. Daily token budget per provider and model. Falls back from Groq to NVIDIA, then to a clear error. | [core/llm.py](core/llm.py) |
| Parsing | Gazette PDFs are split into body and margin columns, because DPDP section titles are printed as margin notes. The penalty Schedule is rebuilt row by row. Running headers are removed from other PDFs. | [ingest/parse.py](ingest/parse.py) |
| Chunking | Splits on Articles, Sections, controls (`AC-2`) and CSF categories (`GV.OC`) first, then into windows of at most 700 tokens with an 80-token overlap. IDs are stable and readable, such as `gdpr::article-33::0`. | [ingest/chunk.py](ingest/chunk.py) |
| Knowledge graph | Covered below under *Knowledge graph model*. | [ingest/kg.py](ingest/kg.py), [ingest/graph.py](ingest/graph.py) |
| Retrieval | Vector top-k, BM25 top-k, and graph top-k. The graph step matches nodes by full-text search, then walks up to 2 hops and never passes through a Regulation node, which links to everything. The three lists are fused with RRF and then reranked. | [rag/retrieve.py](rag/retrieve.py) |
| Answer and verification | The model cites `[E2]`-style handles and code maps them to `[doc, section, chunk_id]`, so the model can't invent a chunk ID. Each cited sentence is checked by embedding similarity and by the small LLM acting as a judge. | [rag/answer.py](rag/answer.py) |
| Pipeline | A LangGraph `StateGraph` with a conditional edge to abstain. `PipelineConfig` switches parts on and off for the evals. | [rag/pipeline.py](rag/pipeline.py) |
| API and UI | FastAPI with `/ask` (SSE), `/ingest`, `/health` and `/stats`. Streamlit chat with clickable citations, a confidence badge, a "How I found this" panel and a pyvis graph view. | [api/main.py](api/main.py), [ui/app.py](ui/app.py) |

### Knowledge graph model

| Edge | How it's made |
|---|---|
| `Regulation -CONTAINS-> Section` | deterministic |
| `Section -REQUIRES-> Obligation`, `-DEFINES-> Definition`, `-CONTAINS-> Right / Penalty` | deterministic, from the chunk's own section |
| `Section -REFERENCES-> Section` | regex on "Article 55", "section 8", "AC-3", "GV.OC" |
| `Obligation / Right -APPLIES_TO-> Entity`, `Penalty -PENALIZES-> ...`, `Definition -DEFINES-> Entity`, `Obligation -REQUIRES-> Obligation` | small LLM, schema-checked, invalid triples dropped |
| `x -SIMILAR_TO- y` across regulations | local embedding similarity of name and description, plus seed pairs (Data Fiduciary ≈ Controller, Data Principal ≈ Data Subject, …) |
| `every node -MENTIONED_IN-> Chunk` | provenance, which makes graph hits citable |

Duplicate entities are merged by normalized name ("The Data Fiduciaries" → `data fiduciary`). Entities are shared across regulations. Obligations, rights, penalties and definitions are scoped to their own regulation. Every relationship records the chunk IDs that produced it, so a changed chunk can be removed from the graph and extracted again.

---

## Quick start

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker. All of it is free.

```bash
# 1. install
uv sync

# 2. keys: create free ones at console.groq.com/keys (and optionally build.nvidia.com)
cp .env.example .env            # then paste GROQ_API_KEY=...
uv run python -m scripts.check_models   # checks the model IDs in .env against /models

# 3. Neo4j (capped at 1 GB RAM)
docker compose up -d neo4j

# 4. corpus + ingestion (embedding ~6 min on CPU; graph extraction spends Groq tokens, see below)
uv run python -m scripts.download_corpus
uv run python -m ingest.pipeline                # or --no-graph for zero LLM calls

# 5. run
uv run uvicorn api.main:app --port 8000
uv run streamlit run ui/app.py                  # http://localhost:8501
```

Without any key, ingestion, vector and BM25 search, reranking and the structural graph all still work. Answering then abstains with a clear message.

### API

```bash
curl -N -X POST localhost:8000/ask -H "content-type: application/json" \
     -d '{"question": "What is the DPDP penalty for failing to protect personal data?", "config": "full"}'
```

`/ask` streams an `event: stage` after each pipeline step, then an `event: result` with the answer, citations, confidence, per-sentence verification and the retrieval trace. `/ask/sync` returns the same result as plain JSON.

---

## Staying inside free tiers

| Mechanism | Where |
|---|---|
| Every LLM call is cached in SQLite, keyed by a hash of the model, messages and parameters. Re-running ingestion or evals costs 0 tokens. | `core/store.py` |
| Daily ledger per `provider:model` of both tokens and requests. Groq's free caps apply per model (1,000 requests/day each), so the large and small models are tracked separately. | `core/llm.py` |
| Client-side pacing keeps each model under its per-minute token cap: a 60 s sliding window counts the prompt plus the full `max_tokens` allowance and waits *before* sending, instead of bouncing off 429s. | `core/llm.py` |
| 429, 5xx and timeouts get exponential backoff with jitter. If `Retry-After` is over 30 s, interactive calls fall back to the next provider. Batch jobs (ingestion, evals) wait up to 2 minutes, and get one 65 s cool-down before giving up. | `core/llm.py`, `ingest/graph.py` |
| `gpt-oss` models get `reasoning_effort=low`, so hidden reasoning tokens stay small. | `core/llm.py` |
| The spaCy pre-filter skips chunks with no normative language (12.5% of the corpus), so they never reach the LLM. | `ingest/kg.py` |
| Ingestion is incremental at two levels. Files with an unchanged SHA-256 and unchanged chunker code are skipped. Only chunks whose content hash changed are re-embedded or re-extracted. | `ingest/pipeline.py`, `ingest/store.py` |
| Graph extraction is resumable. When the budget runs out, the remaining chunks get their deterministic structure now and LLM extraction on the next run. `--kg-limit N` caps calls per run. | `ingest/graph.py` |
| Questions whose best reranked chunk scores under 0.08 abstain before calling the large model. | `rag/pipeline.py` |

**Models (checked against Groq's `/models` on 2026-11-06).** The Llama 3.x models in the original brief have been retired.

| Role | Model | Free-tier limits (measured) |
|---|---|---|
| Answers | `openai/gpt-oss-120b` | 1,000 requests/day, 8,000 tokens/min, **200,000 tokens/day** |
| Extraction, analysis, judging | `openai/gpt-oss-20b` | 1,000 requests/day, 8,000 tokens/min, **200,000 tokens/day** |
| Judging during evals; rejected for extraction | `qwen/qwen3.8-27b` | separate daily quota, but capped at **1,000 output tokens/min**, so only about 1.5 extractions/min |

Two of these limits aren't in Groq's rate-limit headers: the qwen output cap and the 200k tokens/day (a rolling 24 h window). Both only show up in the 429 error message, which is how they were found. The daily token budgets in `.env` are set just below 200k, so the client stops before Groq does.

**Budget arithmetic.**
- **Graph extraction:** about 890 small-model calls in total, about 1.2k tokens each, about 1M tokens. At 200k tokens/day, the full corpus takes about 5 days of free tier. That's why extraction is resumable and can be limited to chosen documents.
- **One full-pipeline answer (measured):** about 2.3k large-model tokens and about 3.2k small-model tokens over 3 calls (analysis, judge, and faithfulness in evals). So one model's daily quota covers about 60–80 eval questions.

To focus extraction on the regulations first, run:

```bash
uv run python -m ingest.pipeline --kg-docs dpdp_act_2023,gdpr,nist_csf_2.0
```

## Low-RAM mode (8 GB laptops)

The defaults target a 7.3 GB Windows laptop:

- Neo4j has `mem_limit: 1g`, a 512 MB heap and a 256 MB page cache.
- The embedder and reranker are loaded once per process and run on CPU, with embedding batches of 16.
- The default reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Measured on this machine, it adds 0.5 GB of RAM and takes 0.7 s per query. `BAAI/bge-reranker-base` adds 1.6 GB and takes 4.5 s. Switch with `RERANK_MODEL` in `.env` if you have 16 GB.
- Ingestion streams: one document at a time, embedding batches of `INGEST_BATCH_SIZE` (64), and paged SQLite reads for the graph step and BM25. Peak ingestion RSS was 775 MB.

---

## Evaluation

`evals/gold.jsonl` holds **44 questions**: 16 simple, 10 multi-hop, 9 cross-regulation comparisons and 9 unanswerable. Each one lists its expected source sections, a reference answer and key facts. Every fact was checked against the indexed text.

```bash
uv run python -m evals.run                    # 3 configs x 44 questions, resumable
uv run python -m evals.run --out evals/results/<run>   # continue a run after a rate limit
GROQ_SMALL_MODEL=qwen/qwen3.8-27b uv run python -m evals.run --configs full   # judge on a model with its own daily quota
```

| Metric | How it's computed |
|---|---|
| Recall@5, MRR | Expected sections among the top-5 evidence chunks, and the reciprocal rank of the first hit. No LLM. |
| Key-fact accuracy | Gold facts (e.g. "72 hours", "250 crore") found in the answer. Deterministic. |
| Citation precision | Share of the cited sentences shown to the user that the judge says their citation supports. |
| Faithfulness | Small model as judge: supported claims divided by total claims, given the evidence. Cached. |
| Abstention | Abstains on unanswerable questions and answers answerable ones. |
| Latency | p50 and p95 seconds, end to end. |

The three configurations compared:

- `vector`: plain vector RAG.
- `hybrid`: vector and BM25 with RRF.
- `full`: decomposition, vector, BM25 and graph, then reranking, verification and abstention.

### Results

**Live run of the full pipeline** (2026-11-07). Answers came from `gpt-oss-120b`, and analysis and judging from `qwen3.8-27b`, all on the Groq free tier. Full results are in [evals/results/2026-11-07-full-groq/](evals/results/2026-11-07-full-groq/report.md).

| Metric | full |
|---|---|
| Recall@5 / MRR | 0.88 / 0.86 |
| Recall@5: simple / multi-hop / comparison | 1.00 / 0.95 / 0.59 |
| Key-fact accuracy | **0.90** |
| Citation precision | **0.97** |
| Faithfulness | **0.95** |
| Abstains on unanswerable (9 questions) | **1.00** |
| Answers answerable (35 questions) | 0.94 |
| Latency p50 / p95 (s) | 59.5 / 67.0 |

What these numbers mean:

- **Unanswerable questions abstain cleanly.** All 9 scored a confidence of 0.34 or lower, and most stopped before calling the large model, in about 2–4 s.
- **Comparisons are the weak spot.** Two of the 9 abstained (c04, c05) because one side's provision never reached the top 5. Comparison recall is 0.59, against 1.00 for simple questions.
- **One miss is a real hallucination by omission.** For m04, about the GDPR fine for breach-notification failures, the retrieved chunk of Article 83 didn't contain paragraph 4 (€10 million / 2 %). The model then said "no fixed amount is specified". Every citation in that answer was correct, so citation-level verification can't catch a missing fact. Only key-fact scoring did.
- **Latency is mostly rate-limit pacing, not compute.** The median stage times were 0.3 s for analysis, 3.3 s for retrieval and reranking, 49.7 s for generation and 4.2 s for verification. During a batch eval, the client paces large-model requests to stay under 8,000 tokens/min. A single interactive question without that pacing takes about 20–25 s.
- **Two fixes to the scorer after this run.** The answers use non-breaking hyphens ("supply‑chain"), so the key-fact matcher now normalizes Unicode and hyphens. Separately, the s15 gold item now also accepts "dies" and "incapacitated", as well as "death" and "incapacity". Saved answers were re-scored with `python -m evals.report <run> --rescore-facts`, which makes no LLM calls; key-fact accuracy went from 0.81 to 0.90.
- **Answer-level comparison with the `vector` and `hybrid` baselines isn't run yet.** It needs about 2 more days of free-tier tokens. `evals.run --configs vector,hybrid --out evals/results/2026-11-07-full-groq` resumes into the same folder.

**Retrieval run without an LLM key** (2026-11-06, CPU laptop, MiniLM reranker). It compares all three configs on retrieval. Full results are in [evals/results/2026-11-06-retrieval-no-llm-v2/](evals/results/2026-11-06-retrieval-no-llm-v2/report.md).

| Metric (answerable questions) | vector | hybrid | full |
|---|---|---|---|
| Recall@5 | 0.76 | 0.80 | **0.88** |
| MRR | 0.75 | 0.70 | **0.82** |
| Recall@5, simple | 0.81 | 0.81 | **1.00** |
| Recall@5, multi-hop | 0.80 | **0.95** | 0.90 |
| Recall@5, comparison | **0.64** | 0.60 | **0.64** |
| Retrieval latency p50 / p95 (s) | 0.04 / 0.05 | 0.05 / 0.05 | 2.1 / 3.4 |

In this run:

- **No key was configured, so every answer abstained.** That's why key-fact accuracy, citation precision and faithfulness are empty, and why abstention accuracy reads 0.20. Add `GROQ_API_KEY` and run `python -m evals.run` to fill those in. The LLM cache makes later runs free.
- **The full config's query analysis fell back to keyword rules.** Comparisons were split per regulation, but each sub-question is just the original question plus "(only the part about X)". That is the main reason comparison recall doesn't beat the baselines here. The LLM analyzer writes self-contained sub-questions instead.
- **All 9 unanswerable questions scored a confidence of 0.34 or lower.** That's below the abstention threshold, so they would abstain even with a key.
- **Latency covers retrieval only.** The full config's 2 s is the cross-encoder reranking 20 candidates on CPU. Generation adds the Groq round trip.

**What the first run taught us.** [evals/results/2026-11-06-retrieval-no-llm/](evals/results/2026-11-06-retrieval-no-llm/report.md) had full-config Recall@5 at 0.81 and MRR at 0.79. Inspecting the misses led to three fixes:

1. **Named provisions.** "What does CP-9 require?" missed CP-9, because the ID appears in dozens of "Related Controls" lists. An exact-reference retriever now always includes provisions the question names.
2. **Sub-question ordering.** Evidence for comparisons was concatenated by sub-question, which buried the GDPR side below the top 5. It's now interleaved.
3. **Definition lists.** About 25 definitions per 700-token chunk diluted each one's embedding. Definitions units now use 250-token windows.

---

## Tests

```bash
uv run pytest            # 62 offline tests: fake LLM, fake models, no network, no Neo4j needed
```

The tests cover:

- **LLM client:** caching, backoff, fallback, budgets and JSON repair.
- **Parsing and chunking:** Gazette margin notes, running headers, GDPR recitals, CSF categories, token caps and stable IDs.
- **Incremental store:** paging and batched embedding.
- **Graph:** triple validation, references, and a resumable extraction loop.
- **Pipeline:** the full LangGraph flow with fakes, covering comparison, abstention and the baselines.
- **API:** SSE contract and validation.
- **Evals:** metric functions.

## Repo layout

```
core/      settings, LLM client (cache, backoff, budgets, fallback), local models
ingest/    parse -> chunk -> store -> Chroma + BM25 + Neo4j graph
rag/       analyze, retrieve (RRF + rerank), answer + verify, LangGraph pipeline
api/       FastAPI service
ui/        Streamlit app
evals/     gold set, metrics, runner, report
scripts/   corpus download, model check
docs/      project brief and phase plan
```

## Limitations

- **LLM graph extraction covers DPDP, GDPR and NIST CSF 2.0.** That's 338 chunks. The graph now has 1,688 nodes, 733 `APPLIES_TO` edges and 14 cross-regulation `SIMILAR_TO` links. SP 800-53 has its deterministic structure (sections, cross-references, provenance), and graph retrieval uses it, but its LLM extraction (about 540 chunks) waits for more free-tier days (200k tokens/day). The live eval above ran before CSF was extracted.
- **Citation verification checks support, not completeness.** A sentence can be faithfully cited and still leave out the key number, as in m04 above. It can also adopt the wording of one provision where another provision differs: in one test, the answer said a breach is reported to the Board "or" the Data Principal, following the Schedule, while Section 8(6) says "and".
- The DPDP Act often says "in such form and manner as may be prescribed" (for example the breach-notification deadline). Those details are in the DPDP Rules, which are not in the corpus, so the system correctly says the Act sets no deadline.
- The Gazette parser is tuned to Gazette of India layout, using margin columns and the Schedule table. Other PDFs use the generic parser.
- The heading rules for 800-53 can match an example control quoted in the narrative. In that case the real control gets a suffixed ID (`au-4-2`), but its citation label stays `AU-4`.
- Latency is CPU-bound (reranking and embeddings) and depends on the LLM cache. Cached questions are much faster than first runs.

## License

MIT. See [LICENSE](LICENSE). The corpus documents in `data/raw/` are not part of this repository; they are public texts downloaded from their official sources and remain under their own terms.
