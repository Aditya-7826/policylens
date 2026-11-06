# PolicyLens: project brief

A hybrid GraphRAG assistant for policy and compliance documents. You ask a compliance question, for example "Under India's DPDP Act, what must a company do after a personal data breach, and how does that compare with GDPR?". The system:

1. breaks the question into sub-questions;
2. retrieves evidence from a knowledge graph, vector search and BM25;
3. reranks the evidence;
4. answers with a citation on every claim;
5. checks those citations and reports a confidence score;
6. says "I don't have enough evidence" when the evidence is weak.

## Hard constraints

- **100% free.** No paid APIs, no paid hosting, no credit card.
  - Primary LLM: Groq free tier (OpenAI-compatible).
  - Fallback LLM: NVIDIA NIM. It has limited trial credits, so it is used only when Groq fails.
  - Embeddings (`BAAI/bge-small-en-v1.5`) and the reranker (`BAAI/bge-reranker-base`) run locally on the CPU.
  - Neo4j Community runs in Docker. ChromaDB is stored locally.
- **Model IDs come from `.env`.** `scripts/check_models.py` checks them against each provider's `/models` endpoint.
- **Free-tier limits are respected:**
  - every LLM call is cached on disk, keyed by a hash of the prompt;
  - 429 responses get exponential backoff;
  - tokens used per day are tracked against a budget;
  - calls fall back from Groq to NVIDIA, then to a clear error message.
- **Hardware:** a normal laptop with 8 to 16 GB RAM and no GPU. The development machine has 7.3 GB.
- **Explainable in an interview.** Clear code beats clever code.

## Corpus

Only public documents:

- India's Digital Personal Data Protection Act 2023
- GDPR
- NIST Cybersecurity Framework 2.0
- NIST SP 800-53 Rev. 5

`scripts/download_corpus.py` fetches them. Ingestion also accepts any PDF, HTML or TXT file you add to `data/raw/`.

## Repo layout

```
core/      config, LLM client (cache, backoff, budgets, fallback), local models
ingest/    parse -> chunk -> vector index, BM25, knowledge graph (incremental)
rag/       LangGraph query pipeline: analyse -> retrieve -> fuse -> rerank -> answer -> verify
api/       FastAPI: /ask (SSE), /ingest, /health, /stats
ui/        Streamlit chat UI with citations, confidence badge, trace panel, graph view
evals/     gold set, metrics, 3-configuration comparison
scripts/   corpus download, model check, helpers
tests/     unit tests (offline: fake LLM, no Neo4j needed)
docs/      this brief, design notes
```

## Phases

| # | Phase | Done when |
|---|-------|-----------|
| 0 | Scaffold, config, LLM layer | LLM tests pass offline |
| 1 | Corpus download | Files are in `data/raw/`, and any failed download is listed with its URL |
| 2 | Parsing and structure-aware chunking | Chunks have stable IDs and metadata for document, section and page |
| 3 | Vector and BM25 indexes | Re-running ingestion is idempotent |
| 4 | Knowledge graph extraction into Neo4j | Extracted data matches a Pydantic schema, has MENTIONED_IN links and resolved duplicates |
| 5 | LangGraph query pipeline | Every claim is cited, citations are verified, and the system abstains on weak evidence |
| 6 | FastAPI and Streamlit UI | `/ask` streams progress and the UI shows citations and the graph |
| 7 | Evals | A 40+ question gold set comparing 3 configurations |
| 8 | README and polish | Someone else can run the project from the README |
