"""FastAPI service.

    uv run uvicorn api.main:app --port 8000

POST /ask        question -> Server-Sent Events: one `stage` event per pipeline step, then `result`
POST /ask/sync   same, but one JSON response (handy for curl and scripts)
POST /ingest     optional file uploads, then incremental ingestion in the background
GET  /ingest/status
GET  /health     what is up: indexes, Neo4j, LLM providers
GET  /stats      chunk counts per document, graph size, today's token usage
GET  /chunks/{chunk_id}   source text for a citation
GET  /graph?chunk_ids=a,b subgraph of the entities mentioned in those chunks
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.config import get_settings
from ingest.parse import SUPPORTED
from rag.pipeline import CONFIGS, ask, stream_ask
from rag.resources import get_resources

log = logging.getLogger("api")
for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = FastAPI(title="PolicyLens", version="0.1.0",
              description="Hybrid GraphRAG assistant for policy and compliance documents")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    config: str = "full"


@app.get("/health")
def health():
    res = get_resources()
    s = res.settings
    graph = res.graph
    llm = res.llm
    return {
        "status": "ok",
        "chunks": sum(d["chunks"] for d in res.store.stats().values()),
        "vector_index": res.vectors.count(),
        "neo4j": graph is not None,
        "llm_providers": [p.name for p in llm.providers if p.configured],
        "models": {"large": s.groq_large_model, "small": s.groq_small_model, "embed": s.embed_model,
                   "rerank": s.rerank_model},
    }


@app.get("/stats")
def stats():
    res = get_resources()
    graph = res.graph
    return {"documents": res.store.stats(), "graph": graph.stats() if graph else None, "llm": res.llm.usage()}


@app.post("/ask/sync")
def ask_sync(req: AskRequest):
    if req.config not in CONFIGS:
        raise HTTPException(400, f"config must be one of {list(CONFIGS)}")
    return ask(req.question, req.config).model_dump()


@app.post("/ask")
async def ask_stream(req: AskRequest):
    if req.config not in CONFIGS:
        raise HTTPException(400, f"config must be one of {list(CONFIGS)}")
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():  # the pipeline is synchronous (CPU models), so run it in a thread
        try:
            for event, data in stream_ask(req.question, req.config):
                loop.call_soon_threadsafe(queue.put_nowait, (event, data))
        except Exception as exc:
            log.exception("ask failed")
            loop.call_soon_threadsafe(queue.put_nowait, ("error", {"message": str(exc)}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def events():
        while (item := await queue.get()) is not None:
            event, data = item
            yield {"event": event, "data": json.dumps(data, default=str)}

    return EventSourceResponse(events(), ping=15)


@app.get("/chunks/{chunk_id:path}")
def get_chunk(chunk_id: str):
    got = get_resources().store.get([chunk_id])
    if chunk_id not in got:
        raise HTTPException(404, "unknown chunk")
    c = got[chunk_id]
    return {**c.model_dump(), "citation": c.citation()}


@app.get("/graph")
def graph_view(chunk_ids: str, max_nodes: int = 40):
    graph = get_resources().graph
    if graph is None:
        return {"nodes": [], "edges": [], "note": "Neo4j unavailable"}
    ids = [c for c in chunk_ids.split(",") if c]
    return graph.subgraph(ids, max_nodes=max_nodes)


# ------------------------------------------------------------------------------- ingestion
_jobs: dict[str, dict] = {}
_ingest_lock = threading.Lock()


@app.post("/ingest")
async def ingest(files: list[UploadFile] | None = File(default=None), graph: bool = Form(default=True),
                 kg_limit: int | None = Form(default=None)):
    raw = get_settings().raw_dir
    saved: list[Path] = []
    for f in files or []:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(400, f"{name}: only {sorted(SUPPORTED)} files are supported")
        target = raw / name
        target.write_bytes(await f.read())
        saved.append(target)
    if _ingest_lock.locked():
        raise HTTPException(409, "an ingestion job is already running")
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", "log": [], "started": time.time(), "files": [p.name for p in saved]}

    def run():
        from ingest.pipeline import run_ingest

        with _ingest_lock:
            job = _jobs[job_id]
            try:
                report = run_ingest(saved or None, graph=graph, kg_limit=kg_limit, progress=job["log"].append)
                job.update(status="done", report=report.__dict__)
                get_resources().refresh()
            except Exception as exc:
                log.exception("ingest failed")
                job.update(status="error", error=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "saved": [p.name for p in saved]}


@app.get("/ingest/status")
def ingest_status(job_id: str | None = None):
    if job_id:
        if job_id not in _jobs:
            raise HTTPException(404, "unknown job")
        return _jobs[job_id]
    return _jobs
