"""Incremental, idempotent ingestion.

    uv run python -m ingest.pipeline                      # everything in data/raw
    uv run python -m ingest.pipeline --no-graph           # indexes only, no LLM calls
    uv run python -m ingest.pipeline --kg-docs gdpr --kg-limit 50

Steps per file: parse -> chunk -> sync into the chunk store (adds / changes / removals)
-> embed only chunks whose content changed -> extract the knowledge graph for chunks
not yet extracted. Re-running with nothing changed does no work and spends no tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.config import get_settings
from ingest.chunk import chunk_document
from ingest.parse import discover, doc_id_for, parse_file
from ingest.store import ChunkStore

log = logging.getLogger("ingest")


@dataclass
class IngestReport:
    files: list[str] = field(default_factory=list)
    added: int = 0
    changed: int = 0
    removed: int = 0
    embedded: int = 0
    kg: dict = field(default_factory=dict)
    seconds: float = 0.0


def chunker_version() -> str:
    """Changes whenever the parsing/chunking code or chunk-size settings change,
    so an unchanged file is re-chunked after a code change but skipped otherwise."""
    s = get_settings()
    h = hashlib.sha256(f"{s.chunk_max_tokens}/{s.chunk_overlap_tokens}".encode())
    for mod in ("parse.py", "chunk.py"):
        h.update((Path(__file__).parent / mod).read_bytes())
    return h.hexdigest()[:12]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):  # 1 MB at a time
            h.update(block)
    return h.hexdigest()


def embed_pending(store: ChunkStore, vindex, batch_size: int, progress=print) -> int:
    """Embed chunks whose content changed, `batch_size` at a time. Each batch is written to
    Chroma and marked done before the next is loaded, so an interrupted run resumes cleanly."""
    total = store.count_pending_embedding()
    if not total:
        return 0
    from core.models import get_embedder

    embedder = get_embedder()
    done = 0
    while batch := store.pending_embedding(limit=batch_size):
        vindex.upsert(batch, embedder.embed_documents([c.context_text for c in batch]))
        store.mark_embedded(batch)
        done += len(batch)
        progress(f"[embed] {done}/{total} chunks")
    return done


def get_chunk_store() -> ChunkStore:
    return ChunkStore(get_settings().processed_dir / "chunks.sqlite")


def run_ingest(files: list[Path] | None = None, graph: bool = True, kg_docs: list[str] | None = None,
               kg_limit: int | None = None, progress=print) -> IngestReport:
    s = get_settings()
    t0 = time.time()
    store = get_chunk_store()
    files = files or discover(s.raw_dir)
    report = IngestReport(files=[f.name for f in files])
    removed_ids: list[str] = []

    from ingest.index import VectorIndex

    vindex = VectorIndex(s.chroma_dir)

    # Documents are processed one at a time: parse -> chunk -> sync -> embed in batches.
    # Only one document's chunks (plus one embedding batch) are in memory at any moment.
    version = chunker_version()
    for f in files:
        sha = _sha256_file(f)
        if store.file_unchanged(doc_id_for(f), sha, version):
            progress(f"[parse] {f.name}: unchanged, skipped")
            continue
        doc = parse_file(f)
        doc_id = doc.doc_id
        chunks = chunk_document(doc, s.chunk_max_tokens, s.chunk_overlap_tokens)
        del doc  # page text is no longer needed
        diff = store.sync_document(doc_id, chunks)
        n_chunks = len(chunks)
        del chunks
        report.added += len(diff["added"])
        report.changed += len(diff["changed"])
        report.removed += len(diff["removed"])
        removed_ids += diff["removed"]
        vindex.delete(diff["removed"])
        progress(f"[parse] {f.name}: {n_chunks} chunks "
                 f"(+{len(diff['added'])} new, ~{len(diff['changed'])} changed, -{len(diff['removed'])} removed)")
        report.embedded += embed_pending(store, vindex, s.ingest_batch_size, progress)
        store.record_file(doc_id_for(f), f.name, sha, version)  # only after its embeddings are stored

    # Anything left over from an interrupted earlier run
    report.embedded += embed_pending(store, vindex, s.ingest_batch_size, progress)
    if not report.embedded:
        progress("[embed] nothing to do")

    # 5. knowledge graph
    if graph:
        from ingest.graph import build_graph

        report.kg = build_graph(store, doc_ids=kg_docs, limit=kg_limit, removed_ids=removed_ids,
                                progress=progress)

    report.seconds = round(time.time() - t0, 1)
    return report


def main():
    ap = argparse.ArgumentParser(description="Ingest documents from data/raw")
    ap.add_argument("--files", nargs="*", help="specific files (default: all in data/raw)")
    ap.add_argument("--no-graph", action="store_true", help="skip knowledge-graph extraction (no LLM calls)")
    ap.add_argument("--kg-docs", help="comma-separated doc ids to extract the graph for (default: all)")
    ap.add_argument("--kg-limit", type=int, help="max chunks to send to the LLM this run (token budget)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    files = [Path(f) for f in args.files] if args.files else None
    kg_docs = [doc_id_for(Path(d)) for d in args.kg_docs.split(",")] if args.kg_docs else None
    r = run_ingest(files, graph=not args.no_graph, kg_docs=kg_docs, kg_limit=args.kg_limit)
    print(f"\nDone in {r.seconds}s: +{r.added} ~{r.changed} -{r.removed} chunks, {r.embedded} embedded")
    if r.kg:
        print("Graph:", r.kg)


if __name__ == "__main__":
    main()
