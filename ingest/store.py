"""The chunk store: one SQLite table that is the source of truth for all chunks.

It also records what has been done to each chunk (embedded? graph-extracted?), keyed by
content hash, so re-running ingestion only processes new or changed chunks.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

from ingest.chunk import Chunk


class ChunkStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                data TEXT NOT NULL,              -- Chunk as JSON
                embedded_hash TEXT,              -- content_hash at the time it was embedded
                kg_hash TEXT,                    -- content_hash at the time the graph was extracted
                kg_status TEXT                   -- 'done' | 'skipped' (spaCy pre-filter) | 'error'
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE TABLE IF NOT EXISTS files (
                doc_id TEXT PRIMARY KEY, filename TEXT, sha256 TEXT, chunker_version TEXT
            );
            """
        )
        self._db.commit()

    # ---- file-level change detection (skip re-parsing unchanged files)
    def file_unchanged(self, doc_id: str, sha256: str, chunker_version: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT sha256, chunker_version FROM files WHERE doc_id=?", (doc_id,)).fetchone()
        return row == (sha256, chunker_version)

    def record_file(self, doc_id: str, filename: str, sha256: str, chunker_version: str):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?)", (doc_id, filename, sha256, chunker_version))
            self._db.commit()

    def sync_document(self, doc_id: str, chunks: list[Chunk]) -> dict:
        """Insert new chunks, update changed ones, delete ones that no longer exist.
        Returns {'added': [...], 'changed': [...], 'removed': [...]} chunk ids."""
        with self._lock:
            existing = dict(self._db.execute(
                "SELECT chunk_id, content_hash FROM chunks WHERE doc_id=?", (doc_id,)).fetchall())
            new_ids = {c.chunk_id for c in chunks}
            added, changed = [], []
            for c in chunks:
                if c.chunk_id not in existing:
                    added.append(c.chunk_id)
                elif existing[c.chunk_id] != c.content_hash:
                    changed.append(c.chunk_id)
                else:
                    continue
                self._db.execute(
                    """INSERT INTO chunks (chunk_id, doc_id, content_hash, data) VALUES (?,?,?,?)
                       ON CONFLICT(chunk_id) DO UPDATE SET content_hash=excluded.content_hash, data=excluded.data""",
                    (c.chunk_id, doc_id, c.content_hash, c.model_dump_json()),
                )
            removed = [cid for cid in existing if cid not in new_ids]
            self._db.executemany("DELETE FROM chunks WHERE chunk_id=?", [(r,) for r in removed])
            self._db.commit()
        return {"added": added, "changed": changed, "removed": removed}

    def all(self) -> list[Chunk]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM chunks ORDER BY doc_id, rowid").fetchall()
        return [Chunk.model_validate_json(r[0]) for r in rows]

    def get(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        with self._lock:
            q = ",".join("?" * len(chunk_ids))
            rows = self._db.execute(f"SELECT data FROM chunks WHERE chunk_id IN ({q})", chunk_ids).fetchall()
        out = [Chunk.model_validate_json(r[0]) for r in rows]
        return {c.chunk_id: c for c in out}

    def iter_all(self, batch: int = 256) -> Iterator[list[Chunk]]:
        """All chunks, one page at a time (keyset pagination on rowid), so callers never hold the corpus."""
        last = 0
        while True:
            with self._lock:
                rows = self._db.execute("SELECT rowid, data FROM chunks WHERE rowid > ? ORDER BY rowid LIMIT ?",
                                        (last, batch)).fetchall()
            if not rows:
                return
            last = rows[-1][0]
            yield [Chunk.model_validate_json(r[1]) for r in rows]

    def pending_embedding(self, limit: int | None = None) -> list[Chunk]:
        sql = "SELECT data FROM chunks WHERE embedded_hash IS NULL OR embedded_hash != content_hash ORDER BY rowid"
        with self._lock:
            rows = self._db.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
        return [Chunk.model_validate_json(r[0]) for r in rows]

    def count_pending_embedding(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM chunks WHERE embedded_hash IS NULL "
                                    "OR embedded_hash != content_hash").fetchone()[0]

    def sections_by_doc(self) -> dict[str, set[str]]:
        """Section labels per document, read straight from SQLite (no Chunk objects)."""
        with self._lock:
            rows = self._db.execute("SELECT DISTINCT doc_id, json_extract(data, '$.section') FROM chunks").fetchall()
        out: dict[str, set[str]] = {}
        for doc, sec in rows:
            out.setdefault(doc, set()).add(sec)
        return out

    def chunks_for_section(self, section: str, doc_ids: list[str] | None = None) -> list[str]:
        """Chunk IDs of a named provision ("Article 33", "CP-9"), in document order."""
        sql = "SELECT chunk_id FROM chunks WHERE json_extract(data, '$.section') = ?"
        params: list = [section]
        if doc_ids:
            sql += f" AND doc_id IN ({','.join('?' * len(doc_ids))})"
            params += list(doc_ids)
        with self._lock:
            return [r[0] for r in self._db.execute(sql + " ORDER BY rowid", params).fetchall()]

    def doc_titles(self) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute("SELECT doc_id, MIN(json_extract(data, '$.doc_title')) FROM chunks "
                                    "GROUP BY doc_id").fetchall()
        return dict(rows)

    def mark_embedded(self, chunks: list[Chunk]):
        with self._lock:
            self._db.executemany("UPDATE chunks SET embedded_hash=? WHERE chunk_id=?",
                                 [(c.content_hash, c.chunk_id) for c in chunks])
            self._db.commit()

    # 'structure' = deterministic graph written, LLM extraction still to do
    _PENDING_KG = "(kg_hash IS NULL OR kg_hash != content_hash OR kg_status IN ('error', 'structure'))"

    def _kg_filter(self, doc_ids):
        if not doc_ids:
            return "", []
        return f" AND doc_id IN ({','.join('?' * len(doc_ids))})", list(doc_ids)

    def pending_kg(self, doc_ids: list[str] | None = None) -> list[Chunk]:
        return [c for page in self.iter_pending_kg(doc_ids) for c in page]

    def iter_pending_kg(self, doc_ids: list[str] | None = None, batch: int = 64) -> Iterator[list[Chunk]]:
        """Pending chunks page by page. Keyset pagination (rowid > last) means chunks that stay
        pending during the run (structure-only) are not fetched again in an endless loop."""
        extra, params = self._kg_filter(doc_ids)
        last = 0
        while True:
            with self._lock:
                rows = self._db.execute(
                    f"SELECT rowid, data FROM chunks WHERE {self._PENDING_KG}{extra} AND rowid > ? "
                    "ORDER BY rowid LIMIT ?", params + [last, batch]).fetchall()
            if not rows:
                return
            last = rows[-1][0]
            yield [Chunk.model_validate_json(r[1]) for r in rows]

    def count_pending_kg(self, doc_ids: list[str] | None = None) -> int:
        extra, params = self._kg_filter(doc_ids)
        with self._lock:
            return self._db.execute(f"SELECT COUNT(*) FROM chunks WHERE {self._PENDING_KG}{extra}",
                                    params).fetchone()[0]

    def needs_structure(self, chunk: Chunk) -> bool:
        """True if this chunk's deterministic graph has never been written for its current content."""
        with self._lock:
            row = self._db.execute("SELECT kg_hash, kg_status FROM chunks WHERE chunk_id=?",
                                   (chunk.chunk_id,)).fetchone()
        return row is None or row[0] != chunk.content_hash or row[1] not in ("structure", "done", "skipped")

    def mark_kg(self, chunk: Chunk, status: str):
        with self._lock:
            self._db.execute("UPDATE chunks SET kg_hash=?, kg_status=? WHERE chunk_id=?",
                             (chunk.content_hash, status, chunk.chunk_id))
            self._db.commit()

    def stats(self) -> dict:
        with self._lock:
            per_doc = self._db.execute(
                """SELECT doc_id, COUNT(*),
                          SUM(embedded_hash = content_hash),
                          SUM(kg_hash = content_hash AND kg_status='done'),
                          SUM(kg_hash = content_hash AND kg_status='skipped')
                   FROM chunks GROUP BY doc_id""").fetchall()
        return {d: {"chunks": n, "embedded": e or 0, "kg_done": k or 0, "kg_skipped": s or 0}
                for d, n, e, k, s in per_doc}
