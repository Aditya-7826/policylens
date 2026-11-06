"""SQLite-backed LLM response cache and per-day token ledger.

One file (data/cache/llm.sqlite) holds both:
  - llm_cache: prompt-hash -> response, so re-running ingestion or evals costs zero tokens.
  - token_usage: tokens spent per provider per UTC day, so we stop before free-tier caps.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


def utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class LLMStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                provider TEXT, model TEXT,
                response TEXT NOT NULL,
                prompt_tokens INTEGER, completion_tokens INTEGER,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS token_usage (
                day TEXT, provider TEXT, tokens INTEGER NOT NULL DEFAULT 0, calls INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (day, provider)
            );
            """
        )
        self._db.commit()

    # ---- cache
    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT provider, model, response, prompt_tokens, completion_tokens FROM llm_cache WHERE key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "provider": row[0], "model": row[1], "text": row[2],
            "prompt_tokens": row[3] or 0, "completion_tokens": row[4] or 0,
        }

    def put(self, key: str, provider: str, model: str, text: str, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?,?,?)",
                (key, provider, model, text, prompt_tokens, completion_tokens,
                 datetime.now(timezone.utc).isoformat()),
            )
            self._db.commit()

    def cache_size(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]

    # ---- token ledger
    def add_usage(self, provider: str, tokens: int, day: str | None = None):
        day = day or utc_day()
        with self._lock:
            self._db.execute(
                """INSERT INTO token_usage (day, provider, tokens, calls) VALUES (?,?,?,1)
                   ON CONFLICT(day, provider) DO UPDATE SET tokens = tokens + excluded.tokens, calls = calls + 1""",
                (day, provider, tokens),
            )
            self._db.commit()

    def used_today(self, provider: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT tokens FROM token_usage WHERE day=? AND provider=?", (utc_day(), provider)
            ).fetchone()
        return row[0] if row else 0

    def calls_today(self, provider: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT calls FROM token_usage WHERE day=? AND provider=?", (utc_day(), provider)
            ).fetchone()
        return row[0] if row else 0

    def usage_report(self) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT provider, tokens, calls FROM token_usage WHERE day=?", (utc_day(),)
            ).fetchall()
        return {"day": utc_day(), "providers": {p: {"tokens": t, "calls": c} for p, t, c in rows},
                "cached_responses": self.cache_size()}


def cache_key(**parts) -> str:
    import hashlib

    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
