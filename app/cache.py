"""Persistent SQLite cache with TTL expiration and single-flight execution locks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Literal

from app.config import CACHE_PATH, CACHE_TTL_H


class Cache:
    """Thread-safe SQLite-backed key-value cache with TTL expiration and single-flight locks."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or CACHE_PATH
        self.ttl_seconds = CACHE_TTL_H * 3600
        self._init_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()
        self._ensure_tables()

    @property
    def is_disabled(self) -> bool:
        """Check if cache is globally disabled via CACHE_DISABLED env var."""
        return os.getenv("CACHE_DISABLED", "0").strip() == "1"

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._init_lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS search (
                        key TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        ts REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS page (
                        key TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        ts REAL NOT NULL
                    )
                    """
                )
                conn.commit()

    def make_key(self, kind: str, raw_key: str) -> str:
        """Create a deterministic SHA-256 hash for kind and normalized key."""
        normalised = raw_key.strip().lower()
        content = f"{kind}|{normalised}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def get_lock(self, key: str) -> threading.Lock:
        """Return a per-key Lock for single-flight deduplication of concurrent calls."""
        with self._key_locks_guard:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def get(
        self, kind: Literal["search", "page"], raw_key: str, *, now: float | None = None
    ) -> Any | None:
        """Retrieve payload if present and not expired; return None otherwise."""
        if self.is_disabled:
            return None

        key = self.make_key(kind, raw_key)
        current_ts = now if now is not None else time.time()

        table = "search" if kind == "search" else "page"
        query = f"SELECT payload, ts FROM {table} WHERE key = ?"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (key,))
            row = cursor.fetchone()
            if row is None:
                return None

            payload_str, ts = row["payload"], row["ts"]
            if current_ts - ts > self.ttl_seconds:
                # Expired: clean up
                conn.execute(f"DELETE FROM {table} WHERE key = ?", (key,))
                conn.commit()
                return None

            try:
                return json.loads(payload_str)
            except Exception:
                return None

    def set(
        self, kind: Literal["search", "page"], raw_key: str, payload: Any, *, now: float | None = None
    ) -> None:
        """Save JSON-serializable payload with current timestamp."""
        if self.is_disabled:
            return

        key = self.make_key(kind, raw_key)
        current_ts = now if now is not None else time.time()
        payload_str = json.dumps(payload)

        table = "search" if kind == "search" else "page"
        query = f"INSERT OR REPLACE INTO {table} (key, payload, ts) VALUES (?, ?, ?)"

        with self._get_connection() as conn:
            conn.execute(query, (key, payload_str, current_ts))
            conn.commit()


# Global default cache instance
cache = Cache()
