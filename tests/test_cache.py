"""Unit tests for SQLite Cache, TTL expiration, single-flight locking, and bypass mode."""

from __future__ import annotations

from pathlib import Path
import threading
import time
import pytest

from app.cache import Cache


def test_cache_get_set(tmp_path: Path) -> None:
    cache = Cache(db_path=tmp_path / "test.db")
    payload = {"results": [{"title": "Qdrant", "url": "https://qdrant.tech"}]}

    cache.set("search", "vector database comparison", payload)
    retrieved = cache.get("search", "vector database comparison")
    assert retrieved == payload

    # Case insensitive key match
    retrieved_case = cache.get("search", "  VECTOR DATABASE COMPARISON  ")
    assert retrieved_case == payload

    # Non-existent key returns None
    assert cache.get("search", "unknown query") is None


def test_cache_ttl_expiry(tmp_path: Path) -> None:
    cache = Cache(db_path=tmp_path / "test_ttl.db")
    payload = {"data": "cached content"}

    base_time = 1_000_000.0
    cache.set("page", "https://example.com/article", payload, now=base_time)

    # 1 hour later (within 24h)
    assert cache.get("page", "https://example.com/article", now=base_time + 3600) == payload

    # 25 hours later (expired)
    assert cache.get("page", "https://example.com/article", now=base_time + 25 * 3600) is None


def test_cache_single_flight(tmp_path: Path) -> None:
    cache = Cache(db_path=tmp_path / "test_sf.db")
    key = "https://example.com/heavy-page"
    call_count = 0
    lock = threading.Lock()

    def fetch_with_cache() -> None:
        nonlocal call_count
        cached = cache.get("page", key)
        if cached is not None:
            return

        flight_lock = cache.get_lock(cache.make_key("page", key))
        with flight_lock:
            # Double-check inside lock
            cached = cache.get("page", key)
            if cached is None:
                with lock:
                    call_count += 1
                time.sleep(0.05)  # Simulate network latency
                cache.set("page", key, {"fetched": True})

    t1 = threading.Thread(target=fetch_with_cache)
    t2 = threading.Thread(target=fetch_with_cache)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert call_count == 1
    assert cache.get("page", key) == {"fetched": True}


def test_cache_disabled_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_DISABLED", "1")
    cache = Cache(db_path=tmp_path / "test_disabled.db")

    cache.set("search", "qdrant vs milvus", {"items": []})
    assert cache.get("search", "qdrant vs milvus") is None
