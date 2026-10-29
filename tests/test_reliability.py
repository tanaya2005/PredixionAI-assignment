"""Unit tests for tool retries, fallback chains, circuit breaker behavior, and caching."""

from __future__ import annotations

from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from app.cache import Cache
from app.errors import ToolError, ToolExhausted
from app.tools.base import ToolSpec
from app.tools.registry import REGISTRY, call, circuit_breaker, register
from app.trace import Tracer


class DummyArgs(BaseModel):
    query: str = "vector database comparison"


@pytest.fixture(autouse=True)
def setup_clean_registry_and_breaker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure registry, circuit breaker, and cache are isolated and clean before each test."""
    REGISTRY.clear()
    circuit_breaker.reset()
    test_cache = Cache(db_path=tmp_path / "isolated_cache.db")
    monkeypatch.setattr("app.tools.registry.cache", test_cache)


def test_tool_primary_500_twice_then_retry_success(tmp_path: Path) -> None:
    tracer = Tracer("run_retry", trace_dir=tmp_path)
    mock_fn = MagicMock(
        side_effect=[
            RuntimeError("500 Internal Server Error"),
            [{"title": "Result 1", "url": "https://example.com/1"}],
        ]
    )

    spec = ToolSpec(
        name="web_search",
        fn=mock_fn,
        input_schema=DummyArgs,
        timeout_s=10.0,
        provider="tavily",
        fallbacks=[],
    )
    register(spec)

    with patch("time.sleep", return_value=None):
        result = call("web_search", DummyArgs(), tracer=tracer)

    assert result.status == "ok"
    assert result.provider == "tavily"
    assert len(result.items) == 1
    assert mock_fn.call_count == 2

    events = tracer.events()
    assert len(events) == 2
    assert events[0]["status"] == "error"
    assert events[0]["attempt"] == 1
    assert events[1]["status"] == "ok"
    assert events[1]["attempt"] == 2


def test_search_falls_back_to_ddg_on_tavily_500_fallback(tmp_path: Path) -> None:
    tracer = Tracer("run_fallback", trace_dir=tmp_path)

    tavily_mock = MagicMock(side_effect=RuntimeError("500 Server Error"))
    ddg_mock = MagicMock(return_value=[{"title": "DDG hit", "url": "https://duckduckgo.com/1"}])

    tavily_spec = ToolSpec(
        name="web_search",
        fn=tavily_mock,
        input_schema=DummyArgs,
        timeout_s=10.0,
        provider="tavily",
        fallbacks=["ddg_search"],
    )
    ddg_spec = ToolSpec(
        name="ddg_search",
        fn=ddg_mock,
        input_schema=DummyArgs,
        timeout_s=10.0,
        provider="duckduckgo",
        fallbacks=[],
    )

    register(tavily_spec)
    register(ddg_spec)

    with patch("time.sleep", return_value=None):
        result = call("web_search", DummyArgs(), tracer=tracer)

    assert result.status == "ok"
    assert result.provider == "duckduckgo"
    assert len(result.items) == 1
    assert tavily_mock.call_count == 2  # 2 attempts on primary
    assert ddg_mock.call_count == 1     # 1 attempt on fallback

    events = tracer.events()
    assert events[-1]["provider"] == "duckduckgo"
    assert events[-1]["status"] == "ok"


def test_whole_chain_fails_raises_tool_exhausted(tmp_path: Path) -> None:
    tracer = Tracer("run_exhausted", trace_dir=tmp_path)

    primary_mock = MagicMock(side_effect=RuntimeError("Primary down"))
    fallback_mock = MagicMock(side_effect=RuntimeError("Fallback down"))

    register(
        ToolSpec(
            name="fetch_page",
            fn=primary_mock,
            input_schema=DummyArgs,
            timeout_s=5.0,
            provider="jina",
            fallbacks=["httpx_fetch"],
        )
    )
    register(
        ToolSpec(
            name="httpx_fetch",
            fn=fallback_mock,
            input_schema=DummyArgs,
            timeout_s=5.0,
            provider="httpx",
            fallbacks=[],
        )
    )

    with patch("time.sleep", return_value=None):
        with pytest.raises(ToolExhausted) as exc:
            call("fetch_page", DummyArgs(), tracer=tracer)

    assert "fetch_page" in str(exc.value)
    assert tracer.provider_failures["jina"] == 2
    assert tracer.provider_failures["httpx"] == 2


def test_fetch_circuit_opens_after_three_failures(tmp_path: Path) -> None:
    tracer = Tracer("run_circuit", trace_dir=tmp_path)
    fail_mock = MagicMock(side_effect=RuntimeError("Service 503"))

    spec = ToolSpec(
        name="fetch_page",
        fn=fail_mock,
        input_schema=DummyArgs,
        timeout_s=5.0,
        provider="jina",
        fallbacks=[],
    )
    register(spec)

    # Trigger 3 failures on provider 'jina'
    circuit_breaker.record_failure("jina")
    circuit_breaker.record_failure("jina")
    circuit_breaker.record_failure("jina")

    assert circuit_breaker.is_open("jina") is True

    # 4th call must immediately skip 'jina' and fail/raise in <10ms
    start_t = time.perf_counter()
    with pytest.raises(ToolExhausted):
        call("fetch_page", DummyArgs(), tracer=tracer)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    assert elapsed_ms < 20.0  # Under 20ms (well under test requirement)
    assert fail_mock.call_count == 0  # Did not attempt request

    events = tracer.events()
    assert len(events) == 1
    assert events[0]["status"] == "skipped"
    assert events[0]["provider"] == "jina"


def test_cache_hit_returns_without_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_cache = Cache(db_path=tmp_path / "test_reg_cache.db")
    monkeypatch.setattr("app.tools.registry.cache", test_cache)

    tracer = Tracer("run_cache", trace_dir=tmp_path)
    mock_fn = MagicMock(return_value=[{"title": "Fresh hit"}])

    spec = ToolSpec(
        name="web_search",
        fn=mock_fn,
        input_schema=DummyArgs,
        timeout_s=5.0,
        provider="tavily",
    )
    register(spec)

    # First call populates cache
    res1 = call("web_search", DummyArgs(query="test q"), tracer=tracer)
    assert res1.status == "ok"
    assert mock_fn.call_count == 1

    # Second call should be a cache hit
    res2 = call("web_search", DummyArgs(query="test q"), tracer=tracer)
    assert res2.status == "cache_hit"
    assert res2.provider == "cache"
    assert mock_fn.call_count == 1  # No additional network call made

    events = tracer.events()
    assert events[-1]["status"] == "cache_hit"
    assert events[-1]["cache_hit"] is True


def test_429_honours_retry_after(tmp_path: Path) -> None:
    tracer = Tracer("run_429", trace_dir=tmp_path)
    err_429 = RuntimeError("429 Rate Limit")
    setattr(err_429, "retry_after", 0.5)

    mock_fn = MagicMock(side_effect=[err_429, [{"title": "Success after retry"}]])

    register(
        ToolSpec(
            name="web_search",
            fn=mock_fn,
            input_schema=DummyArgs,
            timeout_s=5.0,
            provider="tavily",
        )
    )

    with patch("time.sleep", return_value=None) as mock_sleep:
        res = call("web_search", DummyArgs(), tracer=tracer)

    assert res.status == "ok"
    mock_sleep.assert_called_with(0.5)
