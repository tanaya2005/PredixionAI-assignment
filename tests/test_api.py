"""Unit tests for FastAPI endpoints: /research, /trace/{id}, and /healthz."""

from __future__ import annotations

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.contracts import Answer, Confidence, Finding, RunMeta, SourceRef
from app.main import _TRACE_CACHE, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_trace_cache():
    _TRACE_CACHE.clear()
    yield
    _TRACE_CACHE.clear()


def test_healthz_reflects_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure healthz returns ok and dict of providers
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "providers" in data
    providers = data["providers"]
    for p in ("gemini", "groq", "tavily", "exa", "jina"):
        assert p in providers
        assert isinstance(providers[p], bool)


def test_research_happy_path() -> None:
    mock_answer = Answer(
        run_id="run_api_test_123",
        question="Compare vector search methods",
        short_answer="HNSW provides graph-based similarity search while IVF uses inverted file clustering.",
        key_findings=[
            Finding(
                claim="HNSW builds multi-layer proximity graphs for nearest neighbor search.",
                source_ids=["s1"],
                supporting_quote="HNSW builds multi-layer proximity graphs for nearest neighbor search.",
            )
        ],
        sources=[],
        confidence=Confidence(level="medium", score=0.70, reasons=["Grounded finding"]),
        assumptions=[],
        limitations=[],
        next_steps=[],
        status="ok",
        meta=RunMeta(tool_calls=2, latency_ms=250, cost_usd=0.002, grounding_rate=1.0),
    )
    mock_events = [
        {"node": "intake", "status": "ok"},
        {"node": "execute", "status": "ok"},
    ]

    with patch("app.main.run_with_trace", return_value=(mock_answer, mock_events)):
        resp = client.post("/research", json={"question": "Compare vector search methods"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "run_api_test_123"
    assert data["status"] == "ok"
    assert len(data["key_findings"]) == 1

    # Verify trace is retrievable via GET /trace/run_api_test_123
    trace_resp = client.get("/trace/run_api_test_123")
    assert trace_resp.status_code == 200
    trace_data = trace_resp.json()
    assert len(trace_data) == 2


def test_research_validation_errors() -> None:
    # Less than 5 chars -> 422
    resp_short = client.post("/research", json={"question": "xyz"})
    assert resp_short.status_code == 422

    # Blank / whitespace only -> 400
    resp_blank = client.post("/research", json={"question": "     "})
    assert resp_blank.status_code == 400

    # Over 2000 chars -> 422
    resp_too_long = client.post("/research", json={"question": "a" * 2001})
    assert resp_too_long.status_code == 422


def test_research_degraded_answer_on_pipeline_exception() -> None:
    # If run_with_trace raises an unexpected error, /research MUST NOT return 500
    with patch("app.main.run_with_trace", side_effect=RuntimeError("Simulated unhandled pipeline crash")):
        resp = client.post("/research", json={"question": "Valid length question that crashes"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert "Internal error" in data["confidence"]["reasons"][0]
    assert len(data["limitations"]) > 0


def test_trace_unknown_run_id_returns_404() -> None:
    resp = client.get("/trace/non_existent_run_999")
    assert resp.status_code == 404

