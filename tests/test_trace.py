"""Unit tests for Tracer, execution latency tracking, and log rendering."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.trace import Tracer, render_log_line


def test_tracer_event_logging(tmp_path: Path) -> None:
    tracer = Tracer("run_test_01", trace_dir=tmp_path)
    record = tracer.event(
        node="plan",
        step=1,
        tool="web_search",
        provider="tavily",
        attempt=1,
        latency_ms=120,
        status="ok",
        tokens_in=500,
        tokens_out=150,
        cost_usd=0.0005,
        cache_hit=False,
        note="Generated 3 search queries",
    )

    assert record["run_id"] == "run_test_01"
    assert record["node"] == "plan"
    assert record["ts"].endswith("Z")
    assert tracer.cost_usd == pytest.approx(0.0005)
    assert len(tracer.events()) == 1

    # Check file on disk
    trace_file = tmp_path / "run_test_01.jsonl"
    assert trace_file.exists()
    lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["node"] == "plan"


def test_tracer_redacts_api_keys(tmp_path: Path) -> None:
    tracer = Tracer("run_redact", trace_dir=tmp_path)
    record = tracer.event(
        node="execute",
        note="Calling https://api.tavily.com/search?api_key=secret123&query=test",
        extra_url="https://example.com?token=my_secret_token",
    )
    assert "secret123" not in record["note"]
    assert "[REDACTED]" in record["note"]
    assert "my_secret_token" not in record["extra_url"]
    assert "[REDACTED]" in record["extra_url"]


def test_tracer_node_context_manager_logs_on_success(tmp_path: Path) -> None:
    tracer = Tracer("run_ctx_success", trace_dir=tmp_path)
    with tracer.node("intake") as ctx:
        ctx["note"] = "classified query"

    events = tracer.events()
    assert len(events) == 1
    assert events[0]["node"] == "intake"
    assert events[0]["status"] == "ok"
    assert events[0]["latency_ms"] >= 0
    assert events[0]["note"] == "classified query"


def test_tracer_node_context_manager_logs_on_exception(tmp_path: Path) -> None:
    tracer = Tracer("run_ctx_error", trace_dir=tmp_path)
    with pytest.raises(RuntimeError):
        with tracer.node("boom"):
            raise RuntimeError("simulation failure")

    events = tracer.events()
    assert len(events) == 1
    assert events[0]["node"] == "boom"
    assert events[0]["status"] == "error"
    assert events[0]["latency_ms"] >= 0
    assert "RuntimeError" in events[0]["note"]


def test_render_log_line() -> None:
    line = render_log_line("plan", 42, "3 queries")
    assert "plan" in line
    assert "42ms" in line
    assert "3 queries" in line
