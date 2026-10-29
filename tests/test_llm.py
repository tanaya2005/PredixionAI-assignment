"""Unit tests for LLM router, retries, fallback chains, schema repair, and cost accounting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel, Field

from app.llm import gemini, groq
from app.llm.gemini import GeminiResponse
from app.llm.groq import GroqResponse
from app.llm.router import call_json
from app.trace import Tracer


class MockOutput(BaseModel):
    summary: str = Field(min_length=5)
    score: int


def test_router_happy_path(tmp_path: Path) -> None:
    tracer = Tracer("test_happy", trace_dir=tmp_path)
    mock_resp = GeminiResponse(
        text=json.dumps({"summary": "Valid summary text", "score": 9}),
        tokens_in=100,
        tokens_out=50,
    )

    with patch("app.llm.gemini.generate_json", return_value=mock_resp):
        res = call_json("plan", "prompt text", MockOutput, temperature=0.2, trace=tracer)

    assert res is not None
    assert res.summary == "Valid summary text"
    assert res.score == 9

    events = tracer.events()
    assert len(events) == 1
    assert events[0]["status"] == "ok"
    assert events[0]["provider"] == "gemini"
    assert events[0]["attempt"] == 1
    assert events[0]["tokens_in"] == 100
    assert events[0]["tokens_out"] == 50
    assert events[0]["cost_usd"] > 0.0


def test_router_one_500_then_success(tmp_path: Path) -> None:
    tracer = Tracer("test_retry_success", trace_dir=tmp_path)
    mock_success = GeminiResponse(
        text=json.dumps({"summary": "Recovered output", "score": 8}),
        tokens_in=80,
        tokens_out=40,
    )

    with patch("app.llm.gemini.generate_json", side_effect=[RuntimeError("500 Server Error"), mock_success]), \
         patch("time.sleep", return_value=None) as mock_sleep:
        res = call_json("plan", "prompt text", MockOutput, temperature=0.2, trace=tracer)

    assert res is not None
    assert res.summary == "Recovered output"
    mock_sleep.assert_called_once()

    events = tracer.events()
    assert len(events) == 2
    assert events[0]["status"] == "error"
    assert events[0]["attempt"] == 1
    assert events[1]["status"] == "ok"
    assert events[1]["attempt"] == 2


def test_router_primary_fails_twice_then_fallback_succeeds(tmp_path: Path) -> None:
    tracer = Tracer("test_fallback", trace_dir=tmp_path)
    mock_fallback_success = GroqResponse(
        text=json.dumps({"summary": "Output from fallback provider", "score": 7}),
        tokens_in=120,
        tokens_out=60,
    )

    with patch("app.llm.gemini.generate_json", side_effect=RuntimeError("503 Service Unavailable")), \
         patch("app.llm.groq.generate_json", return_value=mock_fallback_success), \
         patch("time.sleep", return_value=None):
        res = call_json("plan", "prompt text", MockOutput, temperature=0.2, trace=tracer)

    assert res is not None
    assert res.summary == "Output from fallback provider"

    events = tracer.events()
    # 2 errors on gemini, 1 success on groq
    assert len(events) == 3
    assert events[0]["provider"] == "gemini"
    assert events[1]["provider"] == "gemini"
    assert events[2]["provider"] == "groq"
    assert events[2]["status"] == "ok"


def test_router_invalid_json_then_successful_repair(tmp_path: Path) -> None:
    tracer = Tracer("test_repair", trace_dir=tmp_path)
    # First response is missing required field 'score'
    bad_resp = GroqResponse(text=json.dumps({"summary": "Missing score field"}), tokens_in=50, tokens_out=20)
    repaired_resp = GroqResponse(text=json.dumps({"summary": "Properly repaired summary", "score": 10}), tokens_in=60, tokens_out=30)

    with patch("app.llm.groq.generate_json", side_effect=[bad_resp, repaired_resp]):
        res = call_json("intake", "prompt text", MockOutput, temperature=0.0, trace=tracer, max_repair=1)

    assert res is not None
    assert res.score == 10
    assert res.summary == "Properly repaired summary"

    events = tracer.events()
    assert len(events) == 1
    assert events[0]["status"] == "repaired"
    assert events[0]["provider"] == "groq"


def test_router_repair_exhausted_falls_back(tmp_path: Path) -> None:
    tracer = Tracer("test_repair_exhausted", trace_dir=tmp_path)
    bad_resp1 = GroqResponse(text=json.dumps({"summary": "Bad 1"}), tokens_in=30, tokens_out=10)
    bad_resp2 = GroqResponse(text=json.dumps({"summary": "Bad 2"}), tokens_in=30, tokens_out=10)
    gemini_resp = GeminiResponse(text=json.dumps({"summary": "Fallback works", "score": 6}), tokens_in=50, tokens_out=25)

    with patch("app.llm.groq.generate_json", side_effect=[bad_resp1, bad_resp2]), \
         patch("app.llm.gemini.generate_json", return_value=gemini_resp):
        res = call_json("intake", "prompt text", MockOutput, temperature=0.0, trace=tracer, max_repair=1)

    assert res is not None
    assert res.summary == "Fallback works"

    events = tracer.events()
    assert any(e["status"] == "validation_error" for e in events)
    assert events[-1]["provider"] == "gemini"
    assert events[-1]["status"] == "ok"


def test_router_429_with_retry_after(tmp_path: Path) -> None:
    tracer = Tracer("test_429", trace_dir=tmp_path)
    rate_limit_err = RuntimeError("429 Too Many Requests")
    setattr(rate_limit_err, "retry_after", 0.25)

    mock_success = GeminiResponse(
        text=json.dumps({"summary": "Success after 429", "score": 9}),
        tokens_in=70,
        tokens_out=35,
    )

    with patch("app.llm.gemini.generate_json", side_effect=[rate_limit_err, mock_success]), \
         patch("time.sleep", return_value=None) as mock_sleep:
        res = call_json("plan", "prompt text", MockOutput, temperature=0.2, trace=tracer)

    assert res is not None
    assert res.summary == "Success after 429"
    mock_sleep.assert_called_with(0.25)


def test_router_cost_accounting(tmp_path: Path) -> None:
    tracer = Tracer("test_cost", trace_dir=tmp_path)
    # Gemini prices: input 0.0001 / 1k, output 0.0004 / 1k
    # 10,000 in ($0.001) + 2,000 out ($0.0008) = $0.0018
    mock_resp = GeminiResponse(
        text=json.dumps({"summary": "Cost testing model output", "score": 100}),
        tokens_in=10000,
        tokens_out=2000,
    )

    with patch("app.llm.gemini.generate_json", return_value=mock_resp):
        res = call_json("synthesize", "prompt text", MockOutput, temperature=0.2, trace=tracer)

    assert res is not None
    assert tracer.cost_usd == pytest.approx(0.0018, abs=1e-5)


def test_router_all_providers_dead_returns_none_never_raises(tmp_path: Path) -> None:
    tracer = Tracer("test_dead", trace_dir=tmp_path)

    with patch("app.llm.gemini.generate_json", side_effect=ConnectionError("Connection refused")), \
         patch("app.llm.groq.generate_json", side_effect=ConnectionError("Host unreachable")), \
         patch("time.sleep", return_value=None):
        res = call_json("plan", "prompt text", MockOutput, temperature=0.2, trace=tracer)

    assert res is None
    events = tracer.events()
    assert events[-1]["status"] == "exhausted"


def test_router_invalid_role_returns_none(tmp_path: Path) -> None:
    tracer = Tracer("test_invalid_role", trace_dir=tmp_path)
    res = call_json("invalid_role", "prompt text", MockOutput, temperature=0.2, trace=tracer)
    assert res is None
    events = tracer.events()
    assert events[0]["status"] == "error"
