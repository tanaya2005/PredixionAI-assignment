"""Unit tests for the command-line interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest

from app.contracts import Answer, Confidence, Finding, Intake, ResearchPlan, RunMeta, SourceRef


def test_cli_validation_error_short_question() -> None:
    res = subprocess.run([sys.executable, "-m", "app.cli", "x"], capture_output=True, text=True)
    assert res.returncode == 1
    assert "Question is too short" in res.stderr


def test_cli_subprocess_happy_path(tmp_path: Path) -> None:
    # Run CLI with a mocked graph runner via small runner script to test exit code 0 and summary format
    runner_code = f"""
import sys, os
sys.path.insert(0, r"{Path.cwd()}")
from unittest.mock import patch
from app.contracts import Answer, Confidence, Finding, RunMeta, SourceRef, ResearchPlan
from app.cli import main
from datetime import datetime, timezone

mock_ans = Answer(
    run_id="run_cli_test",
    question="Compare vector DBs",
    short_answer="Vector databases provide scalable vector indexing and similarity search.",
    key_findings=[
        Finding(
            claim="Qdrant provides payload filtering with HNSW indexing.",
            source_ids=["s1"],
            supporting_quote="Qdrant provides payload filtering with HNSW indexing.",
        )
    ],
    sources=[
        SourceRef(
            id="s1",
            title="Qdrant Docs",
            url="https://qdrant.tech",
            domain="qdrant.tech",
            retrieved_at=datetime.now(timezone.utc),
            trust=1.0,
        )
    ],
    confidence=Confidence(level="medium", score=0.65, reasons=["Grounded finding"]),
    assumptions=[],
    limitations=[],
    next_steps=[],
    status="ok",
    meta=RunMeta(
        plan=ResearchPlan(
            query_type="comparison",
            sub_questions=["Sub question"],
            search_queries=["search query"],
            tools=["web_search"],
            stop_criteria="done",
        ),
        tool_calls=2,
        latency_ms=150,
        cost_usd=0.0012,
        grounding_rate=1.0,
    ),
)

mock_events = [
    {{"node": "filter", "candidates_in": 5, "deduped": 1, "junk_dropped": 1, "below_threshold": 1, "kept": 2}},
    {{"node": "verify", "initial_findings": 1, "kept_findings": 1, "grounding_rate": 1.0, "confidence_level": "medium", "confidence_score": 0.65}},
]

with patch("app.cli.run_with_trace", return_value=(mock_ans, mock_events)):
    main()
"""
    runner_script = tmp_path / "run_cli_mock.py"
    runner_script.write_text(runner_code, encoding="utf-8")

    out_json = tmp_path / "answer.json"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    res = subprocess.run(
        [sys.executable, str(runner_script), "Compare vector DBs", "--json", str(out_json)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    assert res.returncode == 0
    assert "status=ok" in res.stdout
    assert "latency=150ms" in res.stdout
    assert "cost=$0.0012" in res.stdout
    assert "calls=2" in res.stdout
    assert out_json.exists()


def test_cli_subprocess_insufficient_evidence_exits_2(tmp_path: Path) -> None:
    runner_code = f"""
import sys, os
sys.path.insert(0, r"{Path.cwd()}")
from unittest.mock import patch
from app.contracts import Answer, Confidence, RunMeta
from app.cli import main

mock_ans = Answer(
    run_id="run_cli_insufficient",
    question="Tell me about it",
    short_answer="Insufficient verified evidence was found.",
    key_findings=[],
    sources=[],
    confidence=Confidence(level="low", score=0.0, reasons=["0 findings"]),
    assumptions=[],
    limitations=[],
    next_steps=["Broaden query"],
    status="insufficient_evidence",
    meta=RunMeta(tool_calls=1, latency_ms=100, cost_usd=0.0),
)

with patch("app.cli.run_with_trace", return_value=(mock_ans, [])):
    main()
"""
    runner_script = tmp_path / "run_cli_insufficient.py"
    runner_script.write_text(runner_code, encoding="utf-8")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    res = subprocess.run(
        [sys.executable, str(runner_script), "Tell me about it"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    assert res.returncode == 2
    assert "status=insufficient_evidence" in res.stdout
