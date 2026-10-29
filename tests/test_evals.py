"""Unit tests for the evaluation harness (runs with mocked runner, zero network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.contracts import Answer, Confidence, Finding, Intake, ResearchPlan, RunMeta, SourceRef
from evals.run_eval import evaluate_checks, run_evaluation


def test_evaluate_checks_mechanisms() -> None:
    ans = Answer(
        run_id="run_eval_test",
        question="Which vector DB is fastest?",
        short_answer="Benchmarks conflict and show variable performance depending on filtering and dataset size.",
        key_findings=[
            Finding(
                claim="Qdrant claims higher throughput on filtered benchmarks.",
                source_ids=["s1"],
                supporting_quote="Qdrant claims higher throughput on filtered benchmarks.",
            ),
            Finding(
                claim="Milvus benchmarks show faster ingestion speeds on large batches.",
                source_ids=["s2"],
                supporting_quote="Milvus benchmarks show faster ingestion speeds on large batches.",
            ),
        ],
        sources=[
            SourceRef(id="s1", title="Qdrant", url="https://qdrant.tech", domain="qdrant.tech", retrieved_at="2026-10-29T10:00:00Z", trust=1.0),
            SourceRef(id="s2", title="Milvus", url="https://milvus.io", domain="milvus.io", retrieved_at="2026-10-29T10:00:00Z", trust=1.0),
        ],
        confidence=Confidence(level="medium", score=0.60, reasons=["Conflicting benchmark results"]),
        assumptions=[],
        limitations=["Conflicting data from separate vendor benchmarks"],
        next_steps=[],
        status="ok",
        meta=RunMeta(grounding_rate=1.0),
    )

    # Check passes
    passed, fails = evaluate_checks(ans, {
        "has_min_sources": 2,
        "has_min_findings": 2,
        "mentions_disagreement": True,
        "max_confidence": "medium",
    })
    assert passed is True
    assert len(fails) == 0

    # Failing check
    passed_fail, fails_list = evaluate_checks(ans, {"has_min_sources": 5})
    assert passed_fail is False
    assert "sources: 2 < 5" in fails_list[0]


def test_run_eval_produces_report_and_handles_missing_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure missing keys degrade gracefully into skipped rows in report.md
    with patch("evals.run_eval.has_key", return_value=False):
        run_evaluation()

    report_path = Path("evals/report.md")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Evaluation Benchmark Report" in content
    assert "Non-Agentic Baseline Comparison" in content
    assert "What Worked Well, and What Did Not" in content
    assert "skipped (missing API keys)" in content
