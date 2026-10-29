"""Unit tests for individual pipeline execution nodes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest

from app.config import BUDGETS
from app.contracts import Intake
from app.graph import State
from app.nodes.intake import intake


@pytest.mark.parametrize("q_type", ["comparison", "list", "explain", "howto"])
def test_intake_classification_types(tmp_path: Path, q_type: str) -> None:
    state = State(
        run_id="run_intake_type",
        question="Test query",
        trace_path=tmp_path / "run_intake_type.jsonl",
    )

    mock_intake = Intake(
        query_type=q_type,  # type: ignore[arg-type]
        entities=["Entity A", "Entity B"],
        recency_sensitive=False,
        k_sources=6,
        max_steps=6,
    )

    with patch("app.llm.router.call_json", return_value=mock_intake):
        updated_state = intake(state)

    assert updated_state.intake is not None
    assert updated_state.intake.query_type == q_type
    if q_type == "howto":
        assert updated_state.intake.k_sources == 4
    else:
        assert updated_state.intake.k_sources == 6
    assert len(updated_state.intake.entities) == 2


def test_intake_router_returns_none_falls_back(tmp_path: Path) -> None:
    state = State(
        run_id="run_intake_none",
        question="Unclassifiable query",
        trace_path=tmp_path / "run_intake_none.jsonl",
    )

    with patch("app.llm.router.call_json", return_value=None):
        updated_state = intake(state)

    assert updated_state.intake is not None
    assert updated_state.intake.query_type == "explain"
    assert updated_state.intake.entities == []
    assert updated_state.intake.recency_sensitive is False
    assert updated_state.status == "degraded"

    trace_file = tmp_path / "run_intake_none.jsonl"
    assert trace_file.exists()
    content = trace_file.read_text(encoding="utf-8")
    assert "intake fallback" in content


def test_intake_budget_clamping(tmp_path: Path) -> None:
    state = State(
        run_id="run_intake_clamp",
        question="How to optimize vectors",
        trace_path=tmp_path / "run_intake_clamp.jsonl",
    )

    mock_intake = Intake(
        query_type="howto",
        entities=["Vectors"],
        recency_sensitive=True,
        k_sources=10,  # Exceeds max 4 for howto
        max_steps=12,  # Exceeds BUDGETS.max_tool_calls (6)
    )

    with patch("app.llm.router.call_json", return_value=mock_intake):
        updated_state = intake(state)

    assert updated_state.intake is not None
    assert updated_state.intake.k_sources == 4
    assert updated_state.intake.max_steps == BUDGETS.max_tool_calls
    assert updated_state.intake.recency_sensitive is True

    content = (tmp_path / "run_intake_clamp.jsonl").read_text(encoding="utf-8")
    assert "recency_sensitive" in content


def test_intake_unexpected_exception_never_raises(tmp_path: Path) -> None:
    state = State(
        run_id="run_intake_crash",
        question="Corrupted input",
        trace_path=tmp_path / "run_intake_crash.jsonl",
    )

    with patch("app.llm.router.call_json", side_effect=RuntimeError("Catastrophic error")):
        updated_state = intake(state)

    assert updated_state.intake is not None
    assert updated_state.intake.query_type == "explain"
    assert updated_state.status == "degraded"
