"""Unit tests for individual pipeline execution nodes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest

from app.config import BUDGETS, TOOL_POLICY
from app.contracts import Intake, ResearchPlan, Source
from app.graph import State
from app.nodes.execute import execute
from app.nodes.intake import intake
from app.nodes.plan import plan


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


def test_plan_drops_disallowed_and_unknown_tools(tmp_path: Path) -> None:
    state = State(
        run_id="run_plan_tools",
        question="Compare vector databases",
        trace_path=tmp_path / "run_plan_tools.jsonl",
        intake=Intake(query_type="comparison", entities=["Qdrant"]),
    )

    # For comparison, allowed tools are ["web_search", "fetch_page"]
    mock_plan = ResearchPlan(
        query_type="comparison",
        sub_questions=["Sub Q1", "Sub Q2"],
        search_queries=["vector db comparison", "vector db benchmarks"],
        tools=["web_search", "github", "hallucinated_tool"],  # github & hallucinated are disallowed
        stop_criteria="Stop once docs retrieved",
    )

    with patch("app.llm.router.call_json", return_value=mock_plan):
        updated_state = plan(state)

    assert updated_state.plan is not None
    assert updated_state.plan.tools == ["web_search"]
    assert "github" not in updated_state.plan.tools
    assert "hallucinated_tool" not in updated_state.plan.tools


def test_plan_trims_queries_and_subquestions_to_caps(tmp_path: Path) -> None:
    state = State(
        run_id="run_plan_caps",
        question="Comprehensive overview",
        trace_path=tmp_path / "run_plan_caps.jsonl",
        intake=Intake(query_type="explain", entities=[]),
    )

    long_query = "q " * 100  # Exceeds 160 chars
    mock_plan = ResearchPlan(
        query_type="explain",
        sub_questions=["SQ1", "SQ2", "SQ3", "SQ4", "SQ5"],  # 5 sub-questions
        search_queries=["Q1", "Q2", "Q3", "Q4", "Q5", long_query],  # 6 queries, one very long
        tools=["web_search", "wikipedia"],
        stop_criteria="Done",
    )

    # Simulate router returning 7 queries and 6 sub-questions via a dict or relaxed model
    with patch("app.llm.router.call_json", return_value=mock_plan):
        # Add an extra element directly to verify node trimming
        mock_plan.search_queries.append("Q7_extra")
        mock_plan.sub_questions.append("SQ6_extra")
        updated_state = plan(state)

    assert updated_state.plan is not None
    assert len(updated_state.plan.search_queries) <= 6
    assert len(updated_state.plan.sub_questions) <= 5
    for q in updated_state.plan.search_queries:
        assert len(q) <= 160


def test_plan_router_none_falls_back(tmp_path: Path) -> None:
    state = State(
        run_id="run_plan_none",
        question="Unplannable question",
        trace_path=tmp_path / "run_plan_none.jsonl",
        intake=Intake(query_type="comparison", entities=[]),
    )

    with patch("app.llm.router.call_json", return_value=None):
        updated_state = plan(state)

    assert updated_state.plan is not None
    assert updated_state.plan.query_type == "comparison"
    assert updated_state.plan.sub_questions == ["Unplannable question"]
    assert updated_state.plan.search_queries == ["Unplannable question"]
    assert "web_search" in updated_state.plan.tools
    assert updated_state.status == "degraded"

    trace_content = (tmp_path / "run_plan_none.jsonl").read_text(encoding="utf-8")
    assert "plan fallback" in trace_content


def test_plan_written_to_trace_before_tools(tmp_path: Path) -> None:
    state = State(
        run_id="run_plan_trace",
        question="Vector search explain",
        trace_path=tmp_path / "run_plan_trace.jsonl",
        intake=Intake(query_type="explain", entities=["HNSW"]),
    )

    mock_plan = ResearchPlan(
        query_type="explain",
        sub_questions=["How does HNSW work?"],
        search_queries=["HNSW graph index algorithm"],
        tools=["web_search", "wikipedia"],
        stop_criteria="sufficient evidence",
    )

    with patch("app.llm.router.call_json", return_value=mock_plan):
        updated_state = plan(state)

    assert updated_state.plan is not None
    trace_file = tmp_path / "run_plan_trace.jsonl"
    assert trace_file.exists()
    content = trace_file.read_text(encoding="utf-8")
    assert "HNSW graph index algorithm" in content
    assert '"node": "plan"' in content


def test_execute_happy_path_produces_contiguous_sources(tmp_path: Path) -> None:
    state = State(
        run_id="run_exec_happy",
        question="Compare vector databases",
        trace_path=tmp_path / "run_exec_happy.jsonl",
        intake=Intake(query_type="comparison", entities=["Qdrant", "Milvus"], k_sources=6, max_steps=6),
        plan=ResearchPlan(
            query_type="comparison",
            sub_questions=["What is Qdrant?", "What is Milvus?"],
            search_queries=["qdrant vector db", "milvus vector db"],
            tools=["web_search", "fetch_page"],
            stop_criteria="docs found",
        ),
    )

    from app.tools.base import ToolResult

    def fake_call(tool: str, args: Any, *, tracer: Any) -> ToolResult:
        if tool == "web_search":
            q = getattr(args, "query", "")
            return ToolResult(
                items=[
                    {"title": f"Doc for {q}", "url": f"https://example.com/{abs(hash(q))}", "snippet": f"Snippet for {q}"}
                ],
                provider="tavily",
                status="ok",
            )
        elif tool == "fetch_page":
            u = getattr(args, "url", "")
            return ToolResult(
                items=[
                    {"title": f"Page for {u}", "url": u, "text": f"Extensive documentation text for {u}. " * 20}
                ],
                provider="jina",
                status="ok",
            )
        return ToolResult(items=[], provider="unknown", status="empty")

    with patch("app.nodes.execute.call", side_effect=fake_call):
        updated_state = execute(state)

    assert len(updated_state.sources) >= 2
    assert updated_state.sources[0].id == "s1"
    assert updated_state.sources[1].id == "s2"
    assert updated_state.status == "ok"
    assert len(updated_state.calls) <= 6


def test_execute_partial_search_failure_still_gathers_sources(tmp_path: Path) -> None:
    state = State(
        run_id="run_exec_partial",
        question="Vector search analysis",
        trace_path=tmp_path / "run_exec_partial.jsonl",
        intake=Intake(query_type="comparison", max_steps=6),
        plan=ResearchPlan(
            query_type="comparison",
            sub_questions=["Q1", "Q2"],
            search_queries=["failing query", "succeeding query"],
            tools=["web_search", "fetch_page"],
            stop_criteria="done",
        ),
    )

    from app.errors import ToolExhausted
    from app.tools.base import ToolResult

    def fake_call(tool: str, args: Any, *, tracer: Any) -> ToolResult:
        if tool == "web_search":
            if getattr(args, "query") == "failing query":
                raise ToolExhausted("Search provider 500")
            return ToolResult(
                items=[{"title": "Valid Hit", "url": "https://example.com/valid", "snippet": "Text"}],
                provider="tavily",
                status="ok",
            )
        return ToolResult(
            items=[{"title": "Valid Hit", "url": "https://example.com/valid", "text": "Full text content " * 15}],
            provider="jina",
            status="ok",
        )

    with patch("app.nodes.execute.call", side_effect=fake_call):
        updated_state = execute(state)

    assert len(updated_state.sources) >= 1
    assert any(c.status == "error" for c in updated_state.calls)
    assert any(c.status == "ok" for c in updated_state.calls)


def test_execute_all_tools_exhausted_results_in_insufficient_evidence(tmp_path: Path) -> None:
    state = State(
        run_id="run_exec_all_fail",
        question="Dead network scenario",
        trace_path=tmp_path / "run_exec_all_fail.jsonl",
        intake=Intake(query_type="explain", max_steps=6),
        plan=ResearchPlan(
            query_type="explain",
            sub_questions=["Q1"],
            search_queries=["search query"],
            tools=["web_search"],
            stop_criteria="done",
        ),
    )

    from app.errors import ToolExhausted

    with patch("app.nodes.execute.call", side_effect=ToolExhausted("All down")):
        updated_state = execute(state)

    assert len(updated_state.sources) == 0
    assert updated_state.status == "insufficient_evidence"


def test_execute_budget_exhaustion_after_six_calls(tmp_path: Path) -> None:
    state = State(
        run_id="run_exec_budget",
        question="Heavy search query",
        trace_path=tmp_path / "run_exec_budget.jsonl",
        intake=Intake(query_type="comparison", max_steps=6),
        plan=ResearchPlan(
            query_type="comparison",
            sub_questions=["SQ1", "SQ2"],
            search_queries=["q1", "q2", "q3", "q4", "q5", "q6"],
            tools=["web_search"],
            stop_criteria="done",
        ),
    )

    from app.tools.base import ToolResult

    call_count = 0

    def fake_call(tool: str, args: Any, *, tracer: Any) -> ToolResult:
        nonlocal call_count
        call_count += 1
        return ToolResult(
            items=[
                {"title": f"Hit {call_count}a", "url": f"https://example.com/hit_{call_count}a", "snippet": "Content " * 15},
                {"title": f"Hit {call_count}b", "url": f"https://example.com/hit_{call_count}b", "snippet": "Content " * 15},
            ],
            provider="tavily",
            status="ok",
        )

    with patch("app.nodes.execute.call", side_effect=fake_call):
        updated_state = execute(state)

    # Hard cap of 6 calls must be strictly honoured
    assert len(updated_state.calls) == 6
    assert updated_state.status == "degraded"


def test_execute_wall_clock_breach_degrades_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = State(
        run_id="run_exec_timeout",
        question="Slow query",
        trace_path=tmp_path / "run_exec_timeout.jsonl",
        started_at=1_000_000.0,
        intake=Intake(query_type="comparison", max_steps=6),
        plan=ResearchPlan(
            query_type="comparison",
            sub_questions=["SQ1"],
            search_queries=["q1", "q2", "q3"],
            tools=["web_search"],
            stop_criteria="done",
        ),
    )

    # Simulate clock exceeding 60s
    monkeypatch.setattr("time.time", lambda: 1_000_070.0)

    updated_state = execute(state)
    assert updated_state.status in ("degraded", "insufficient_evidence")
    assert len(updated_state.calls) == 0  # Exited immediately before making calls


