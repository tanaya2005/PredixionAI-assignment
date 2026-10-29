"""End-to-end mocked integration tests across all pipeline nodes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest

from app.contracts import Answer, Finding, Intake, ResearchPlan, SynthesisDraft
from app.graph import run, run_with_trace
from app.relevance import RelevanceItem, RelevanceOutput
from app.tools.base import ToolResult


def test_e2e_happy_path(tmp_path: Path) -> None:
    question = "Compare Qdrant vs Milvus vector databases"

    # 1. Mock intake
    mock_intake = Intake(
        query_type="comparison",
        entities=["Qdrant", "Milvus"],
        recency_sensitive=False,
        k_sources=6,
        max_steps=6,
    )

    # 2. Mock planner
    mock_plan = ResearchPlan(
        query_type="comparison",
        sub_questions=[
            "What is Qdrant architecture?",
            "What is Milvus architecture?",
        ],
        search_queries=[
            "qdrant architecture vector search official documentation",
            "milvus architecture vector database documentation",
        ],
        tools=["web_search", "fetch_page"],
        stop_criteria="Gathered architecture details for both",
    )

    # 3. Mock tool calls in execute
    def fake_tool_call(tool: str, args: dict, *_, **__):
        if tool == "web_search":
            q = getattr(args, "query", "") if hasattr(args, "query") else (args.get("query", "") if isinstance(args, dict) else "")
            if "qdrant" in q.lower():
                return ToolResult(
                    items=[
                        {
                            "title": "Qdrant Architecture",
                            "url": "https://qdrant.tech/documentation/architecture",
                            "snippet": "Qdrant is written in Rust and provides payload-based filtering alongside fast HNSW vector indexing. " * 5,
                        }
                    ],
                    provider="tavily",
                    status="ok",
                )
            else:
                return ToolResult(
                    items=[
                        {
                            "title": "Milvus Overview",
                            "url": "https://milvus.io/docs/overview",
                            "snippet": "Milvus is a distributed vector database separating compute and storage into decoupled components. " * 5,
                        }
                    ],
                    provider="tavily",
                    status="ok",
                )
        return ToolResult(items=[], provider="none", status="empty")

    # 4. Mock relevance
    mock_relevance = RelevanceOutput(
        items=[
            RelevanceItem(id="s1", score=0.95, reason="Qdrant official arch"),
            RelevanceItem(id="s2", score=0.92, reason="Milvus official arch"),
        ]
    )

    # 5. Mock synthesis
    mock_draft = SynthesisDraft(
        short_answer="Qdrant is a single-binary Rust engine while Milvus is a distributed cloud-native architecture.",
        findings=[
            Finding(
                claim="Qdrant is built with Rust and integrates payload filtering with HNSW.",
                source_ids=["s1"],
                supporting_quote="Qdrant is written in Rust and provides payload-based filtering alongside fast HNSW vector indexing.",
            ),
            Finding(
                claim="Milvus separates compute and storage into decoupled microservice components.",
                source_ids=["s2"],
                supporting_quote="Milvus is a distributed vector database separating compute and storage into decoupled components.",
            ),
        ],
        assumptions=["Default cloud-native configuration applies."],
        limitations=["Benchmark figures depend on workload and hardware."],
        next_steps=["Test both databases with your target dimensionality."],
        contradictions=[],
    )

    def fake_router_call_json(*args, **kwargs):
        role = kwargs.get("role")
        schema = kwargs.get("schema")
        if role == "intake" or schema == Intake:
            return mock_intake
        if role == "plan" or schema == ResearchPlan:
            return mock_plan
        if role == "filter" or schema == RelevanceOutput:
            return mock_relevance
        if role == "synthesize" or schema == SynthesisDraft:
            return mock_draft
        return None

    with patch("app.llm.router.call_json", side_effect=fake_router_call_json), \
         patch("app.nodes.execute.call", side_effect=fake_tool_call):

        answer, events = run_with_trace(question, trace_dir=tmp_path)

    # Assertions
    assert isinstance(answer, Answer)
    assert answer.status == "ok"
    assert len(answer.key_findings) >= 2
    assert answer.meta.grounding_rate == 1.0
    assert len(answer.sources) >= 2
    assert answer.confidence.score >= 0.5
    assert len(events) >= 5

    # Check that traces/<run_id>.json was created
    trace_json = tmp_path / f"{answer.run_id}.json"
    assert trace_json.exists()


def test_e2e_all_tools_fail_results_in_insufficient_evidence(tmp_path: Path) -> None:
    from app.errors import ToolExhausted

    question = "What is the secret sauce of Project X?"

    mock_intake = Intake(
        query_type="explain",
        entities=["Project X"],
        recency_sensitive=False,
        k_sources=6,
        max_steps=6,
    )

    mock_plan = ResearchPlan(
        query_type="explain",
        sub_questions=["What is Project X?"],
        search_queries=["project x documentation"],
        tools=["web_search"],
        stop_criteria="Found details",
    )

    with patch("app.nodes.intake.router.call_json", return_value=mock_intake), \
         patch("app.nodes.plan.router.call_json", return_value=mock_plan), \
         patch("app.nodes.execute.call", side_effect=ToolExhausted("All search providers failed")):

        answer = run(question, trace_dir=tmp_path)

    assert isinstance(answer, Answer)
    assert answer.status == "insufficient_evidence"
    assert len(answer.key_findings) == 0
    assert "Insufficient verified evidence" in answer.short_answer
    assert len(answer.next_steps) > 0


def test_e2e_fabricated_quote_dropped_and_degraded_when_grounding_low(tmp_path: Path) -> None:
    question = "Explain HNSW graph indexing"

    mock_intake = Intake(
        query_type="explain",
        entities=["HNSW"],
        recency_sensitive=False,
        k_sources=6,
        max_steps=6,
    )

    mock_plan = ResearchPlan(
        query_type="explain",
        sub_questions=["How does HNSW work?"],
        search_queries=["hnsw graph indexing algorithm vector search"],
        tools=["web_search"],
        stop_criteria="Found HNSW details",
    )

    def fake_search(tool: str, args: dict, *_, **__):
        return ToolResult(
            items=[
                {
                    "title": "HNSW Algorithm Overview",
                    "url": "https://example.com/hnsw",
                    "snippet": "Hierarchical Navigable Small World graphs create multi-layer proximity graphs for approximate nearest neighbor search. " * 5,
                }
            ],
            provider="tavily",
            status="ok",
        )

    mock_relevance = RelevanceOutput(
        items=[RelevanceItem(id="s1", score=0.9, reason="Relevant")]
    )

    # Synthesis invents an ungrounded fabricated quote that will fail verification
    mock_draft = SynthesisDraft(
        short_answer="HNSW uses multi-layer graphs.",
        findings=[
            Finding(
                claim="HNSW uses a proprietary quantum algorithm to calculate similarities instantly.",
                source_ids=["s1"],
                supporting_quote="This quote was completely made up by an overconfident model and does not exist in the source.",
            )
        ],
    )

    def fake_router_call_json(*args, **kwargs):
        role = kwargs.get("role")
        schema = kwargs.get("schema")
        if role == "intake" or schema == Intake:
            return mock_intake
        if role == "plan" or schema == ResearchPlan:
            return mock_plan
        if role == "filter" or schema == RelevanceOutput:
            return mock_relevance
        if role == "synthesize" or schema == SynthesisDraft:
            return mock_draft
        return None

    with patch("app.llm.router.call_json", side_effect=fake_router_call_json), \
         patch("app.nodes.execute.call", side_effect=fake_search):

        answer = run(question, trace_dir=tmp_path)

    assert isinstance(answer, Answer)
    # The fabricated finding must be dropped by verify node
    assert len(answer.key_findings) == 0
    # 0 surviving findings -> status is insufficient_evidence (or degraded if grounding < 0.5)
    assert answer.status in ("degraded", "insufficient_evidence")
    assert answer.meta.dropped_findings == 1
    assert answer.meta.grounding_rate == 0.0
