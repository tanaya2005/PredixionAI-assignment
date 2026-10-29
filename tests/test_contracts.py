"""Unit tests for Pydantic data contracts and validation constraints."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

from app.contracts import (
    Answer,
    Confidence,
    Finding,
    Intake,
    ResearchPlan,
    RunMeta,
    Source,
    SourceRef,
    SynthesisDraft,
    ToolCallLog,
)
from app.graph import State


def test_research_plan_valid() -> None:
    plan = ResearchPlan(
        query_type="comparison",
        sub_questions=["What is vector DB A?", "What is vector DB B?"],
        search_queries=["vector db comparison", "vector db benchmarks"],
        tools=["web_search", "fetch_page"],
        stop_criteria="sufficient documentation retrieved",
    )
    assert plan.query_type == "comparison"
    assert len(plan.sub_questions) == 2
    assert len(plan.search_queries) == 2
    assert len(plan.tools) == 2


def test_research_plan_rejects_empty_tools() -> None:
    with pytest.raises(ValidationError) as exc:
        ResearchPlan(
            query_type="comparison",
            sub_questions=["q1"],
            search_queries=["sq1"],
            tools=[],
            stop_criteria="stop",
        )
    assert "tools" in str(exc.value)


def test_research_plan_rejects_too_many_tools() -> None:
    with pytest.raises(ValidationError) as exc:
        ResearchPlan(
            query_type="comparison",
            sub_questions=["q1"],
            search_queries=["sq1"],
            tools=["t1", "t2", "t3", "t4", "t5"],
            stop_criteria="stop",
        )
    assert "tools" in str(exc.value)


def test_research_plan_rejects_empty_sub_questions() -> None:
    with pytest.raises(ValidationError) as exc:
        ResearchPlan(
            query_type="comparison",
            sub_questions=[],
            search_queries=["sq1"],
            tools=["web_search"],
            stop_criteria="stop",
        )
    assert "sub_questions" in str(exc.value)


def test_research_plan_rejects_too_many_sub_questions() -> None:
    with pytest.raises(ValidationError) as exc:
        ResearchPlan(
            query_type="comparison",
            sub_questions=["q1", "q2", "q3", "q4", "q5", "q6"],
            search_queries=["sq1"],
            tools=["web_search"],
            stop_criteria="stop",
        )
    assert "sub_questions" in str(exc.value)


def test_research_plan_rejects_empty_search_queries() -> None:
    with pytest.raises(ValidationError) as exc:
        ResearchPlan(
            query_type="comparison",
            sub_questions=["q1"],
            search_queries=[],
            tools=["web_search"],
            stop_criteria="stop",
        )
    assert "search_queries" in str(exc.value)


def test_research_plan_rejects_too_many_search_queries() -> None:
    with pytest.raises(ValidationError) as exc:
        ResearchPlan(
            query_type="comparison",
            sub_questions=["q1"],
            search_queries=["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
            tools=["web_search"],
            stop_criteria="stop",
        )
    assert "search_queries" in str(exc.value)


def test_finding_valid() -> None:
    finding = Finding(
        claim="Qdrant provides a single binary deployment option for low operational overhead.",
        source_ids=["s1"],
        supporting_quote="Qdrant is distributed as a single binary and can run without external dependencies.",
        corroboration=1,
    )
    assert len(finding.claim) >= 20
    assert len(finding.supporting_quote) >= 40


def test_finding_rejects_short_claim() -> None:
    with pytest.raises(ValidationError) as exc:
        Finding(
            claim="Too short",
            source_ids=["s1"],
            supporting_quote="A sufficiently long quote that exceeds forty characters in total length.",
            corroboration=1,
        )
    assert "claim" in str(exc.value)


def test_finding_rejects_short_quote() -> None:
    with pytest.raises(ValidationError) as exc:
        Finding(
            claim="A valid claim that easily exceeds the minimum twenty character length.",
            source_ids=["s1"],
            supporting_quote="Short quote.",
            corroboration=1,
        )
    assert "supporting_quote" in str(exc.value)


def test_finding_rejects_empty_sources() -> None:
    with pytest.raises(ValidationError) as exc:
        Finding(
            claim="A valid claim that easily exceeds the minimum twenty character length.",
            source_ids=[],
            supporting_quote="A sufficiently long quote that exceeds forty characters in total length.",
            corroboration=1,
        )
    assert "source_ids" in str(exc.value)


def test_intake_valid() -> None:
    intake = Intake(
        query_type="comparison",
        entities=["Qdrant", "Milvus"],
        recency_sensitive=True,
        k_sources=6,
        max_steps=6,
    )
    assert intake.query_type == "comparison"
    assert intake.k_sources == 6


def test_intake_rejects_bad_query_type() -> None:
    with pytest.raises(ValidationError) as exc:
        Intake(query_type="invalid_type")  # type: ignore[arg-type]
    assert "query_type" in str(exc.value)


def test_extra_field_forbidden_on_research_plan() -> None:
    with pytest.raises(ValidationError) as exc:
        ResearchPlan(
            query_type="comparison",
            sub_questions=["q1"],
            search_queries=["sq1"],
            tools=["web_search"],
            stop_criteria="stop",
            unexpected="forbidden",  # type: ignore[call-arg]
        )
    assert "extra" in str(exc.value).lower()


def test_extra_field_forbidden_on_intake() -> None:
    with pytest.raises(ValidationError) as exc:
        Intake(query_type="explain", rogue_field=123)  # type: ignore[call-arg]
    assert "extra" in str(exc.value).lower()


def test_extra_field_forbidden_on_finding() -> None:
    with pytest.raises(ValidationError) as exc:
        Finding(
            claim="A valid claim that easily exceeds the minimum twenty character length.",
            source_ids=["s1"],
            supporting_quote="A sufficiently long quote that exceeds forty characters in total length.",
            corroboration=1,
            unwanted=True,  # type: ignore[call-arg]
        )
    assert "extra" in str(exc.value).lower()


def test_source_serialization_iso_z() -> None:
    now = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
    pub = date(2026, 10, 15)
    source = Source(
        id="s1",
        url="https://example.com/doc",
        title="Doc Title",
        domain="example.com",
        published=pub,
        retrieved_at=now,
        text="Sample document text.",
        trust=1.0,
        tool="web_search",
        provider="tavily",
        untrusted=False,
    )
    dumped = source.model_dump(mode="json")
    assert dumped["published"] == "2026-10-15"
    assert dumped["retrieved_at"] == "2026-10-30T12:00:00Z"


def test_answer_model_dump_json() -> None:
    dt = datetime(2026, 10, 29, 19, 40, 22, tzinfo=timezone.utc)
    d = date(2026, 4, 2)
    answer = Answer(
        run_id="r_123",
        question="What is Qdrant?",
        short_answer="Qdrant is a vector database.",
        key_findings=[
            Finding(
                claim="Qdrant ships single-binary deployment with no external dependencies.",
                source_ids=["s1"],
                supporting_quote="Qdrant is distributed as a single binary and can run without any external dependencies",
                corroboration=1,
            )
        ],
        sources=[
            SourceRef(
                id="s1",
                title="Qdrant Docs",
                url="https://qdrant.tech",
                domain="qdrant.tech",
                published=d,
                retrieved_at=dt,
                trust=1.0,
                untrusted=False,
            )
        ],
        confidence=Confidence(
            level="high",
            score=0.9,
            reasons=["All findings verified"],
        ),
        assumptions=["Standard architecture assumption"],
        limitations=["Only English documentation reviewed"],
        next_steps=["Deploy on test cluster"],
        status="ok",
        meta=RunMeta(
            tool_calls=2,
            latency_ms=1200,
            cost_usd=0.002,
            grounding_rate=1.0,
            dropped_findings=0,
            cache_hits=1,
            models={"synthesize": "gemini-2.5-flash"},
            llm_calls=1,
        ),
    )
    data = answer.model_dump(mode="json")
    assert data["status"] == "ok"
    assert data["sources"][0]["published"] == "2026-04-02"
    assert data["sources"][0]["retrieved_at"] == "2026-10-29T19:40:22Z"


def test_state_initialization() -> None:
    state = State(
        run_id="run_001",
        question="What is the architecture?",
        trace_path=Path("traces/run_001.jsonl"),
    )
    assert state.run_id == "run_001"
    assert state.question == "What is the architecture?"
    assert state.status == "ok"
    assert state.sources == []
    assert state.calls == []
    assert state.kept == []
    assert state.dropped == []
    assert state.cost_usd == 0.0
