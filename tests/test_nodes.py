"""Unit tests for individual pipeline execution nodes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest

from app.config import BUDGETS, TOOL_POLICY
from app.contracts import Intake, ResearchPlan, Source
from app.graph import State
from app.nodes.execute import execute
from app.nodes.filter import filter_evidence
from app.nodes.intake import intake
from app.nodes.plan import plan
from app.nodes.synthesize import synthesize


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


def test_filter_scorer_drops_irrelevant_and_keeps_relevant(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    s_rel = Source(
        id="s1",
        url="https://docs.python.org/3/whatsnew/3.11.html",
        title="Python 3.11 Release Notes",
        text="Python 3.11 includes significant performance improvements and fine-grained error locations. " * 10,
        domain="docs.python.org",
        retrieved_at=now,
        trust=1.0,
        tool="web_search",
        provider="tavily",
    )
    s_irrel = Source(
        id="s2",
        url="https://example.com/unrelated-shoes",
        title="Buy Discount Shoes",
        text="Buy running shoes online today with fast shipping and huge discounts. " * 10,
        domain="example.com",
        retrieved_at=now,
        trust=0.5,
        tool="web_search",
        provider="tavily",
    )
    state = State(
        run_id="run_filter_rel",
        question="What is new in Python 3.11?",
        trace_path=tmp_path / "run_filter_rel.jsonl",
        sources=[s_rel, s_irrel],
        plan=ResearchPlan(
            query_type="explain",
            sub_questions=["What are Python 3.11 features?"],
            search_queries=["python 3.11 new features"],
            tools=["web_search"],
            stop_criteria="done",
        ),
    )

    with patch("app.relevance.router.call_json") as mock_router:
        from app.relevance import RelevanceItem, RelevanceOutput

        mock_router.return_value = RelevanceOutput(
            items=[
                RelevanceItem(id="s1", score=0.95, reason="Directly answers features"),
                RelevanceItem(id="s2", score=0.05, reason="Irrelevant shoe store"),
            ]
        )
        updated_state = filter_evidence(state)

    assert len(updated_state.kept) == 1
    assert updated_state.kept[0].url == s_rel.url
    assert updated_state.kept[0].id == "s1"


def test_filter_scorer_failure_preserves_trust_ordering(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    s_low_trust = Source(
        id="s1",
        url="https://medium.com/post1",
        title="Post 1",
        text="Good content discussing the question with enough characters to pass junk filtering. " * 8,
        domain="medium.com",
        retrieved_at=now,
        trust=0.5,
        tool="web_search",
        provider="tavily",
    )
    s_high_trust = Source(
        id="s2",
        url="https://docs.python.org/post2",
        title="Post 2",
        text="High trust official doc content discussing the question in detail with plenty of text. " * 8,
        domain="docs.python.org",
        retrieved_at=now,
        trust=1.0,
        tool="web_search",
        provider="tavily",
    )
    state = State(
        run_id="run_filter_fail",
        question="What is new?",
        trace_path=tmp_path / "run_filter_fail.jsonl",
        sources=[s_low_trust, s_high_trust],
    )

    with patch("app.relevance.router.call_json", return_value=None):
        updated_state = filter_evidence(state)

    assert len(updated_state.kept) == 2
    # High trust source should be ranked first and renamed s1
    assert updated_state.kept[0].url == s_high_trust.url
    assert updated_state.kept[0].id == "s1"
    assert updated_state.kept[1].url == s_low_trust.url
    assert updated_state.kept[1].id == "s2"


def test_filter_all_zeros_scorer_still_keeps_at_least_two(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    texts = [
        "Vector databases index high-dimensional embeddings efficiently using hierarchical navigable small world graphs and inverted file indexes with product quantization for similarity queries in machine learning applications. " * 4,
        "Photosynthesis in green plants converts light energy, usually from the sun, into chemical energy that can later be released to fuel the organisms' metabolic activities through cellular respiration processes. " * 4,
        "The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth following the Middle Ages, generally described as taking place from the 14th century to the 17th century. " * 4,
    ]
    sources = [
        Source(
            id=f"s{i}",
            url=f"https://site{i}.com/page{i}",
            title=f"Page {i}",
            text=texts[i - 1],
            domain=f"site{i}.com",
            retrieved_at=now,
            trust=0.5 + (0.1 * i),
            tool="web_search",
            provider="tavily",
        )
        for i in range(1, 4)
    ]
    state = State(
        run_id="run_filter_zeros",
        question="Question?",
        trace_path=tmp_path / "run_filter_zeros.jsonl",
        sources=sources,
    )

    with patch("app.relevance.router.call_json") as mock_router:
        from app.relevance import RelevanceItem, RelevanceOutput

        mock_router.return_value = RelevanceOutput(
            items=[RelevanceItem(id=f"s{i}", score=0.0, reason="zero") for i in range(1, 4)]
        )
        updated_state = filter_evidence(state)

    # Scorer returned 0 for everything, but guard keeps >= 2
    assert len(updated_state.kept) >= 2
    assert updated_state.kept[0].id == "s1"
    assert updated_state.kept[1].id == "s2"


def test_filter_junk_drop_reasons_aggregated(tmp_path: Path) -> None:
    from datetime import datetime, timezone
    from app.trace import Tracer

    now = datetime.now(timezone.utc)
    s_short = Source(
        id="s1",
        url="https://site1.com/short",
        title="Short",
        text="Too short",
        domain="site1.com",
        retrieved_at=now,
        trust=0.5,
        tool="web_search",
        provider="tavily",
    )
    s_404 = Source(
        id="s2",
        url="https://site2.com/404",
        title="404",
        text="404 Page Not Found error occurred. " * 15,
        domain="site2.com",
        retrieved_at=now,
        trust=0.5,
        tool="web_search",
        provider="tavily",
    )
    s_good = Source(
        id="s3",
        url="https://site3.com/good",
        title="Good",
        text="A completely valid source body that provides meaningful explanations and analysis. " * 8,
        domain="site3.com",
        retrieved_at=now,
        trust=0.8,
        tool="web_search",
        provider="tavily",
    )
    state = State(
        run_id="run_filter_junk",
        question="Question?",
        trace_path=tmp_path / "run_filter_junk.jsonl",
        sources=[s_short, s_404, s_good],
    )

    with patch("app.relevance.router.call_json") as mock_router:
        from app.relevance import RelevanceItem, RelevanceOutput

        mock_router.return_value = RelevanceOutput(
            items=[RelevanceItem(id="s3", score=0.9, reason="valid")]
        )
        updated_state = filter_evidence(state)

    assert len(updated_state.kept) == 1
    assert updated_state.kept[0].url == s_good.url

    tracer = Tracer(run_id="run_filter_junk", trace_dir=tmp_path)
    # Read from the file on disk as Tracer instance in test_nodes is separate from filter_evidence's tracer
    trace_file = state.trace_path
    assert trace_file.exists()
    import json
    lines = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").strip().splitlines() if line]
    filter_events = [e for e in lines if e.get("node") == "filter"]
    assert len(filter_events) == 1
    fe = filter_events[0]
    assert fe["candidates_in"] == 3
    assert fe["junk_dropped"] == 2
    assert "text_under_300_chars" in fe["junk_reasons"]
    assert "html_error_marker" in fe["junk_reasons"]
    assert fe["kept"] == 1


def test_synthesize_untrusted_wrappers_and_char_budget(tmp_path: Path) -> None:
    from datetime import datetime, timezone
    from app.contracts import Finding, SynthesisDraft

    now = datetime.now(timezone.utc)
    # Create sources exceeding 40,000 chars total
    # Each source has 15,000 chars. 3 sources = 45,000 chars
    text_content = "This is a detailed analysis of high dimensional vector databases and indexing techniques. " * 160
    sources = [
        Source(
            id=f"s{i}",
            url=f"https://source{i}.com/docs",
            title=f"Source {i}",
            text=text_content,
            domain=f"source{i}.com",
            retrieved_at=now,
            trust=0.9 - (0.1 * i),
            tool="web_search",
            provider="tavily",
        )
        for i in range(1, 4)
    ]
    state = State(
        run_id="run_synth_budget",
        question="How do vector databases index embeddings?",
        trace_path=tmp_path / "run_synth_budget.jsonl",
        kept=sources,
    )

    captured_prompt = ""

    def fake_router_call(*args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = kwargs.get("prompt", "")
        return SynthesisDraft(
            short_answer="Vector databases index embeddings using graphs and quantization.",
            findings=[
                Finding(
                    claim="Vector databases use graph-based indexes for similarity search.",
                    source_ids=["s1"],
                    supporting_quote="This is a detailed analysis of high dimensional vector databases and indexing techniques.",
                )
            ],
            contradictions=["One paper claims IVFPQ is faster than HNSW."],
        )

    with patch("app.nodes.synthesize.router.call_json", side_effect=fake_router_call):
        updated_state = synthesize(state)

    assert updated_state.draft is not None
    assert len(updated_state.draft.findings) == 1
    # Check untrusted wrappers exist in prompt
    assert '<untrusted_document id="s1" url="https://source1.com/docs">' in captured_prompt
    assert "</untrusted_document>" in captured_prompt
    # Char budget MAX_EVIDENCE_CHARS (40,000) was respected: s3 was omitted
    assert "https://source3.com/docs" not in captured_prompt
    assert len(captured_prompt) < 45000
    # Contradictions are carried over
    assert "One paper claims IVFPQ is faster than HNSW." in updated_state.draft.contradictions
    assert any("Contradiction:" in lim for lim in updated_state.draft.limitations)


def test_synthesize_unknown_source_id_survives_synthesize(tmp_path: Path) -> None:
    from datetime import datetime, timezone
    from app.contracts import Finding, SynthesisDraft

    now = datetime.now(timezone.utc)
    s1 = Source(
        id="s1",
        url="https://source1.com/docs",
        title="Source 1",
        text="Valid text body describing vector search architectures and algorithms. " * 8,
        domain="source1.com",
        retrieved_at=now,
        trust=0.9,
        tool="web_search",
        provider="tavily",
    )
    state = State(
        run_id="run_synth_unknown_id",
        question="Question?",
        trace_path=tmp_path / "run_synth_unknown_id.jsonl",
        kept=[s1],
    )

    mock_draft = SynthesisDraft(
        short_answer="Summary answer.",
        findings=[
            Finding(
                claim="Finding citing unknown invented source id s99.",
                source_ids=["s99"],
                supporting_quote="Valid text body describing vector search architectures and algorithms.",
            )
        ],
    )

    with patch("app.nodes.synthesize.router.call_json", return_value=mock_draft):
        updated_state = synthesize(state)

    # Verification is Node 6's job — synthesis MUST NOT drop it here
    assert updated_state.draft is not None
    assert len(updated_state.draft.findings) == 1
    assert updated_state.draft.findings[0].source_ids == ["s99"]


def test_synthesize_router_none_sets_degraded(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    s1 = Source(
        id="s1",
        url="https://source1.com/docs",
        title="Source 1",
        text="Valid text body describing vector search architectures and algorithms. " * 8,
        domain="source1.com",
        retrieved_at=now,
        trust=0.9,
        tool="web_search",
        provider="tavily",
    )
    state = State(
        run_id="run_synth_fail",
        question="Question?",
        trace_path=tmp_path / "run_synth_fail.jsonl",
        kept=[s1],
    )

    with patch("app.nodes.synthesize.router.call_json", return_value=None):
        updated_state = synthesize(state)

    assert updated_state.draft is None
    assert updated_state.status == "degraded"




