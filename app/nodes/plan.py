"""Node 2 (Plan): Generates structured research sub-questions and keyword search queries."""

from __future__ import annotations

import logging
import time

from app.config import TOOL_POLICY
from app.contracts import QueryType, ResearchPlan
from app.graph import State
from app.llm import router
from app.prompts import PLANNER
from app.trace import Tracer

logger = logging.getLogger(__name__)


def _make_fallback_plan(question: str, q_type: QueryType, allowed: list[str]) -> ResearchPlan:
    """Construct a safe minimal fallback plan."""
    default_tools = [t for t in ["web_search", "fetch_page"] if t in allowed] or allowed[:2] or ["web_search"]
    q_trimmed = question.strip()[:160] or "Research question"
    return ResearchPlan(
        query_type=q_type,
        sub_questions=[q_trimmed],
        search_queries=[q_trimmed],
        tools=default_tools,
        stop_criteria="single pass",
    )


def plan(state: State) -> State:
    """Generate and validate a structured research plan before tool execution. Never raises."""
    tracer = Tracer(run_id=state.run_id, trace_dir=state.trace_path.parent)
    start_time = time.perf_counter()

    query_type: QueryType = state.intake.query_type if state.intake else "explain"
    allowed_tools: list[str] = TOOL_POLICY.get(query_type, ["web_search", "fetch_page"])

    try:
        prompt = PLANNER.format(
            question=state.question,
            query_type=query_type,
            allowed_tools=", ".join(allowed_tools),
        )

        result = router.call_json(
            role="plan",
            prompt=prompt,
            schema=ResearchPlan,
            temperature=0.2,
            trace=tracer,
        )

        latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))

        if result is None:
            fallback = _make_fallback_plan(state.question, query_type, allowed_tools)
            state.plan = fallback
            state.status = "degraded"
            tracer.event(
                node="plan",
                latency_ms=latency_ms,
                status="error",
                note="plan fallback (router returned None)",
                plan=fallback.model_dump(),
            )
            return state

        # Filter tools against allowed_tools
        valid_tools: list[str] = []
        for t in result.tools:
            if t in allowed_tools:
                if t not in valid_tools:
                    valid_tools.append(t)
            else:
                logger.warning("Dropping disallowed or unknown tool from plan: %s", t)

        if not valid_tools:
            valid_tools = allowed_tools[:2]

        # Deduplicate and trim sub_questions (<= 5)
        seen_sq: set[str] = set()
        sub_questions: list[str] = []
        for sq in result.sub_questions:
            clean_sq = sq.strip()
            if clean_sq and clean_sq not in seen_sq:
                seen_sq.add(clean_sq)
                sub_questions.append(clean_sq)
        sub_questions = sub_questions[:5]

        # Deduplicate and trim search_queries (<= 6, <= 160 chars each)
        seen_q: set[str] = set()
        search_queries: list[str] = []
        for q in result.search_queries:
            clean_q = q.strip()[:160]
            if clean_q and clean_q not in seen_q:
                seen_q.add(clean_q)
                search_queries.append(clean_q)
        search_queries = search_queries[:6]

        # If sub_questions or search_queries ended up empty, use fallback
        if not sub_questions or not search_queries:
            fallback = _make_fallback_plan(state.question, query_type, allowed_tools)
            state.plan = fallback
            state.status = "degraded"
            tracer.event(
                node="plan",
                latency_ms=latency_ms,
                status="error",
                note="plan fallback (empty queries or sub-questions)",
                plan=fallback.model_dump(),
            )
            return state

        validated_plan = ResearchPlan(
            query_type=result.query_type,
            sub_questions=sub_questions,
            search_queries=search_queries,
            tools=valid_tools,
            stop_criteria=result.stop_criteria or "sufficient evidence gathered",
        )

        state.plan = validated_plan
        tracer.event(
            node="plan",
            latency_ms=latency_ms,
            status="ok",
            note=f"sub_questions={len(sub_questions)} queries={len(search_queries)} tools={','.join(valid_tools)}",
            plan=validated_plan.model_dump(),
        )

    except Exception as exc:
        latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))
        logger.exception("Unexpected error in plan node: %s", exc)
        fallback = _make_fallback_plan(state.question, query_type, allowed_tools)
        state.plan = fallback
        state.status = "degraded"
        tracer.event(
            node="plan",
            latency_ms=latency_ms,
            status="error",
            note=f"Unexpected exception: {type(exc).__name__}",
            plan=fallback.model_dump(),
        )

    return state
