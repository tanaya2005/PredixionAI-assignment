"""Node 1 (Intake): Classifies question type, extracts entities, and configures pipeline bounds."""

from __future__ import annotations

import logging
import time

from app.config import BUDGETS
from app.contracts import Intake
from app.graph import State
from app.llm import router
from app.prompts import INTAKE
from app.trace import Tracer

logger = logging.getLogger(__name__)


def intake(state: State) -> State:
    """Classify the input question and configure budget bounds. Never raises."""
    tracer = Tracer(run_id=state.run_id, trace_dir=state.trace_path.parent)
    start_time = time.perf_counter()

    try:
        prompt = INTAKE.format(question=state.question)
        result = router.call_json(
            role="intake",
            prompt=prompt,
            schema=Intake,
            temperature=0.0,
            trace=tracer,
        )

        latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))

        if result is None:
            # Router failed or returned invalid JSON
            state.intake = Intake(
                query_type="explain",
                entities=[],
                recency_sensitive=False,
                k_sources=6,
                max_steps=BUDGETS.max_tool_calls,
            )
            state.status = "degraded"
            tracer.event(
                node="intake",
                latency_ms=latency_ms,
                status="error",
                note="intake fallback",
            )
            return state

        # Local deterministic clamping
        target_k = 4 if result.query_type == "howto" else 6
        clamped_k = min(result.k_sources, target_k)
        clamped_steps = min(result.max_steps, BUDGETS.max_tool_calls)

        state.intake = Intake(
            query_type=result.query_type,
            entities=result.entities,
            recency_sensitive=result.recency_sensitive,
            k_sources=clamped_k,
            max_steps=clamped_steps,
        )

        note_text = "recency_sensitive" if result.recency_sensitive else f"type={result.query_type}"
        tracer.event(
            node="intake",
            latency_ms=latency_ms,
            status="ok",
            note=note_text,
        )

    except Exception as exc:
        latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))
        logger.exception("Unexpected error in intake node: %s", exc)
        state.status = "degraded"
        state.intake = Intake(
            query_type="explain",
            entities=[],
            recency_sensitive=False,
            k_sources=6,
            max_steps=BUDGETS.max_tool_calls,
        )
        tracer.event(
            node="intake",
            latency_ms=latency_ms,
            status="error",
            note=f"Unexpected exception: {type(exc).__name__}",
        )

    return state
