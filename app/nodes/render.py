"""Node 7 (Render): Formats and validates the final Answer contract in strict key order."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any
from pydantic import ValidationError

from app.config import MODELS, TRACE_DIR
from app.contracts import Answer, Confidence, Finding, RunMeta, SourceRef
from app.graph import State
from app.trace import Tracer

logger = logging.getLogger(__name__)


def render(state: State) -> Answer:
    """Format and validate the final Answer contract in strict key order. Never raises."""
    tracer = Tracer(run_id=state.run_id, trace_dir=state.trace_path.parent)
    now_ms = max(0, int((time.time() - state.started_at) * 1000)) if state.started_at else 0

    # 1. Collect trace events for metadata
    trace_events = tracer.events()
    total_cost = tracer.cost_usd
    cache_hits = sum(1 for e in trace_events if e.get("cache_hit") is True)
    llm_calls = sum(1 for e in trace_events if e.get("node") in ("intake", "plan", "filter", "synthesize", "llm"))

    provider_failures: dict[str, int] = {}
    for c in state.calls:
        if c.status in ("error", "failed"):
            provider_failures[c.provider] = provider_failures.get(c.provider, 0) + 1

    # Grounding rate
    total_draft_findings = len(state.draft.findings) + len(state.dropped) if state.draft else len(state.dropped)
    if total_draft_findings > 0:
        grounding_rate = len(state.draft.findings if state.draft else []) / total_draft_findings
    else:
        grounding_rate = 1.0 if (state.draft and state.draft.findings) else 0.0

    # 2. Build SourceRef objects from state.kept (or state.sources if state.kept empty)
    source_refs: list[SourceRef] = []
    sources_to_render = state.kept if state.kept else state.sources
    for s in sources_to_render:
        source_refs.append(
            SourceRef(
                id=s.id,
                title=s.title,
                url=s.url,
                domain=s.domain,
                published=s.published,
                retrieved_at=s.retrieved_at,
                trust=s.trust,
                untrusted=s.untrusted,
            )
        )

    # 3. Key findings
    key_findings: list[Finding] = state.draft.findings if state.draft else []

    # 4. Confidence
    confidence = state.confidence or Confidence(
        level="low",
        score=0.0,
        reasons=["Confidence was not computed during pipeline run"],
    )

    # 5. Handle status specifics
    status = state.status
    short_answer = state.draft.short_answer if state.draft else ""
    assumptions = list(state.draft.assumptions) if state.draft else []
    limitations = list(state.draft.limitations) if state.draft else []
    next_steps = list(state.draft.next_steps) if state.draft else []

    if status == "insufficient_evidence":
        if not short_answer:
            short_answer = (
                "Insufficient verified evidence was found to answer this research question reliably. "
                "The available search and retrieval results yielded zero or uncorroborated sources."
            )
        if not next_steps:
            next_steps = [
                "Broaden or rephrase the query terms to include general industry terminology.",
                "Verify whether the subject technology or entity has official public documentation.",
            ]
    elif status == "degraded":
        deg_note = "Execution operated in degraded mode (e.g. wall-clock timeout, ungrounded findings, or tool exhaustion)."
        if deg_note not in limitations:
            limitations.append(deg_note)

    # 6. Construct RunMeta
    meta = RunMeta(
        plan=state.plan,
        tool_calls=len(state.calls),
        latency_ms=now_ms,
        cost_usd=round(total_cost, 6),
        grounding_rate=round(grounding_rate, 4),
        dropped_findings=len(state.dropped),
        cache_hits=cache_hits,
        provider_failures=provider_failures,
        models=dict(MODELS),
        llm_calls=llm_calls,
    )

    # 7. Construct Answer with explicit key order (dict insertion order)
    raw_answer: dict[str, Any] = {
        "run_id": state.run_id,
        "question": state.question,
        "short_answer": short_answer,
        "key_findings": key_findings,
        "sources": source_refs,
        "confidence": confidence,
        "assumptions": assumptions,
        "limitations": limitations,
        "next_steps": next_steps,
        "status": status,
        "meta": meta,
    }

    try:
        validated_answer = Answer.model_validate(raw_answer)
    except ValidationError as err:
        logger.error("Answer failed contract validation: %s. Emitting minimal degraded answer.", err)
        validated_answer = Answer(
            run_id=state.run_id,
            question=state.question,
            short_answer="Pipeline encountered a validation error during rendering; outputting degraded summary.",
            key_findings=[],
            sources=[],
            confidence=Confidence(level="low", score=0.0, reasons=[f"ValidationError: {err}"]),
            assumptions=[],
            limitations=[f"Validation error: {err}"],
            next_steps=["Inspect system trace and report schema discrepancy."],
            status="degraded",
            meta=meta,
        )

    # 8. Write pretty JSON trace to traces/<run_id>.json
    try:
        trace_dir = state.trace_path.parent if state.trace_path else TRACE_DIR
        trace_dir.mkdir(parents=True, exist_ok=True)
        json_path = trace_dir / f"{state.run_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(validated_answer.model_dump_json(indent=2))
    except Exception as exc:
        logger.warning("Failed to write traces/%s.json: %s", state.run_id, exc)

    return validated_answer
