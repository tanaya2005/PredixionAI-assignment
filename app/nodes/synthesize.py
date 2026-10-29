"""Node 5 (Synthesize): Synthesizes structured findings backed by verbatim quotes from evidence."""

from __future__ import annotations

import logging
import time

from app.config import MAX_EVIDENCE_CHARS
from app.contracts import SynthesisDraft
from app.graph import State
from app.guardrails import wrap_untrusted
from app.llm import router
from app.prompts import SYNTHESIS
from app.trace import Tracer

logger = logging.getLogger(__name__)


def synthesize(state: State) -> State:
    """Synthesize structured findings backed by verbatim quotes from evidence. Never raises."""
    tracer = Tracer(run_id=state.run_id, trace_dir=state.trace_path.parent)
    start_time = time.perf_counter()

    # If no kept sources, degrade immediately
    if not state.kept:
        state.draft = None
        state.status = "insufficient_evidence" if state.status == "insufficient_evidence" else "degraded"
        latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))
        tracer.event(
            node="synthesize",
            latency_ms=latency_ms,
            status="empty",
            note="no kept sources to synthesize",
        )
        return state

    # Build evidence block respecting MAX_EVIDENCE_CHARS budget
    # Truncate lowest-ranked sources first (sources in state.kept are sorted by priority: s1, s2...)
    # We trim whole trailing sources rather than splitting mid-source so quotes remain intact.
    wrapped_docs: list[str] = []
    current_chars = 0

    for s in state.kept:
        wrapped = wrap_untrusted(text=s.text, id=s.id, url=s.url)
        # Separator length when joining with "\n\n"
        additional_len = len(wrapped) + (2 if wrapped_docs else 0)
        if current_chars + additional_len > MAX_EVIDENCE_CHARS:
            # Check if at least 1 doc was wrapped; if not, wrap at least the first doc
            if not wrapped_docs:
                wrapped_docs.append(wrapped[:MAX_EVIDENCE_CHARS])
                current_chars = len(wrapped_docs[0])
            break
        wrapped_docs.append(wrapped)
        current_chars += additional_len

    evidence_text = "\n\n".join(wrapped_docs)
    prompt = SYNTHESIS.format(question=state.question, evidence=evidence_text)

    # Call LLM router with role="synthesize"
    result = router.call_json(
        role="synthesize",
        prompt=prompt,
        schema=SynthesisDraft,
        temperature=0.2,
        trace=tracer,
    )

    latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))

    if result is None:
        logger.warning("Synthesize failed or returned None; setting state.draft=None and status='degraded'")
        state.draft = None
        state.status = "degraded"
        tracer.event(
            node="synthesize",
            latency_ms=latency_ms,
            status="error",
            note="synthesis failed or invalid schema",
        )
        return state

    # On success:
    # Cap findings to 8
    capped_findings = result.findings[:8]

    # Trim short_answer to 1200 chars
    trimmed_short_answer = result.short_answer[:1200]

    # Contradictions are kept to feed limitations if applicable
    combined_limitations = list(result.limitations)
    for c in result.contradictions:
        if c not in combined_limitations:
            combined_limitations.append(f"Contradiction: {c}")

    state.draft = SynthesisDraft(
        short_answer=trimmed_short_answer,
        findings=capped_findings,
        assumptions=result.assumptions,
        limitations=combined_limitations,
        next_steps=result.next_steps,
        contradictions=result.contradictions,
    )

    tracer.event(
        node="synthesize",
        latency_ms=latency_ms,
        status="ok",
        findings_count=len(capped_findings),
        evidence_chars=len(evidence_text),
        note=f"{len(capped_findings)} findings synthesized",
    )

    return state
