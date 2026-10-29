"""Node 6 (Verify): Deterministically verifies quote presence, computes confidence, and drops ungrounded claims."""

from __future__ import annotations

import logging
import time

from app.graph import State
from app.trace import Tracer
from app.verify import compute_confidence, verify_findings

logger = logging.getLogger(__name__)


def verify(state: State) -> State:
    """Deterministically verify findings against sources, compute confidence, and update state status.

    Never calls an LLM. Never raises.
    """
    tracer = Tracer(run_id=state.run_id, trace_dir=state.trace_path.parent)
    start_time = time.perf_counter()

    initial_findings_count = len(state.draft.findings) if state.draft else 0

    # 1. Verify findings deterministically
    kept_findings, dropped_findings = verify_findings(
        draft=state.draft,
        sources=state.kept,
    )

    state.dropped = dropped_findings

    # Compute grounding rate
    if initial_findings_count > 0:
        grounding_rate = len(kept_findings) / initial_findings_count
    else:
        grounding_rate = 0.0

    # 2. Compute confidence score and level
    confidence = compute_confidence(
        kept_findings=kept_findings,
        sources=state.kept,
        grounding_rate=grounding_rate,
    )
    state.confidence = confidence

    # 3. Update state.draft with kept findings
    if state.draft is not None:
        state.draft.findings = kept_findings

    # 4. Status updates:
    # - insufficient_evidence if 0 findings survive
    # - degraded if grounding_rate < 0.5 (or if already degraded)
    if len(kept_findings) == 0:
        state.status = "insufficient_evidence"
    elif grounding_rate < 0.5:
        state.status = "degraded"

    latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))

    tracer.event(
        node="verify",
        latency_ms=latency_ms,
        status="ok",
        initial_findings=initial_findings_count,
        kept_findings=len(kept_findings),
        dropped=len(dropped_findings),
        grounding_rate=round(grounding_rate, 4),
        confidence_level=confidence.level,
        confidence_score=confidence.score,
        note=f"{len(kept_findings)}/{initial_findings_count} grounded ({confidence.level} confidence)",
    )

    return state
