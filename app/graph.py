"""Explicit pipeline graph wiring and execution runner across all 7 nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
import uuid

from app.config import TRACE_DIR
from app.contracts import (
    Answer,
    Confidence,
    Intake,
    ResearchPlan,
    Source,
    Status,
    SynthesisDraft,
    ToolCallLog,
)
from app.trace import Tracer

logger = logging.getLogger(__name__)


@dataclass
class State:
    run_id: str
    question: str
    trace_path: Path
    intake: Intake | None = None
    plan: ResearchPlan | None = None
    sources: list[Source] = field(default_factory=list)
    calls: list[ToolCallLog] = field(default_factory=list)
    kept: list[Source] = field(default_factory=list)
    draft: SynthesisDraft | None = None
    dropped: list[dict] = field(default_factory=list)  # {finding, reason}
    confidence: Confidence | None = None
    status: Status = "ok"
    started_at: float = 0.0
    cost_usd: float = 0.0


# Import nodes after State definition to prevent circular imports
from app.nodes.execute import execute
from app.nodes.filter import filter_evidence
from app.nodes.intake import intake
from app.nodes.plan import plan
from app.nodes.render import render
from app.nodes.synthesize import synthesize
from app.nodes.verify import verify


def _build_initial_state(question: str, trace_dir: Path | None = None) -> State:
    """Instantiate a new run State with unique run_id and trace path."""
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    t_dir = trace_dir or TRACE_DIR
    t_dir.mkdir(parents=True, exist_ok=True)
    trace_path = t_dir / f"{run_id}.jsonl"

    return State(
        run_id=run_id,
        question=question,
        trace_path=trace_path,
        started_at=time.time(),
    )


def run_with_trace(
    question: str,
    *,
    trace_dir: Path | None = None,
) -> tuple[Answer, list[dict]]:
    """Run the 7-node pipeline and return both the final Answer and the trace events."""
    state = _build_initial_state(question, trace_dir=trace_dir)
    tracer = Tracer(run_id=state.run_id, trace_dir=state.trace_path.parent)

    pipeline_nodes = [
        ("intake", intake),
        ("plan", plan),
        ("execute", execute),
        ("filter", filter_evidence),
        ("synthesize", synthesize),
        ("verify", verify),
    ]

    for name, node_fn in pipeline_nodes:
        # Check early exit conditions:
        # If execute sets insufficient_evidence, skip directly to render
        if state.status == "insufficient_evidence" and name in ("filter", "synthesize", "verify"):
            logger.info("Pipeline early exit to render on status='insufficient_evidence' before node '%s'", name)
            break

        try:
            state = node_fn(state)
        except Exception as exc:
            logger.error("Node '%s' failed unexpectedly: %s. Setting status='degraded' and continuing.", name, exc)
            state.status = "degraded"
            tracer.event(
                node=name,
                status="error",
                note=f"Unhandled node exception: {type(exc).__name__}: {str(exc)}",
            )

    # Final node is always render
    answer = render(state)
    return answer, tracer.events()


def run(question: str, *, trace_dir: Path | None = None) -> Answer:
    """Execute the full 7-node research pipeline graph for a question. Never raises."""
    answer, _ = run_with_trace(question, trace_dir=trace_dir)
    return answer
