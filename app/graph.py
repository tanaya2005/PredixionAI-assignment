"""Explicit pipeline graph wiring and execution runner across all 7 nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


def run(question: str, *, trace_dir: Path | None = None) -> Answer:
    """Stub runner for T1, fully wired in T18."""
    raise NotImplementedError("Pipeline graph runner will be wired in T18")
