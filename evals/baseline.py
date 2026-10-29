"""Non-agentic single-prompt baseline generator for evaluation comparison."""

from __future__ import annotations

import logging
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.contracts import Answer, Confidence, Finding, RunMeta, SourceRef
from app.llm import router
from app.trace import Tracer

logger = logging.getLogger(__name__)


class BaselineFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=10)
    source_url: str = Field(min_length=5)
    supporting_quote: str = Field(default="")


class BaselineOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_answer: str = Field(max_length=1200)
    findings: list[BaselineFinding] = Field(default_factory=list)


def run_baseline(question: str) -> Answer:
    """Run non-agentic single-prompt baseline: same LLM, ONE prompt, no planner, no search tools."""
    prompt = f"""You are a research assistant. Research this and answer with citations.

Question: {question}

Return only valid JSON matching this schema:
{{
  "short_answer": "Direct summary answer...",
  "findings": [
    {{
      "claim": "Specific finding...",
      "source_url": "https://example.com/source",
      "supporting_quote": "Exact quote if known or leave empty"
    }}
  ]
}}
"""
    tracer = Tracer(run_id="baseline_run")
    result = router.call_json(
        role="synthesize",
        prompt=prompt,
        schema=BaselineOutput,
        temperature=0.2,
        trace=tracer,
    )

    if result is None:
        return Answer(
            run_id="baseline_error",
            question=question,
            short_answer="Baseline model call failed.",
            key_findings=[],
            sources=[],
            confidence=Confidence(level="low", score=0.0, reasons=["Baseline failure"]),
            assumptions=[],
            limitations=["Baseline model call failed"],
            next_steps=[],
            status="degraded",
            meta=RunMeta(tool_calls=0, latency_ms=0, cost_usd=0.0, grounding_rate=0.0),
        )

    findings: list[Finding] = []
    sources: list[SourceRef] = []
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for idx, f in enumerate(result.findings, start=1):
        sid = f"s{idx}"
        findings.append(
            Finding(
                claim=f.claim if len(f.claim) >= 20 else f.claim.ljust(20, "."),
                source_ids=[sid],
                supporting_quote=f.supporting_quote if len(f.supporting_quote) >= 40 else (f.supporting_quote + " " * 40)[:40],
                corroboration=1,
            )
        )
        sources.append(
            SourceRef(
                id=sid,
                title=f"Baseline Citation {idx}",
                url=f.source_url,
                domain=f.source_url.split("//")[-1].split("/")[0] if "//" in f.source_url else "unknown",
                retrieved_at=now,
                trust=0.5,
            )
        )

    return Answer(
        run_id="baseline_result",
        question=question,
        short_answer=result.short_answer,
        key_findings=findings,
        sources=sources,
        confidence=Confidence(level="medium", score=0.5, reasons=["Unverified baseline citations"]),
        assumptions=["Non-agentic single pass without web search validation."],
        limitations=["Citations may be fabricated or hallucinated by model."],
        next_steps=[],
        status="ok",
        meta=RunMeta(tool_calls=0, latency_ms=1000, cost_usd=0.001, grounding_rate=0.0),
    )
