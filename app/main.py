"""FastAPI application entrypoint providing /research, /trace/{id}, and /healthz endpoints."""

from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import has_key
from app.contracts import Answer, Confidence, RunMeta
from app.graph import run_with_trace

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Agent API",
    description="Deterministic 7-node agentic research pipeline.",
    version="0.1.0",
)

# Bounded in-memory dict keeping the last 100 runs: run_id -> list[trace_events]
_TRACE_CACHE: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_MAX_TRACE_CACHE_SIZE = 100


def _store_trace(run_id: str, events: list[dict[str, Any]]) -> None:
    """Store trace events in bounded in-memory cache."""
    _TRACE_CACHE[run_id] = events
    while len(_TRACE_CACHE) > _MAX_TRACE_CACHE_SIZE:
        _TRACE_CACHE.popitem(last=False)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    session_id: str | None = None


@app.post("/research", response_model=Answer)
def research_endpoint(req: ResearchRequest) -> Answer:
    """Run research pipeline for question and return validated Answer. Never returns 500."""
    raw_q = req.question
    stripped = raw_q.strip()

    # Question length validation
    if len(stripped) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be blank or whitespace-only.",
        )
    if len(stripped) < 5 or len(raw_q) > 2000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Question length must be between 5 and 2000 characters (got {len(raw_q)}).",
        )

    try:
        answer, events = run_with_trace(stripped)
        _store_trace(answer.run_id, events)
        return answer
    except Exception as exc:
        logger.error("Unhandled exception in pipeline execution: %s", exc, exc_info=True)
        # Never 500: return a degraded Answer
        degraded_answer = Answer(
            run_id="run_error_fallback",
            question=stripped,
            short_answer="The research pipeline encountered an unexpected internal error and operated in degraded mode.",
            key_findings=[],
            sources=[],
            confidence=Confidence(
                level="low",
                score=0.0,
                reasons=[f"Internal error during execution: {type(exc).__name__}: {str(exc)}"],
            ),
            assumptions=[],
            limitations=[f"Pipeline error: {type(exc).__name__}: {str(exc)}"],
            next_steps=["Retry the query or contact support."],
            status="degraded",
            meta=RunMeta(
                tool_calls=0,
                latency_ms=0,
                cost_usd=0.0,
                grounding_rate=0.0,
            ),
        )
        return degraded_answer


@app.get("/trace/{run_id}")
def trace_endpoint(run_id: str) -> list[dict[str, Any]]:
    """Retrieve in-memory execution trace events for run_id."""
    if run_id not in _TRACE_CACHE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No trace found for run_id '{run_id}'.",
        )
    return _TRACE_CACHE[run_id]


@app.get("/healthz")
def healthz_endpoint() -> dict[str, Any]:
    """Health check endpoint indicating service status and configured tool/LLM providers."""
    return {
        "status": "ok",
        "providers": {
            "gemini": has_key("gemini"),
            "groq": has_key("groq"),
            "tavily": has_key("tavily"),
            "exa": has_key("exa"),
            "jina": has_key("jina"),
        },
    }
