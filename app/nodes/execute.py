"""Node 3 (Execute): Runs a deterministic, budget-capped loop executing search and fetch tools."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit
from pydantic import ValidationError

from app.config import BUDGETS
from app.contracts import ToolCallLog
from app.errors import ToolError, ToolExhausted
from app.evidence import build_sources, canonical_url
from app.graph import State
from app.guardrails import domain_trust
from app.tools.base import ToolResult, ToolSpec
from app.tools.fetch import FetchArgs
from app.tools.registry import REGISTRY, call
from app.tools.search import SearchArgs
from app.trace import Tracer

logger = logging.getLogger(__name__)


def _quarantine_bad_payload(tool: str, args: Any, err: str) -> None:
    """Quarantine malformed tool arguments to disk for auditing."""
    try:
        bad_dir = Path("traces/bad_payloads")
        bad_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        target = bad_dir / f"{ts}.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump({"tool": tool, "args": str(args), "error": err, "timestamp": ts}, f)
    except Exception as exc:
        logger.warning("Failed to write quarantined payload: %s", exc)


def execute(state: State) -> State:
    """Execute search and fetch tools deterministically under strict step and wall-clock caps."""
    tracer = Tracer(run_id=state.run_id, trace_dir=state.trace_path.parent)

    if not state.started_at:
        state.started_at = time.time()
    loop_start = state.started_at

    max_steps = (
        min(state.intake.max_steps, BUDGETS.max_tool_calls)
        if state.intake
        else BUDGETS.max_tool_calls
    )

    plan = state.plan
    search_queries = plan.search_queries[:3] if plan and plan.search_queries else [state.question]
    plan_tools = plan.tools if plan and plan.tools else ["web_search", "fetch_page"]

    # 1. Deterministic queue: web_search first
    queue: list[tuple[str, dict[str, Any]]] = []
    for q in search_queries:
        queue.append(("web_search", {"query": q, "max_results": 10}))

    # 2. Wikipedia if in plan
    if "wikipedia" in plan_tools:
        wiki_title = (
            state.intake.entities[0]
            if state.intake and state.intake.entities
            else state.question.split()[0]
        )
        queue.append(("wikipedia", {"title": wiki_title}))

    # 3. Arxiv if in plan
    if "arxiv" in plan_tools:
        arxiv_q = search_queries[0] if search_queries else state.question
        queue.append(("arxiv", {"query": arxiv_q, "max_results": 5}))

    search_hits: list[dict[str, Any]] = []
    fetched_pages: list[dict[str, Any]] = []
    fetched_urls: set[str] = set()
    url_frequency: dict[str, int] = {}

    step = 0
    wall_clock_breached = False

    while step < max_steps:
        # Check wall-clock budget
        elapsed_s = time.time() - loop_start
        if elapsed_s >= BUDGETS.wall_clock_s:
            wall_clock_breached = True
            state.status = "degraded"
            tracer.event(
                node="execute",
                step=step + 1,
                status="degraded",
                note=f"wall-clock timeout ({elapsed_s:.1f}s >= {BUDGETS.wall_clock_s}s)",
            )
            break

        # If queue is empty, populate fetch_page calls from discovered search hits
        if not queue:
            unfetched_candidates: list[str] = []
            for hit in search_hits:
                u = hit.get("url")
                if u and isinstance(u, str):
                    c_u = canonical_url(u)
                    if c_u and c_u not in fetched_urls:
                        unfetched_candidates.append(u)

            if not unfetched_candidates:
                # No more tools or URLs to fetch
                break

            # Rank unfetched URLs by (appears_in_multiple_results desc, trust desc)
            def rank_url(url_str: str) -> tuple[int, float]:
                c_u = canonical_url(url_str)
                freq = url_frequency.get(c_u, 1)
                domain = urlsplit(url_str).hostname or ""
                trust = domain_trust(domain)
                return (freq, trust)

            unique_unfetched = list(dict.fromkeys(unfetched_candidates))
            ranked_urls = sorted(unique_unfetched, key=rank_url, reverse=True)

            for u in ranked_urls:
                queue.append(("fetch_page", {"url": u, "max_chars": 8000}))

            if not queue:
                break

        # Pick next (tool, args) from queue
        tool_name, raw_args = queue.pop(0)
        step += 1

        if tool_name not in REGISTRY:
            tracer.event(
                node="execute",
                step=step,
                tool=tool_name,
                status="skipped",
                note=f"Tool '{tool_name}' not in registry",
            )
            state.calls.append(
                ToolCallLog(
                    step=step,
                    tool=tool_name,
                    provider="unknown",
                    args=raw_args,
                    attempt=1,
                    latency_ms=0,
                    status="skipped",
                    error="Not registered",
                )
            )
            continue

        spec = REGISTRY[tool_name]
        provider_name = spec.provider or tool_name

        # Validate arguments against input_schema
        try:
            validated_args = spec.input_schema.model_validate(raw_args)
        except ValidationError as val_err:
            _quarantine_bad_payload(tool_name, raw_args, str(val_err))
            tracer.event(
                node="execute",
                step=step,
                tool=tool_name,
                provider=provider_name,
                status="skipped",
                note="Invalid args quarantined",
            )
            state.calls.append(
                ToolCallLog(
                    step=step,
                    tool=tool_name,
                    provider=provider_name,
                    args=raw_args,
                    attempt=1,
                    latency_ms=0,
                    status="skipped",
                    error="Validation error",
                )
            )
            continue

        # Execute call via registry
        call_start = time.perf_counter()
        try:
            result = call(tool_name, validated_args, tracer=tracer)
            call_ms = max(0, int((time.perf_counter() - call_start) * 1000))

            log_status = "cache_hit" if result.status == "cache_hit" else "ok"
            state.calls.append(
                ToolCallLog(
                    step=step,
                    tool=tool_name,
                    provider=result.provider,
                    args=validated_args.model_dump(),
                    attempt=1,
                    latency_ms=result.latency_ms or call_ms,
                    status=log_status,
                    items=len(result.items),
                )
            )

            # Route items to search hits or fetched pages
            if tool_name in ("web_search", "wikipedia", "arxiv", "github"):
                for item in result.items:
                    search_hits.append(item)
                    hit_url = item.get("url")
                    if hit_url:
                        c_u = canonical_url(hit_url)
                        url_frequency[c_u] = url_frequency.get(c_u, 0) + 1

            elif tool_name in ("fetch_page", "httpx_fetch"):
                for item in result.items:
                    fetched_pages.append(item)
                    p_url = item.get("url")
                    if p_url:
                        fetched_urls.add(canonical_url(p_url))

        except ToolExhausted as exc:
            call_ms = max(0, int((time.perf_counter() - call_start) * 1000))
            logger.warning("Tool %s exhausted at step %d: %s", tool_name, step, exc)
            state.calls.append(
                ToolCallLog(
                    step=step,
                    tool=tool_name,
                    provider=provider_name,
                    args=validated_args.model_dump(),
                    attempt=2,
                    latency_ms=call_ms,
                    status="error",
                    error=str(exc),
                )
            )

        except Exception as exc:
            call_ms = max(0, int((time.perf_counter() - call_start) * 1000))
            logger.exception("Unexpected tool execution failure on %s: %s", tool_name, exc)
            state.calls.append(
                ToolCallLog(
                    step=step,
                    tool=tool_name,
                    provider=provider_name,
                    args=validated_args.model_dump(),
                    attempt=1,
                    latency_ms=call_ms,
                    status="error",
                    error=str(exc),
                )
            )

    # Budget exhaustion check
    if step >= max_steps and queue:
        state.status = "degraded"
        tracer.event(
            node="execute",
            status="degraded",
            note="step budget exhausted (6 tool calls cap)",
        )

    # Assemble Sources via evidence.build_sources
    sources = build_sources(search_hits, fetched_pages)
    state.sources = sources
    state.kept = sources

    if len(sources) == 0:
        state.status = "insufficient_evidence"
        tracer.event(
            node="execute",
            status="insufficient_evidence",
            note="zero sources gathered",
        )

    return state
