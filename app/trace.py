"""JSONL execution tracing with latency tracking, cost accounting, and rich console logging."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterator

from app.config import TRACE_DIR

# Sensitive query params and patterns to redact
_REDACT_PATTERNS = [
    re.compile(r"([?&](?:api_?key|key|token|auth)=)[^&]+", re.IGNORECASE),
    re.compile(r"((?:Bearer|token)\s+)[A-Za-z0-9_\-\.]{10,}", re.IGNORECASE),
]


def _sanitize_value(val: Any) -> Any:
    """Recursively sanitize strings, urls, and dicts to remove secrets."""
    if isinstance(val, str):
        cleaned = val
        for pat in _REDACT_PATTERNS:
            cleaned = pat.sub(r"\1[REDACTED]", cleaned)
        return cleaned
    if isinstance(val, dict):
        return {k: _sanitize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_value(item) for item in val]
    return val


def render_log_line(node: str, ms: int, note: str | None = None) -> str:
    """Render a single formatted console line for CLI display."""
    note_str = f" — {note}" if note else ""
    return f"[{node:<10}] {ms:>5}ms{note_str}"


class Tracer:
    """Per-run tracer writing JSONL events and maintaining an in-memory event timeline."""

    def __init__(self, run_id: str, trace_dir: Path | None = None) -> None:
        self.run_id = run_id
        self.trace_dir = trace_dir or TRACE_DIR
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.trace_dir / f"{self.run_id}.jsonl"
        self._events: list[dict[str, Any]] = []
        self.cost_usd: float = 0.0

    def event(
        self,
        node: str,
        *,
        step: int | None = None,
        tool: str | None = None,
        provider: str | None = None,
        attempt: int | None = None,
        latency_ms: int = 0,
        status: str = "ok",
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
        note: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Record an event to the in-memory log and append a JSON line to disk."""
        now_z = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "ts": now_z,
            "node": node,
            "step": step,
            "tool": tool,
            "provider": provider,
            "attempt": attempt,
            "latency_ms": latency_ms,
            "status": status,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "cache_hit": cache_hit,
            "note": note,
        }
        if extra:
            record.update(extra)

        sanitized = _sanitize_value(record)
        self._events.append(sanitized)
        if cost_usd > 0.0:
            self.cost_usd += cost_usd

        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized) + "\n")

        return sanitized

    def events(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded events."""
        return list(self._events)

    @contextmanager
    def node(self, node: str, **initial_fields: Any) -> Iterator[dict[str, Any]]:
        """Context manager tracking node latency and logging an event in finally block."""
        ctx: dict[str, Any] = dict(initial_fields)
        if "status" not in ctx:
            ctx["status"] = "ok"

        start_time = time.perf_counter()
        try:
            yield ctx
        except Exception as exc:
            if ctx.get("status") == "ok":
                ctx["status"] = "error"
            if not ctx.get("note"):
                ctx["note"] = f"{type(exc).__name__}: {str(exc)}"
            raise
        finally:
            elapsed_ms = max(0, int((time.perf_counter() - start_time) * 1000))
            self.event(node=node, latency_ms=elapsed_ms, **ctx)
