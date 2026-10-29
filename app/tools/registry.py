"""Tool registry providing validation, caching, retries, fallback chains, and circuit breaking."""

from __future__ import annotations

import logging
import random
import time
from typing import Any
from pydantic import BaseModel, ValidationError

from app.cache import cache
from app.errors import ToolError, ToolExhausted
from app.tools.base import ToolResult, ToolSpec
from app.trace import Tracer

logger = logging.getLogger(__name__)

# Global tool registry mapping tool/provider name to its ToolSpec
REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    """Register a ToolSpec into the global registry."""
    REGISTRY[spec.name] = spec


class CircuitBreaker:
    """Tracks consecutive provider failures and opens circuit for 60s after 3 failures."""

    def __init__(self, failure_threshold: int = 3, open_duration_s: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.open_duration_s = open_duration_s
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def is_open(self, provider: str, *, now: float | None = None) -> bool:
        """Check if circuit is currently open for provider."""
        current_time = now if now is not None else time.time()
        if self._failures.get(provider, 0) >= self.failure_threshold:
            opened_time = self._opened_at.get(provider, 0.0)
            if current_time - opened_time < self.open_duration_s:
                return True
            # Duration expired: half-open
            return False
        return False

    def record_success(self, provider: str) -> None:
        """Reset failure count on success."""
        self._failures[provider] = 0
        self._opened_at.pop(provider, None)

    def record_failure(self, provider: str, *, now: float | None = None) -> None:
        """Increment failure count and record timestamp when threshold is reached."""
        current_time = now if now is not None else time.time()
        count = self._failures.get(provider, 0) + 1
        self._failures[provider] = count
        if count >= self.failure_threshold and provider not in self._opened_at:
            self._opened_at[provider] = current_time

    def reset(self) -> None:
        """Reset all circuit breaker states."""
        self._failures.clear()
        self._opened_at.clear()


# Global circuit breaker instance
circuit_breaker = CircuitBreaker()


def _extract_retry_after(exc: Exception) -> float | None:
    retry_val = getattr(exc, "retry_after", None)
    if retry_val is not None:
        try:
            return float(retry_val)
        except (ValueError, TypeError):
            pass
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "headers"):
        header = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if header:
            try:
                return float(header)
            except (ValueError, TypeError):
                pass
    return None


def call(tool: str, args: BaseModel | dict[str, Any], *, tracer: Tracer) -> ToolResult:
    """Execute a tool with argument validation, cache lookup, fallback chain, and circuit breaker."""
    if tool not in REGISTRY:
        raise ToolError(f"Tool '{tool}' not found in registry", context={"tool": tool})

    spec = REGISTRY[tool]

    # Validate argument schema
    if isinstance(args, spec.input_schema):
        validated_args = args
    elif isinstance(args, dict):
        try:
            validated_args = spec.input_schema.model_validate(args)
        except ValidationError as e:
            raise ToolError(f"Invalid arguments for tool '{tool}': {e}", context={"tool": tool, "args": args}) from e
    else:
        raise ToolError(
            f"Arguments for '{tool}' must be {spec.input_schema.__name__} or dict",
            context={"tool": tool, "provided_type": type(args).__name__},
        )

    # Cache lookup
    cache_kind = "page" if hasattr(validated_args, "url") else "search"
    cache_raw_key = getattr(
        validated_args,
        "url",
        getattr(validated_args, "query", str(validated_args.model_dump())),
    )

    cached_payload = cache.get(cache_kind, cache_raw_key)
    if cached_payload is not None:
        tracer.event(
            node="execute",
            tool=tool,
            provider="cache",
            attempt=1,
            latency_ms=0,
            status="cache_hit",
            cache_hit=True,
            items=len(cached_payload) if isinstance(cached_payload, list) else 1,
            note="retrieved from cache",
        )
        return ToolResult(
            items=cached_payload if isinstance(cached_payload, list) else [cached_payload],
            provider="cache",
            status="cache_hit",
            latency_ms=0,
        )

    # Provider failure accounting on tracer
    if not hasattr(tracer, "provider_failures"):
        tracer.provider_failures = {}

    # Build execution chain: primary spec + fallback specs
    chain: list[ToolSpec] = [spec]
    for fb_name in spec.fallbacks:
        if fb_name in REGISTRY:
            chain.append(REGISTRY[fb_name])

    last_error: str | None = None

    for target_spec in chain:
        provider_name = target_spec.provider or target_spec.name

        # Circuit breaker check
        if circuit_breaker.is_open(provider_name):
            tracer.event(
                node="execute",
                tool=tool,
                provider=provider_name,
                attempt=1,
                latency_ms=0,
                status="skipped",
                note="circuit breaker open",
            )
            continue

        # Retry loop per provider: 2 attempts
        for attempt in range(1, 3):
            start_time = time.perf_counter()
            try:
                result_items = target_spec.fn(validated_args)
                latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))

                circuit_breaker.record_success(provider_name)

                # Standardize to list of dicts
                if isinstance(result_items, list):
                    items_list = result_items
                elif isinstance(result_items, dict):
                    items_list = [result_items]
                else:
                    items_list = [{"result": result_items}]

                # Store in cache
                cache.set(cache_kind, cache_raw_key, items_list)

                tracer.event(
                    node="execute",
                    tool=tool,
                    provider=provider_name,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    status="ok",
                    items=len(items_list),
                    note=f"tool={tool} provider={provider_name}",
                )

                return ToolResult(
                    items=items_list,
                    provider=provider_name,
                    status="ok",
                    latency_ms=latency_ms,
                )

            except Exception as exc:
                latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))
                last_error = f"{type(exc).__name__}: {str(exc)}"

                circuit_breaker.record_failure(provider_name)
                tracer.provider_failures[provider_name] = (
                    tracer.provider_failures.get(provider_name, 0) + 1
                )

                tracer.event(
                    node="execute",
                    tool=tool,
                    provider=provider_name,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    status="error",
                    note=last_error,
                )

                # Retry backoff
                if attempt < 2:
                    retry_after = _extract_retry_after(exc)
                    if retry_after is not None:
                        time.sleep(min(retry_after, 5.0))
                    else:
                        jitter = random.uniform(0.5, 1.5)
                        backoff = (0.5 * (2 ** (attempt - 1))) * jitter
                        time.sleep(backoff)

    # All providers and fallbacks failed
    raise ToolExhausted(
        f"All providers exhausted for tool '{tool}': {last_error}",
        context={"tool": tool, "chain": [s.provider or s.name for s in chain]},
    )
