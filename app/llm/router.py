"""Central LLM router managing provider failover, retry with jitter, schema repair, and token accounting."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, TypeVar
from pydantic import BaseModel, ValidationError

from app.config import MODEL_PRICES, MODELS
from app.llm import gemini, groq
from app.llm.schemas import parse_and_validate
from app.prompts import REPAIR
from app.trace import Tracer

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

VALID_ROLES = {"intake", "plan", "filter", "synthesize", "judge"}


def _compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    prices = MODEL_PRICES.get(model, {"input": 0.0001, "output": 0.0004})
    in_cost = (tokens_in / 1000.0) * prices.get("input", 0.0)
    out_cost = (tokens_out / 1000.0) * prices.get("output", 0.0)
    return round(in_cost + out_cost, 6)


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


def call_json(
    role: str,
    prompt: str,
    schema: type[T],
    *,
    temperature: float,
    trace: Tracer,
    max_repair: int = 1,
) -> T | None:
    """Route a JSON generation request to primary provider with failover, retries, and repair."""
    if role not in VALID_ROLES:
        trace.event(
            node="llm",
            status="error",
            note=f"Invalid LLM role requested: {role}",
        )
        return None

    # Determine provider chain (primary and fallback)
    if role in ("intake", "filter", "judge"):
        providers = [
            ("groq", MODELS.get(role, "llama-3.3-70b-versatile")),
            ("gemini", "gemini-2.5-flash"),
        ]
    else:
        providers = [
            ("gemini", MODELS.get(role, "gemini-2.5-flash")),
            ("groq", "llama-3.3-70b-versatile"),
        ]

    for provider_name, model_name in providers:
        for attempt in range(1, 3):
            start_time = time.perf_counter()
            try:
                if provider_name == "gemini":
                    resp = gemini.generate_json(
                        prompt=prompt,
                        schema=schema,
                        model=model_name,
                        temperature=temperature,
                    )
                else:
                    resp = groq.generate_json(
                        prompt=prompt,
                        schema=schema,
                        model=model_name,
                        temperature=temperature,
                    )

                latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))

                # Parse and validate schema
                try:
                    parsed = parse_and_validate(resp.text, schema)
                    cost_usd = _compute_cost(model_name, resp.tokens_in, resp.tokens_out)
                    trace.event(
                        node="llm",
                        provider=provider_name,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        status="ok",
                        tokens_in=resp.tokens_in,
                        tokens_out=resp.tokens_out,
                        cost_usd=cost_usd,
                        note=f"role={role} model={model_name}",
                    )
                    return parsed

                except (ValidationError, Exception) as val_err:
                    logger.warning(
                        "Schema validation failed on %s (attempt %d): %s",
                        provider_name,
                        attempt,
                        val_err,
                    )
                    # Attempt schema repair if budget allows
                    if max_repair > 0:
                        repair_prompt = f"{prompt}\n\n{REPAIR.format(err=str(val_err))}"
                        repair_start = time.perf_counter()
                        try:
                            if provider_name == "gemini":
                                rep_resp = gemini.generate_json(
                                    prompt=repair_prompt,
                                    schema=schema,
                                    model=model_name,
                                    temperature=temperature,
                                )
                            else:
                                rep_resp = groq.generate_json(
                                    prompt=repair_prompt,
                                    schema=schema,
                                    model=model_name,
                                    temperature=temperature,
                                )

                            rep_latency = max(0, int((time.perf_counter() - repair_start) * 1000))
                            parsed_repair = parse_and_validate(rep_resp.text, schema)
                            rep_cost = _compute_cost(
                                model_name, rep_resp.tokens_in, rep_resp.tokens_out
                            )

                            trace.event(
                                node="llm",
                                provider=provider_name,
                                attempt=attempt,
                                latency_ms=latency_ms + rep_latency,
                                status="repaired",
                                tokens_in=resp.tokens_in + rep_resp.tokens_in,
                                tokens_out=resp.tokens_out + rep_resp.tokens_out,
                                cost_usd=rep_cost,
                                note=f"role={role} repaired schema validation",
                            )
                            return parsed_repair

                        except Exception as rep_err:
                            logger.warning("Schema repair failed on %s: %s", provider_name, rep_err)

                    # Repair exhausted or failed
                    cost_usd = _compute_cost(model_name, resp.tokens_in, resp.tokens_out)
                    trace.event(
                        node="llm",
                        provider=provider_name,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        status="validation_error",
                        tokens_in=resp.tokens_in,
                        tokens_out=resp.tokens_out,
                        cost_usd=cost_usd,
                        note=f"role={role} schema validation error",
                    )
                    # Break out to next provider on schema failure
                    break

            except Exception as api_err:
                latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))
                retry_after = _extract_retry_after(api_err)
                logger.warning(
                    "LLM provider %s error on attempt %d: %s", provider_name, attempt, api_err
                )

                trace.event(
                    node="llm",
                    provider=provider_name,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    status="error",
                    note=f"{type(api_err).__name__}: {str(api_err)}",
                )

                if attempt < 2:
                    if retry_after is not None:
                        sleep_s = min(retry_after, 5.0)
                    else:
                        jitter = random.uniform(0.5, 1.5)
                        sleep_s = (0.5 * (2 ** (attempt - 1))) * jitter
                    time.sleep(sleep_s)

    # All providers and attempts failed
    trace.event(
        node="llm",
        status="exhausted",
        note=f"All providers exhausted for role={role}",
    )
    return None
