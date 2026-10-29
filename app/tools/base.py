"""Base models and contracts for tool specifications and standardized execution results."""

from __future__ import annotations

from typing import Any, Callable
from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """Specification of an executable tool registered in the pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    fn: Callable[..., Any]
    input_schema: type[BaseModel]
    timeout_s: float
    cost_per_call: float = 0.0
    fallbacks: list[str] = Field(default_factory=list)
    provider: str | None = None


class ToolResult(BaseModel):
    """Standardized result returned by the tool execution layer."""

    model_config = ConfigDict(extra="forbid")

    items: list[dict[str, Any]] = Field(default_factory=list)
    provider: str
    status: str
    latency_ms: int = 0
    error: str | None = None
