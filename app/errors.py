"""Domain-specific exception definitions for tool execution, schema validation, and pipeline budgets."""

from __future__ import annotations

from typing import Any


class ResearchAgentError(Exception):
    """Base exception for research-agent errors carrying structured context."""

    def __init__(self, message: str = "", *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}


class ToolError(ResearchAgentError):
    """Raised when an individual tool invocation fails."""


class ToolExhausted(ResearchAgentError):
    """Raised when all providers and fallbacks in a tool chain have failed."""


class SchemaValidationError(ResearchAgentError):
    """Raised when structured JSON outputs fail Pydantic validation."""


class BudgetExhausted(ResearchAgentError):
    """Raised when execution budget (tool call count or wall-clock time) is exceeded."""
