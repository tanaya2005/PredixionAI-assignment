"""Central prompt templates and schema instructions for pipeline nodes and evaluation judge."""

from __future__ import annotations

# Placeholder templates to be populated in detail in T11
INTAKE: str = ""
PLANNER: str = ""
RELEVANCE: str = ""
SYNTHESIS: str = ""
JUDGE: str = ""

# Schema repair template used by LLM router
REPAIR: str = "The previous output failed validation with: {err}. Return only corrected JSON."
