"""Application settings, environment variable management, model specifications, and execution budgets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Load .env if present
load_dotenv(override=False)


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tool_calls: int = 6
    wall_clock_s: float = 60.0
    k_sources: int = 6
    max_source_chars: int = 8000
    max_evidence_chars: int = 40000


class RetrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    times: int = 2
    base_delay: float = 0.5
    jitter: bool = True


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gemini_api_key: str | None = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    groq_api_key: str | None = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    tavily_api_key: str | None = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    exa_api_key: str | None = Field(default_factory=lambda: os.getenv("EXA_API_KEY"))
    jina_api_key: str | None = Field(default_factory=lambda: os.getenv("JINA_API_KEY"))
    github_token: str | None = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))

    def has_key(self, name: str) -> bool:
        """Check whether an API key/token is present and non-empty."""
        clean = name.strip()
        lower = clean.lower()
        # Direct attribute match
        if hasattr(self, lower):
            val = getattr(self, lower)
            if val and str(val).strip():
                return True
        # Provider shortcut (e.g. 'gemini' -> 'gemini_api_key')
        candidate_attr = f"{lower}_api_key"
        if hasattr(self, candidate_attr):
            val = getattr(self, candidate_attr)
            if val and str(val).strip():
                return True
        # Environment fallback
        env_val = os.getenv(clean.upper()) or os.getenv(clean)
        return bool(env_val and env_val.strip())


# Instantiate global settings
settings = Settings()

# Individual exposed keys
GEMINI_API_KEY: str | None = settings.gemini_api_key
GROQ_API_KEY: str | None = settings.groq_api_key
TAVILY_API_KEY: str | None = settings.tavily_api_key
EXA_API_KEY: str | None = settings.exa_api_key
JINA_API_KEY: str | None = settings.jina_api_key
GITHUB_TOKEN: str | None = settings.github_token


def has_key(name: str) -> bool:
    """Module-level helper to check key presence."""
    return settings.has_key(name)


# Models specification
MODELS: dict[str, str] = {
    "intake": "llama-3.3-70b-versatile",
    "plan": "gemini-2.5-flash",
    "filter": "llama-3.3-70b-versatile",
    "synthesize": "gemini-2.5-flash",
    "judge": "llama-3.3-70b-versatile",  # Must differ from synthesize
}

# Execution budgets and constants
BUDGETS = BudgetConfig()
MAX_SOURCE_CHARS: int = BUDGETS.max_source_chars
MAX_EVIDENCE_CHARS: int = BUDGETS.max_evidence_chars

# Tool routing policy by query type
TOOL_POLICY: dict[str, list[str]] = {
    "comparison": ["web_search", "fetch_page"],
    "list": ["web_search", "fetch_page"],
    "explain": ["web_search", "wikipedia", "fetch_page"],
    "howto": ["web_search", "fetch_page", "arxiv"],
}

# Retry configuration
RETRY = RetrySpec(times=2, base_delay=0.5, jitter=True)

# Approximate free-tier-equivalent token prices in $/1k tokens
MODEL_PRICES: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {
        "input": 0.0001,
        "output": 0.0004,
    },
    "llama-3.3-70b-versatile": {
        "input": 0.0001,
        "output": 0.0004,
    },
}

# Storage and cache configuration
TRACE_DIR: Path = Path("traces")
CACHE_PATH: Path = Path(".cache/runs.db")
CACHE_TTL_H: int = 24
