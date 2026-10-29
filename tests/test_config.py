"""Unit tests for configuration loading, budgets, tool policy, and error classes."""

from __future__ import annotations

import os
from unittest.mock import patch
import pytest

from app.config import (
    BUDGETS,
    CACHE_PATH,
    CACHE_TTL_H,
    MAX_EVIDENCE_CHARS,
    MAX_SOURCE_CHARS,
    MODEL_PRICES,
    MODELS,
    RETRY,
    TOOL_POLICY,
    TRACE_DIR,
    Settings,
    has_key,
    settings,
)
from app.errors import (
    BudgetExhausted,
    ResearchAgentError,
    SchemaValidationError,
    ToolError,
    ToolExhausted,
)


def test_config_imports_with_empty_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    empty_settings = Settings()
    assert empty_settings.gemini_api_key is None
    assert empty_settings.groq_api_key is None
    assert empty_settings.has_key("GEMINI_API_KEY") is False
    assert empty_settings.has_key("tavily") is False


def test_tool_policy_structure() -> None:
    assert sorted(TOOL_POLICY.keys()) == ["comparison", "explain", "howto", "list"]
    assert "github" not in TOOL_POLICY["comparison"]
    assert "github" not in sum(TOOL_POLICY.values(), [])
    assert "web_search" in TOOL_POLICY["comparison"]
    assert "wikipedia" in TOOL_POLICY["explain"]
    assert "arxiv" in TOOL_POLICY["howto"]


def test_models_differentiation() -> None:
    required_roles = {"intake", "plan", "filter", "synthesize", "judge"}
    assert set(MODELS.keys()) == required_roles
    assert MODELS["judge"] != MODELS["synthesize"]
    assert MODELS["synthesize"] == "gemini-2.5-flash"
    assert MODELS["judge"] == "llama-3.3-70b-versatile"


def test_budgets_and_constants() -> None:
    assert BUDGETS.max_tool_calls == 6
    assert BUDGETS.wall_clock_s == 60.0
    assert BUDGETS.k_sources == 6
    assert BUDGETS.max_source_chars == 8000
    assert BUDGETS.max_evidence_chars == 40000
    assert MAX_SOURCE_CHARS == 8000
    assert MAX_EVIDENCE_CHARS == 40000


def test_retry_and_cache_config() -> None:
    assert RETRY.times == 2
    assert RETRY.base_delay == 0.5
    assert RETRY.jitter is True
    assert CACHE_TTL_H == 24
    assert str(CACHE_PATH).replace("\\", "/") == ".cache/runs.db"
    assert str(TRACE_DIR).replace("\\", "/") == "traces"


def test_model_prices_exist() -> None:
    for model_name in ["gemini-2.5-flash", "llama-3.3-70b-versatile"]:
        assert model_name in MODEL_PRICES
        assert "input" in MODEL_PRICES[model_name]
        assert "output" in MODEL_PRICES[model_name]
        assert MODEL_PRICES[model_name]["input"] > 0
        assert MODEL_PRICES[model_name]["output"] > 0


def test_has_key_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_PROV_API_KEY", "secret_value")
    custom_settings = Settings()
    assert custom_settings.has_key("TEST_PROV_API_KEY") is True
    assert custom_settings.has_key("non_existent_key") is False


def test_errors_hierarchy() -> None:
    ctx = {"provider": "tavily", "attempt": 2}
    err = ToolError("Failed to fetch", context=ctx)
    assert isinstance(err, ResearchAgentError)
    assert isinstance(err, Exception)
    assert err.context == ctx

    exhausted = ToolExhausted("All tools down", context={"chain": ["tavily", "exa"]})
    assert isinstance(exhausted, ResearchAgentError)

    schema_err = SchemaValidationError("Invalid schema")
    assert isinstance(schema_err, ResearchAgentError)
    assert schema_err.context == {}

    budget_err = BudgetExhausted("Time exceeded", context={"elapsed_s": 61.2})
    assert isinstance(budget_err, ResearchAgentError)
