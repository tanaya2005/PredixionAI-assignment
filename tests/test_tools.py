"""Unit tests for individual tool providers and payload parsing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest
import respx

from app.cache import Cache
from app.errors import ProviderSkipped, ToolExhausted
from app.tools import search
from app.tools.registry import REGISTRY, call, register
from app.tools.search import SearchArgs, ddg_search, exa_search, tavily_search
from app.trace import Tracer


@pytest.fixture(autouse=True)
def clean_tool_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_cache = Cache(db_path=tmp_path / "search_test_cache.db")
    monkeypatch.setattr("app.tools.registry.cache", test_cache)
    # Re-register tools in case registry was cleared
    register(search.web_search_spec)
    register(search.exa_spec)
    register(search.ddg_spec)


@respx.mock
def test_tavily_search_payload_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.search.TAVILY_API_KEY", "tvly-test-key")

    mock_response = {
        "results": [
            {
                "title": "Qdrant Vector DB",
                "url": "https://qdrant.tech/documentation/",
                "content": "Qdrant is an open-source vector database.",
                "published_date": "2026-04-01",
            },
            {
                "title": "Blocked SSRF Result",
                "url": "http://169.254.169.254/meta-data",
                "content": "Secret cloud metadata.",
                "published_date": None,
            },
        ]
    }
    respx.post("https://api.tavily.com/search").respond(200, json=mock_response)

    results = tavily_search(SearchArgs(query="vector database", max_results=5))

    assert len(results) == 1  # SSRF URL filtered out
    assert results[0]["title"] == "Qdrant Vector DB"
    assert results[0]["url"] == "https://qdrant.tech/documentation"
    assert results[0]["snippet"] == "Qdrant is an open-source vector database."
    assert results[0]["published"] == "2026-04-01"


@respx.mock
def test_exa_search_payload_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.search.EXA_API_KEY", "exa-test-key")

    mock_response = {
        "results": [
            {
                "title": "Milvus Vector Database",
                "url": "https://milvus.io/docs",
                "text": "Milvus is a distributed vector database.",
                "publishedDate": "2026-03-15",
            }
        ]
    }
    respx.post("https://api.exa.ai/search").respond(200, json=mock_response)

    results = exa_search(SearchArgs(query="milvus vector database", max_results=5))

    assert len(results) == 1
    assert results[0]["title"] == "Milvus Vector Database"
    assert results[0]["snippet"] == "Milvus is a distributed vector database."
    assert results[0]["published"] == "2026-03-15"


def test_ddg_search_payload_parsing() -> None:
    mock_ddg_data = [
        {
            "title": "Weaviate Documentation",
            "href": "https://weaviate.io/developers/weaviate",
            "body": "Weaviate is an open-source vector search engine.",
        }
    ]

    with patch("ddgs.DDGS") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.text.return_value = mock_ddg_data
        mock_cls.return_value = mock_instance
        results = ddg_search(SearchArgs(query="weaviate vector db", max_results=5))

    assert len(results) == 1
    assert results[0]["title"] == "Weaviate Documentation"
    assert results[0]["url"] == "https://weaviate.io/developers/weaviate"
    assert results[0]["snippet"] == "Weaviate is an open-source vector search engine."
    assert results[0]["published"] is None


def test_missing_api_key_skips_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.search.TAVILY_API_KEY", None)
    monkeypatch.setattr("app.tools.search.EXA_API_KEY", "")

    with pytest.raises(ProviderSkipped):
        tavily_search(SearchArgs(query="test"))

    with pytest.raises(ProviderSkipped):
        exa_search(SearchArgs(query="test"))


def test_web_search_fallback_to_ddg_when_keys_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tavily and Exa keys are missing
    monkeypatch.setattr("app.tools.search.TAVILY_API_KEY", None)
    monkeypatch.setattr("app.tools.search.EXA_API_KEY", None)

    tracer = Tracer("run_ddg_fallback", trace_dir=tmp_path)
    mock_ddg_data = [{"title": "DDG Result", "href": "https://example.com/ddg", "body": "DDG Snippet"}]

    with patch("ddgs.DDGS") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.text.return_value = mock_ddg_data
        mock_cls.return_value = mock_instance
        res = call("web_search", SearchArgs(query="python vector db"), tracer=tracer)

    assert res.status == "ok"
    assert res.provider == "duckduckgo"
    assert len(res.items) == 1

    events = tracer.events()
    # Tavily and Exa were skipped, DDG succeeded
    statuses = [e["status"] for e in events]
    assert "skipped" in statuses
    assert events[-1]["provider"] == "duckduckgo"
    assert events[-1]["status"] == "ok"


@respx.mock
def test_all_search_providers_down_raises_tool_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.tools.search.TAVILY_API_KEY", "tvly-key")
    monkeypatch.setattr("app.tools.search.EXA_API_KEY", "exa-key")

    respx.post("https://api.tavily.com/search").respond(500)
    respx.post("https://api.exa.ai/search").respond(500)

    tracer = Tracer("run_all_down", trace_dir=tmp_path)

    with patch("ddgs.DDGS") as mock_cls, patch("time.sleep", return_value=None):
        mock_instance = MagicMock()
        mock_instance.text.side_effect = RuntimeError("DDG Blocked")
        mock_cls.return_value = mock_instance
        with pytest.raises(ToolExhausted) as exc:
            call("web_search", SearchArgs(query="fail query"), tracer=tracer)

    assert "web_search" in str(exc.value)


@respx.mock
def test_search_dedupes_canonical_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.tools.search.TAVILY_API_KEY", "tvly-key")

    mock_response = {
        "results": [
            {
                "title": "Page 1",
                "url": "https://example.com/article?utm_source=twitter&b=2&a=1",
                "content": "Content 1",
            },
            {
                "title": "Page 1 Duplicate",
                "url": "https://www.example.com/article/?utm_medium=email&a=1&b=2#section",
                "content": "Content duplicate",
            },
        ]
    }
    respx.post("https://api.tavily.com/search").respond(200, json=mock_response)

    results = tavily_search(SearchArgs(query="dedupe test"))
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/article?a=1&b=2"
