"""Unit tests for individual tool providers and payload parsing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest
import respx

from app.cache import Cache
from app.errors import ProviderSkipped, ToolError, ToolExhausted
from app.tools import fetch, search
from app.tools.fetch import FetchArgs, httpx_fetch, jina_fetch
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
    register(fetch.fetch_page_spec)
    register(fetch.httpx_fetch_spec)


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


@respx.mock
def test_fetch_jina_markdown_returned() -> None:
    target_url = "https://example.com/docs/qdrant"
    jina_endpoint = f"https://r.jina.ai/{target_url}"
    markdown_content = "Title: Qdrant Guide\n\n# Qdrant Guide\n\nDetailed installation and quickstart guide."
    respx.get(jina_endpoint).respond(200, text=markdown_content, headers={"content-type": "text/markdown"})

    result = jina_fetch(FetchArgs(url=target_url))
    assert result["url"] == target_url
    assert result["title"] == "Qdrant Guide"
    assert "Detailed installation" in result["text"]
    assert result["fetched_via"] == "jina"


@respx.mock
def test_fetch_httpx_trafilatura_extraction() -> None:
    target_url = "https://example.com/article"
    html_content = (
        "<!DOCTYPE html><html><head><title>Vector DBs in 2026</title></head>"
        "<body><article><h1>Vector DBs</h1><p>Modern vector search architectures rely on HNSW graphs "
        "and inverted file indexes for high throughput.</p></article></body></html>"
    )
    respx.get(target_url).respond(200, text=html_content, headers={"content-type": "text/html; charset=utf-8"})

    result = httpx_fetch(FetchArgs(url=target_url))
    assert result["url"] == target_url
    assert "Vector DBs" in result["title"]
    assert "Modern vector search architectures" in result["text"]
    assert result["fetched_via"] == "httpx"


@respx.mock
def test_fetch_js_walled_page_under_300_chars_returned() -> None:
    target_url = "https://example.com/paywall"
    wall_html = "<html><body><p>Please enable JavaScript or subscribe to continue reading.</p></body></html>"
    respx.get(target_url).respond(200, text=wall_html, headers={"content-type": "text/html"})

    # Must NOT raise; the filter drops short pages, not the fetcher
    result = httpx_fetch(FetchArgs(url=target_url))
    assert len(result["text"]) < 300
    assert "subscribe to continue" in result["text"]


@respx.mock
def test_fetch_2mb_body_rejected() -> None:
    target_url = "https://example.com/huge-file"
    huge_data = "x" * (2 * 1024 * 1024 + 100)
    respx.get(target_url).respond(200, text=huge_data, headers={"content-type": "text/html"})

    with pytest.raises(ToolError) as exc:
        httpx_fetch(FetchArgs(url=target_url))
    assert "exceeds 2MB" in str(exc.value)


@respx.mock
def test_fetch_non_html_content_type_rejected() -> None:
    target_url = "https://example.com/image.png"
    respx.get(target_url).respond(200, content=b"\x89PNG\r\n\x1a\n", headers={"content-type": "image/png"})

    with pytest.raises(ToolError) as exc:
        httpx_fetch(FetchArgs(url=target_url))
    assert "Unsupported content-type" in str(exc.value)


@respx.mock
def test_fetch_redirect_followed() -> None:
    initial_url = "https://example.com/old-link"
    final_url = "https://example.com/new-link"
    html_content = "<html><head><title>Redirected Target</title></head><body><p>Redirected content.</p></body></html>"

    respx.get(initial_url).respond(301, headers={"Location": final_url})
    respx.get(final_url).respond(200, text=html_content, headers={"content-type": "text/html"})

    result = httpx_fetch(FetchArgs(url=initial_url))
    assert "Redirected Target" in result["title"]
    assert "Redirected content." in result["text"]


def test_fetch_truncation_respects_word_boundaries() -> None:
    from app.tools.fetch import _truncate_text

    text = "The quick brown fox jumps over the lazy dog"
    # Cut at 15 chars: "The quick brown" is 15 chars, " fox" starts at 16
    truncated = _truncate_text(text, 17)
    # 17 chars would be "The quick brown f", cut at last space is "The quick brown"
    assert truncated == "The quick brown"
    assert not truncated.endswith("f")


def test_fetch_blocks_ssrf_before_request() -> None:
    ssrf_url = "http://169.254.169.254/latest/meta-data"
    with pytest.raises(ToolError):
        jina_fetch(FetchArgs(url=ssrf_url))
    with pytest.raises(ToolError):
        httpx_fetch(FetchArgs(url=ssrf_url))

