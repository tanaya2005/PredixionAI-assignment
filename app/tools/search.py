"""Web search tool wrapping Tavily, Exa, and DuckDuckGo search providers."""

from __future__ import annotations

import logging
from typing import Any
import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.config import EXA_API_KEY, TAVILY_API_KEY
from app.errors import ProviderSkipped, ToolError
from app.evidence import canonical_url
from app.guardrails import assert_safe_url
from app.tools.base import ToolSpec
from app.tools.registry import register

logger = logging.getLogger(__name__)


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    max_results: int = Field(default=10, ge=1, le=20)


def _filter_and_dedupe_hits(raw_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out non-http(s) and unsafe URLs, dedupe by canonical URL within one response."""
    seen_canonical: set[str] = set()
    cleaned_hits: list[dict[str, Any]] = []

    for hit in raw_hits:
        raw_url = hit.get("url")
        if not raw_url or not isinstance(raw_url, str):
            continue
        try:
            safe_url = assert_safe_url(raw_url)
        except Exception:
            # Skip SSRF or unsafe URLs
            continue

        c_url = canonical_url(safe_url)
        if c_url in seen_canonical:
            continue
        seen_canonical.add(c_url)

        cleaned_hits.append(
            {
                "title": (hit.get("title") or "").strip(),
                "url": c_url,
                "snippet": (hit.get("snippet") or "").strip(),
                "published": hit.get("published"),
            }
        )

    return cleaned_hits


def tavily_search(args: SearchArgs) -> list[dict[str, Any]]:
    """Search via Tavily API; skips cleanly if API key is not configured."""
    api_key = TAVILY_API_KEY
    if not api_key or not api_key.strip():
        raise ProviderSkipped("TAVILY_API_KEY is not configured")

    payload = {
        "api_key": api_key.strip(),
        "query": args.query,
        "search_depth": "basic",
        "max_results": args.max_results,
        "include_raw_content": False,
        "include_answer": False,
    }

    with httpx.Client(timeout=20.0) as client:
        resp = client.post("https://api.tavily.com/search", json=payload)
        resp.raise_for_status()
        data = resp.json()

    hits: list[dict[str, Any]] = []
    for item in data.get("results", []):
        hits.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "published": item.get("published_date"),
            }
        )

    return _filter_and_dedupe_hits(hits)


def exa_search(args: SearchArgs) -> list[dict[str, Any]]:
    """Search via Exa API; skips cleanly if API key is not configured."""
    api_key = EXA_API_KEY
    if not api_key or not api_key.strip():
        raise ProviderSkipped("EXA_API_KEY is not configured")

    headers = {
        "x-api-key": api_key.strip(),
        "Content-Type": "application/json",
    }
    payload = {
        "query": args.query,
        "numResults": args.max_results,
        "type": "auto",
        "contents": {
            "text": {
                "maxCharacters": 1200,
            }
        },
    }

    with httpx.Client(timeout=20.0) as client:
        resp = client.post("https://api.exa.ai/search", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    hits: list[dict[str, Any]] = []
    for item in data.get("results", []):
        contents = item.get("contents") or {}
        text = item.get("text") or contents.get("text") or ""
        hits.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": text,
                "published": item.get("publishedDate"),
            }
        )

    return _filter_and_dedupe_hits(hits)


def ddg_search(args: SearchArgs) -> list[dict[str, Any]]:
    """Search via DuckDuckGo keyless search."""
    try:
        from ddgs import DDGS
    except ImportError as e:
        raise ToolError(f"ddgs package is not installed: {e}") from e

    try:
        results = DDGS().text(args.query, max_results=args.max_results)
    except Exception as exc:
        raise ToolError(f"DuckDuckGo search failed: {exc}") from exc

    hits: list[dict[str, Any]] = []
    for r in results or []:
        hits.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": r.get("body", ""),
                "published": None,
            }
        )

    return _filter_and_dedupe_hits(hits)


# Register primary tool and fallbacks
web_search_spec = ToolSpec(
    name="web_search",
    fn=tavily_search,
    input_schema=SearchArgs,
    timeout_s=20.0,
    provider="tavily",
    fallbacks=["exa_search", "ddg_search"],
)

exa_spec = ToolSpec(
    name="exa_search",
    fn=exa_search,
    input_schema=SearchArgs,
    timeout_s=20.0,
    provider="exa",
    fallbacks=[],
)

ddg_spec = ToolSpec(
    name="ddg_search",
    fn=ddg_search,
    input_schema=SearchArgs,
    timeout_s=20.0,
    provider="duckduckgo",
    fallbacks=[],
)

register(web_search_spec)
register(exa_spec)
register(ddg_spec)
