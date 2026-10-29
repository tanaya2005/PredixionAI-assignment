"""Page fetching tool wrapping Jina Reader and local httpx/trafilatura fallback."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit
from typing import Any
import httpx
from pydantic import BaseModel, ConfigDict, Field
import trafilatura

from app.config import JINA_API_KEY
from app.errors import ToolError
from app.guardrails import assert_safe_url
from app.tools.base import ToolSpec
from app.tools.registry import register

logger = logging.getLogger(__name__)

MAX_BODY_BYTES: int = 2 * 1024 * 1024  # 2 MB limit
DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 ResearchAgent/1.0"
)


class FetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    max_chars: int = Field(default=8000, ge=100, le=50000)


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars without cutting mid-word."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        return cut[:last_space].rstrip()
    return cut.rstrip()


def _extract_title_from_html(html: str, url: str) -> str:
    """Extract page title via trafilatura, title tag, or URL path."""
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and meta.title:
            return meta.title.strip()
    except Exception:
        pass

    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        clean = re.sub(r"\s+", " ", match.group(1)).strip()
        if clean:
            return clean

    path = urlsplit(url).path.strip("/")
    return path.split("/")[-1] if path else urlsplit(url).netloc


def jina_fetch(args: FetchArgs) -> dict[str, Any]:
    """Fetch page markdown via Jina Reader API."""
    safe_url = assert_safe_url(args.url)

    headers = {
        "X-Return-Format": "markdown",
        "X-Timeout": "20",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if JINA_API_KEY and JINA_API_KEY.strip():
        headers["Authorization"] = f"Bearer {JINA_API_KEY.strip()}"

    jina_url = f"https://r.jina.ai/{safe_url}"

    with httpx.Client(timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)) as client:
        resp = client.get(jina_url, headers=headers)
        resp.raise_for_status()

        body_bytes = len(resp.content)
        if body_bytes > MAX_BODY_BYTES:
            raise ToolError(
                f"Page size {body_bytes} bytes exceeds 2MB limit",
                context={"url": safe_url, "size_bytes": body_bytes},
            )

        content_type = resp.headers.get("content-type", "text/markdown").lower()
        text = resp.text or ""

        # Extract title from markdown header or metadata
        title = ""
        lines = text.splitlines()
        for line in lines[:5]:
            if line.startswith("Title:"):
                title = line[6:].strip()
                break
            if line.startswith("# "):
                title = line[2:].strip()
                break
        if not title:
            path = urlsplit(safe_url).path.strip("/")
            title = path.split("/")[-1] if path else urlsplit(safe_url).netloc

        truncated = _truncate_text(text, args.max_chars)
        return {
            "url": safe_url,
            "title": title,
            "text": truncated,
            "content_type": content_type,
            "fetched_via": "jina",
            "bytes": body_bytes,
        }


def httpx_fetch(args: FetchArgs) -> dict[str, Any]:
    """Fetch page HTML locally via httpx and extract main text with trafilatura."""
    safe_url = assert_safe_url(args.url)

    headers = {"User-Agent": DEFAULT_USER_AGENT}

    with httpx.Client(
        timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
        follow_redirects=True,
    ) as client:
        resp = client.get(safe_url, headers=headers)
        resp.raise_for_status()

        body_bytes = len(resp.content)
        if body_bytes > MAX_BODY_BYTES:
            raise ToolError(
                f"Page size {body_bytes} bytes exceeds 2MB limit",
                context={"url": safe_url, "size_bytes": body_bytes},
            )

        content_type = resp.headers.get("content-type", "").lower()
        if not any(t in content_type for t in ["html", "text", "pdf"]):
            raise ToolError(
                f"Unsupported content-type: {content_type}",
                context={"url": safe_url, "content_type": content_type},
            )

        raw_html = resp.text
        title = _extract_title_from_html(raw_html, safe_url)

        # Extract main text with trafilatura; fallback to raw text if JS-walled/minimal
        extracted = trafilatura.extract(raw_html, include_comments=False, include_tables=True)
        final_text = extracted if extracted and extracted.strip() else raw_html

        truncated = _truncate_text(final_text, args.max_chars)
        return {
            "url": safe_url,
            "title": title,
            "text": truncated,
            "content_type": content_type,
            "fetched_via": "httpx",
            "bytes": body_bytes,
        }


fetch_page_spec = ToolSpec(
    name="fetch_page",
    fn=jina_fetch,
    input_schema=FetchArgs,
    timeout_s=25.0,
    provider="jina",
    fallbacks=["httpx_fetch"],
)

httpx_fetch_spec = ToolSpec(
    name="httpx_fetch",
    fn=httpx_fetch,
    input_schema=FetchArgs,
    timeout_s=25.0,
    provider="httpx",
    fallbacks=[],
)

register(fetch_page_spec)
register(httpx_fetch_spec)
