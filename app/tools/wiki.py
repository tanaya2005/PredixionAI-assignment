"""Wikipedia tool providing encyclopedic summary retrieval and title resolution."""

from __future__ import annotations

import logging
from urllib.parse import quote
from typing import Any
import httpx
from pydantic import BaseModel, ConfigDict

from app.tools.base import ToolSpec
from app.tools.registry import register

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "ResearchAgent/1.0 (academic research pipeline; contact@example.com)"


class WikiArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str


def wiki_summary(args: WikiArgs) -> list[dict[str, Any]]:
    """Retrieve Wikipedia summary for a topic with search-based title resolution."""
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    raw_title = args.title.strip()
    resolved_title = raw_title

    with httpx.Client(timeout=15.0) as client:
        # Step 1: Resolve title via MediaWiki search API
        search_url = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(raw_title)}"
            "&format=json&utf8=1"
        )
        try:
            search_resp = client.get(search_url, headers=headers)
            if search_resp.status_code == 200:
                data = search_resp.json()
                results = data.get("query", {}).get("search", [])
                if results:
                    resolved_title = results[0].get("title", raw_title)
        except Exception as exc:
            logger.warning("Wikipedia title resolution failed: %s", exc)

        # Step 2: Fetch page summary
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(resolved_title.replace(' ', '_'))}"
        resp = client.get(summary_url, headers=headers)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        summary_data = resp.json()

    title = summary_data.get("title", resolved_title)
    urls = summary_data.get("content_urls", {}).get("desktop", {})
    page_url = urls.get("page") or f"https://en.wikipedia.org/wiki/{quote(resolved_title.replace(' ', '_'))}"
    extract = summary_data.get("extract") or ""

    return [
        {
            "title": title,
            "url": page_url,
            "text": extract,
            "snippet": extract[:400],
            "published": None,
        }
    ]


wiki_spec = ToolSpec(
    name="wikipedia",
    fn=wiki_summary,
    input_schema=WikiArgs,
    timeout_s=15.0,
    provider="wikipedia",
    fallbacks=[],
)

register(wiki_spec)
