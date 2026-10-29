"""ArXiv tool providing scientific paper search and abstract retrieval."""

from __future__ import annotations

import logging
from urllib.parse import quote
import xml.etree.ElementTree as ET
from typing import Any
import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.tools.base import ToolSpec
from app.tools.registry import register

logger = logging.getLogger(__name__)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    max_results: int = Field(default=5, ge=1, le=20)


def arxiv_search(args: ArxivArgs) -> list[dict[str, Any]]:
    """Search ArXiv API and parse Atom XML entries into standardized hits."""
    encoded_query = quote(args.query.strip())
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results={args.max_results}"

    with httpx.Client(timeout=20.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        xml_text = resp.text

    hits: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
        for entry in root.findall("atom:entry", ATOM_NS):
            title_elem = entry.find("atom:title", ATOM_NS)
            summary_elem = entry.find("atom:summary", ATOM_NS)
            id_elem = entry.find("atom:id", ATOM_NS)
            published_elem = entry.find("atom:published", ATOM_NS)

            title = " ".join((title_elem.text or "").split()) if title_elem is not None else ""
            summary = " ".join((summary_elem.text or "").split()) if summary_elem is not None else ""
            paper_url = (id_elem.text or "").strip() if id_elem is not None else ""

            # Check alternative HTML link
            for link in entry.findall("atom:link", ATOM_NS):
                if link.attrib.get("type") == "text/html":
                    paper_url = link.attrib.get("href") or paper_url
                    break

            pub_str = None
            if published_elem is not None and published_elem.text:
                pub_str = published_elem.text.strip().split("T")[0]

            truncated_summary = summary[:1200]
            hits.append(
                {
                    "title": title,
                    "url": paper_url,
                    "snippet": truncated_summary,
                    "text": summary,
                    "published": pub_str,
                }
            )
    except Exception as exc:
        logger.warning("Error parsing ArXiv Atom XML: %s", exc)

    return hits


arxiv_spec = ToolSpec(
    name="arxiv",
    fn=arxiv_search,
    input_schema=ArxivArgs,
    timeout_s=20.0,
    provider="arxiv",
    fallbacks=[],
)

register(arxiv_spec)
