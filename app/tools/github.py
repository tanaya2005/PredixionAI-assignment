"""GitHub repository statistics tool (registered, disabled by default in policy)."""

from __future__ import annotations

import logging
from typing import Any
import httpx
from pydantic import BaseModel, ConfigDict

from app.config import GITHUB_TOKEN
from app.tools.base import ToolSpec
from app.tools.registry import register

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "ResearchAgent/1.0"


class RepoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str
    repo: str


def github_repo(args: RepoArgs) -> list[dict[str, Any]]:
    """Retrieve repository metadata from GitHub API; returns empty list on 404."""
    url = f"https://api.github.com/repos/{args.owner.strip()}/{args.repo.strip()}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if GITHUB_TOKEN and GITHUB_TOKEN.strip():
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN.strip()}"

    with httpx.Client(timeout=15.0) as client:
        resp = client.get(url, headers=headers)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()

    license_info = data.get("license") or {}
    license_id = license_info.get("spdx_id") or license_info.get("key")

    full_name = data.get("full_name") or f"{args.owner}/{args.repo}"
    html_url = data.get("html_url") or url
    stars = data.get("stargazers_count", 0)
    description = data.get("description") or ""

    return [
        {
            "title": full_name,
            "url": html_url,
            "stars": stars,
            "pushed_at": data.get("pushed_at"),
            "description": description,
            "license": license_id,
            "open_issues": data.get("open_issues_count", 0),
            "snippet": f"{full_name}: {description} ({stars} stars)",
            "published": (data.get("pushed_at") or "").split("T")[0] or None,
        }
    ]


github_spec = ToolSpec(
    name="github",
    fn=github_repo,
    input_schema=RepoArgs,
    timeout_s=15.0,
    provider="github",
    fallbacks=[],
)

# Registered in registry but NOT added to config.TOOL_POLICY
register(github_spec)
