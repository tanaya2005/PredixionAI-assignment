"""Security guardrails: SSRF protection, prompt injection detection, and untrusted wrapping."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit

from app.errors import ToolError

# Known content-farm and spam domains
CONTENT_FARM_BLOCKLIST: set[str] = {
    "contentfarm.com",
    "clickbaitnews.com",
    "spamydomain.com",
    "autoseoarticle.com",
    "cheapcontenthub.net",
    "scraperblog.org",
    "fakefactsnetwork.com",
    "keywordstuffingdaily.com",
    "junkpress.info",
    "adwarearticles.biz",
    "contentfarms.net",
    "spamsite.xyz",
    "scraperzone.info",
}

# Injection regex patterns compiled with ignorecase
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"ignore\s+(?:all\s+|any\s+|your\s+|the\s+)*(?:previous|prior)?\s*instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(?:all\s+|any\s+|your\s+|the\s+)*(?:previous|prior)?\s*instructions",
        re.IGNORECASE,
    ),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bexfiltrate\b", re.IGNORECASE),
    re.compile(r"reveal\s+your\b", re.IGNORECASE),
    re.compile(r"call\s+the\s+tool\b", re.IGNORECASE),
    re.compile(r"\btool_call\b", re.IGNORECASE),
    re.compile(r"print\s+your\s+instructions\b", re.IGNORECASE),
    re.compile(r"output\s+CONFIDENT\b", re.IGNORECASE),
    re.compile(r"high\s+confidence\s+and\s+no\s+citations", re.IGNORECASE),
    re.compile(r"no\s+citations\s+and\s+high\s+confidence", re.IGNORECASE),
    re.compile(r"^\s*SYSTEM:\s*", re.IGNORECASE | re.MULTILINE),
]


def assert_safe_url(url: str) -> str:
    """Validate that a URL is safe to fetch (SSRF guard) and return normalized URL.

    Raises ToolError for invalid schemes, IP literals, reserved/private ranges,
    userinfo, non-standard ports, or blocklisted domains.
    """
    if not url or not isinstance(url, str):
        raise ToolError("URL must be a non-empty string", context={"url": str(url)})

    try:
        parts = urlsplit(url.strip())
    except Exception as exc:
        raise ToolError(f"Malformed URL: {exc}", context={"url": url}) from exc

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ToolError(f"Forbidden scheme: {parts.scheme}", context={"url": url, "scheme": parts.scheme})

    # Reject userinfo (e.g. http://user:pass@example.com)
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ToolError("URLs containing userinfo are forbidden", context={"url": url})

    hostname = parts.hostname
    if not hostname:
        raise ToolError("URL is missing hostname", context={"url": url})

    hostname = hostname.lower()

    # Reject local/internal domain suffixes
    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
        or hostname.endswith(".internal")
    ):
        raise ToolError(f"Forbidden local hostname: {hostname}", context={"url": url, "host": hostname})

    # Reject IP literals (IPv4 and IPv6)
    clean_host = hostname.strip("[]")
    try:
        ip = ipaddress.ip_address(clean_host)
        # Any IP literal is forbidden
        raise ToolError(
            f"Direct IP literals are forbidden: {ip}",
            context={"url": url, "ip": str(ip), "is_private": ip.is_private},
        )
    except ValueError:
        # Not an IP literal, proceed to domain checks
        pass

    # Port restriction: only default ports (80 for http, 443 for https) or omitted
    port = parts.port
    if port is not None and port not in (80, 443):
        raise ToolError(f"Forbidden port: {port}", context={"url": url, "port": port})

    # Domain blocklist check
    host_parts = hostname.split(".")
    for i in range(len(host_parts) - 1):
        domain_candidate = ".".join(host_parts[i:])
        if domain_candidate in CONTENT_FARM_BLOCKLIST:
            raise ToolError(
                f"Blocklisted domain: {domain_candidate}",
                context={"url": url, "domain": domain_candidate},
            )

    # Reconstruct normalized URL (lowercase scheme and host)
    netloc = hostname
    if port and ((scheme == "http" and port != 80) or (scheme == "https" and port != 443)):
        netloc = f"{hostname}:{port}"

    return urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment))


def scan_injection(text: str) -> bool:
    """Scan first 8000 characters of text for prompt injection patterns."""
    if not text:
        return False
    sample = text[:8000]
    return any(pattern.search(sample) is not None for pattern in INJECTION_PATTERNS)


def strip_injection(text: str) -> tuple[str, list[str]]:
    """Remove lines matching injection patterns and return (cleaned_text, hit_descriptions)."""
    if not text:
        return "", []

    lines = text.splitlines(keepends=True)
    clean_lines: list[str] = []
    hits: list[str] = []

    for line in lines:
        matched = False
        for pattern in INJECTION_PATTERNS:
            if pattern.search(line):
                matched = True
                hits.append(pattern.pattern)
                break
        if not matched:
            clean_lines.append(line)

    return "".join(clean_lines), hits


def wrap_untrusted(text: str, id: str, url: str) -> str:
    """Wrap untrusted text in XML-style boundary tag and escape inner boundary tags."""
    # Escape any closing boundary tag occurring inside the untrusted text
    escaped_text = text.replace("</untrusted_document>", "&lt;/untrusted_document&gt;")
    return f'<untrusted_document id="{id}" url="{url}">\n{escaped_text}\n</untrusted_document>'


def domain_trust(host: str) -> float:
    """Compute trust score for a domain per multi-tier domain reputation rules."""
    if not host:
        return 0.5

    clean = host.lower().strip()
    if ":" in clean:
        clean = clean.split(":")[0]
    if clean.startswith("www."):
        clean = clean[4:]

    # Blocklisted domains
    if clean in CONTENT_FARM_BLOCKLIST:
        return 0.0

    # Official / documentation / .edu / .gov
    if (
        clean.endswith(".gov")
        or clean.endswith(".edu")
        or ".gov." in clean
        or ".edu." in clean
        or clean.endswith(".ac.uk")
    ):
        return 1.0

    # Official docs and developer portals
    if (
        clean.startswith("docs.")
        or clean.startswith("documentation.")
        or clean.startswith("developer.")
        or clean
        in {
            "github.com",
            "qdrant.tech",
            "weaviate.io",
            "milvus.io",
            "python.org",
            "postgresql.org",
            "elastic.co",
            "redis.io",
            "sqlite.org",
            "apache.org",
        }
    ):
        return 1.0

    # Established news & Wikipedia
    if clean == "wikipedia.org" or clean.endswith(".wikipedia.org"):
        return 0.8
    if clean in {
        "nytimes.com",
        "bbc.com",
        "bbc.co.uk",
        "reuters.com",
        "theguardian.com",
        "bloomberg.com",
        "techcrunch.com",
        "wired.com",
        "wsj.com",
        "economist.com",
        "arstechnica.com",
    }:
        return 0.8

    # Forums
    if clean in {
        "reddit.com",
        "news.ycombinator.com",
        "stackoverflow.com",
        "quora.com",
    } or clean.startswith("forum.") or clean.startswith("community."):
        return 0.3

    # Blogs and user-generated articles
    if clean in {"medium.com", "substack.com"} or clean.endswith(".substack.com") or clean.endswith(".medium.com"):
        return 0.5

    # Default general web tier
    return 0.5
