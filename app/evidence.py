"""Evidence normalization, canonical URL deduplication, simhash near-deduping, and junk filtering."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import ipaddress
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.config import MAX_SOURCE_CHARS
from app.contracts import Source
from app.guardrails import (
    CONTENT_FARM_BLOCKLIST,
    assert_safe_url,
    domain_trust,
    scan_injection,
)

_TRACKING_PARAMS: set[str] = {"ref", "fbclid", "gclid", "mc_cid"}
_WORD_REGEX = re.compile(r"\b[a-zA-Z0-9]{2,}\b")


def canonical_url(url: str) -> str:
    """Normalize URL by lowercasing scheme/host, stripping www., removing tracking params, and sorting query params."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except Exception:
        return url

    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    port = parts.port
    if port and ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = host
    elif port:
        netloc = f"{host}:{port}"
    else:
        netloc = host

    path = parts.path.rstrip("/")

    query_tuples = parse_qsl(parts.query, keep_blank_values=True)
    filtered: list[tuple[str, str]] = []
    for k, v in query_tuples:
        kl = k.lower()
        if kl.startswith("utm_") or kl in _TRACKING_PARAMS:
            continue
        filtered.append((k, v))
    filtered.sort(key=lambda x: x[0])
    query = urlencode(filtered)

    return urlunsplit((scheme, netloc, path, query, ""))


def simhash64(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint for near-duplicate text detection."""
    tokens = _WORD_REGEX.findall(text.lower())
    if not tokens:
        return 0

    v = [0] * 64
    for token in tokens:
        # Compute 64-bit hash of token using md5
        digest = hashlib.md5(token.encode("utf-8")).digest()
        h = int.from_bytes(digest[:8], byteorder="big", signed=False)
        for i in range(64):
            bit = (h >> i) & 1
            if bit == 1:
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= 1 << i

    return fingerprint


def hamming(a: int, b: int) -> int:
    """Compute the Hamming distance (number of differing bits) between two 64-bit integers."""
    return (a ^ b).bit_count()


def is_junk(source: Source | dict[str, Any]) -> str | None:
    """Evaluate source against deterministic junk rules in priority order.

    Returns the rejection reason string if junk, or None if acceptable.
    """
    if isinstance(source, Source):
        text = source.text
        url = source.url
        content_type = "text/html"
        byte_size = len(text.encode("utf-8"))
        domain = source.domain
    else:
        text = source.get("text") or source.get("snippet") or ""
        url = source.get("url") or ""
        content_type = source.get("content_type") or "text/html"
        byte_size = source.get("bytes") or len(text.encode("utf-8"))
        domain = source.get("domain") or (urlsplit(url).hostname or "")

    clean_text = text.strip()

    # Rule 1: Text shorter than 300 characters
    if len(clean_text) < 300:
        return "text_under_300_chars"

    first_400 = clean_text[:400].lower()

    # Rule 2: HTML error markers in the first 400 characters
    error_markers = ["404", "page not found", "not found", "403 forbidden"]
    for marker in error_markers:
        if marker in first_400:
            return "html_error_marker"

    # Rule 3: JS and paywall walls in the first 400 characters
    wall_markers = [
        "enable javascript",
        "subscribe to continue",
        "cookie",
        "are you a robot",
        "sign in to continue",
    ]
    for marker in wall_markers:
        if marker in first_400:
            return "paywall_or_cookie_wall"

    # Rule 4: Scheme not http or https
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in ("http", "https"):
            return "invalid_scheme"
    except Exception:
        return "invalid_scheme"

    # Rule 5: Private or IP-literal host
    clean_host = (parts.hostname or "").strip("[]").lower()
    try:
        ip = ipaddress.ip_address(clean_host)
        return "ip_literal_host"
    except ValueError:
        pass

    # Rule 6: Content-type not in {text/html, text/plain, application/pdf}
    ct_lower = content_type.lower()
    valid_cts = ["html", "text", "pdf"]
    if not any(valid in ct_lower for valid in valid_cts):
        return "unsupported_content_type"

    # Rule 7: Body over 2MB
    if byte_size > 2 * 1024 * 1024:
        return "body_over_2mb"

    # Rule 8: Blocklisted domain
    clean_domain = domain.lower()
    if clean_domain.startswith("www."):
        clean_domain = clean_domain[4:]
    if clean_domain in CONTENT_FARM_BLOCKLIST:
        return "blocklisted_domain"

    return None


def dedupe(sources: list[Source]) -> list[Source]:
    """Perform exact canonical-URL deduplication, followed by SimHash-64 near-deduplication."""
    # Step 1: Exact canonical URL deduplication
    url_groups: dict[str, Source] = {}
    for s in sources:
        c_url = canonical_url(s.url)
        if c_url not in url_groups:
            url_groups[c_url] = s
        else:
            existing = url_groups[c_url]
            # Keep higher trust; if equal, keep longer text
            if (s.trust, len(s.text)) > (existing.trust, len(existing.text)):
                url_groups[c_url] = s

    exact_deduped = list(url_groups.values())

    # Step 2: SimHash-64 near-deduplication (Hamming distance <= 3)
    final_sources: list[Source] = []
    fingerprints: list[tuple[int, Source]] = []

    for s in exact_deduped:
        fp = simhash64(s.text)
        is_near_dup = False
        for idx, (existing_fp, existing_s) in enumerate(fingerprints):
            if hamming(fp, existing_fp) <= 3:
                is_near_dup = True
                # Keep the longer text
                if len(s.text) > len(existing_s.text):
                    fingerprints[idx] = (fp, s)
                    final_sources[idx] = s
                break

        if not is_near_dup:
            fingerprints.append((fp, s))
            final_sources.append(s)

    return final_sources


def rank_candidates(sources: list[Source]) -> list[Source]:
    """Sort candidate sources by trust descending, then text length descending."""
    return sorted(sources, key=lambda s: (s.trust, len(s.text)), reverse=True)


def build_sources(
    raw_hits: list[dict[str, Any]],
    fetched_pages: list[dict[str, Any]] | None = None,
) -> list[Source]:
    """Assemble standardized Source objects, dedupe, rank, and assign stable contiguous ids."""
    fetched_dict: dict[str, dict[str, Any]] = {}
    for p in fetched_pages or []:
        c_url = canonical_url(p.get("url", ""))
        if c_url:
            fetched_dict[c_url] = p

    candidates: list[Source] = []
    now = datetime.now(timezone.utc)

    # Process all raw hits
    for hit in raw_hits:
        url = hit.get("url") or ""
        if not url:
            continue
        try:
            safe_url = assert_safe_url(url)
        except Exception:
            continue

        c_url = canonical_url(safe_url)
        page_info = fetched_dict.get(c_url, {})

        title = page_info.get("title") or hit.get("title") or c_url
        full_text = page_info.get("text") or hit.get("snippet") or ""
        tool_name = hit.get("tool") or "web_search"
        provider_name = hit.get("provider") or "tavily"

        # Check prompt injection
        untrusted = scan_injection(full_text)

        domain = urlsplit(safe_url).hostname or ""
        base_trust = domain_trust(domain)
        trust = min(base_trust, 0.3) if untrusted else base_trust

        # Trim text to MAX_SOURCE_CHARS
        trimmed_text = full_text[:MAX_SOURCE_CHARS]

        candidates.append(
            Source(
                id="temp",
                url=safe_url,
                title=title.strip(),
                domain=domain,
                published=hit.get("published"),
                retrieved_at=now,
                text=trimmed_text,
                trust=round(trust, 2),
                tool=tool_name,
                provider=provider_name,
                untrusted=untrusted,
            )
        )

    # Add any fetched pages that weren't in raw_hits
    for c_url, p in fetched_dict.items():
        if not any(canonical_url(c.url) == c_url for c in candidates):
            url = p.get("url", "")
            try:
                safe_url = assert_safe_url(url)
            except Exception:
                continue

            full_text = p.get("text", "")
            untrusted = scan_injection(full_text)
            domain = urlsplit(safe_url).hostname or ""
            base_trust = domain_trust(domain)
            trust = min(base_trust, 0.3) if untrusted else base_trust

            candidates.append(
                Source(
                    id="temp",
                    url=safe_url,
                    title=(p.get("title") or c_url).strip(),
                    domain=domain,
                    published=p.get("published"),
                    retrieved_at=now,
                    text=full_text[:MAX_SOURCE_CHARS],
                    trust=round(trust, 2),
                    tool=p.get("tool") or "fetch_page",
                    provider=p.get("provider") or p.get("fetched_via") or "jina",
                    untrusted=untrusted,
                )
            )

    # Step 1: Deduplicate
    deduped = dedupe(candidates)

    # Step 2: Rank by (trust desc, len(text) desc)
    ranked = rank_candidates(deduped)

    # Step 3: Assign stable, contiguous ids s1..sN
    final_sources: list[Source] = []
    for idx, s in enumerate(ranked, start=1):
        final_sources.append(
            Source(
                id=f"s{idx}",
                url=s.url,
                title=s.title,
                domain=s.domain,
                published=s.published,
                retrieved_at=s.retrieved_at,
                text=s.text,
                trust=s.trust,
                tool=s.tool,
                provider=s.provider,
                untrusted=s.untrusted,
            )
        )

    return final_sources
