"""Evidence normalization, canonical URL deduplication, simhash near-deduping, and junk filtering."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = {"ref", "fbclid", "gclid", "mc_cid"}


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

    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = ""

    query_tuples = parse_qsl(parts.query, keep_blank_values=True)
    filtered = []
    for k, v in query_tuples:
        kl = k.lower()
        if kl.startswith("utm_") or kl in _TRACKING_PARAMS:
            continue
        filtered.append((k, v))
    filtered.sort(key=lambda x: x[0])
    query = urlencode(filtered)

    return urlunsplit((scheme, netloc, path, query, ""))
