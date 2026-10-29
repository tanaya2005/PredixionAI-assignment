"""Unit tests for evidence canonicalization, deduplication, and junk filtering."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from app.contracts import Source
from app.evidence import (
    build_sources,
    canonical_url,
    dedupe,
    hamming,
    is_junk,
    rank_candidates,
    simhash64,
)


@pytest.mark.parametrize(
    "raw_url, expected",
    [
        (
            "https://WWW.Example.com/a/?utm_source=x&b=2&a=1#frag",
            "https://example.com/a?a=1&b=2",
        ),
        ("http://example.com/", "http://example.com"),
        ("https://example.com/path/?ref=twitter", "https://example.com/path"),
        ("https://example.com?fbclid=12345&q=test", "https://example.com?q=test"),
        ("https://example.com:443/docs/", "https://example.com/docs"),
        ("http://example.com:80/", "http://example.com"),
        (
            "https://example.com/item?gclid=abc&utm_medium=email&id=99",
            "https://example.com/item?id=99",
        ),
        ("https://example.com/path?mc_cid=xyz&category=ai", "https://example.com/path?category=ai"),
        ("https://WWW.QDRANT.TECH/documentation/", "https://qdrant.tech/documentation"),
        ("https://example.com/a/b/?", "https://example.com/a/b"),
        ("https://example.com/search?z=3&a=1&m=2", "https://example.com/search?a=1&m=2&z=3"),
        ("https://example.com/#frag-only", "https://example.com"),
    ],
)
def test_canonical_url_table_driven(raw_url: str, expected: str) -> None:
    assert canonical_url(raw_url) == expected


def test_simhash_merges_identical_and_near_duplicates() -> None:
    article1 = (
        "Qdrant is an open-source vector search engine and database written in Rust. "
        "It provides fast and scalable vector similarity search with rich filtering capabilities. "
    ) * 10
    article2 = (
        "qdrant is an open-source vector search engine and database written in rust. "
        "it provides fast and scalable vector similarity search with rich filtering capabilities. "
    ) * 10

    fp1 = simhash64(article1)
    fp2 = simhash64(article2)
    assert hamming(fp1, fp2) <= 3


def test_simhash_keeps_different_articles_apart() -> None:
    article_tech = (
        "Hierarchical Navigable Small World graphs optimize high-dimensional nearest neighbor queries. "
        "Quantization reduces memory footprint across distributed clusters. "
    ) * 10
    article_cooking = (
        "Bake the sourdough bread in a preheated Dutch oven at 450 degrees Fahrenheit. "
        "Ensure the crust is deeply caramelized and the internal crumb is fully set. "
    ) * 10

    fp_tech = simhash64(article_tech)
    fp_cook = simhash64(article_cooking)
    assert hamming(fp_tech, fp_cook) > 10


def test_junk_rules_each_fires_once() -> None:
    # Rule 1: Text shorter than 300 chars
    short_doc = {"text": "Short snippet only 50 chars long.", "url": "https://example.com"}
    assert is_junk(short_doc) == "text_under_300_chars"

    # Rule 2: HTML error marker in first 400 chars
    err_doc = {
        "text": "<html><head><title>404 Page Not Found</title></head><body>Sorry, the requested page was not found. "
        + ("extra padding text content " * 20),
        "url": "https://example.com",
    }
    assert is_junk(err_doc) == "html_error_marker"

    # Rule 3: Paywall / cookie wall in first 400 chars
    wall_doc = {
        "text": "Please enable JavaScript or subscribe to continue reading our exclusive subscriber analysis. "
        + ("extra padding text content " * 20),
        "url": "https://example.com",
    }
    assert is_junk(wall_doc) == "paywall_or_cookie_wall"

    # Rule 4: Invalid scheme
    ftp_doc = {
        "text": ("Valid lengthy text content for the article. " * 20),
        "url": "ftp://example.com/file",
    }
    assert is_junk(ftp_doc) == "invalid_scheme"

    # Rule 5: IP literal host
    ip_doc = {
        "text": ("Valid lengthy text content for the article. " * 20),
        "url": "http://192.168.1.1/doc",
    }
    assert is_junk(ip_doc) == "ip_literal_host"

    # Rule 6: Unsupported content-type
    img_doc = {
        "text": ("Valid lengthy text content for the article. " * 20),
        "url": "https://example.com/image",
        "content_type": "image/jpeg",
    }
    assert is_junk(img_doc) == "unsupported_content_type"

    # Rule 7: Body over 2MB
    huge_doc = {
        "text": ("Valid lengthy text content for the article. " * 20),
        "url": "https://example.com/huge",
        "bytes": 3 * 1024 * 1024,
    }
    assert is_junk(huge_doc) == "body_over_2mb"

    # Rule 8: Blocklisted domain
    spam_doc = {
        "text": ("Valid lengthy text content for the article. " * 20),
        "url": "https://contentfarm.com/article",
        "domain": "contentfarm.com",
    }
    assert is_junk(spam_doc) == "blocklisted_domain"

    # Clean document returns None
    clean_doc = {
        "text": (
            "Comprehensive technical documentation about vector database architectures, "
            "indexing strategies, query performance benchmarks, and storage considerations. " * 10
        ),
        "url": "https://qdrant.tech/documentation",
        "content_type": "text/html",
        "bytes": 5000,
        "domain": "qdrant.tech",
    }
    assert is_junk(clean_doc) is None


def test_dedupe_exact_and_near_duplicates() -> None:
    now = datetime.now(timezone.utc)
    base_text = "Vector database indexing and clustering algorithms for high-throughput AI workloads. " * 10

    # Two exact URL duplicates: one with lower trust, one with higher trust
    s1 = Source(
        id="s1",
        url="https://example.com/doc?utm_source=twitter",
        title="Doc 1 Low Trust",
        domain="example.com",
        retrieved_at=now,
        text=base_text,
        trust=0.5,
        tool="web_search",
        provider="tavily",
    )
    s2 = Source(
        id="s2",
        url="https://example.com/doc",
        title="Doc 1 High Trust",
        domain="example.com",
        retrieved_at=now,
        text=base_text,
        trust=1.0,
        tool="web_search",
        provider="tavily",
    )
    # Near duplicate with longer text
    s3 = Source(
        id="s3",
        url="https://another-domain.com/mirrored-doc",
        title="Mirrored Doc",
        domain="another-domain.com",
        retrieved_at=now,
        text=base_text + " Extra detailed implementation notes and benchmarks.",
        trust=0.8,
        tool="web_search",
        provider="tavily",
    )

    deduped = dedupe([s1, s2, s3])
    # Exact dedupe keeps s2 over s1. Near-dedupe between s2 and s3 merges and keeps s3 (longer text)
    assert len(deduped) == 1
    assert "Extra detailed implementation notes" in deduped[0].text


def test_build_sources_stable_contiguous_ids() -> None:
    raw_hits = [
        {
            "title": "Low Trust Blog",
            "url": "https://medium.com/blog/vector-db",
            "snippet": "Short blog overview of vector databases. " * 15,
            "published": "2026-01-01",
            "tool": "web_search",
            "provider": "tavily",
        },
        {
            "title": "Official Qdrant Docs",
            "url": "https://qdrant.tech/documentation/",
            "snippet": "Official documentation for Qdrant vector database. " * 15,
            "published": "2026-02-01",
            "tool": "web_search",
            "provider": "tavily",
        },
    ]

    fetched_pages = [
        {
            "url": "https://qdrant.tech/documentation/",
            "title": "Official Qdrant Docs",
            "text": "Full comprehensive guide to Qdrant vector search architecture. " * 30,
            "fetched_via": "jina",
        }
    ]

    sources = build_sources(raw_hits, fetched_pages)
    assert len(sources) == 2
    # s1 must be highest trust (qdrant.tech = 1.0)
    assert sources[0].id == "s1"
    assert sources[0].domain == "qdrant.tech"
    assert sources[0].trust == 1.0
    # s2 must be medium.com (0.5)
    assert sources[1].id == "s2"
    assert sources[1].domain == "medium.com"
    assert sources[1].trust == 0.5
