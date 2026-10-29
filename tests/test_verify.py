"""Unit tests for deterministic quote presence and citation verification."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from app.contracts import Finding, Source, SynthesisDraft
from app.verify import normalise, quote_present, verify_findings


def test_quote_present_exact_and_normalised() -> None:
    src = "Qdrant is distributed as a single binary and can run without any external dependencies."

    # Exact match
    q1 = "distributed as a single binary and can run without any external dependencies."
    assert quote_present(q1, src) is True

    # Shorter than 20 chars must be rejected
    q_short = "distributed"
    assert quote_present(q_short, src) is False

    # Normalised differences (case, excessive whitespace, markdown bold)
    q_norm = "**QDRANT**   is  distributed as   a single binary and can run without any external dependencies."
    assert quote_present(q_norm, src) is True


def test_quote_present_rejects_paraphrase() -> None:
    src = "Qdrant is distributed as a single binary and can run without any external dependencies."
    # Paraphrased version with same meaning but different words
    paraphrase = "Qdrant only needs a single binary file to execute properly and has zero other external dependencies."
    assert quote_present(paraphrase, src) is False


def test_quote_present_handles_unicode_and_smart_quotes() -> None:
    src = 'The database uses "HNSW" algorithm for fast hierarchical search indexing.'
    # Quote with curly smart quotes and zero-width spaces
    smart_quote = "The database uses \u200B“HNSW” algorithm for fast hierarchical search indexing."
    assert quote_present(smart_quote, src) is True


def test_verify_findings_drops_unknown_source_id() -> None:
    now = datetime.now(timezone.utc)
    s1 = Source(
        id="s1",
        url="https://qdrant.tech",
        title="Qdrant",
        text="Qdrant is a vector similarity search engine and vector database written in Rust." * 2,
        domain="qdrant.tech",
        retrieved_at=now,
        trust=1.0,
        tool="web_search",
        provider="tavily",
    )
    draft = SynthesisDraft(
        short_answer="Answer",
        findings=[
            Finding(
                claim="This claim cites a non-existent source id in the pipeline.",
                source_ids=["s99"],
                supporting_quote="Qdrant is a vector similarity search engine and vector database written in Rust.",
            )
        ],
    )

    kept, dropped = verify_findings(draft, [s1])
    assert len(kept) == 0
    assert len(dropped) == 1
    assert "unknown_source_id" in dropped[0]["reason"]


def test_verify_findings_drops_ungrounded_quote() -> None:
    now = datetime.now(timezone.utc)
    s1 = Source(
        id="s1",
        url="https://qdrant.tech",
        title="Qdrant",
        text="Qdrant is a vector similarity search engine and vector database written in Rust." * 2,
        domain="qdrant.tech",
        retrieved_at=now,
        trust=1.0,
        tool="web_search",
        provider="tavily",
    )
    draft = SynthesisDraft(
        short_answer="Answer",
        findings=[
            Finding(
                claim="This claim invents a quote not found anywhere in source text.",
                source_ids=["s1"],
                supporting_quote="Milvus uses a distributed architecture with separate query coordinators and workers.",
            )
        ],
    )

    kept, dropped = verify_findings(draft, [s1])
    assert len(kept) == 0
    assert len(dropped) == 1
    assert "quote_not_grounded" in dropped[0]["reason"]


def test_verify_findings_drops_short_claim() -> None:
    now = datetime.now(timezone.utc)
    s1 = Source(
        id="s1",
        url="https://qdrant.tech",
        title="Qdrant",
        text="Qdrant is a vector similarity search engine and vector database written in Rust." * 2,
        domain="qdrant.tech",
        retrieved_at=now,
        trust=1.0,
        tool="web_search",
        provider="tavily",
    )
    # Finding model validates min_length=20, so construct with valid finding then mutate or mock
    finding = Finding(
        claim="Claim is long enough initially for model validation.",
        source_ids=["s1"],
        supporting_quote="Qdrant is a vector similarity search engine and vector database written in Rust.",
    )
    # Artificially shorten claim to simulate an edge-case or unvalidated data
    object.__setattr__(finding, "claim", "Short claim.")
    draft = SynthesisDraft(
        short_answer="Answer",
        findings=[finding],
    )

    kept, dropped = verify_findings(draft, [s1])
    assert len(kept) == 0
    assert len(dropped) == 1
    assert "claim_under_20_chars" in dropped[0]["reason"]


def test_verify_findings_recomputes_corroboration_from_distinct_domains() -> None:
    now = datetime.now(timezone.utc)
    s1 = Source(
        id="s1",
        url="https://docs.qdrant.tech/page1",
        title="Qdrant Docs",
        text="Qdrant provides payload filtering alongside vector search operations." * 2,
        domain="qdrant.tech",
        retrieved_at=now,
        trust=1.0,
        tool="web_search",
        provider="tavily",
    )
    s2 = Source(
        id="s2",
        url="https://blog.qdrant.tech/page2",
        title="Qdrant Blog",
        text="Qdrant provides payload filtering alongside vector search operations." * 2,
        domain="qdrant.tech",
        retrieved_at=now,
        trust=0.8,
        tool="web_search",
        provider="tavily",
    )
    s3 = Source(
        id="s3",
        url="https://weaviate.io/page",
        title="Weaviate",
        text="Weaviate comparison notes that Qdrant provides payload filtering alongside vector search operations.",
        domain="weaviate.io",
        retrieved_at=now,
        trust=1.0,
        tool="web_search",
        provider="tavily",
    )

    # Finding 1 cites s1 and s2 (same domain: qdrant.tech) -> corroboration = 1
    # Finding 2 cites s1 and s3 (two distinct domains: qdrant.tech, weaviate.io) -> corroboration = 2
    draft = SynthesisDraft(
        short_answer="Answer",
        findings=[
            Finding(
                claim="Qdrant provides payload filtering directly in its vector search engine.",
                source_ids=["s1", "s2"],
                supporting_quote="Qdrant provides payload filtering alongside vector search operations.",
                corroboration=5,  # Deliberately bogus; should be overwritten
            ),
            Finding(
                claim="Both Qdrant and Weaviate benchmark comparison confirm payload filtering capabilities.",
                source_ids=["s1", "s3"],
                supporting_quote="Qdrant provides payload filtering alongside vector search operations.",
                corroboration=1,
            ),
        ],
    )

    kept, dropped = verify_findings(draft, [s1, s2, s3])
    assert len(kept) == 2
    assert len(dropped) == 0
    assert kept[0].corroboration == 1
    assert kept[1].corroboration == 2
