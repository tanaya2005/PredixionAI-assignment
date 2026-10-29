"""Unit tests for deterministic confidence calculation and threshold overrides."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from app.contracts import Finding, Source
from app.verify import compute_confidence


@pytest.fixture
def sample_sources() -> list[Source]:
    now = datetime.now(timezone.utc)
    return [
        Source(
            id=f"s{i}",
            url=f"https://domain{i}.com/path",
            title=f"Source {i}",
            text="Evidence body text here...",
            domain=f"domain{i}.com",
            retrieved_at=now,
            trust=0.9,
            tool="web_search",
            provider="tavily",
        )
        for i in range(1, 5)
    ]


def test_confidence_monotonicity(sample_sources: list[Source]) -> None:
    # 2 findings from 2 distinct domains
    findings_base = [
        Finding(
            claim="Claim number one with sufficient characters.",
            source_ids=["s1"],
            supporting_quote="Evidence body text here with enough characters to pass validation.",
            corroboration=1,
        ),
        Finding(
            claim="Claim number two with sufficient characters.",
            source_ids=["s2"],
            supporting_quote="Evidence body text here with enough characters to pass validation.",
            corroboration=1,
        ),
    ]

    conf_base = compute_confidence(findings_base, sample_sources, grounding_rate=1.0)

    # Adding a 3rd and 4th distinct domain should increase confidence
    findings_more_domains = findings_base + [
        Finding(
            claim="Claim number three with sufficient characters.",
            source_ids=["s3", "s4"],
            supporting_quote="Evidence body text here with enough characters to pass validation.",
            corroboration=2,
        ),
    ]
    conf_higher = compute_confidence(findings_more_domains, sample_sources, grounding_rate=1.0)
    assert conf_higher.score > conf_base.score

    # Lower grounding rate should decrease confidence
    conf_lower_grounding = compute_confidence(findings_more_domains, sample_sources, grounding_rate=0.5)
    assert conf_lower_grounding.score < conf_higher.score


def test_confidence_hard_override_single_domain(sample_sources: list[Source]) -> None:
    # 3 findings all from s1 (single domain) with 100% grounding
    findings = [
        Finding(
            claim=f"Claim number {i} with more than twenty characters for validation.",
            source_ids=["s1"],
            supporting_quote="Evidence body text here with enough characters to pass validation.",
            corroboration=1,
        )
        for i in range(1, 4)
    ]
    conf = compute_confidence(findings, sample_sources, grounding_rate=1.0)

    # Score must be capped at 0.60
    assert conf.score <= 0.60
    assert any("Single domain across all findings" in r for r in conf.reasons)


def test_confidence_hard_override_single_finding(sample_sources: list[Source]) -> None:
    # Exactly 1 finding even with multi-domain corroboration and high trust
    findings = [
        Finding(
            claim="A single solitary claim that has more than twenty characters.",
            source_ids=["s1", "s2"],
            supporting_quote="Evidence body text here with enough characters to pass validation.",
            corroboration=2,
        )
    ]
    conf = compute_confidence(findings, sample_sources, grounding_rate=1.0)

    # Must be capped at "low" (score < 0.50)
    assert conf.level == "low"
    assert conf.score < 0.50
    assert any("Single finding caps confidence level at low" in r for r in conf.reasons)


def test_confidence_zero_findings_returns_zero(sample_sources: list[Source]) -> None:
    conf = compute_confidence([], sample_sources, grounding_rate=0.0)
    assert conf.level == "low"
    assert conf.score == 0.0
    assert "0 findings survived verification" in conf.reasons[0]
