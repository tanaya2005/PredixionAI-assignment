"""Deterministic quote verification, confidence scoring, and citation grounding without an LLM."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from app.contracts import Confidence, Finding, Source, SynthesisDraft

logger = logging.getLogger(__name__)

# Zero-width characters regex
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
# Whitespace collapsing regex
_WHITESPACE_RE = re.compile(r"\s+")
# Markdown emphasis (*, _, `)
_MARKDOWN_EMPHASIS_RE = re.compile(r"[\*_`]")
# Smart quotes mapping
_SMART_QUOTES = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
    "‹": "'",
    "›": "'",
})


def normalise(s: str) -> str:
    """Normalize text: NFKC, strip zero-width chars, collapse whitespace, lowercase, strip markdown and smart quotes."""
    if not s:
        return ""
    # 1. NFKC normalization
    normalized = unicodedata.normalize("NFKC", s)
    # 2. Smart quotes to straight quotes
    normalized = normalized.translate(_SMART_QUOTES)
    # 3. Strip zero-width chars
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    # 4. Strip markdown emphasis (*, _, `)
    normalized = _MARKDOWN_EMPHASIS_RE.sub("", normalized)
    # 5. Lowercase
    normalized = normalized.lower()
    # 6. Collapse whitespace and strip ends
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def quote_present(quote: str, text: str) -> bool:
    """Check if quote is present in text.

    Requires quote length >= 20 chars (or 40 for findings). Checks exact normalized substring match,
    or tolerates sentence-spanning quotes split by fetcher using fragment matching.
    """
    if len(quote) < 20:
        return False

    q = normalise(quote)
    t = normalise(text)

    # 1. Exact substring after normalization
    if q in t:
        return True

    # 2. Fragment tolerance rule from PROJECT_DETAILS.md §6 node 6:
    # Tolerate sentence-spanning quotes split by the fetcher: try long fragments
    words = [w for w in q.split(" ") if w]
    frags = [f for f in words if len(f) > 6]
    if len(frags) >= 5 and all(f in t for f in frags[:5]):
        # Additional check: ensure the words in frags[:5] appear in relative order in t
        pos = 0
        ordered = True
        for f in frags[:5]:
            idx = t.find(f, pos)
            if idx == -1:
                ordered = False
                break
            pos = idx + len(f)
        if ordered:
            return True

    return False


def verify_findings(
    draft: SynthesisDraft | None,
    sources: list[Source],
) -> tuple[list[Finding], list[dict[str, str]]]:
    """Verify each finding against cited sources deterministically.

    Drops on:
    - unknown source_id
    - quote not present in ANY cited source
    - claim under 20 chars

    Recomputes corroboration as the count of DISTINCT domains among cited sources.
    """
    if not draft or not draft.findings:
        return [], []

    source_map: dict[str, Source] = {s.id: s for s in sources}
    kept: list[Finding] = []
    dropped: list[dict[str, str]] = []

    for finding in draft.findings:
        # Check claim length
        if len(finding.claim.strip()) < 20:
            dropped.append({"claim": finding.claim, "reason": "claim_under_20_chars"})
            continue

        # Check unknown source_ids
        unknown_ids = [sid for sid in finding.source_ids if sid not in source_map]
        if unknown_ids:
            dropped.append({
                "claim": finding.claim,
                "reason": f"unknown_source_id: {', '.join(unknown_ids)}",
            })
            continue

        # Check if supporting_quote is present in ANY of the cited sources
        cited_sources = [source_map[sid] for sid in finding.source_ids]
        quote_found = any(quote_present(finding.supporting_quote, s.text) for s in cited_sources)

        if not quote_found:
            dropped.append({
                "claim": finding.claim,
                "reason": "quote_not_grounded_in_sources",
            })
            continue

        # Recompute corroboration as the count of distinct domains among cited sources
        distinct_domains = {s.domain.lower() for s in cited_sources if s.domain}
        corroboration = max(1, len(distinct_domains))

        verified_finding = Finding(
            claim=finding.claim,
            source_ids=finding.source_ids,
            supporting_quote=finding.supporting_quote,
            corroboration=corroboration,
        )
        kept.append(verified_finding)

    return kept, dropped


def compute_confidence(
    kept_findings: list[Finding],
    sources: list[Source],
    grounding_rate: float,
) -> Confidence:
    """Compute confidence using the exact documented weighted formula and hard overrides.

    Formula:
    0.35 * grounding_rate + 0.25 * min(domains, 4) / 4 + 0.20 * (corroborated / total) + 0.20 * mean(trust)

    Hard overrides:
    1. Single domain across all findings -> cap score at 0.60 ("medium" at best, or lower)
    2. Single finding -> cap level at "low" (score capped at 0.49)
    """
    reasons: list[str] = []

    if not kept_findings:
        return Confidence(
            level="low",
            score=0.0,
            reasons=["0 findings survived verification"],
        )

    # 1. Grounding rate term (weight 0.35)
    g_rate = max(0.0, min(1.0, grounding_rate))
    reasons.append(f"Grounding rate: {g_rate:.1%} ({len(kept_findings)} findings kept)")

    # 2. Distinct domains across cited sources for kept findings (weight 0.25)
    source_map = {s.id: s for s in sources}
    cited_domains: set[str] = set()
    for f in kept_findings:
        for sid in f.source_ids:
            if sid in source_map and source_map[sid].domain:
                clean_d = source_map[sid].domain.lower().strip()
                if clean_d.startswith("www."):
                    clean_d = clean_d[4:]
                cited_domains.add(clean_d)

    domain_count = len(cited_domains)
    domain_term = min(domain_count, 4) / 4.0
    reasons.append(f"Distinct cited domains: {domain_count} ({domain_term:.2f} domain diversity)")

    # 3. Corroborated findings ratio (weight 0.20)
    # A finding is corroborated if corroboration >= 2 (multiple distinct domains)
    total_findings = len(kept_findings)
    corroborated_count = sum(1 for f in kept_findings if f.corroboration >= 2)
    corroborated_ratio = corroborated_count / total_findings if total_findings > 0 else 0.0
    reasons.append(f"Corroborated findings: {corroborated_count}/{total_findings} ({corroborated_ratio:.1%})")

    # 4. Mean trust of cited sources (weight 0.20)
    cited_source_ids = {sid for f in kept_findings for sid in f.source_ids if sid in source_map}
    if cited_source_ids:
        mean_trust = sum(source_map[sid].trust for sid in cited_source_ids) / len(cited_source_ids)
    else:
        mean_trust = 0.5
    mean_trust = max(0.0, min(1.0, mean_trust))
    reasons.append(f"Mean source trust: {mean_trust:.2f}")

    # Weighted calculation
    score = (
        0.35 * g_rate
        + 0.25 * domain_term
        + 0.20 * corroborated_ratio
        + 0.20 * mean_trust
    )
    score = round(max(0.0, min(1.0, score)), 4)

    # Determine level from score before overrides (thresholds 0.75 / 0.50)
    if score >= 0.75:
        level = "high"
    elif score >= 0.50:
        level = "medium"
    else:
        level = "low"

    # Hard override 1: Single domain across all findings -> cap score at 0.60
    if domain_count <= 1:
        reasons.append("Hard override: Single domain across all findings caps score at 0.60")
        if score > 0.60:
            score = 0.60
            level = "medium"

    # Hard override 2: Single finding -> cap level at "low"
    if total_findings == 1:
        reasons.append("Hard override: Single finding caps confidence level at low")
        level = "low"
        if score >= 0.50:
            score = 0.49

    return Confidence(
        level=level,  # type: ignore[arg-type]
        score=score,
        reasons=reasons,
    )
