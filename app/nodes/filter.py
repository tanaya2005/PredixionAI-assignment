"""Node 4 (Filter): Performs deduplication, junk filtering, and batched relevance scoring."""

from __future__ import annotations

from collections import Counter
import copy
import logging
import time

from app.config import BUDGETS
from app.contracts import Source
from app.evidence import dedupe, is_junk, rank_candidates
from app.graph import State
from app.relevance import score as score_relevance
from app.trace import Tracer

logger = logging.getLogger(__name__)


def filter_evidence(state: State) -> State:
    """Filter raw sources via exact/simhash deduplication, junk checks, and relevance scoring. Never raises."""
    tracer = Tracer(run_id=state.run_id, trace_dir=state.trace_path.parent)
    start_time = time.perf_counter()

    raw_candidates = list(state.sources)
    candidates_in = len(raw_candidates)

    # 1. Exact & SimHash deduplication
    deduped_sources = dedupe(raw_candidates)
    deduped_dropped_count = candidates_in - len(deduped_sources)

    # 2. Junk filtering
    surviving_junk: list[Source] = []
    junk_reasons: Counter[str] = Counter()
    for s in deduped_sources:
        reason = is_junk(s)
        if reason:
            junk_reasons[reason] += 1
        else:
            surviving_junk.append(s)

    junk_dropped_count = sum(junk_reasons.values())

    # 3. Sort by trust descending, then text length descending; cap at 12 candidates
    ranked = rank_candidates(surviving_junk)
    candidates_for_scoring = ranked[:12]

    # Assign temporary contiguous IDs s1..sN for scoring if IDs are not unique or missing
    candidates_for_scoring = [
        copy.deepcopy(s) for s in candidates_for_scoring
    ]
    for idx, s in enumerate(candidates_for_scoring, start=1):
        s.id = f"s{idx}"

    # Target number of kept sources: from intake.k_sources or BUDGETS.k_sources (default 6)
    k_target = state.intake.k_sources if state.intake and state.intake.k_sources else BUDGETS.k_sources

    sub_questions = state.plan.sub_questions if state.plan else []

    # 4. Relevance scoring (batched LLM call)
    scores: dict[str, float] = {}
    scorer_failed = False
    try:
        scores = score_relevance(
            sources=candidates_for_scoring,
            question=state.question,
            sub_questions=sub_questions,
            tracer=tracer,
        )
        if not scores:
            scorer_failed = True
    except Exception as exc:
        logger.warning("Relevance scoring raised exception: %s", exc)
        scorer_failed = True

    # 5. Filter score < 0.3, keep top k_target
    below_threshold_count = 0
    kept: list[Source] = []

    if scorer_failed:
        # Fall back to trust ordering directly
        kept = candidates_for_scoring[:k_target]
    else:
        # We have scores
        scored_candidates: list[tuple[float, Source]] = []
        for s in candidates_for_scoring:
            s_score = scores.get(s.id, 0.5)
            if s_score < 0.3:
                below_threshold_count += 1
            else:
                scored_candidates.append((s_score, s))

        # Sort by score desc, then trust desc, then text length desc
        scored_candidates.sort(key=lambda item: (item[0], item[1].trust, len(item[1].text)), reverse=True)
        kept = [s for _, s in scored_candidates[:k_target]]

        # Guard: always keep at least 2 sources if >= 2 survived junk filtering
        # (e.g. if a bad scorer returned all zeros or all below 0.3)
        if len(kept) < 2 and len(candidates_for_scoring) >= 2 and len(scored_candidates) == 0:
            kept = candidates_for_scoring[: min(k_target, 2)]

    # 6. Reassign contiguous IDs s1..sN to kept sources
    final_kept: list[Source] = []
    for idx, s in enumerate(kept, start=1):
        s_copy = copy.deepcopy(s)
        s_copy.id = f"s{idx}"
        final_kept.append(s_copy)

    state.kept = final_kept

    latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))

    # Record trace event
    tracer.event(
        node="filter",
        latency_ms=latency_ms,
        status="error" if scorer_failed else "ok",
        candidates_in=candidates_in,
        deduped=deduped_dropped_count,
        junk_dropped=junk_dropped_count,
        junk_reasons=dict(junk_reasons),
        below_threshold=below_threshold_count,
        kept=len(final_kept),
    )

    return state
