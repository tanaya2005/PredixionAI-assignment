"""Batched LLM relevance scoring of candidate evidence against research questions."""

from __future__ import annotations

import logging
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.contracts import Source
from app.llm import router
from app.prompts import RELEVANCE
from app.trace import Tracer

logger = logging.getLogger(__name__)


class RelevanceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class RelevanceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RelevanceItem]


def score(
    sources: list[Source],
    question: str,
    sub_questions: list[str],
    tracer: Tracer,
) -> dict[str, float]:
    """Score all candidate documents in ONE batched LLM call against the research question."""
    if not sources:
        return {}

    docs_text: list[str] = []
    for s in sources:
        snippet = s.text[:500].replace("\n", " ").strip()
        docs_text.append(f"ID: {s.id}\nTitle: {s.title}\nDomain: {s.domain}\nSnippet: {snippet}\n")
    docs_formatted = "\n".join(docs_text)

    sub_q_text = "\n".join(f"- {sq}" for sq in sub_questions) if sub_questions else "None specified"
    prompt = RELEVANCE.format(
        question=question,
        sub_questions=sub_q_text,
        documents=docs_formatted,
    )

    result = router.call_json(
        role="filter",
        prompt=prompt,
        schema=RelevanceOutput,
        temperature=0.0,
        trace=tracer,
    )

    if result is None:
        logger.warning("Relevance scoring failed or returned None; returning empty score dict")
        return {}

    scores: dict[str, float] = {}
    for item in result.items:
        scores[item.id] = float(item.score)

    # Fill any missing IDs with default 0.5 and log warning
    for s in sources:
        if s.id not in scores:
            logger.warning("Relevance scorer omitted ID '%s'; defaulting score to 0.5", s.id)
            scores[s.id] = 0.5

    return scores
