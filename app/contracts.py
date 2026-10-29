"""Single source of truth for every data shape that crosses a node boundary or reaches a user."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_serializer

QueryType = Literal["comparison", "list", "explain", "howto"]
Status = Literal["ok", "degraded", "insufficient_evidence"]
ConfidenceLevel = Literal["low", "medium", "high"]


def _format_datetime_z(dt: datetime) -> str:
    """Format datetime as ISO-8601 UTC with Z suffix."""
    if dt.tzinfo is None:
        return dt.isoformat() + "Z"
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- node 1 ----------
class Intake(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_type: QueryType
    entities: list[str] = Field(default_factory=list)  # e.g. ["vector databases"]
    recency_sensitive: bool = False
    k_sources: int = 6
    max_steps: int = 6


# ---------- node 2 ----------
class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_type: QueryType
    sub_questions: list[str] = Field(min_length=1, max_length=5)
    search_queries: list[str] = Field(min_length=1, max_length=6)
    tools: list[str] = Field(min_length=1, max_length=4)  # validated against registry
    stop_criteria: str


# ---------- node 3 ----------
class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str  # s1, s2 …
    url: str
    title: str
    domain: str
    published: date | None = None
    retrieved_at: datetime
    text: str  # trimmed to MAX_SOURCE_CHARS (8000)
    trust: float = Field(ge=0.0, le=1.0)
    tool: str  # web_search | wikipedia | arxiv | github
    provider: str  # tavily | exa | ddg | jina | httpx …
    untrusted: bool = False  # injection pattern detected

    @field_serializer("retrieved_at", when_used="json")
    def serialize_retrieved_at(self, dt: datetime) -> str:
        return _format_datetime_z(dt)


class ToolCallLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    tool: str
    provider: str
    args: dict
    attempt: int
    latency_ms: int
    status: Literal["ok", "empty", "error", "skipped", "cache_hit"]
    items: int = 0
    error: str | None = None


# ---------- node 5 ----------
class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=20)
    source_ids: list[str] = Field(min_length=1)
    supporting_quote: str = Field(min_length=40)
    corroboration: int = 1


class SynthesisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_answer: str = Field(max_length=1200)
    findings: list[Finding] = Field(min_length=1, max_length=8)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


# ---------- node 6 / 7 ----------
class Confidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: ConfidenceLevel
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str]


class SourceRef(BaseModel):
    """Public view of a source — no raw text in the answer body."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    url: str
    domain: str
    published: date | None = None
    retrieved_at: datetime
    trust: float
    untrusted: bool = False

    @field_serializer("retrieved_at", when_used="json")
    def serialize_retrieved_at(self, dt: datetime) -> str:
        return _format_datetime_z(dt)


class RunMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: ResearchPlan | None = None
    tool_calls: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    grounding_rate: float = 0.0
    dropped_findings: int = 0
    cache_hits: int = 0
    provider_failures: dict[str, int] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    llm_calls: int = 0


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    question: str
    short_answer: str
    key_findings: list[Finding]
    sources: list[SourceRef]
    confidence: Confidence
    assumptions: list[str]
    limitations: list[str]
    next_steps: list[str]
    status: Status
    meta: RunMeta
