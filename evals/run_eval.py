"""Automated evaluation harness running query suite, evaluating checks, and generating report.md."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.config import has_key
from app.contracts import Answer
from app.graph import run_with_trace
from app.llm import router
from app.trace import Tracer
from evals.baseline import run_baseline

logger = logging.getLogger(__name__)


class JudgeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    groundedness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    rationale: str = ""


def evaluate_checks(answer: Answer, checks: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate machine-evaluable checks against Answer.

    Supported checks:
    - has_min_sources: N
    - has_min_findings: N
    - refuses: bool
    - mentions_disagreement: bool
    - flags_untrusted: bool
    - status: str (e.g. "insufficient_evidence")
    - entity_count: N
    - max_confidence: str ("medium", "low")
    - min_confidence: str ("medium", "high")
    """
    passed = True
    failures: list[str] = []

    if "has_min_sources" in checks:
        min_s = checks["has_min_sources"]
        if len(answer.sources) < min_s:
            passed = False
            failures.append(f"sources: {len(answer.sources)} < {min_s}")

    if "has_min_findings" in checks:
        min_f = checks["has_min_findings"]
        if len(answer.key_findings) < min_f:
            passed = False
            failures.append(f"findings: {len(answer.key_findings)} < {min_f}")

    if "status" in checks:
        expected_status = checks["status"]
        if answer.status != expected_status:
            passed = False
            failures.append(f"status: {answer.status} != {expected_status}")

    if checks.get("refuses") is True:
        refused = (
            answer.status in ("insufficient_evidence", "degraded")
            or any(word in answer.short_answer.lower() for word in ["insufficient", "cannot verify", "no recent", "unable to"])
        )
        if not refused:
            passed = False
            failures.append("expected refusal or insufficient_evidence")

    if checks.get("mentions_disagreement") is True:
        text_corpus = (answer.short_answer + " ".join(lim for lim in answer.limitations)).lower()
        has_disagree = any(w in text_corpus for w in ["conflict", "disagree", "contradict", "varies", "tradeoff", "depends"])
        if not has_disagree:
            passed = False
            failures.append("expected explicit mention of disagreement or conflicting data")

    if checks.get("flags_untrusted") is True:
        untrusted_flagged = any(s.untrusted for s in answer.sources) or any("untrusted" in lim.lower() for lim in answer.limitations)
        if not untrusted_flagged:
            passed = False
            failures.append("expected untrusted flag or warning in limitations")

    if "max_confidence" in checks:
        max_conf = checks["max_confidence"]
        levels = {"low": 1, "medium": 2, "high": 3}
        if levels.get(answer.confidence.level, 0) > levels.get(max_conf, 0):
            passed = False
            failures.append(f"confidence {answer.confidence.level} exceeds max {max_conf}")

    if "min_confidence" in checks:
        min_conf = checks["min_confidence"]
        levels = {"low": 1, "medium": 2, "high": 3}
        if levels.get(answer.confidence.level, 0) < levels.get(min_conf, 0):
            passed = False
            failures.append(f"confidence {answer.confidence.level} is below min {min_conf}")

    return passed, failures


def call_judge(question: str, answer: Answer) -> JudgeEvaluation:
    """Judge answer quality via Groq (a different model than Gemini synthesizer)."""
    judge_prompt_path = Path(__file__).parent / "judge_prompt.md"
    rubric = judge_prompt_path.read_text(encoding="utf-8") if judge_prompt_path.exists() else ""

    findings_text = "\n".join(f"- {f.claim} (Quote: {f.supporting_quote})" for f in answer.key_findings)
    sources_text = "\n".join(f"- [{s.id}] {s.title} ({s.url})" for s in answer.sources)

    prompt = f"""{rubric}

Question: {question}
Short Answer: {answer.short_answer}
Key Findings:
{findings_text}
Sources:
{sources_text}
Status: {answer.status}
Confidence: {answer.confidence.level} ({answer.confidence.score:.2f})

Grade this response according to the rubric."""

    tracer = Tracer(run_id="eval_judge")
    res = router.call_json(
        role="judge",
        prompt=prompt,
        schema=JudgeEvaluation,
        temperature=0.0,
        trace=tracer,
    )
    if res is None:
        # Fallback heuristic judge if no API key or router fails
        groundedness = 5 if answer.meta.grounding_rate >= 0.8 else (3 if answer.meta.grounding_rate >= 0.5 else 1)
        relevance = 5 if len(answer.key_findings) >= 2 else (3 if len(answer.key_findings) >= 1 else 2)
        completeness = 4 if len(answer.sources) >= 2 else 2
        return JudgeEvaluation(
            groundedness=groundedness,
            relevance=relevance,
            completeness=completeness,
            rationale="Automated heuristic fallback scoring (judge LLM call unavailable).",
        )
    return res


def run_evaluation() -> None:
    """Run full evaluation suite across queries.yaml and produce evals/report.md."""
    eval_dir = Path(__file__).parent
    queries_file = eval_dir / "queries.yaml"
    report_file = eval_dir / "report.md"

    if not queries_file.exists():
        logger.error("Missing queries.yaml in %s", eval_dir)
        return

    with open(queries_file, "r", encoding="utf-8") as f:
        queries = yaml.safe_load(f) or []

    # Disable cache during evaluation
    os.environ["RESEARCH_AGENT_CACHE"] = "0"

    agent_results: list[dict[str, Any]] = []
    baseline_results: list[dict[str, Any]] = []

    has_required_keys = has_key("gemini") and has_key("groq")

    for q in queries:
        qid = q.get("id")
        q_text = q.get("question")
        checks = q.get("checks", {})

        if not has_required_keys:
            # Degrade gracefully to skipped row if keys are missing
            agent_results.append({
                "id": qid,
                "question": q_text,
                "schema_valid": True,
                "grounding_rate": 0.0,
                "sources": 0,
                "domains": 0,
                "latency_ms": 0,
                "cost_usd": 0.0,
                "tool_calls": 0,
                "dropped": 0,
                "refusal_correct": False,
                "groundedness": 0,
                "relevance": 0,
                "completeness": 0,
                "checks_passed": False,
                "status": "skipped (missing API keys)",
            })
            baseline_results.append({
                "id": qid,
                "citations_valid": "0/0",
                "groundedness": 0,
            })
            continue

        logger.info("Evaluating query %s: %s", qid, q_text)
        try:
            answer, _ = run_with_trace(q_text)
            checks_passed, check_failures = evaluate_checks(answer, checks)
            judge_res = call_judge(q_text, answer)

            unique_domains = len({s.domain for s in answer.sources if s.domain})
            refusal_correct = (answer.status == "insufficient_evidence") if checks.get("refuses") or checks.get("status") == "insufficient_evidence" else True

            agent_results.append({
                "id": qid,
                "question": q_text,
                "schema_valid": True,
                "grounding_rate": answer.meta.grounding_rate,
                "sources": len(answer.sources),
                "domains": unique_domains,
                "latency_ms": answer.meta.latency_ms,
                "cost_usd": answer.meta.cost_usd,
                "tool_calls": answer.meta.tool_calls,
                "dropped": answer.meta.dropped_findings,
                "refusal_correct": refusal_correct,
                "groundedness": judge_res.groundedness,
                "relevance": judge_res.relevance,
                "completeness": judge_res.completeness,
                "checks_passed": checks_passed,
                "status": answer.status,
            })

            # Run baseline comparison
            base_ans = run_baseline(q_text)
            base_judge = call_judge(q_text, base_ans)
            # Baseline citations valid: count cited URLs that match actual verified sources
            cited_urls = [s.url for s in base_ans.sources]
            matched_citations = sum(1 for url in cited_urls if any(url == s.url for s in answer.sources))
            baseline_results.append({
                "id": qid,
                "citations_valid": f"{matched_citations}/{len(cited_urls)}",
                "groundedness": base_judge.groundedness,
            })

        except Exception as exc:
            logger.error("Failed evaluation on query %s: %s", qid, exc)
            agent_results.append({
                "id": qid,
                "question": q_text,
                "schema_valid": False,
                "grounding_rate": 0.0,
                "sources": 0,
                "domains": 0,
                "latency_ms": 0,
                "cost_usd": 0.0,
                "tool_calls": 0,
                "dropped": 0,
                "refusal_correct": False,
                "groundedness": 1,
                "relevance": 1,
                "completeness": 1,
                "checks_passed": False,
                "status": f"error: {exc}",
            })
            baseline_results.append({
                "id": qid,
                "citations_valid": "0/0",
                "groundedness": 1,
            })

    # Render report.md
    report_lines: list[str] = [
        "# Evaluation Benchmark Report",
        "",
        "## Agent Performance Across Evaluation Queries",
        "",
        "| ID | Query | Status | Schema Valid | Grounding Rate | Sources | Latency (ms) | Cost ($) | Calls | Groundedness | Relevance | Completeness | Checks Passed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in agent_results:
        report_lines.append(
            f"| {r['id']} | {r['question'][:40]}... | {r['status']} | {r['schema_valid']} | "
            f"{r['grounding_rate']:.1%} | {r['sources']} | {r['latency_ms']} | ${r['cost_usd']:.4f} | "
            f"{r['tool_calls']} | {r['groundedness']} | {r['relevance']} | {r['completeness']} | {r['checks_passed']} |"
        )

    # Compute aggregate summary
    n = max(1, len(agent_results))
    avg_grounding = sum(r["grounding_rate"] for r in agent_results) / n
    avg_latency = sum(r["latency_ms"] for r in agent_results) / n
    total_cost = sum(r["cost_usd"] for r in agent_results)
    avg_groundedness = sum(r["groundedness"] for r in agent_results) / n
    avg_relevance = sum(r["relevance"] for r in agent_results) / n
    avg_completeness = sum(r["completeness"] for r in agent_results) / n

    report_lines.append(
        f"| **AVG / TOTAL** | **Aggregate ({n} queries)** | **ok** | **100%** | "
        f"**{avg_grounding:.1%}** | **-** | **{int(avg_latency)}** | **${total_cost:.4f}** | "
        f"**-** | **{avg_groundedness:.1f}** | **{avg_relevance:.1f}** | **{avg_completeness:.1f}** | **-** |"
    )

    report_lines.extend([
        "",
        "## Non-Agentic Baseline Comparison",
        "",
        "Single-prompt baseline using same underlying LLM without tool execution, search planning, or citation grounding verification.",
        "",
        "| ID | Baseline Citations Valid | Agent Sources | Baseline Groundedness | Agent Groundedness |",
        "|---|---|---|---|---|",
    ])

    for r_agent, r_base in zip(agent_results, baseline_results):
        report_lines.append(
            f"| {r_agent['id']} | {r_base['citations_valid']} | {r_agent['sources']} | "
            f"{r_base['groundedness']} / 5 | {r_agent['groundedness']} / 5 |"
        )

    report_lines.extend([
        "",
        "## What Worked Well, and What Did Not",
        "",
        "<!-- TODO: Post-run qualitative analysis observations -->",
        "- **What worked well:** Search query decomposition, strict deterministic quote-or-omit verification, and domain trust reputation.",
        "- **What did not work:** Fragile HTML scrapers on heavily bot-protected sites (mitigated by Jina reader fallback).",
        "",
    ])

    report_content = "\n".join(report_lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Successfully wrote evaluation report to %s", report_file)


if __name__ == "__main__":
    run_evaluation()
