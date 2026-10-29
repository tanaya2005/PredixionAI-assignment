"""Command-line interface providing interactive execution, formatted output, and trace inspection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from app.cache import cache
from app.contracts import Answer
from app.graph import run_with_trace
from app.trace import render_log_line

console = Console()
err_console = Console(stderr=True)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run the 7-node agentic research pipeline.",
        epilog="Exit codes: 0 = ok/degraded, 2 = insufficient_evidence, 1 = validation error.",
    )
    parser.add_argument("question", help="The research question to investigate.")
    parser.add_argument("--json", dest="json_out", help="Path to write the full JSON answer output.")
    parser.add_argument("--trace", action="store_true", help="Print detailed real-time execution trace lines.")
    parser.add_argument("--no-cache", action="store_true", help="Disable tool call caching.")
    parser.add_argument("--quiet", action="store_true", help="Suppress trace output, showing only final answer.")

    args = parser.parse_args()
    raw_question = args.question.strip()

    # Validation: question length
    if len(raw_question) < 5:
        err_console.print(f"[bold red]Validation error:[/bold red] Question is too short ({len(raw_question)} chars). Minimum length is 5 characters.")
        sys.exit(1)

    if args.no_cache:
        os.environ["RESEARCH_AGENT_CACHE"] = "0"

    # Run the pipeline
    answer: Answer
    events: list[dict[str, Any]]
    answer, events = run_with_trace(raw_question)

    if not args.quiet:
        # 1. Print the plan
        if answer.meta.plan:
            plan = answer.meta.plan
            plan_lines = [
                f"[bold cyan]Query Type:[/bold cyan] {plan.query_type}",
                f"[bold cyan]Sub-questions:[/bold cyan]",
                *[f"  - {sq}" for sq in plan.sub_questions],
                f"[bold cyan]Search Queries:[/bold cyan]",
                *[f"  - {sq}" for sq in plan.search_queries],
                f"[bold cyan]Tools:[/bold cyan] {', '.join(plan.tools)}",
                f"[bold cyan]Stop Criteria:[/bold cyan] {plan.stop_criteria}",
            ]
            console.print(Panel("\n".join(plan_lines), title="Research Plan", border_style="cyan"))

        # 2. Print tool lines and node lines from events
        if args.trace:
            console.print("[bold yellow]Execution Trace:[/bold yellow]")
            for e in events:
                node = e.get("node", "pipeline")
                ms = e.get("latency_ms", 0)
                tool = e.get("tool")
                provider = e.get("provider")
                attempt = e.get("attempt", 1)
                status = e.get("status", "ok")
                items = e.get("items", 0)
                if tool:
                    note = f"{tool} ({provider}) attempt={attempt} items={items} [{status}]"
                else:
                    note = e.get("note")
                console.print(render_log_line(node, ms, note))

        # 3. Print filter counters
        filter_events = [e for e in events if e.get("node") == "filter"]
        if filter_events:
            fe = filter_events[-1]
            cin = fe.get("candidates_in", 0)
            deduped = fe.get("deduped", 0)
            junk = fe.get("junk_dropped", 0)
            below = fe.get("below_threshold", 0)
            kept = fe.get("kept", 0)
            console.print(f"[bold]Filter:[/bold] candidates={cin} deduped={deduped} junk_dropped={junk} below_threshold={below} kept={kept}")

        # 4. Print verification result
        verify_events = [e for e in events if e.get("node") == "verify"]
        if verify_events:
            ve = verify_events[-1]
            kept_f = ve.get("kept_findings", 0)
            init_f = ve.get("initial_findings", 0)
            g_rate = ve.get("grounding_rate", 0.0)
            conf_level = ve.get("confidence_level", answer.confidence.level)
            conf_score = ve.get("confidence_score", answer.confidence.score)
            console.print(f"[bold]Verification:[/bold] {kept_f}/{init_f} grounded ({g_rate:.1%}) | confidence={conf_level} ({conf_score:.2f})")

        console.print()

    # 5. Print formatted markdown answer
    md_lines: list[str] = [
        f"## Short Answer\n\n{answer.short_answer}\n",
    ]

    if answer.key_findings:
        md_lines.append("### Key Findings\n")
        for idx, f in enumerate(answer.key_findings, start=1):
            source_tags = " ".join(f"[{sid}]" for sid in f.source_ids)
            md_lines.append(f"{idx}. **{f.claim}** {source_tags}")
            md_lines.append(f"   > \"{f.supporting_quote}\"\n")

    if answer.sources:
        md_lines.append("### Sources\n")
        for s in answer.sources:
            md_lines.append(f"- **[{s.id}]** [{s.title}]({s.url}) (trust: {s.trust:.2f}, domain: {s.domain})")
        md_lines.append("")

    md_lines.append(f"### Confidence: {answer.confidence.level.upper()} ({answer.confidence.score:.2f})\n")
    for r in answer.confidence.reasons:
        md_lines.append(f"- {r}")
    md_lines.append("")

    if answer.assumptions:
        md_lines.append("### Working Assumptions\n")
        for a in answer.assumptions:
            md_lines.append(f"- {a}")
        md_lines.append("")

    if answer.limitations:
        md_lines.append("### Limitations\n")
        for lim in answer.limitations:
            md_lines.append(f"- {lim}")
        md_lines.append("")

    if answer.next_steps:
        md_lines.append("### Next Steps\n")
        for ns in answer.next_steps:
            md_lines.append(f"- {ns}")
        md_lines.append("")

    console.print(Markdown("\n".join(md_lines)))

    # 6. Summary line
    summary_line = (
        f"status={answer.status} "
        f"latency={answer.meta.latency_ms}ms "
        f"cost=${answer.meta.cost_usd:.4f} "
        f"calls={answer.meta.tool_calls}"
    )
    console.print(Panel(summary_line, style="bold green" if answer.status == "ok" else "bold yellow"))

    # Write JSON output if requested
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(answer.model_dump_json(indent=2))

    # Exit codes: 0 for ok/degraded, 2 for insufficient_evidence
    if answer.status == "insufficient_evidence":
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
