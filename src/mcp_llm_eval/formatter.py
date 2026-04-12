"""Generate markdown PR comments from evaluation results."""

from __future__ import annotations

from typing import Any


def format_pr_comment(
    summary: dict[str, Any],
    comparison: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> str:
    """Generate a markdown PR comment from evaluation results.

    Args:
        summary: Evaluation summary JSON dict.
        comparison: Optional compare_runs output dict.
        thresholds: Optional threshold config dict for pass/fail badges.

    Returns:
        Markdown string ready to post as a GitHub PR comment.
    """
    lines: list[str] = []
    lines.append("## LLM Eval Results\n")

    overall = summary.get("overall", {})
    threshold_results = _check_thresholds_inline(overall, thresholds) if thresholds else {}

    # Main results table
    lines.append("| Model | Faithfulness | Relevance | TTFT (avg) | Cost/Query | Status |")
    lines.append("|-------|-------------|-----------|------------|------------|--------|")

    model_statuses: list[bool] = []
    for model_name, metrics in sorted(overall.items()):
        faith = metrics.get("avg_faithfulness")
        rel = metrics.get("avg_relevance")
        ttft = metrics.get("avg_ttft_ms")
        cost = metrics.get("avg_cost_per_query")

        faith_str = f"{faith:.2f}" if faith is not None else "N/A"
        rel_str = f"{rel:.2f}" if rel is not None else "N/A"
        ttft_str = f"{int(ttft)}ms" if ttft is not None else "N/A"
        cost_str = f"${cost:.4f}" if cost is not None else "N/A"

        model_pass = threshold_results.get(model_name, True)
        model_statuses.append(model_pass)
        status = "\u2705 PASS" if model_pass else "\u274c FAIL"

        lines.append(f"| {model_name} | {faith_str} | {rel_str} | {ttft_str} | {cost_str} | {status} |")

    # Overall summary line
    overall_pass = all(model_statuses) if model_statuses else True
    overall_status = "\u2705 **Overall: PASS**" if overall_pass else "\u274c **Overall: FAIL**"
    total_cost = summary.get("total_estimated_cost", 0)
    total_questions = summary.get("total_questions", 0)
    judge_model = summary.get("judge_model", "N/A")

    lines.append("")
    lines.append(
        f"{overall_status} | Total cost: ${total_cost:.2f} | "
        f"Questions: {total_questions} | Judge: {judge_model}"
    )

    # Regressions section
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Regressions vs baseline</summary>")
    lines.append("")
    lines.append("| Model | Metric | Baseline | Current | Delta |")
    lines.append("|-------|--------|----------|---------|-------|")

    if comparison and comparison.get("regressions"):
        for reg in comparison["regressions"]:
            delta_str = _format_delta(reg["metric"], reg["delta"])
            baseline_str = _format_value(reg["metric"], reg["baseline"])
            current_str = _format_value(reg["metric"], reg["current"])
            lines.append(
                f"| {reg['model']} | {reg['metric']} | "
                f"{baseline_str} | {current_str} | {delta_str} |"
            )
    else:
        lines.append("| (none detected) | | | | |")

    lines.append("")
    lines.append("</details>")

    # Threshold details section
    if thresholds:
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Threshold details</summary>")
        lines.append("")
        lines.append("| Metric | Threshold | Actual | Status |")
        lines.append("|--------|-----------|--------|--------|")

        threshold_checks = _compute_threshold_details(overall, thresholds)
        for check in threshold_checks:
            status = "\u2705 PASS" if check["passed"] else "\u274c FAIL"
            lines.append(
                f"| {check['metric']} | {check['threshold_str']} | "
                f"{check['actual_str']} | {status} |"
            )

        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def _check_thresholds_inline(
    overall: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, bool]:
    """Check each model against thresholds, return per-model pass/fail."""
    results: dict[str, bool] = {}

    for model_name, metrics in overall.items():
        passed = True

        if thresholds.get("avg_faithfulness") is not None:
            faith = metrics.get("avg_faithfulness")
            if faith is not None and faith < thresholds["avg_faithfulness"]:
                passed = False

        if thresholds.get("avg_relevance") is not None:
            rel = metrics.get("avg_relevance")
            if rel is not None and rel < thresholds["avg_relevance"]:
                passed = False

        if thresholds.get("p95_ttft_ms") is not None:
            ttft = metrics.get("avg_ttft_ms")
            if ttft is not None and ttft > thresholds["p95_ttft_ms"]:
                passed = False

        if thresholds.get("max_cost_per_query") is not None:
            cost = metrics.get("avg_cost_per_query")
            if cost is not None and cost > thresholds["max_cost_per_query"]:
                passed = False

        results[model_name] = passed

    return results


def _compute_threshold_details(
    overall: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute threshold check details for the details section."""
    checks: list[dict[str, Any]] = []

    # Average across all models
    all_faith = [
        m["avg_faithfulness"] for m in overall.values()
        if m.get("avg_faithfulness") is not None
    ]
    all_rel = [
        m["avg_relevance"] for m in overall.values()
        if m.get("avg_relevance") is not None
    ]
    all_ttft = [
        m["avg_ttft_ms"] for m in overall.values()
        if m.get("avg_ttft_ms") is not None
    ]
    all_cost = [
        m["avg_cost_per_query"] for m in overall.values()
        if m.get("avg_cost_per_query") is not None
    ]

    if thresholds.get("avg_faithfulness") is not None:
        actual = sum(all_faith) / len(all_faith) if all_faith else 0.0
        thresh_val = thresholds["avg_faithfulness"]
        checks.append({
            "metric": "avg_faithfulness",
            "threshold_str": f">= {thresh_val:.2f}",
            "actual_str": f"{actual:.2f}",
            "passed": actual >= thresh_val,
        })

    if thresholds.get("avg_relevance") is not None:
        actual = sum(all_rel) / len(all_rel) if all_rel else 0.0
        thresh_val = thresholds["avg_relevance"]
        checks.append({
            "metric": "avg_relevance",
            "threshold_str": f">= {thresh_val:.2f}",
            "actual_str": f"{actual:.2f}",
            "passed": actual >= thresh_val,
        })

    if thresholds.get("p95_ttft_ms") is not None:
        actual = max(all_ttft) if all_ttft else 0
        thresh_val = thresholds["p95_ttft_ms"]
        checks.append({
            "metric": "p95_ttft_ms",
            "threshold_str": f"<= {thresh_val}ms",
            "actual_str": f"{int(actual)}ms",
            "passed": actual <= thresh_val,
        })

    if thresholds.get("max_cost_per_query") is not None:
        actual = max(all_cost) if all_cost else 0.0
        thresh_val = thresholds["max_cost_per_query"]
        checks.append({
            "metric": "max_cost_per_query",
            "threshold_str": f"<= ${thresh_val:.4f}",
            "actual_str": f"${actual:.4f}",
            "passed": actual <= thresh_val,
        })

    return checks


def _format_delta(metric: str, delta: float) -> str:
    """Format a delta value for display."""
    if metric in ("faithfulness", "relevance"):
        return f"{delta:+.4f}"
    if metric == "avg_ttft_ms":
        return f"{delta:+.0f}ms"
    if metric == "avg_cost_per_query":
        return f"{delta:+.6f}"
    return str(delta)


def _format_value(metric: str, value: float) -> str:
    """Format a metric value for display."""
    if metric in ("faithfulness", "relevance"):
        return f"{value:.4f}"
    if metric == "avg_ttft_ms":
        return f"{value:.0f}ms"
    if metric == "avg_cost_per_query":
        return f"${value:.4f}"
    return str(value)
