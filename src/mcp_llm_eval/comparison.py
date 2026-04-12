"""Compare two evaluation runs and detect regressions."""

from __future__ import annotations

from typing import Any


def compare_runs(
    baseline: dict[str, Any],
    current: dict[str, Any],
    tolerance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two evaluation summary dicts and detect regressions.

    Args:
        baseline: Baseline evaluation summary JSON (e.g., main branch results).
        current: Current evaluation summary JSON (e.g., PR branch results).
        tolerance: Per-metric regression tolerance before flagging.

    Returns:
        Comparison result dict with per-model metrics, deltas, and regression flags.
    """
    tol = _apply_tolerance_defaults(tolerance or {})

    baseline_overall = baseline.get("overall", {})
    current_overall = current.get("overall", {})

    all_models = set(baseline_overall.keys()) | set(current_overall.keys())

    models_result: dict[str, Any] = {}
    regressions: list[dict[str, Any]] = []

    for model_name in sorted(all_models):
        b_metrics = baseline_overall.get(model_name, {})
        c_metrics = current_overall.get(model_name, {})

        model_comparison: dict[str, Any] = {}

        # faithfulness
        b_faith = b_metrics.get("avg_faithfulness")
        c_faith = c_metrics.get("avg_faithfulness")
        if b_faith is not None and c_faith is not None:
            delta = round(c_faith - b_faith, 4)
            is_regression = delta < -tol["faithfulness"]
            model_comparison["faithfulness"] = {
                "baseline": b_faith,
                "current": c_faith,
                "delta": delta,
                "regression": is_regression,
            }
            if is_regression:
                regressions.append({
                    "model": model_name,
                    "metric": "faithfulness",
                    "baseline": b_faith,
                    "current": c_faith,
                    "delta": delta,
                })

        # relevance
        b_rel = b_metrics.get("avg_relevance")
        c_rel = c_metrics.get("avg_relevance")
        if b_rel is not None and c_rel is not None:
            delta = round(c_rel - b_rel, 4)
            is_regression = delta < -tol["relevance"]
            model_comparison["relevance"] = {
                "baseline": b_rel,
                "current": c_rel,
                "delta": delta,
                "regression": is_regression,
            }
            if is_regression:
                regressions.append({
                    "model": model_name,
                    "metric": "relevance",
                    "baseline": b_rel,
                    "current": c_rel,
                    "delta": delta,
                })

        # avg_ttft_ms
        b_ttft = b_metrics.get("avg_ttft_ms")
        c_ttft = c_metrics.get("avg_ttft_ms")
        if b_ttft is not None and c_ttft is not None:
            delta = round(c_ttft - b_ttft, 1)
            is_regression = delta > tol["ttft_ms"]
            model_comparison["avg_ttft_ms"] = {
                "baseline": b_ttft,
                "current": c_ttft,
                "delta": delta,
                "regression": is_regression,
            }
            if is_regression:
                regressions.append({
                    "model": model_name,
                    "metric": "avg_ttft_ms",
                    "baseline": b_ttft,
                    "current": c_ttft,
                    "delta": delta,
                })

        # avg_cost_per_query
        b_cost = b_metrics.get("avg_cost_per_query")
        c_cost = c_metrics.get("avg_cost_per_query")
        if b_cost is not None and c_cost is not None:
            delta = round(c_cost - b_cost, 6)
            is_regression = delta > tol["cost"]
            model_comparison["avg_cost_per_query"] = {
                "baseline": b_cost,
                "current": c_cost,
                "delta": delta,
                "regression": is_regression,
            }
            if is_regression:
                regressions.append({
                    "model": model_name,
                    "metric": "avg_cost_per_query",
                    "baseline": b_cost,
                    "current": c_cost,
                    "delta": delta,
                })

        if model_comparison:
            models_result[model_name] = model_comparison

    return {
        "baseline_timestamp": baseline.get("timestamp", ""),
        "current_timestamp": current.get("timestamp", ""),
        "models": models_result,
        "regressions": regressions,
        "has_regressions": len(regressions) > 0,
    }


def _apply_tolerance_defaults(tolerance: dict[str, Any]) -> dict[str, Any]:
    """Apply default tolerance values."""
    return {
        "faithfulness": tolerance.get("faithfulness", 0.05),
        "relevance": tolerance.get("relevance", 0.05),
        "ttft_ms": tolerance.get("ttft_ms", 200),
        "cost": tolerance.get("cost", 0.005),
    }
