"""Eval engine: dataset loader, LLM runner, judge, threshold checker."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import judge as judge_module
from .types import (
    AggregateMetrics,
    EvalEntry,
    EvalResult,
    MetricCheck,
    ModelConfig,
    RunSummary,
    ThresholdConfig,
    ThresholdResult,
)


# ---------------------------------------------------------------------------
# Provider registry — lazy imports
# ---------------------------------------------------------------------------

def _get_runner(provider: str):
    """Return the run() function for a provider, importing lazily."""
    if provider == "anthropic":
        from .providers import anthropic as mod
        return mod
    elif provider == "openai":
        from .providers import openai as mod
        return mod
    elif provider == "google":
        from .providers import google as mod
        return mod
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: anthropic, openai, google")


def _get_client(provider: str) -> Any:
    """Create a client for the given provider."""
    mod = _get_runner(provider)
    return mod._get_client()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[EvalEntry]:
    """Load and validate a JSON evaluation dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of eval entries")
    if len(data) == 0:
        raise ValueError("Dataset is empty")

    required_keys = {"id", "category", "context", "question", "expected_response"}
    entries = []
    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Entry {i} missing required keys: {missing}")
        entries.append(EvalEntry.from_dict(item))

    return entries


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(subset: list[EvalResult], provider: str, model: str) -> AggregateMetrics | None:
    """Compute aggregate metrics for a list of eval results."""
    if not subset:
        return None

    n = len(subset)
    scored = [r for r in subset if r.faithfulness_score is not None]

    return AggregateMetrics(
        provider=provider,
        model=model,
        runs=n,
        avg_ttft_ms=round(sum(r.time_to_first_token_ms or 0 for r in subset) / n, 1),
        avg_latency_ms=round(sum(r.total_latency_ms for r in subset) / n, 1),
        avg_input_tokens=round(sum(r.input_tokens for r in subset) / n, 1),
        avg_output_tokens=round(sum(r.output_tokens for r in subset) / n, 1),
        avg_cost_per_query=round(sum(r.cost_per_query for r in subset) / n, 6),
        avg_faithfulness=round(sum(r.faithfulness_score for r in scored) / len(scored), 4) if scored else None,
        avg_relevance=round(sum(r.relevance_score for r in scored) / len(scored), 4) if scored else None,
    )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset: list[EvalEntry],
    models: list[ModelConfig],
    judge_config: dict[str, Any] | None = None,
    output_dir: str | None = None,
    tracing_config: dict[str, Any] | None = None,
) -> RunSummary:
    """Orchestrate a full evaluation run.

    Steps: dataset -> provider runners -> judge scoring -> aggregation -> result storage.
    """
    judge_config = judge_config or {}
    judge_model = judge_config.get("model", judge_module.DEFAULT_JUDGE_MODEL)
    judge_temperature = judge_config.get("temperature", 0)
    judge_provider = judge_config.get("provider", "openai")

    # Instantiate provider clients
    clients: dict[str, Any] = {}
    needed_providers = {m.provider for m in models}
    # Judge needs openai by default
    if judge_provider:
        needed_providers.add(judge_provider)

    for provider in needed_providers:
        try:
            clients[provider] = _get_client(provider)
        except ImportError as e:
            raise ImportError(f"Provider '{provider}' SDK not available: {e}")

    # Tracing setup (optional langsmith)
    ls_client = None
    if tracing_config and tracing_config.get("enabled"):
        try:
            import langsmith
            ls_client = langsmith.Client(
                api_url=tracing_config.get("endpoint", "https://api.smith.langchain.com"),
            )
        except Exception:
            pass

    results: list[EvalResult] = []
    run_start = time.monotonic()

    # Phase 1: Run models
    for entry in dataset:
        for model_cfg in models:
            provider = model_cfg.provider
            model = model_cfg.model

            if provider not in clients:
                result = EvalResult(
                    eval_id=entry.id,
                    category=entry.category,
                    model=model,
                    provider=provider,
                    error=f"{provider} client not available",
                )
                results.append(result)
                continue

            try:
                runner_mod = _get_runner(provider)
                metrics = runner_mod.run(
                    clients[provider],
                    model,
                    entry.context,
                    entry.question,
                    model_cfg.max_tokens,
                )

                cost = (
                    metrics["input_tokens"] * model_cfg.input_cost_per_mtok
                    + metrics["output_tokens"] * model_cfg.output_cost_per_mtok
                ) / 1_000_000

                result = EvalResult(
                    eval_id=entry.id,
                    category=entry.category,
                    model=model,
                    provider=provider,
                    response=metrics["response"],
                    input_tokens=metrics["input_tokens"],
                    output_tokens=metrics["output_tokens"],
                    stop_reason=metrics.get("stop_reason"),
                    time_to_first_token_ms=metrics.get("time_to_first_token_ms"),
                    total_latency_ms=metrics["total_latency_ms"],
                    cost_per_query=round(cost, 6),
                )
            except Exception as e:
                result = EvalResult(
                    eval_id=entry.id,
                    category=entry.category,
                    model=model,
                    provider=provider,
                    error=str(e),
                )

            results.append(result)

    # Phase 2: Judge scoring
    judge_client = clients.get(judge_provider)
    if judge_client:
        entry_map = {e.id: e for e in dataset}
        scorable = [r for r in results if r.error is None and r.response]

        for result in scorable:
            entry = entry_map[result.eval_id]
            try:
                scores = judge_module.evaluate_response(
                    client=judge_client,
                    context=entry.context,
                    question=entry.question,
                    expected_response=entry.expected_response,
                    response=result.response,
                    judge_model=judge_model,
                    temperature=judge_temperature,
                )
                result.faithfulness_score = scores["faithfulness_score"]
                result.faithfulness_reason = scores["faithfulness_reason"]
                result.relevance_score = scores["relevance_score"]
                result.relevance_reason = scores["relevance_reason"]
                result.judge_model = scores["judge_model"]
            except Exception:
                result.judge_model = judge_model

    run_elapsed = round(time.monotonic() - run_start, 1)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Phase 3: Aggregation
    ok_results = [r for r in results if r.error is None]
    categorys = sorted({e.category for e in dataset})

    overall: dict[str, Any] = {}
    for model_cfg in models:
        subset = [r for r in ok_results if r.model == model_cfg.model]
        agg = _aggregate(subset, model_cfg.provider, model_cfg.model)
        if agg:
            overall[model_cfg.model] = agg.to_dict()

    by_category: dict[str, Any] = {}
    for mt in categorys:
        by_category[mt] = {}
        for model_cfg in models:
            subset = [r for r in ok_results if r.model == model_cfg.model and r.category == mt]
            agg = _aggregate(subset, model_cfg.provider, model_cfg.model)
            if agg:
                by_category[mt][model_cfg.model] = agg.to_dict()

    total_cost = sum(r.cost_per_query for r in ok_results)

    summary = RunSummary(
        timestamp=timestamp,
        total_questions=len(dataset),
        total_model_runs=len(results),
        total_errors=len(results) - len(ok_results),
        total_elapsed_sec=run_elapsed,
        total_estimated_cost=round(total_cost, 6),
        judge_model=judge_model,
        overall=overall,
        by_category=by_category,
        results=[r.to_dict() for r in results],
    )

    # Phase 4: Save results
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary_path = out / f"{timestamp}_summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

        benchmark_path = out / f"{timestamp}_benchmark.json"
        benchmark_data = {
            "timestamp": timestamp,
            "total_entries": len(dataset),
            "models": [m.model for m in models],
            "judge_model": judge_model,
            "total_runs": len(results),
            "total_elapsed_sec": run_elapsed,
            "results": [r.to_dict() for r in results],
        }
        benchmark_path.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")

    return summary


# ---------------------------------------------------------------------------
# Threshold checking
# ---------------------------------------------------------------------------

def check_thresholds(summary: RunSummary, thresholds: ThresholdConfig) -> ThresholdResult:
    """Validate a run summary against quality gate thresholds."""
    checks: list[MetricCheck] = []

    # Collect all ok results for p95 and per-query checks
    ok_results = [EvalResult.from_dict(r) for r in summary.results if r.get("error") is None]

    # avg_faithfulness — average across all models in overall
    if thresholds.avg_faithfulness is not None:
        faith_scores = [
            v["avg_faithfulness"]
            for v in summary.overall.values()
            if isinstance(v, dict) and v.get("avg_faithfulness") is not None
        ]
        actual = round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else 0.0
        checks.append(MetricCheck(
            metric="avg_faithfulness",
            threshold=thresholds.avg_faithfulness,
            actual=actual,
            passed=actual >= thresholds.avg_faithfulness,
        ))

    # avg_relevance
    if thresholds.avg_relevance is not None:
        rel_scores = [
            v["avg_relevance"]
            for v in summary.overall.values()
            if isinstance(v, dict) and v.get("avg_relevance") is not None
        ]
        actual = round(sum(rel_scores) / len(rel_scores), 4) if rel_scores else 0.0
        checks.append(MetricCheck(
            metric="avg_relevance",
            threshold=thresholds.avg_relevance,
            actual=actual,
            passed=actual >= thresholds.avg_relevance,
        ))

    # p95_ttft_ms
    if thresholds.p95_ttft_ms is not None:
        ttft_values = sorted(
            r.time_to_first_token_ms
            for r in ok_results
            if r.time_to_first_token_ms is not None
        )
        if ttft_values:
            idx = int(len(ttft_values) * 0.95)
            idx = min(idx, len(ttft_values) - 1)
            p95 = ttft_values[idx]
        else:
            p95 = 0
        checks.append(MetricCheck(
            metric="p95_ttft_ms",
            threshold=float(thresholds.p95_ttft_ms),
            actual=float(p95),
            passed=p95 <= thresholds.p95_ttft_ms,
        ))

    # max_cost_per_query
    if thresholds.max_cost_per_query is not None:
        costs = [r.cost_per_query for r in ok_results if r.cost_per_query > 0]
        max_cost = max(costs) if costs else 0.0
        checks.append(MetricCheck(
            metric="max_cost_per_query",
            threshold=thresholds.max_cost_per_query,
            actual=max_cost,
            passed=max_cost <= thresholds.max_cost_per_query,
        ))

    overall_pass = all(c.passed for c in checks) if checks else True

    return ThresholdResult(overall_pass=overall_pass, per_metric=checks)
