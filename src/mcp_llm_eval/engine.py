"""Eval engine: dataset loader, LLM runner, judge, threshold checker."""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import judge as judge_module
from . import retrieval_metrics
from .retrieval import BM25Adapter, RetrievalAdapter
from .types import (
    AggregateMetrics,
    EvalEntry,
    EvalResult,
    MetricCheck,
    ModelConfig,
    RAGResult,
    RetrievalResult,
    RetrievedChunk,
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
            raise ImportError(f"Provider '{provider}' SDK not available: {e}") from e

    if tracing_config and tracing_config.get("enabled"):
        try:
            import langsmith
            langsmith.Client(
                api_url=tracing_config.get("endpoint", "https://api.smith.langchain.com"),
            )
        except ImportError:
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

    # ------------------------------------------------------------------
    # v0.5.0 retrieval / RAG thresholds
    # ------------------------------------------------------------------

    def _retrieval_metric_actual(name: str) -> float | None:
        """Average a retrieval/RAG metric across per-model ``overall`` blocks.

        Both retrieval-only summaries (single synthetic ``_retrieval`` key) and
        RAG summaries (one key per model) put the metric in ``overall[*][name]``,
        so a single averaging path handles both.
        """
        per_model = [
            v[name]
            for v in summary.overall.values()
            if isinstance(v, dict) and v.get(name) is not None
        ]
        if per_model:
            return round(sum(per_model) / len(per_model), 4)
        return None

    _retrieval_min_thresholds = (
        ("avg_recall_at_k", thresholds.avg_recall_at_k),
        ("avg_precision_at_k", thresholds.avg_precision_at_k),
        ("avg_mrr", thresholds.avg_mrr),
        ("avg_ndcg_at_k", thresholds.avg_ndcg_at_k),
        ("avg_context_relevance", thresholds.avg_context_relevance),
        ("avg_citation_faithfulness", thresholds.avg_citation_faithfulness),
    )
    for metric_name, threshold in _retrieval_min_thresholds:
        if threshold is None:
            continue
        actual = _retrieval_metric_actual(metric_name)
        if actual is None:
            actual = 0.0
        checks.append(MetricCheck(
            metric=metric_name,
            threshold=float(threshold),
            actual=float(actual),
            passed=actual >= threshold,
        ))

    if thresholds.p95_retrieval_latency_ms is not None:
        actual = _retrieval_metric_actual("p95_retrieval_latency_ms")
        if actual is None:
            actual = 0.0
        checks.append(MetricCheck(
            metric="p95_retrieval_latency_ms",
            threshold=float(thresholds.p95_retrieval_latency_ms),
            actual=float(actual),
            passed=actual <= thresholds.p95_retrieval_latency_ms,
        ))

    overall_pass = all(c.passed for c in checks) if checks else True

    return ThresholdResult(overall_pass=overall_pass, per_metric=checks)


# ---------------------------------------------------------------------------
# v0.5.0 — retrieval and RAG evaluation
# ---------------------------------------------------------------------------


def _build_retrieval_adapter(adapter: str, corpus_path: str) -> RetrievalAdapter:
    """Construct a retrieval adapter by name. Only ``bm25`` is supported in v0.5.0."""
    if adapter == "bm25":
        return BM25Adapter.from_jsonl(corpus_path)
    raise ValueError(f"Unsupported retrieval adapter: {adapter}")


def load_jsonl_dataset(path: str) -> list[EvalEntry]:
    """Load a JSONL evaluation dataset (one EvalEntry per line)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    required_keys = {"id", "category", "context", "question", "expected_response"}
    entries: list[EvalEntry] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {lineno}: invalid JSON ({e})") from e
        if not isinstance(item, dict):
            raise ValueError(f"Line {lineno}: expected JSON object")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Line {lineno}: missing required keys: {missing}")
        entries.append(EvalEntry.from_dict(item))

    if not entries:
        raise ValueError("Dataset is empty")
    return entries


def _format_rag_system_prompt(context: str, chunks: list[RetrievedChunk]) -> str:
    """Build the system prompt for RAG generation with retrieved context."""
    rendered = "\n\n".join(
        f"[chunk {i + 1}]\n{c.content}" for i, c in enumerate(chunks)
    )
    return (
        f"{context}\n\n"
        f"Retrieved context:\n{rendered}\n\n"
        f"Answer the question using only the retrieved context above."
    )


def _percentiles(values: list[float]) -> tuple[float, float]:
    """Return (p50, p95) of a list of values. Falls back to max for <2 values."""
    if not values:
        return 0.0, 0.0
    if len(values) < 2:
        v = float(max(values))
        return v, v
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return float(qs[49]), float(qs[94])


def _aggregate_rag(
    results: list[RAGResult],
    provider: str,
    model: str,
    k: int,
) -> dict[str, Any]:
    """Aggregate per-model RAGResults into the ``overall`` block shape."""
    ok = [r for r in results if r.error is None]
    if not ok:
        return {
            "provider": provider,
            "model": model,
            "runs": 0,
            "k": k,
        }

    n = len(ok)
    latencies = [r.retrieval_latency_ms for r in ok]
    p50, p95 = _percentiles(latencies)

    def avg(key: str) -> float:
        vals = [r.retrieval_metrics.get(key, 0.0) for r in ok]
        return round(sum(vals) / n, 4)

    def avg_score(attr: str) -> float | None:
        vals = [getattr(r, attr) for r in ok if getattr(r, attr) is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 4)

    def avg_gen(key: str) -> float:
        vals = [r.generation_metrics.get(key, 0) or 0 for r in ok]
        return round(sum(vals) / n, 4)

    return {
        "provider": provider,
        "model": model,
        "runs": n,
        "k": k,
        "avg_recall_at_k": avg("recall_at_k"),
        "avg_precision_at_k": avg("precision_at_k"),
        "avg_mrr": avg("mrr"),
        "avg_ndcg_at_k": avg("ndcg_at_k"),
        "avg_context_relevance": avg_score("context_relevance_score"),
        "avg_citation_faithfulness": avg_score("citation_faithfulness_score"),
        "avg_retrieval_latency_ms": round(sum(latencies) / n, 1),
        "p50_retrieval_latency_ms": round(p50, 1),
        "p95_retrieval_latency_ms": round(p95, 1),
        "avg_ttft_ms": avg_gen("time_to_first_token_ms"),
        "avg_latency_ms": avg_gen("total_latency_ms"),
        "avg_input_tokens": avg_gen("input_tokens"),
        "avg_output_tokens": avg_gen("output_tokens"),
        "avg_cost_per_query": round(sum(
            r.generation_metrics.get("cost_per_query", 0.0) for r in ok
        ) / n, 6),
    }


def run_retrieval_evaluation(
    dataset_path: str,
    corpus_path: str,
    k: int = 5,
    adapter: str = "bm25",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run retrieval-only evaluation against a labelled dataset.

    Returns a dict with timestamp, k, adapter, total_queries, total_errors,
    aggregate (avg metrics + p50/p95 latency), per_query (RetrievalResult dicts),
    and skipped (entries omitted because they have no relevant_chunk_ids).
    """
    try:
        all_entries = load_jsonl_dataset(dataset_path)
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"dataset load failed: {e}"}

    skipped = [e.id for e in all_entries if not e.relevant_chunk_ids]
    entries = [e for e in all_entries if e.relevant_chunk_ids]
    if skipped:
        print(
            f"Warning: skipping {len(skipped)} entries without relevant_chunk_ids: "
            f"{skipped}",
            file=sys.stderr,
        )
    if not entries:
        return {"error": "no entries with relevant_chunk_ids"}

    try:
        retriever = _build_retrieval_adapter(adapter, corpus_path)
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"adapter init failed: {e}"}

    results: list[RetrievalResult] = []
    for entry in entries:
        result = _retrieve_one(retriever, entry, k)
        results.append(result)

    aggregate = _aggregate_retrieval(results, k)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    total_errors = sum(1 for r in results if r.error is not None)
    per_query = [r.to_dict() for r in results]

    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "k": k,
        "adapter": adapter,
        "total_queries": len(results),
        "total_questions": len(results),
        "total_model_runs": len(results),
        "total_errors": total_errors,
        "total_elapsed_sec": 0.0,
        "total_estimated_cost": 0.0,
        "judge_model": None,
        "skipped": skipped,
        "aggregate": aggregate,
        "overall": {"_retrieval": aggregate},
        "by_category": {},
        "per_query": per_query,
        "results": per_query,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_only = {
            k_: v for k_, v in summary.items() if k_ != "per_query"
        }
        (out / f"{timestamp}_retrieval_summary.json").write_text(
            json.dumps(summary_only, indent=2), encoding="utf-8"
        )
        (out / f"{timestamp}_retrieval_detail.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (out / "latest_retrieval_summary.json").write_text(
            json.dumps(summary_only, indent=2), encoding="utf-8"
        )

    return summary


def _retrieve_one(
    retriever: RetrievalAdapter, entry: EvalEntry, k: int,
) -> RetrievalResult:
    """Run retrieval for a single query and compute the four metrics."""
    relevant = entry.relevant_chunk_ids or []
    try:
        t0 = time.monotonic()
        chunks = retriever.retrieve(entry.question, k)
        latency_ms = (time.monotonic() - t0) * 1000.0
        retrieved_ids = [c.chunk_id for c in chunks]
        metrics = {
            "recall_at_k": retrieval_metrics.recall_at_k(retrieved_ids, relevant, k),
            "precision_at_k": retrieval_metrics.precision_at_k(retrieved_ids, relevant, k),
            "mrr": retrieval_metrics.mean_reciprocal_rank(retrieved_ids, relevant),
            "ndcg_at_k": retrieval_metrics.ndcg_at_k(retrieved_ids, relevant, k),
        }
        return RetrievalResult(
            query_id=entry.id,
            query=entry.question,
            retrieved_chunk_ids=retrieved_ids,
            retrieval_latency_ms=round(latency_ms, 3),
            metrics={k_: round(v, 4) for k_, v in metrics.items()},
            k=k,
        )
    except Exception as e:
        return RetrievalResult(
            query_id=entry.id,
            query=entry.question,
            retrieved_chunk_ids=[],
            retrieval_latency_ms=0.0,
            metrics={"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0},
            k=k,
            error=str(e),
        )


def _aggregate_retrieval(
    results: list[RetrievalResult], k: int,
) -> dict[str, Any]:
    ok = [r for r in results if r.error is None]
    if not ok:
        return {
            "k": k,
            "avg_recall_at_k": 0.0,
            "avg_precision_at_k": 0.0,
            "avg_mrr": 0.0,
            "avg_ndcg_at_k": 0.0,
            "p50_retrieval_latency_ms": 0.0,
            "p95_retrieval_latency_ms": 0.0,
        }
    n = len(ok)
    latencies = [r.retrieval_latency_ms for r in ok]
    p50, p95 = _percentiles(latencies)
    return {
        "k": k,
        "avg_recall_at_k": round(sum(r.metrics["recall_at_k"] for r in ok) / n, 4),
        "avg_precision_at_k": round(sum(r.metrics["precision_at_k"] for r in ok) / n, 4),
        "avg_mrr": round(sum(r.metrics["mrr"] for r in ok) / n, 4),
        "avg_ndcg_at_k": round(sum(r.metrics["ndcg_at_k"] for r in ok) / n, 4),
        "p50_retrieval_latency_ms": round(p50, 1),
        "p95_retrieval_latency_ms": round(p95, 1),
    }


def run_rag_evaluation(
    dataset_path: str,
    corpus_path: str,
    models: list[ModelConfig],
    k: int = 5,
    adapter: str = "bm25",
    judge_config: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run the full RAG pipeline (retrieve + generate + judge) over a dataset.

    Returns a summary dict with timestamp, models, total_queries, total_errors,
    overall (per-model aggregate block), per_query (RAGResult dicts), skipped.
    """
    judge_config = judge_config or {}
    judge_model = judge_config.get("model")
    judge_temperature = judge_config.get("temperature", 0)
    judge_provider = judge_config.get("provider", "openai")

    try:
        all_entries = load_jsonl_dataset(dataset_path)
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"dataset load failed: {e}"}

    skipped = [e.id for e in all_entries if not e.relevant_chunk_ids]
    entries = [e for e in all_entries if e.relevant_chunk_ids]
    if skipped:
        print(
            f"Warning: skipping {len(skipped)} entries without relevant_chunk_ids: "
            f"{skipped}",
            file=sys.stderr,
        )
    if not entries:
        return {"error": "no entries with relevant_chunk_ids"}

    try:
        retriever = _build_retrieval_adapter(adapter, corpus_path)
    except (FileNotFoundError, ValueError) as e:
        return {"error": f"adapter init failed: {e}"}

    needed_providers = {m.provider for m in models}
    needed_providers.add(judge_provider)
    clients: dict[str, Any] = {}
    for provider in needed_providers:
        try:
            clients[provider] = _get_client(provider)
        except ImportError as e:
            raise ImportError(f"Provider '{provider}' SDK not available: {e}") from e

    judge_client = clients.get(judge_provider)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_start = time.monotonic()
    all_rag_results: list[RAGResult] = []

    for entry in entries:
        retrieval = _retrieve_one(retriever, entry, k)
        chunks_for_query: list[RetrievedChunk] = []
        if retrieval.error is None:
            try:
                t0 = time.monotonic()
                fresh = retriever.retrieve(entry.question, k)
                chunks_for_query = fresh
                # If the second retrieve disagrees on ordering for any reason,
                # the timed metrics from _retrieve_one still take precedence.
                _ = (time.monotonic() - t0) * 1000.0
            except Exception:
                chunks_for_query = []

        per_chunk_scores: list[float] = []
        context_score: float | None = None
        if judge_client and chunks_for_query and retrieval.error is None:
            for chunk in chunks_for_query:
                try:
                    res = judge_module.judge_context_relevance(
                        client=judge_client,
                        query=entry.question,
                        chunk=chunk.content,
                        judge_model=judge_model,
                        temperature=judge_temperature,
                    )
                    per_chunk_scores.append(res["score"])
                except Exception:
                    pass
            if per_chunk_scores:
                context_score = round(
                    sum(per_chunk_scores) / len(per_chunk_scores), 4
                )

        for model_cfg in models:
            rag_result = _run_rag_for_model(
                entry=entry,
                retrieval=retrieval,
                chunks=chunks_for_query,
                model_cfg=model_cfg,
                clients=clients,
                judge_client=judge_client,
                judge_model=judge_model,
                judge_temperature=judge_temperature,
                per_chunk_scores=per_chunk_scores,
                context_score=context_score,
                k=k,
            )
            all_rag_results.append(rag_result)

    run_elapsed = round(time.monotonic() - run_start, 1)

    overall: dict[str, Any] = {}
    for model_cfg in models:
        subset = [r for r in all_rag_results if r.model == model_cfg.model]
        overall[model_cfg.model] = _aggregate_rag(
            subset, model_cfg.provider, model_cfg.model, k,
        )

    total_errors = sum(1 for r in all_rag_results if r.error is not None)
    total_cost = sum(
        r.generation_metrics.get("cost_per_query", 0.0) or 0.0
        for r in all_rag_results
        if r.error is None
    )
    resolved_judge_model = judge_module._resolve_judge_model(judge_model)

    per_query = [r.to_dict() for r in all_rag_results]
    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "k": k,
        "adapter": adapter,
        "total_queries": len(entries),
        "total_questions": len(entries),
        "total_model_runs": len(all_rag_results),
        "total_errors": total_errors,
        "total_elapsed_sec": run_elapsed,
        "total_estimated_cost": round(total_cost, 6),
        "judge_model": resolved_judge_model,
        "skipped": skipped,
        "overall": overall,
        "by_category": {},
        "per_query": per_query,
        "results": per_query,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_only = {k_: v for k_, v in summary.items() if k_ != "per_query"}
        benchmark = {**summary_only, "results": summary["per_query"]}
        (out / f"{timestamp}_rag_summary.json").write_text(
            json.dumps(summary_only, indent=2), encoding="utf-8"
        )
        (out / f"{timestamp}_rag_benchmark.json").write_text(
            json.dumps(benchmark, indent=2), encoding="utf-8"
        )
        (out / "latest_rag_summary.json").write_text(
            json.dumps(summary_only, indent=2), encoding="utf-8"
        )

    return summary


def _run_rag_for_model(
    *,
    entry: EvalEntry,
    retrieval: RetrievalResult,
    chunks: list[RetrievedChunk],
    model_cfg: ModelConfig,
    clients: dict[str, Any],
    judge_client: Any,
    judge_model: str | None,
    judge_temperature: float,
    per_chunk_scores: list[float],
    context_score: float | None,
    k: int,
) -> RAGResult:
    """Run generation + citation faithfulness for a single (query, model) pair."""
    base = RAGResult(
        query_id=entry.id,
        query=entry.question,
        retrieved_chunks=chunks,
        retrieval_latency_ms=retrieval.retrieval_latency_ms,
        retrieval_metrics=dict(retrieval.metrics),
        model=model_cfg.model,
        provider=model_cfg.provider,
    )
    if retrieval.error is not None:
        base.error = f"retrieval failed: {retrieval.error}"
        return base

    if model_cfg.provider not in clients:
        base.error = f"{model_cfg.provider} client not available"
        return base

    system_prompt = _format_rag_system_prompt(entry.context, chunks)
    try:
        runner_mod = _get_runner(model_cfg.provider)
        gen = runner_mod.run(
            clients[model_cfg.provider],
            model_cfg.model,
            system_prompt,
            entry.question,
            model_cfg.max_tokens,
        )
        cost = (
            gen["input_tokens"] * model_cfg.input_cost_per_mtok
            + gen["output_tokens"] * model_cfg.output_cost_per_mtok
        ) / 1_000_000
        base.answer = gen["response"]
        base.generation_metrics = {
            "input_tokens": gen["input_tokens"],
            "output_tokens": gen["output_tokens"],
            "stop_reason": gen.get("stop_reason"),
            "time_to_first_token_ms": gen.get("time_to_first_token_ms"),
            "total_latency_ms": gen["total_latency_ms"],
            "cost_per_query": round(cost, 6),
            "per_chunk_context_scores": list(per_chunk_scores),
        }
    except Exception as e:
        base.error = f"generation failed: {e}"
        base.generation_metrics = {"per_chunk_context_scores": list(per_chunk_scores)}
        return base

    base.context_relevance_score = context_score
    if context_score is not None:
        base.context_relevance_reason = (
            f"Mean of {len(per_chunk_scores)} per-chunk context-relevance scores"
        )

    if judge_client and base.answer:
        try:
            faith = judge_module.judge_citation_faithfulness(
                client=judge_client,
                answer=base.answer,
                cited_chunks=[c.content for c in chunks],
                judge_model=judge_model,
                temperature=judge_temperature,
            )
            base.citation_faithfulness_score = faith["score"]
            base.citation_faithfulness_reason = faith["reason"]
            base.judge_model = faith["judge_model"]
        except Exception:
            base.judge_model = judge_module._resolve_judge_model(judge_model)
    else:
        base.judge_model = judge_module._resolve_judge_model(judge_model)

    return base


def check_retrieval_drift(
    baseline: dict[str, Any],
    current: dict[str, Any],
    tolerance: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare two retrieval/RAG result summaries and flag regressions.

    Quality metrics regress when ``current - baseline < -tolerance``; latency
    metrics regress when ``current - baseline > +tolerance``.
    """
    tol = tolerance or {}
    quality_metrics = (
        ("avg_recall_at_k", tol.get("recall_at_k", 0.05)),
        ("avg_precision_at_k", tol.get("precision_at_k", 0.05)),
        ("avg_mrr", tol.get("mrr", 0.05)),
        ("avg_ndcg_at_k", tol.get("ndcg_at_k", 0.05)),
        ("avg_context_relevance", tol.get("context_relevance", 0.05)),
        ("avg_citation_faithfulness", tol.get("citation_faithfulness", 0.05)),
    )
    latency_metric = ("p95_retrieval_latency_ms", tol.get("p95_latency_ms", 50))

    def lookup(summary: dict[str, Any], name: str) -> float | None:
        agg = summary.get("aggregate")
        if isinstance(agg, dict) and agg.get(name) is not None:
            return float(agg[name])
        per_model = [
            v[name]
            for v in summary.get("overall", {}).values()
            if isinstance(v, dict) and v.get(name) is not None
        ]
        if per_model:
            return round(sum(per_model) / len(per_model), 4)
        return None

    metrics_out: dict[str, Any] = {}
    has_regressions = False

    for name, threshold in quality_metrics:
        b = lookup(baseline, name)
        c = lookup(current, name)
        if b is None or c is None:
            continue
        delta = round(c - b, 4)
        regression = delta < -float(threshold)
        if regression:
            has_regressions = True
        metrics_out[name] = {
            "baseline": b,
            "current": c,
            "delta": delta,
            "regression": regression,
        }

    name, threshold = latency_metric
    b = lookup(baseline, name)
    c = lookup(current, name)
    if b is not None and c is not None:
        delta = round(c - b, 3)
        regression = delta > float(threshold)
        if regression:
            has_regressions = True
        metrics_out[name] = {
            "baseline": b,
            "current": c,
            "delta": delta,
            "regression": regression,
        }

    return {
        "baseline_timestamp": baseline.get("timestamp"),
        "current_timestamp": current.get("timestamp"),
        "has_regressions": has_regressions,
        "metrics": metrics_out,
    }
