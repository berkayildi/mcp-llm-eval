"""MCP server that exposes LLM evaluation gates as reusable CI/CD primitives."""

import json
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from . import engine
from .comparison import compare_runs
from .formatter import format_pr_comment
from .types import ModelConfig, RunSummary, ThresholdConfig

app = Server("mcp-llm-eval")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_evaluation",
            description=(
                "Run an LLM evaluation: load a dataset, query models via streaming, "
                "score responses with an LLM-as-judge, and return per-question scores, "
                "aggregate summary, and pass/fail status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the JSON evaluation dataset file.",
                    },
                    "models": {
                        "type": "array",
                        "description": "Models to evaluate.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "provider": {
                                    "type": "string",
                                    "enum": ["anthropic", "openai", "google"],
                                    "description": "LLM provider.",
                                },
                                "model": {
                                    "type": "string",
                                    "description": "Model identifier.",
                                },
                                "max_tokens": {
                                    "type": "integer",
                                    "description": "Maximum output tokens (default 500).",
                                    "default": 500,
                                },
                            },
                            "required": ["provider", "model"],
                        },
                    },
                    "judge": {
                        "type": "object",
                        "description": "Judge configuration (optional).",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Judge provider (default openai).",
                            },
                            "model": {
                                "type": "string",
                                "description": "Judge model (default gpt-4o-mini).",
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Judge temperature (default 0).",
                            },
                        },
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save results JSON files.",
                    },
                    "tracing": {
                        "type": "object",
                        "description": "Optional tracing configuration.",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "project": {"type": "string"},
                            "endpoint": {"type": "string"},
                        },
                    },
                },
                "required": ["dataset_path", "models"],
            },
        ),
        types.Tool(
            name="check_thresholds",
            description=(
                "Check evaluation results against quality gate thresholds. "
                "Returns pass/fail per metric and overall gate status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "results_path": {
                        "type": "string",
                        "description": "Path to an evaluation results JSON file (summary).",
                    },
                    "thresholds": {
                        "type": "object",
                        "description": "Quality gate thresholds.",
                        "properties": {
                            "avg_faithfulness": {
                                "type": "number",
                                "description": "Minimum average faithfulness score (0-1).",
                            },
                            "avg_relevance": {
                                "type": "number",
                                "description": "Minimum average relevance score (0-1).",
                            },
                            "p95_ttft_ms": {
                                "type": "integer",
                                "description": "Maximum p95 time-to-first-token in ms.",
                            },
                            "max_cost_per_query": {
                                "type": "number",
                                "description": "Maximum cost per query in USD.",
                            },
                        },
                    },
                },
                "required": ["results_path", "thresholds"],
            },
        ),
        types.Tool(
            name="list_evaluations",
            description=(
                "List past evaluation runs in a directory. Returns metadata for each run: "
                "timestamp, dataset, models, pass/fail, and cost."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "results_dir": {
                        "type": "string",
                        "description": "Directory containing evaluation result files.",
                    },
                },
                "required": ["results_dir"],
            },
        ),
        types.Tool(
            name="get_evaluation",
            description=(
                "Retrieve the full details of a specific evaluation run: per-question "
                "per-model scores, responses, and judge reasoning."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "results_path": {
                        "type": "string",
                        "description": "Path to a specific evaluation result file.",
                    },
                },
                "required": ["results_path"],
            },
        ),
        types.Tool(
            name="compare_runs",
            description=(
                "Compare two evaluation runs and detect regressions. "
                "Flags metrics that worsened beyond configurable tolerance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "baseline_path": {
                        "type": "string",
                        "description": "Path to baseline evaluation summary JSON.",
                    },
                    "current_path": {
                        "type": "string",
                        "description": "Path to current evaluation summary JSON.",
                    },
                    "tolerance": {
                        "type": "object",
                        "description": "Per-metric regression tolerance.",
                        "properties": {
                            "faithfulness": {
                                "type": "number",
                                "description": "Allowed drop in avg faithfulness (default 0.05).",
                            },
                            "relevance": {
                                "type": "number",
                                "description": "Allowed drop in avg relevance (default 0.05).",
                            },
                            "ttft_ms": {
                                "type": "integer",
                                "description": "Allowed increase in avg TTFT ms (default 200).",
                            },
                            "cost": {
                                "type": "number",
                                "description": "Allowed increase in avg cost per query (default 0.005).",
                            },
                        },
                    },
                },
                "required": ["baseline_path", "current_path"],
            },
        ),
        types.Tool(
            name="evaluate_retrieval",
            description=(
                "Run retrieval metrics (recall@k, precision@k, MRR, nDCG@k) against "
                "a labelled dataset with a configurable retrieval adapter. Returns "
                "per-query metrics, dataset-level aggregate, and p50/p95 retrieval "
                "latency."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to JSONL dataset with relevant_chunk_ids on each entry.",
                    },
                    "corpus_path": {
                        "type": "string",
                        "description": "Path to JSONL corpus file.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Top-k cutoff for all metrics (default 5).",
                        "default": 5,
                    },
                    "adapter": {
                        "type": "string",
                        "enum": ["bm25"],
                        "description": "Retrieval adapter to use (default bm25).",
                        "default": "bm25",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save results (optional).",
                    },
                },
                "required": ["dataset_path", "corpus_path"],
            },
        ),
        types.Tool(
            name="evaluate_rag_end_to_end",
            description=(
                "Run the full RAG pipeline: retrieve chunks, generate answers using "
                "the retrieved chunks as context, and score with context_relevance "
                "and citation_faithfulness judges. Returns retrieval metrics, "
                "generation metrics, and judge scores per query, plus an aggregate."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_path": {"type": "string"},
                    "corpus_path": {"type": "string"},
                    "models": {
                        "type": "array",
                        "description": "Models to evaluate.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "provider": {
                                    "type": "string",
                                    "enum": ["anthropic", "openai", "google"],
                                },
                                "model": {"type": "string"},
                                "max_tokens": {"type": "integer", "default": 500},
                                "input_cost_per_mtok": {"type": "number", "default": 0.0},
                                "output_cost_per_mtok": {"type": "number", "default": 0.0},
                            },
                            "required": ["provider", "model"],
                        },
                    },
                    "k": {"type": "integer", "default": 5},
                    "adapter": {
                        "type": "string",
                        "enum": ["bm25"],
                        "default": "bm25",
                    },
                    "judge": {
                        "type": "object",
                        "properties": {
                            "provider": {"type": "string"},
                            "model": {"type": "string"},
                            "temperature": {"type": "number"},
                        },
                    },
                    "output_dir": {"type": "string"},
                },
                "required": ["dataset_path", "corpus_path", "models"],
            },
        ),
        types.Tool(
            name="check_retrieval_drift",
            description=(
                "Compare two retrieval evaluation result files and detect drift. "
                "Flags metrics that have regressed beyond configurable tolerance. "
                "Takes two result-set paths; does not persist history itself."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "baseline_path": {"type": "string"},
                    "current_path": {"type": "string"},
                    "tolerance": {
                        "type": "object",
                        "properties": {
                            "recall_at_k": {"type": "number", "default": 0.05},
                            "precision_at_k": {"type": "number", "default": 0.05},
                            "mrr": {"type": "number", "default": 0.05},
                            "ndcg_at_k": {"type": "number", "default": 0.05},
                            "context_relevance": {"type": "number", "default": 0.05},
                            "citation_faithfulness": {"type": "number", "default": 0.05},
                            "p95_latency_ms": {"type": "integer", "default": 50},
                        },
                    },
                },
                "required": ["baseline_path", "current_path"],
            },
        ),
        types.Tool(
            name="simulate_poisoned_corpus",
            description=(
                "[STUB - not implemented in v0.5.0] Inject poisoned chunks into a "
                "corpus and re-run retrieval evaluation. Returns a clear "
                "not-implemented response."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_path": {"type": "string"},
                    "corpus_path": {"type": "string"},
                    "poisoning_strategy": {
                        "type": "string",
                        "enum": ["contradiction", "injection", "noise"],
                    },
                    "poison_ratio": {"type": "number"},
                },
                "required": ["dataset_path", "corpus_path", "poisoning_strategy"],
            },
        ),
        types.Tool(
            name="format_pr_comment",
            description=(
                "Generate a markdown PR comment from evaluation results. "
                "Includes results table, regression details, and threshold status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "summary_path": {
                        "type": "string",
                        "description": "Path to evaluation summary JSON.",
                    },
                    "comparison_path": {
                        "type": "string",
                        "description": "Path to compare_runs output JSON (optional).",
                    },
                    "thresholds": {
                        "type": "object",
                        "description": "Quality gate thresholds for pass/fail badges.",
                        "properties": {
                            "avg_faithfulness": {
                                "type": "number",
                                "description": "Minimum average faithfulness score (0-1).",
                            },
                            "avg_relevance": {
                                "type": "number",
                                "description": "Minimum average relevance score (0-1).",
                            },
                            "p95_ttft_ms": {
                                "type": "integer",
                                "description": "Maximum p95 time-to-first-token in ms.",
                            },
                            "max_cost_per_query": {
                                "type": "number",
                                "description": "Maximum cost per query in USD.",
                            },
                        },
                    },
                },
                "required": ["summary_path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name == "run_evaluation":
        return await _run_evaluation(arguments)
    if name == "check_thresholds":
        return await _check_thresholds(arguments)
    if name == "list_evaluations":
        return await _list_evaluations(arguments)
    if name == "get_evaluation":
        return await _get_evaluation(arguments)
    if name == "compare_runs":
        return await _compare_runs(arguments)
    if name == "format_pr_comment":
        return await _format_pr_comment(arguments)
    if name == "evaluate_retrieval":
        return await _evaluate_retrieval(arguments)
    if name == "evaluate_rag_end_to_end":
        return await _evaluate_rag_end_to_end(arguments)
    if name == "check_retrieval_drift":
        return await _check_retrieval_drift_tool(arguments)
    if name == "simulate_poisoned_corpus":
        return await _simulate_poisoned_corpus(arguments)
    raise ValueError(f"Unknown tool: {name}")


async def _run_evaluation(arguments: dict[str, Any]) -> list[types.TextContent]:
    dataset_path = arguments["dataset_path"]
    models_raw = arguments["models"]
    judge_config = arguments.get("judge")
    output_dir = arguments.get("output_dir")
    tracing_config = arguments.get("tracing")

    try:
        dataset = engine.load_dataset(dataset_path)
    except (FileNotFoundError, ValueError) as e:
        return [types.TextContent(type="text", text=f"Error loading dataset: {e}")]

    model_configs = []
    for m in models_raw:
        model_configs.append(ModelConfig(
            provider=m["provider"],
            model=m["model"],
            max_tokens=m.get("max_tokens", 500),
            input_cost_per_mtok=m.get("input_cost_per_mtok", 0.0),
            output_cost_per_mtok=m.get("output_cost_per_mtok", 0.0),
        ))

    try:
        summary = engine.run_evaluation(
            dataset=dataset,
            models=model_configs,
            judge_config=judge_config,
            output_dir=output_dir,
            tracing_config=tracing_config,
        )
    except Exception as e:
        return [types.TextContent(type="text", text=f"Evaluation error: {e}")]

    return [types.TextContent(type="text", text=json.dumps(summary.to_dict(), indent=2))]


async def _check_thresholds(arguments: dict[str, Any]) -> list[types.TextContent]:
    results_path = arguments["results_path"]
    thresholds_raw = arguments["thresholds"]

    p = Path(results_path)
    if not p.exists():
        return [types.TextContent(type="text", text=f"File not found: {results_path}")]

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        summary = RunSummary.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
        return [types.TextContent(type="text", text=f"Error reading results: {e}")]

    thresholds = ThresholdConfig.from_dict(thresholds_raw)
    result = engine.check_thresholds(summary, thresholds)

    return [types.TextContent(type="text", text=json.dumps(result.to_dict(), indent=2))]


async def _list_evaluations(arguments: dict[str, Any]) -> list[types.TextContent]:
    results_dir = arguments["results_dir"]
    p = Path(results_dir)

    if not p.exists():
        return [types.TextContent(type="text", text=f"Directory not found: {results_dir}")]

    runs = []
    for f in sorted(p.glob("*_summary.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            runs.append({
                "file": str(f),
                "timestamp": data.get("timestamp"),
                "total_questions": data.get("total_questions"),
                "total_model_runs": data.get("total_model_runs"),
                "total_errors": data.get("total_errors", 0),
                "total_estimated_cost": data.get("total_estimated_cost"),
                "judge_model": data.get("judge_model"),
                "models": list(data.get("overall", {}).keys()),
            })
        except (json.JSONDecodeError, OSError):
            continue

    return [types.TextContent(type="text", text=json.dumps(runs, indent=2))]


async def _get_evaluation(arguments: dict[str, Any]) -> list[types.TextContent]:
    results_path = arguments["results_path"]
    p = Path(results_path)

    if not p.exists():
        return [types.TextContent(type="text", text=f"File not found: {results_path}")]

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [types.TextContent(type="text", text=f"Error reading file: {e}")]

    return [types.TextContent(type="text", text=json.dumps(data, indent=2))]


async def _compare_runs(arguments: dict[str, Any]) -> list[types.TextContent]:
    baseline_path = arguments["baseline_path"]
    current_path = arguments["current_path"]
    tolerance = arguments.get("tolerance")

    bp = Path(baseline_path)
    cp = Path(current_path)

    if not bp.exists():
        return [types.TextContent(type="text", text=f"File not found: {baseline_path}")]
    if not cp.exists():
        return [types.TextContent(type="text", text=f"File not found: {current_path}")]

    try:
        baseline = json.loads(bp.read_text(encoding="utf-8"))
        current = json.loads(cp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [types.TextContent(type="text", text=f"Error reading files: {e}")]

    result = compare_runs(baseline, current, tolerance)
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _format_pr_comment(arguments: dict[str, Any]) -> list[types.TextContent]:
    summary_path = arguments["summary_path"]
    comparison_path = arguments.get("comparison_path")
    thresholds = arguments.get("thresholds")

    sp = Path(summary_path)
    if not sp.exists():
        return [types.TextContent(type="text", text=f"File not found: {summary_path}")]

    try:
        summary = json.loads(sp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [types.TextContent(type="text", text=f"Error reading summary: {e}")]

    comparison = None
    if comparison_path:
        cp = Path(comparison_path)
        if not cp.exists():
            return [types.TextContent(type="text", text=f"File not found: {comparison_path}")]
        try:
            comparison = json.loads(cp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            return [types.TextContent(type="text", text=f"Error reading comparison: {e}")]

    markdown = format_pr_comment(summary, comparison=comparison, thresholds=thresholds)
    return [types.TextContent(type="text", text=markdown)]


async def _evaluate_retrieval(arguments: dict[str, Any]) -> list[types.TextContent]:
    dataset_path = arguments["dataset_path"]
    corpus_path = arguments["corpus_path"]
    k = int(arguments.get("k", 5))
    adapter = arguments.get("adapter", "bm25")
    output_dir = arguments.get("output_dir")

    try:
        summary = engine.run_retrieval_evaluation(
            dataset_path=dataset_path,
            corpus_path=corpus_path,
            k=k,
            adapter=adapter,
            output_dir=output_dir,
        )
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {e}")]

    if "error" in summary:
        return [types.TextContent(type="text", text=f"Error: {summary['error']}")]

    return [types.TextContent(type="text", text=json.dumps(summary, indent=2))]


async def _evaluate_rag_end_to_end(
    arguments: dict[str, Any],
) -> list[types.TextContent]:
    dataset_path = arguments["dataset_path"]
    corpus_path = arguments["corpus_path"]
    models_raw = arguments["models"]
    k = int(arguments.get("k", 5))
    adapter = arguments.get("adapter", "bm25")
    judge_config = arguments.get("judge")
    output_dir = arguments.get("output_dir")

    model_configs = [ModelConfig.from_dict(m) for m in models_raw]

    try:
        summary = engine.run_rag_evaluation(
            dataset_path=dataset_path,
            corpus_path=corpus_path,
            models=model_configs,
            k=k,
            adapter=adapter,
            judge_config=judge_config,
            output_dir=output_dir,
        )
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {e}")]

    if "error" in summary:
        return [types.TextContent(type="text", text=f"Error: {summary['error']}")]

    return [types.TextContent(type="text", text=json.dumps(summary, indent=2))]


async def _check_retrieval_drift_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent]:
    baseline_path = arguments["baseline_path"]
    current_path = arguments["current_path"]
    tolerance = arguments.get("tolerance")

    bp = Path(baseline_path)
    cp = Path(current_path)
    if not bp.exists():
        return [types.TextContent(type="text", text=f"File not found: {baseline_path}")]
    if not cp.exists():
        return [types.TextContent(type="text", text=f"File not found: {current_path}")]

    try:
        baseline = json.loads(bp.read_text(encoding="utf-8"))
        current = json.loads(cp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [types.TextContent(type="text", text=f"Error reading files: {e}")]

    result = engine.check_retrieval_drift(baseline, current, tolerance)
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _simulate_poisoned_corpus(
    arguments: dict[str, Any],
) -> list[types.TextContent]:
    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "not_implemented",
            "message": (
                "simulate_poisoned_corpus is a reserved stub in v0.5.0. "
                "Implementation lands in v0.6.x."
            ),
            "accepted_arguments": list(arguments.keys()),
        }, indent=2),
    )]


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        from mcp_llm_eval.cli import cli_main
        cli_main()
    else:
        import asyncio

        async def _run():
            async with stdio_server() as (read_stream, write_stream):
                await app.run(read_stream, write_stream, app.create_initialization_options())

        asyncio.run(_run())


if __name__ == "__main__":
    main()
