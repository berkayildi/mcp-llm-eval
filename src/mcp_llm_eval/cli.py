"""CLI entry point for mcp-llm-eval: run, check, compare, comment subcommands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import engine
from .comparison import compare_runs
from .config import load_config
from .formatter import format_pr_comment
from .types import ModelConfig, RunSummary, ThresholdConfig


def cli_main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mcp-llm-eval",
        description="LLM evaluation gates for CI/CD pipelines",
    )
    subparsers = parser.add_subparsers(dest="command")

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run a full evaluation")
    run_parser.add_argument("--dataset", help="Path to evaluation dataset JSON")
    run_parser.add_argument("--config", help="Path to .eval-gate.yml config file")
    run_parser.add_argument("--models", help="JSON array of model config objects (alternative to --config)")
    run_parser.add_argument("--judge-model", help="Judge model name (default: gpt-4o-mini)")
    run_parser.add_argument("--output-dir", help="Directory to save results")

    # check subcommand
    check_parser = subparsers.add_parser("check", help="Check thresholds against results")
    check_parser.add_argument("--results", required=True, help="Path to evaluation summary JSON")
    check_parser.add_argument("--config", help="Path to .eval-gate.yml config file")
    check_parser.add_argument("--fail-under-faithfulness", type=float, help="Min avg faithfulness")
    check_parser.add_argument("--fail-under-relevance", type=float, help="Min avg relevance")
    check_parser.add_argument("--fail-over-ttft", type=int, help="Max p95 TTFT in ms")
    check_parser.add_argument("--fail-over-cost", type=float, help="Max cost per query")

    # compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare two evaluation runs")
    compare_parser.add_argument("--baseline", required=True, help="Path to baseline summary JSON")
    compare_parser.add_argument("--current", required=True, help="Path to current summary JSON")
    compare_parser.add_argument("--tolerance-faithfulness", type=float, help="Faithfulness tolerance")
    compare_parser.add_argument("--tolerance-relevance", type=float, help="Relevance tolerance")
    compare_parser.add_argument("--tolerance-ttft", type=int, help="TTFT tolerance in ms")
    compare_parser.add_argument("--tolerance-cost", type=float, help="Cost tolerance")

    # comment subcommand
    comment_parser = subparsers.add_parser("comment", help="Generate PR comment markdown")
    comment_parser.add_argument("--summary", required=True, help="Path to evaluation summary JSON")
    comment_parser.add_argument("--comparison", help="Path to comparison JSON")
    comment_parser.add_argument("--config", help="Path to .eval-gate.yml config file")
    comment_parser.add_argument("--output", help="Output file path (stdout if omitted)")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "check":
        _cmd_check(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "comment":
        _cmd_comment(args)


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the run subcommand."""
    config: dict[str, Any] = {}
    if args.config:
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    dataset_path = args.dataset or config.get("dataset")
    if not dataset_path:
        print("Error: --dataset is required (or set 'dataset' in config)", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or config.get("output_dir", "eval/results")

    # Models from --models JSON string or --config file
    models_raw = None
    if args.models:
        try:
            models_raw = json.loads(args.models)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid --models JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        models_raw = config.get("models")

    if not models_raw:
        print("Error: --models or --config with 'models' is required for run", file=sys.stderr)
        sys.exit(1)

    try:
        dataset = engine.load_dataset(dataset_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    model_configs = [ModelConfig.from_dict(m) for m in models_raw]

    judge_config = config.get("judge", {})
    if args.judge_model:
        judge_config["model"] = args.judge_model
    tracing_config = config.get("tracing")

    try:
        summary = engine.run_evaluation(
            dataset=dataset,
            models=model_configs,
            judge_config=judge_config,
            output_dir=output_dir,
            tracing_config=tracing_config,
        )
    except Exception as e:
        print(f"Error running evaluation: {e}", file=sys.stderr)
        sys.exit(1)

    # Write a latest_summary.json convenience symlink
    if output_dir:
        _write_latest_summary(output_dir, summary)

    # Print summary table to stdout
    _print_summary_table(summary)


def _cmd_check(args: argparse.Namespace) -> None:
    """Execute the check subcommand."""
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        summary = RunSummary.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error: Invalid results file: {e}", file=sys.stderr)
        sys.exit(1)

    # Build thresholds from config or CLI flags
    thresholds_dict: dict[str, Any] = {}

    if args.config:
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        thresholds_dict = config.get("thresholds", {})

    # CLI flags override config
    if args.fail_under_faithfulness is not None:
        thresholds_dict["avg_faithfulness"] = args.fail_under_faithfulness
    if args.fail_under_relevance is not None:
        thresholds_dict["avg_relevance"] = args.fail_under_relevance
    if args.fail_over_ttft is not None:
        thresholds_dict["p95_ttft_ms"] = args.fail_over_ttft
    if args.fail_over_cost is not None:
        thresholds_dict["max_cost_per_query"] = args.fail_over_cost

    thresholds = ThresholdConfig.from_dict(thresholds_dict)
    result = engine.check_thresholds(summary, thresholds)

    # Print results
    for check in result.per_metric:
        status = "PASS" if check.passed else "FAIL"
        print(f"  {check.metric}: {check.actual} (threshold: {check.threshold}) [{status}]")

    if result.overall_pass:
        print("\nAll thresholds passed.")
        sys.exit(0)
    else:
        print("\nThreshold check FAILED.", file=sys.stderr)
        sys.exit(1)


def _cmd_compare(args: argparse.Namespace) -> None:
    """Execute the compare subcommand."""
    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not current_path.exists():
        print(f"Error: Current file not found: {args.current}", file=sys.stderr)
        sys.exit(1)

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        current = json.loads(current_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    tolerance: dict[str, Any] = {}
    if args.tolerance_faithfulness is not None:
        tolerance["faithfulness"] = args.tolerance_faithfulness
    if args.tolerance_relevance is not None:
        tolerance["relevance"] = args.tolerance_relevance
    if args.tolerance_ttft is not None:
        tolerance["ttft_ms"] = args.tolerance_ttft
    if args.tolerance_cost is not None:
        tolerance["cost"] = args.tolerance_cost

    result = compare_runs(baseline, current, tolerance if tolerance else None)

    # Print comparison table
    print(f"Baseline: {result['baseline_timestamp']}")
    print(f"Current:  {result['current_timestamp']}")
    print()

    for model_name, metrics in result["models"].items():
        print(f"  {model_name}:")
        for metric, vals in metrics.items():
            flag = " \u26a0\ufe0f REGRESSION" if vals["regression"] else ""
            print(f"    {metric}: {vals['baseline']} -> {vals['current']} (delta: {vals['delta']}){flag}")
    print()

    if result["has_regressions"]:
        print("Regressions detected!", file=sys.stderr)
        sys.exit(1)
    else:
        print("No regressions detected.")
        sys.exit(0)


def _cmd_comment(args: argparse.Namespace) -> None:
    """Execute the comment subcommand."""
    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Error: Summary file not found: {args.summary}", file=sys.stderr)
        sys.exit(1)

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: Invalid summary file: {e}", file=sys.stderr)
        sys.exit(1)

    comparison = None
    if args.comparison:
        comp_path = Path(args.comparison)
        if not comp_path.exists():
            print(f"Error: Comparison file not found: {args.comparison}", file=sys.stderr)
            sys.exit(1)
        try:
            comparison = json.loads(comp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Error: Invalid comparison file: {e}", file=sys.stderr)
            sys.exit(1)

    thresholds = None
    if args.config:
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        thresholds = config.get("thresholds")

    markdown = format_pr_comment(summary, comparison=comparison, thresholds=thresholds)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"PR comment written to: {args.output}")
    else:
        print(markdown)


def _write_latest_summary(output_dir: str, summary: RunSummary) -> None:
    """Write a latest_summary.json file for CI convenience."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    latest_path = out / "latest_summary.json"
    latest_path.write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )


def _print_summary_table(summary: RunSummary) -> None:
    """Print a summary table to stdout."""
    print(f"\nEvaluation complete: {summary.timestamp}")
    print(f"Questions: {summary.total_questions} | Runs: {summary.total_model_runs} | "
          f"Errors: {summary.total_errors} | Cost: ${summary.total_estimated_cost:.4f}")
    print()

    if summary.overall:
        print(f"{'Model':<30} {'Faithfulness':>12} {'Relevance':>10} {'TTFT':>8} {'Cost':>10}")
        print("-" * 75)
        for model_name, metrics in summary.overall.items():
            faith = metrics.get("avg_faithfulness")
            rel = metrics.get("avg_relevance")
            ttft = metrics.get("avg_ttft_ms")
            cost = metrics.get("avg_cost_per_query")
            faith_str = f"{faith:.4f}" if faith is not None else "N/A"
            rel_str = f"{rel:.4f}" if rel is not None else "N/A"
            ttft_str = f"{int(ttft)}ms" if ttft is not None else "N/A"
            cost_str = f"${cost:.4f}" if cost is not None else "N/A"
            print(f"{model_name:<30} {faith_str:>12} {rel_str:>10} {ttft_str:>8} {cost_str:>10}")
