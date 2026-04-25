"""Tests for cli.py: argument parsing, subcommand routing, exit codes."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_llm_eval.cli import cli_main


# ---------------------------------------------------------------------------
# CLI argument parsing and routing
# ---------------------------------------------------------------------------


class TestCliRouting:
    def test_no_command_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main([])
        assert exc_info.value.code == 1

    def test_help_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--help"])
        assert exc_info.value.code == 0

    def test_run_help(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["run", "--help"])
        assert exc_info.value.code == 0

    def test_check_help(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["check", "--help"])
        assert exc_info.value.code == 0

    def test_compare_help(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["compare", "--help"])
        assert exc_info.value.code == 0

    def test_comment_help(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["comment", "--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


class TestCmdRun:
    def test_run_no_dataset_no_config(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["run"])
        assert exc_info.value.code == 1

    def test_run_no_models_in_config(self):
        content = "dataset: test.json\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(content)
            f.flush()
            try:
                with pytest.raises(SystemExit) as exc_info:
                    cli_main(["run", "--config", f.name])
                # Should fail with validation error about models
            finally:
                os.unlink(f.name)

    @patch("mcp_llm_eval.cli.engine")
    def test_run_success(self, mock_engine):
        config_content = """
dataset: /tmp/test_dataset.json
models:
  - provider: openai
    model: gpt-4o-mini
"""
        mock_engine.load_dataset.return_value = [MagicMock()]
        mock_summary = MagicMock()
        mock_summary.timestamp = "20250101_000000"
        mock_summary.total_questions = 1
        mock_summary.total_model_runs = 1
        mock_summary.total_errors = 0
        mock_summary.total_estimated_cost = 0.001
        mock_summary.overall = {}
        mock_summary.to_dict.return_value = {"timestamp": "20250101_000000"}
        mock_engine.run_evaluation.return_value = mock_summary

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(config_content)
            f.flush()
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    cli_main(["run", "--config", f.name, "--output-dir", tmpdir])
            finally:
                os.unlink(f.name)

    @patch("mcp_llm_eval.cli.engine")
    def test_run_with_dataset_override(self, mock_engine):
        config_content = """
models:
  - provider: openai
    model: gpt-4o-mini
"""
        mock_engine.load_dataset.return_value = [MagicMock()]
        mock_summary = MagicMock()
        mock_summary.timestamp = "20250101_000000"
        mock_summary.total_questions = 1
        mock_summary.total_model_runs = 1
        mock_summary.total_errors = 0
        mock_summary.total_estimated_cost = 0.001
        mock_summary.overall = {}
        mock_summary.to_dict.return_value = {"timestamp": "20250101_000000"}
        mock_engine.run_evaluation.return_value = mock_summary

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(config_content)
            f.flush()
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    cli_main(["run", "--config", f.name, "--dataset", "/tmp/data.json", "--output-dir", tmpdir])
                    mock_engine.load_dataset.assert_called_with("/tmp/data.json")
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# check subcommand
# ---------------------------------------------------------------------------


class TestCmdCheck:
    def _make_summary_file(self):
        data = {
            "timestamp": "20250101_000000",
            "total_questions": 3,
            "total_model_runs": 3,
            "total_errors": 0,
            "total_elapsed_sec": 10.0,
            "total_estimated_cost": 0.01,
            "judge_model": "gpt-4o-mini",
            "overall": {"model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85}},
            "results": [
                {"eval_id": "e1", "category": "factual", "model": "m", "provider": "p",
                 "time_to_first_token_ms": 100, "cost_per_query": 0.001},
            ],
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.flush()
        f.close()
        return f.name

    def test_check_pass(self):
        path = self._make_summary_file()
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli_main(["check", "--results", path, "--fail-under-faithfulness", "0.8"])
            assert exc_info.value.code == 0
        finally:
            os.unlink(path)

    def test_check_fail(self):
        path = self._make_summary_file()
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli_main(["check", "--results", path, "--fail-under-faithfulness", "0.95"])
            assert exc_info.value.code == 1
        finally:
            os.unlink(path)

    def test_check_file_not_found(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["check", "--results", "/nonexistent/file.json"])
        assert exc_info.value.code == 1

    def test_check_with_config(self):
        summary_path = self._make_summary_file()
        config_content = """
models:
  - provider: openai
    model: gpt-4o
thresholds:
  avg_faithfulness: 0.8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(config_content)
            f.flush()
            try:
                with pytest.raises(SystemExit) as exc_info:
                    cli_main(["check", "--results", summary_path, "--config", f.name])
                assert exc_info.value.code == 0
            finally:
                os.unlink(f.name)
                os.unlink(summary_path)

    def test_check_cli_overrides_config(self):
        summary_path = self._make_summary_file()
        config_content = """
models:
  - provider: openai
    model: gpt-4o
thresholds:
  avg_faithfulness: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(config_content)
            f.flush()
            try:
                # CLI flag says 0.95 which should fail
                with pytest.raises(SystemExit) as exc_info:
                    cli_main(["check", "--results", summary_path,
                             "--config", f.name,
                             "--fail-under-faithfulness", "0.95"])
                assert exc_info.value.code == 1
            finally:
                os.unlink(f.name)
                os.unlink(summary_path)

    def test_check_missing_required_results(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["check"])
        assert exc_info.value.code == 2  # argparse error


# ---------------------------------------------------------------------------
# compare subcommand
# ---------------------------------------------------------------------------


class TestCmdCompare:
    def _make_summary_file(self, overall):
        data = {"timestamp": "20250101_000000", "overall": overall}
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.flush()
        f.close()
        return f.name

    def test_compare_no_regression(self):
        baseline = self._make_summary_file({"m": {"avg_faithfulness": 0.9}})
        current = self._make_summary_file({"m": {"avg_faithfulness": 0.9}})
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli_main(["compare", "--baseline", baseline, "--current", current])
            assert exc_info.value.code == 0
        finally:
            os.unlink(baseline)
            os.unlink(current)

    def test_compare_with_regression(self):
        baseline = self._make_summary_file({"m": {"avg_faithfulness": 0.9}})
        current = self._make_summary_file({"m": {"avg_faithfulness": 0.7}})
        try:
            with pytest.raises(SystemExit) as exc_info:
                cli_main(["compare", "--baseline", baseline, "--current", current])
            assert exc_info.value.code == 1
        finally:
            os.unlink(baseline)
            os.unlink(current)

    def test_compare_file_not_found(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["compare", "--baseline", "/nonexistent", "--current", "/nonexistent2"])
        assert exc_info.value.code == 1

    def test_compare_with_custom_tolerance(self):
        baseline = self._make_summary_file({"m": {"avg_faithfulness": 0.9}})
        current = self._make_summary_file({"m": {"avg_faithfulness": 0.8}})  # drop of 0.1
        try:
            # Tolerance of 0.15, so no regression
            with pytest.raises(SystemExit) as exc_info:
                cli_main(["compare", "--baseline", baseline, "--current", current,
                         "--tolerance-faithfulness", "0.15"])
            assert exc_info.value.code == 0
        finally:
            os.unlink(baseline)
            os.unlink(current)

    def test_compare_missing_required_args(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["compare"])
        assert exc_info.value.code == 2  # argparse error


# ---------------------------------------------------------------------------
# comment subcommand
# ---------------------------------------------------------------------------


class TestCmdComment:
    def _make_summary_file(self):
        data = {
            "timestamp": "20250101_000000",
            "total_questions": 3,
            "total_model_runs": 3,
            "total_errors": 0,
            "total_estimated_cost": 0.01,
            "judge_model": "gpt-4o-mini",
            "overall": {
                "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
            },
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.flush()
        f.close()
        return f.name

    def test_comment_stdout(self, capsys):
        path = self._make_summary_file()
        try:
            cli_main(["comment", "--summary", path])
            captured = capsys.readouterr()
            assert "## LLM Eval Results" in captured.out
        finally:
            os.unlink(path)

    def test_comment_to_file(self):
        summary_path = self._make_summary_file()
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as out:
            out_path = out.name
        try:
            cli_main(["comment", "--summary", summary_path, "--output", out_path])
            content = Path(out_path).read_text()
            assert "## LLM Eval Results" in content
        finally:
            os.unlink(summary_path)
            os.unlink(out_path)

    def test_comment_file_not_found(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["comment", "--summary", "/nonexistent/file.json"])
        assert exc_info.value.code == 1

    def test_comment_with_comparison(self, capsys):
        summary_path = self._make_summary_file()
        comparison = {
            "regressions": [
                {"model": "model-a", "metric": "faithfulness", "baseline": 0.95, "current": 0.9, "delta": -0.05},
            ],
            "has_regressions": True,
        }
        comp_f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(comparison, comp_f)
        comp_f.flush()
        comp_f.close()
        try:
            cli_main(["comment", "--summary", summary_path, "--comparison", comp_f.name])
            captured = capsys.readouterr()
            assert "faithfulness" in captured.out
        finally:
            os.unlink(summary_path)
            os.unlink(comp_f.name)

    def test_comment_with_config_thresholds(self, capsys):
        summary_path = self._make_summary_file()
        config_content = """
models:
  - provider: openai
    model: gpt-4o
thresholds:
  avg_faithfulness: 0.8
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(config_content)
            f.flush()
            try:
                cli_main(["comment", "--summary", summary_path, "--config", f.name])
                captured = capsys.readouterr()
                assert "Threshold details" in captured.out
            finally:
                os.unlink(f.name)
                os.unlink(summary_path)

    def test_comment_missing_required_summary(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["comment"])
        assert exc_info.value.code == 2  # argparse error


# ---------------------------------------------------------------------------
# v0.5.0 — evaluate-retrieval subcommand
# ---------------------------------------------------------------------------


class TestCmdEvaluateRetrieval:
    def test_help(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["evaluate-retrieval", "--help"])
        assert exc_info.value.code == 0

    def test_missing_required_args(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["evaluate-retrieval"])
        assert exc_info.value.code == 2  # argparse error

    @patch("mcp_llm_eval.cli.engine")
    def test_happy_path(self, mock_engine, capsys):
        mock_engine.run_retrieval_evaluation.return_value = {
            "timestamp": "20260424_120000",
            "k": 5,
            "adapter": "bm25",
            "total_queries": 3,
            "total_errors": 0,
            "aggregate": {
                "avg_recall_at_k": 0.82,
                "avg_precision_at_k": 0.64,
                "avg_mrr": 0.76,
                "avg_ndcg_at_k": 0.79,
                "p50_retrieval_latency_ms": 3.2,
                "p95_retrieval_latency_ms": 7.8,
            },
        }
        cli_main([
            "evaluate-retrieval",
            "--dataset", "ds.jsonl",
            "--corpus", "corpus.jsonl",
            "--k", "5",
        ])
        out = capsys.readouterr().out
        assert "Retrieval evaluation" in out
        assert "0.8200" in out
        mock_engine.run_retrieval_evaluation.assert_called_once()

    @patch("mcp_llm_eval.cli.engine")
    def test_engine_error_exits_1(self, mock_engine):
        mock_engine.run_retrieval_evaluation.return_value = {"error": "boom"}
        with pytest.raises(SystemExit) as exc_info:
            cli_main([
                "evaluate-retrieval",
                "--dataset", "ds.jsonl",
                "--corpus", "corpus.jsonl",
            ])
        assert exc_info.value.code == 1

    @patch("mcp_llm_eval.cli.engine")
    def test_threshold_breach_exits_1(self, mock_engine, capsys):
        # Engine returns a summary; thresholds in config break.
        mock_engine.run_retrieval_evaluation.return_value = {
            "timestamp": "t", "k": 5, "adapter": "bm25",
            "total_queries": 1, "total_questions": 1,
            "total_model_runs": 1, "total_errors": 0,
            "total_elapsed_sec": 0.0, "total_estimated_cost": 0.0,
            "judge_model": None,
            "aggregate": {"avg_recall_at_k": 0.50},
            "overall": {"_retrieval": {"avg_recall_at_k": 0.50}},
            "by_category": {},
            "results": [],
            "per_query": [],
        }
        # Make engine.check_thresholds work normally
        from mcp_llm_eval import engine as real_engine
        mock_engine.check_thresholds.side_effect = real_engine.check_thresholds

        config_yaml = """
models:
  - provider: openai
    model: gpt-4o-mini
thresholds:
  avg_faithfulness: 0.99
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(config_yaml)
            f.flush()
            try:
                with pytest.raises(SystemExit) as exc_info:
                    cli_main([
                        "evaluate-retrieval",
                        "--dataset", "ds.jsonl",
                        "--corpus", "corpus.jsonl",
                        "--config", f.name,
                    ])
                assert exc_info.value.code == 1
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# v0.5.0 — evaluate-rag subcommand
# ---------------------------------------------------------------------------


class TestCmdEvaluateRag:
    def _rag_summary(self):
        return {
            "timestamp": "t",
            "k": 5,
            "adapter": "bm25",
            "total_queries": 1,
            "total_model_runs": 1,
            "total_errors": 0,
            "total_elapsed_sec": 1.0,
            "total_estimated_cost": 0.001,
            "judge_model": "gpt-4o-mini",
            "overall": {"gpt-4o-mini": {
                "avg_recall_at_k": 1.0,
                "avg_ndcg_at_k": 1.0,
                "avg_context_relevance": 0.9,
                "avg_citation_faithfulness": 0.85,
                "avg_ttft_ms": 100,
                "avg_cost_per_query": 0.001,
            }},
            "by_category": {},
            "per_query": [],
            "results": [],
        }

    def test_no_models_no_config_exits_1(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli_main([
                "evaluate-rag",
                "--dataset", "ds.jsonl",
                "--corpus", "corpus.jsonl",
            ])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "--model" in err and "--config" in err

    def test_invalid_model_format(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli_main([
                "evaluate-rag",
                "--dataset", "ds.jsonl",
                "--corpus", "corpus.jsonl",
                "--model", "no-colon-format",
            ])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "provider:model" in err

    @patch("mcp_llm_eval.cli.engine")
    def test_model_flag_happy_path(self, mock_engine, capsys):
        mock_engine.run_rag_evaluation.return_value = self._rag_summary()
        cli_main([
            "evaluate-rag",
            "--dataset", "ds.jsonl",
            "--corpus", "corpus.jsonl",
            "--model", "openai:gpt-4o-mini",
        ])
        call_kwargs = mock_engine.run_rag_evaluation.call_args[1]
        assert call_kwargs["models"][0].provider == "openai"
        assert call_kwargs["models"][0].model == "gpt-4o-mini"

    @patch("mcp_llm_eval.cli.engine")
    def test_cli_model_overrides_config_models(self, mock_engine):
        mock_engine.run_rag_evaluation.return_value = self._rag_summary()
        config_yaml = """
models:
  - provider: anthropic
    model: claude-sonnet-4-6
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(config_yaml)
            f.flush()
            try:
                cli_main([
                    "evaluate-rag",
                    "--dataset", "ds.jsonl",
                    "--corpus", "corpus.jsonl",
                    "--config", f.name,
                    "--model", "openai:gpt-4o-mini",
                ])
                models_arg = mock_engine.run_rag_evaluation.call_args[1]["models"]
                # CLI model wins, no merging
                assert len(models_arg) == 1
                assert models_arg[0].provider == "openai"
                assert models_arg[0].model == "gpt-4o-mini"
            finally:
                os.unlink(f.name)

    @patch("mcp_llm_eval.cli.engine")
    def test_judge_model_flag_overrides_config(self, mock_engine):
        mock_engine.run_rag_evaluation.return_value = self._rag_summary()
        config_yaml = """
models:
  - provider: openai
    model: gpt-4o-mini
judge:
  model: gpt-4o-mini
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(config_yaml)
            f.flush()
            try:
                cli_main([
                    "evaluate-rag",
                    "--dataset", "ds.jsonl",
                    "--corpus", "corpus.jsonl",
                    "--config", f.name,
                    "--judge-model", "gpt-4o",
                ])
                jc = mock_engine.run_rag_evaluation.call_args[1]["judge_config"]
                assert jc["model"] == "gpt-4o"
            finally:
                os.unlink(f.name)
