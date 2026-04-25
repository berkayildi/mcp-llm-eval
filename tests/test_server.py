"""MCP tool integration tests: registration, routing, input validation, error handling."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_llm_eval.server import (
    _check_retrieval_drift_tool,
    _check_thresholds,
    _compare_runs,
    _evaluate_rag_end_to_end,
    _evaluate_retrieval,
    _format_pr_comment,
    _get_evaluation,
    _list_evaluations,
    _run_evaluation,
    _simulate_poisoned_corpus,
    app,
    call_tool,
    list_tools,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DATASET = FIXTURES_DIR / "sample_dataset.json"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    async def test_list_tools_returns_ten(self):
        tools = await list_tools()
        assert len(tools) == 10

    async def test_tool_names(self):
        tools = await list_tools()
        names = {t.name for t in tools}
        assert names == {
            # v0.4.x
            "run_evaluation", "check_thresholds", "list_evaluations",
            "get_evaluation", "compare_runs", "format_pr_comment",
            # v0.5.0
            "evaluate_retrieval", "evaluate_rag_end_to_end",
            "check_retrieval_drift", "simulate_poisoned_corpus",
        }

    async def test_run_evaluation_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "run_evaluation")
        schema = tool.inputSchema
        assert "dataset_path" in schema["properties"]
        assert "models" in schema["properties"]
        assert "dataset_path" in schema["required"]
        assert "models" in schema["required"]

    async def test_check_thresholds_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "check_thresholds")
        schema = tool.inputSchema
        assert "results_path" in schema["properties"]
        assert "thresholds" in schema["properties"]
        assert "results_path" in schema["required"]

    async def test_list_evaluations_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "list_evaluations")
        schema = tool.inputSchema
        assert "results_dir" in schema["properties"]
        assert "results_dir" in schema["required"]

    async def test_get_evaluation_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "get_evaluation")
        schema = tool.inputSchema
        assert "results_path" in schema["properties"]
        assert "results_path" in schema["required"]

    async def test_run_evaluation_models_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "run_evaluation")
        models_schema = tool.inputSchema["properties"]["models"]
        assert models_schema["type"] == "array"
        item_props = models_schema["items"]["properties"]
        assert "provider" in item_props
        assert "model" in item_props

    async def test_run_evaluation_optional_fields(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "run_evaluation")
        schema = tool.inputSchema
        assert "judge" in schema["properties"]
        assert "output_dir" in schema["properties"]
        assert "tracing" in schema["properties"]

    async def test_check_thresholds_threshold_properties(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "check_thresholds")
        thresh_props = tool.inputSchema["properties"]["thresholds"]["properties"]
        assert "avg_faithfulness" in thresh_props
        assert "avg_relevance" in thresh_props
        assert "p95_ttft_ms" in thresh_props
        assert "max_cost_per_query" in thresh_props

    async def test_all_tools_have_descriptions(self):
        tools = await list_tools()
        for tool in tools:
            assert len(tool.description) > 10


# ---------------------------------------------------------------------------
# Tool routing
# ---------------------------------------------------------------------------


class TestToolRouting:
    async def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            await call_tool("nonexistent_tool", {})

    @patch("mcp_llm_eval.server.engine")
    async def test_routes_to_run_evaluation(self, mock_engine):
        mock_engine.load_dataset.side_effect = FileNotFoundError("not found")
        result = await call_tool("run_evaluation", {
            "dataset_path": "/fake/path.json",
            "models": [{"provider": "openai", "model": "gpt-4o"}],
        })
        assert "Error" in result[0].text

    async def test_routes_to_check_thresholds(self):
        result = await call_tool("check_thresholds", {
            "results_path": "/nonexistent/results.json",
            "thresholds": {"avg_faithfulness": 0.8},
        })
        assert "not found" in result[0].text.lower() or "File not found" in result[0].text

    async def test_routes_to_list_evaluations(self):
        result = await call_tool("list_evaluations", {
            "results_dir": "/nonexistent/dir",
        })
        assert "not found" in result[0].text.lower() or "Directory not found" in result[0].text

    async def test_routes_to_get_evaluation(self):
        result = await call_tool("get_evaluation", {
            "results_path": "/nonexistent/file.json",
        })
        assert "not found" in result[0].text.lower() or "File not found" in result[0].text


# ---------------------------------------------------------------------------
# _run_evaluation
# ---------------------------------------------------------------------------


class TestRunEvaluationTool:
    async def test_dataset_not_found(self):
        result = await _run_evaluation({
            "dataset_path": "/nonexistent/dataset.json",
            "models": [{"provider": "openai", "model": "gpt-4o"}],
        })
        assert "Error loading dataset" in result[0].text

    @patch("mcp_llm_eval.server.engine")
    async def test_evaluation_error(self, mock_engine):
        mock_engine.load_dataset.return_value = [MagicMock()]
        mock_engine.run_evaluation.side_effect = RuntimeError("boom")
        result = await _run_evaluation({
            "dataset_path": str(SAMPLE_DATASET),
            "models": [{"provider": "openai", "model": "gpt-4o"}],
        })
        assert "Evaluation error" in result[0].text

    @patch("mcp_llm_eval.server.engine")
    async def test_successful_evaluation(self, mock_engine):
        mock_summary = MagicMock()
        mock_summary.to_dict.return_value = {
            "timestamp": "20250101_000000",
            "total_questions": 3,
            "total_model_runs": 3,
        }
        mock_engine.load_dataset.return_value = [MagicMock()]
        mock_engine.run_evaluation.return_value = mock_summary
        result = await _run_evaluation({
            "dataset_path": str(SAMPLE_DATASET),
            "models": [{"provider": "openai", "model": "gpt-4o"}],
        })
        data = json.loads(result[0].text)
        assert data["total_questions"] == 3

    @patch("mcp_llm_eval.server.engine")
    async def test_passes_judge_config(self, mock_engine):
        mock_summary = MagicMock()
        mock_summary.to_dict.return_value = {}
        mock_engine.load_dataset.return_value = [MagicMock()]
        mock_engine.run_evaluation.return_value = mock_summary
        await _run_evaluation({
            "dataset_path": str(SAMPLE_DATASET),
            "models": [{"provider": "openai", "model": "gpt-4o"}],
            "judge": {"model": "gpt-4o", "temperature": 0.5},
        })
        call_kwargs = mock_engine.run_evaluation.call_args[1]
        assert call_kwargs["judge_config"]["model"] == "gpt-4o"

    @patch("mcp_llm_eval.server.engine")
    async def test_passes_output_dir(self, mock_engine):
        mock_summary = MagicMock()
        mock_summary.to_dict.return_value = {}
        mock_engine.load_dataset.return_value = [MagicMock()]
        mock_engine.run_evaluation.return_value = mock_summary
        await _run_evaluation({
            "dataset_path": str(SAMPLE_DATASET),
            "models": [{"provider": "openai", "model": "gpt-4o"}],
            "output_dir": "/tmp/results",
        })
        call_kwargs = mock_engine.run_evaluation.call_args[1]
        assert call_kwargs["output_dir"] == "/tmp/results"


# ---------------------------------------------------------------------------
# _check_thresholds
# ---------------------------------------------------------------------------


class TestCheckThresholdsTool:
    async def test_file_not_found(self):
        result = await _check_thresholds({
            "results_path": "/nonexistent/results.json",
            "thresholds": {"avg_faithfulness": 0.8},
        })
        assert "not found" in result[0].text.lower() or "File not found" in result[0].text

    async def test_valid_results(self):
        summary_data = {
            "timestamp": "20250101_000000",
            "total_questions": 3,
            "total_model_runs": 3,
            "total_errors": 0,
            "total_elapsed_sec": 10.0,
            "total_estimated_cost": 0.01,
            "judge_model": "gpt-4o-mini",
            "overall": {"model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85}},
            "results": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(summary_data, f)
            f.flush()
            result = await _check_thresholds({
                "results_path": f.name,
                "thresholds": {"avg_faithfulness": 0.8},
            })
        data = json.loads(result[0].text)
        assert data["overall_pass"] is True

    async def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            f.flush()
            result = await _check_thresholds({
                "results_path": f.name,
                "thresholds": {},
            })
        assert "Error" in result[0].text


# ---------------------------------------------------------------------------
# _list_evaluations
# ---------------------------------------------------------------------------


class TestListEvaluationsTool:
    async def test_directory_not_found(self):
        result = await _list_evaluations({"results_dir": "/nonexistent/dir"})
        assert "not found" in result[0].text.lower() or "Directory not found" in result[0].text

    async def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await _list_evaluations({"results_dir": tmpdir})
            data = json.loads(result[0].text)
            assert data == []

    async def test_lists_summary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = {
                "timestamp": "20250101_000000",
                "total_questions": 3,
                "total_model_runs": 6,
                "total_errors": 0,
                "total_estimated_cost": 0.01,
                "judge_model": "gpt-4o-mini",
                "overall": {"model-a": {}},
            }
            p = Path(tmpdir) / "20250101_000000_summary.json"
            p.write_text(json.dumps(summary))

            result = await _list_evaluations({"results_dir": tmpdir})
            data = json.loads(result[0].text)
            assert len(data) == 1
            assert data[0]["timestamp"] == "20250101_000000"
            assert data[0]["models"] == ["model-a"]

    async def test_ignores_non_summary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "benchmark.json").write_text("{}")
            (Path(tmpdir) / "random.txt").write_text("hello")
            result = await _list_evaluations({"results_dir": tmpdir})
            data = json.loads(result[0].text)
            assert data == []

    async def test_handles_corrupt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "20250101_000000_summary.json").write_text("not json")
            result = await _list_evaluations({"results_dir": tmpdir})
            data = json.loads(result[0].text)
            assert data == []  # corrupt file skipped

    async def test_multiple_runs_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for ts in ["20250101_000000", "20250102_000000", "20250103_000000"]:
                p = Path(tmpdir) / f"{ts}_summary.json"
                p.write_text(json.dumps({
                    "timestamp": ts, "total_questions": 1,
                    "total_model_runs": 1, "overall": {},
                }))
            result = await _list_evaluations({"results_dir": tmpdir})
            data = json.loads(result[0].text)
            assert len(data) == 3
            # Should be sorted newest first (reverse)
            assert data[0]["timestamp"] == "20250103_000000"


# ---------------------------------------------------------------------------
# _get_evaluation
# ---------------------------------------------------------------------------


class TestGetEvaluationTool:
    async def test_file_not_found(self):
        result = await _get_evaluation({"results_path": "/nonexistent/file.json"})
        assert "not found" in result[0].text.lower() or "File not found" in result[0].text

    async def test_returns_full_json(self):
        data = {
            "timestamp": "20250101_000000",
            "results": [{"eval_id": "e1", "model": "gpt-4o"}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = await _get_evaluation({"results_path": f.name})
        parsed = json.loads(result[0].text)
        assert parsed["timestamp"] == "20250101_000000"
        assert len(parsed["results"]) == 1

    async def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("broken json {")
            f.flush()
            result = await _get_evaluation({"results_path": f.name})
        assert "Error" in result[0].text

    async def test_returns_text_content(self):
        data = {"key": "value"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = await _get_evaluation({"results_path": f.name})
        assert result[0].type == "text"


# ---------------------------------------------------------------------------
# Server app
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# compare_runs tool
# ---------------------------------------------------------------------------


class TestCompareRunsTool:
    async def test_compare_runs_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "compare_runs")
        schema = tool.inputSchema
        assert "baseline_path" in schema["properties"]
        assert "current_path" in schema["properties"]
        assert "tolerance" in schema["properties"]
        assert "baseline_path" in schema["required"]
        assert "current_path" in schema["required"]

    async def test_routes_to_compare_runs(self):
        result = await call_tool("compare_runs", {
            "baseline_path": "/nonexistent/baseline.json",
            "current_path": "/nonexistent/current.json",
        })
        assert "not found" in result[0].text.lower() or "File not found" in result[0].text

    async def test_baseline_not_found(self):
        result = await _compare_runs({
            "baseline_path": "/nonexistent/baseline.json",
            "current_path": "/nonexistent/current.json",
        })
        assert "File not found" in result[0].text

    async def test_current_not_found(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"timestamp": "t1", "overall": {}}, f)
            f.flush()
            result = await _compare_runs({
                "baseline_path": f.name,
                "current_path": "/nonexistent/current.json",
            })
        assert "File not found" in result[0].text

    async def test_successful_comparison(self):
        baseline_data = {
            "timestamp": "t1",
            "overall": {"model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85}},
        }
        current_data = {
            "timestamp": "t2",
            "overall": {"model-a": {"avg_faithfulness": 0.88, "avg_relevance": 0.87}},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as bf:
            json.dump(baseline_data, bf)
            bf.flush()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cf:
            json.dump(current_data, cf)
            cf.flush()

        result = await _compare_runs({
            "baseline_path": bf.name,
            "current_path": cf.name,
        })
        data = json.loads(result[0].text)
        assert data["has_regressions"] is False
        assert "model-a" in data["models"]

    async def test_comparison_with_tolerance(self):
        baseline_data = {"timestamp": "t1", "overall": {"m": {"avg_faithfulness": 0.9}}}
        current_data = {"timestamp": "t2", "overall": {"m": {"avg_faithfulness": 0.8}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as bf:
            json.dump(baseline_data, bf)
            bf.flush()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cf:
            json.dump(current_data, cf)
            cf.flush()

        # With large tolerance, no regression
        result = await _compare_runs({
            "baseline_path": bf.name,
            "current_path": cf.name,
            "tolerance": {"faithfulness": 0.2},
        })
        data = json.loads(result[0].text)
        assert data["has_regressions"] is False


# ---------------------------------------------------------------------------
# format_pr_comment tool
# ---------------------------------------------------------------------------


class TestFormatPrCommentTool:
    async def test_format_pr_comment_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "format_pr_comment")
        schema = tool.inputSchema
        assert "summary_path" in schema["properties"]
        assert "comparison_path" in schema["properties"]
        assert "thresholds" in schema["properties"]
        assert "summary_path" in schema["required"]

    async def test_routes_to_format_pr_comment(self):
        result = await call_tool("format_pr_comment", {
            "summary_path": "/nonexistent/summary.json",
        })
        assert "not found" in result[0].text.lower() or "File not found" in result[0].text

    async def test_summary_not_found(self):
        result = await _format_pr_comment({
            "summary_path": "/nonexistent/summary.json",
        })
        assert "File not found" in result[0].text

    async def test_successful_format(self):
        summary_data = {
            "timestamp": "20250101_000000",
            "total_questions": 3,
            "total_estimated_cost": 0.01,
            "judge_model": "gpt-4o-mini",
            "overall": {
                "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85, "avg_ttft_ms": 150, "avg_cost_per_query": 0.001},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(summary_data, f)
            f.flush()
            result = await _format_pr_comment({"summary_path": f.name})
        assert "## LLM Eval Results" in result[0].text
        assert "model-a" in result[0].text

    async def test_format_with_thresholds(self):
        summary_data = {
            "timestamp": "20250101_000000",
            "total_questions": 3,
            "total_estimated_cost": 0.01,
            "judge_model": "gpt-4o-mini",
            "overall": {
                "model-a": {"avg_faithfulness": 0.9, "avg_relevance": 0.85},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(summary_data, f)
            f.flush()
            result = await _format_pr_comment({
                "summary_path": f.name,
                "thresholds": {"avg_faithfulness": 0.8},
            })
        assert "Threshold details" in result[0].text

    async def test_format_with_comparison(self):
        summary_data = {
            "timestamp": "20250101_000000",
            "total_questions": 3,
            "total_estimated_cost": 0.01,
            "judge_model": "gpt-4o-mini",
            "overall": {"model-a": {"avg_faithfulness": 0.7}},
        }
        comparison_data = {
            "regressions": [
                {"model": "model-a", "metric": "faithfulness", "baseline": 0.9, "current": 0.7, "delta": -0.2},
            ],
            "has_regressions": True,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as sf:
            json.dump(summary_data, sf)
            sf.flush()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cf:
            json.dump(comparison_data, cf)
            cf.flush()

        result = await _format_pr_comment({
            "summary_path": sf.name,
            "comparison_path": cf.name,
        })
        assert "faithfulness" in result[0].text
        assert "-0.2000" in result[0].text

    async def test_comparison_file_not_found(self):
        summary_data = {"timestamp": "t", "overall": {}, "total_questions": 0, "total_estimated_cost": 0, "judge_model": "x"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as sf:
            json.dump(summary_data, sf)
            sf.flush()
            result = await _format_pr_comment({
                "summary_path": sf.name,
                "comparison_path": "/nonexistent/comparison.json",
            })
        assert "File not found" in result[0].text


# ---------------------------------------------------------------------------
# Server app
# ---------------------------------------------------------------------------


class TestServerApp:
    def test_app_name(self):
        assert app.name == "mcp-llm-eval"


# ---------------------------------------------------------------------------
# v0.5.0 — schema for new MCP tools
# ---------------------------------------------------------------------------


class TestRetrievalToolSchemas:
    async def test_evaluate_retrieval_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "evaluate_retrieval")
        schema = tool.inputSchema
        assert "dataset_path" in schema["required"]
        assert "corpus_path" in schema["required"]
        assert schema["properties"]["adapter"]["enum"] == ["bm25"]

    async def test_evaluate_rag_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "evaluate_rag_end_to_end")
        schema = tool.inputSchema
        assert {"dataset_path", "corpus_path", "models"} <= set(schema["required"])

    async def test_drift_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "check_retrieval_drift")
        schema = tool.inputSchema
        assert "baseline_path" in schema["required"]
        assert "current_path" in schema["required"]
        assert "tolerance" in schema["properties"]

    async def test_poisoned_stub_schema(self):
        tools = await list_tools()
        tool = next(t for t in tools if t.name == "simulate_poisoned_corpus")
        schema = tool.inputSchema
        assert "STUB" in tool.description
        assert "poisoning_strategy" in schema["properties"]


# ---------------------------------------------------------------------------
# v0.5.0 — _evaluate_retrieval, _evaluate_rag_end_to_end, drift, stub
# ---------------------------------------------------------------------------


class TestEvaluateRetrievalTool:
    @patch("mcp_llm_eval.server.engine")
    async def test_returns_summary_json(self, mock_engine):
        mock_engine.run_retrieval_evaluation.return_value = {
            "timestamp": "20260424_120000",
            "k": 5,
            "total_queries": 3,
            "total_errors": 0,
            "aggregate": {"avg_recall_at_k": 0.9},
            "per_query": [],
        }
        result = await _evaluate_retrieval({
            "dataset_path": "ds.jsonl",
            "corpus_path": "corpus.jsonl",
            "k": 5,
        })
        body = json.loads(result[0].text)
        assert body["aggregate"]["avg_recall_at_k"] == 0.9
        mock_engine.run_retrieval_evaluation.assert_called_once()

    @patch("mcp_llm_eval.server.engine")
    async def test_engine_error_returned_as_text(self, mock_engine):
        mock_engine.run_retrieval_evaluation.return_value = {"error": "boom"}
        result = await _evaluate_retrieval({
            "dataset_path": "ds.jsonl",
            "corpus_path": "corpus.jsonl",
        })
        assert result[0].text.startswith("Error:")
        assert "boom" in result[0].text

    @patch("mcp_llm_eval.server.engine")
    async def test_engine_exception_caught(self, mock_engine):
        mock_engine.run_retrieval_evaluation.side_effect = RuntimeError("kaboom")
        result = await _evaluate_retrieval({
            "dataset_path": "ds.jsonl",
            "corpus_path": "corpus.jsonl",
        })
        assert result[0].text.startswith("Error:")


class TestEvaluateRagTool:
    @patch("mcp_llm_eval.server.engine")
    async def test_runs_with_models(self, mock_engine):
        mock_engine.run_rag_evaluation.return_value = {
            "timestamp": "t",
            "total_queries": 1,
            "total_model_runs": 1,
            "total_errors": 0,
            "overall": {"gpt-4o-mini": {"avg_recall_at_k": 1.0}},
            "per_query": [],
        }
        result = await _evaluate_rag_end_to_end({
            "dataset_path": "ds.jsonl",
            "corpus_path": "corpus.jsonl",
            "models": [{"provider": "openai", "model": "gpt-4o-mini"}],
            "k": 3,
        })
        body = json.loads(result[0].text)
        assert "overall" in body
        call_kwargs = mock_engine.run_rag_evaluation.call_args[1]
        assert call_kwargs["k"] == 3
        # ModelConfig was constructed
        assert call_kwargs["models"][0].provider == "openai"

    @patch("mcp_llm_eval.server.engine")
    async def test_engine_error_returned(self, mock_engine):
        mock_engine.run_rag_evaluation.return_value = {"error": "no labels"}
        result = await _evaluate_rag_end_to_end({
            "dataset_path": "ds.jsonl",
            "corpus_path": "corpus.jsonl",
            "models": [{"provider": "openai", "model": "gpt-4o-mini"}],
        })
        assert "no labels" in result[0].text


class TestCheckRetrievalDriftTool:
    async def test_compares_two_files(self, tmp_path):
        baseline_path = tmp_path / "baseline.json"
        current_path = tmp_path / "current.json"
        baseline_path.write_text(json.dumps({
            "timestamp": "b",
            "aggregate": {"avg_recall_at_k": 0.80, "p95_retrieval_latency_ms": 5.0},
        }))
        current_path.write_text(json.dumps({
            "timestamp": "c",
            "aggregate": {"avg_recall_at_k": 0.65, "p95_retrieval_latency_ms": 5.0},
        }))
        result = await _check_retrieval_drift_tool({
            "baseline_path": str(baseline_path),
            "current_path": str(current_path),
        })
        body = json.loads(result[0].text)
        assert body["has_regressions"] is True
        assert body["metrics"]["avg_recall_at_k"]["regression"] is True

    async def test_baseline_missing_returns_error(self, tmp_path):
        current_path = tmp_path / "c.json"
        current_path.write_text("{}")
        result = await _check_retrieval_drift_tool({
            "baseline_path": str(tmp_path / "missing.json"),
            "current_path": str(current_path),
        })
        assert "not found" in result[0].text


class TestSimulatePoisonedCorpus:
    async def test_returns_not_implemented(self):
        result = await _simulate_poisoned_corpus({
            "dataset_path": "ds.jsonl",
            "corpus_path": "corpus.jsonl",
            "poisoning_strategy": "contradiction",
        })
        body = json.loads(result[0].text)
        assert body["status"] == "not_implemented"
        assert "v0.6.x" in body["message"]
        assert "poisoning_strategy" in body["accepted_arguments"]


class TestRoutingRetrievalTools:
    async def test_routes_evaluate_retrieval(self):
        with patch("mcp_llm_eval.server.engine") as mock_engine:
            mock_engine.run_retrieval_evaluation.return_value = {
                "timestamp": "t", "aggregate": {}, "per_query": [],
                "k": 5, "total_queries": 0, "total_errors": 0,
            }
            await call_tool("evaluate_retrieval", {
                "dataset_path": "ds.jsonl", "corpus_path": "corpus.jsonl",
            })
            mock_engine.run_retrieval_evaluation.assert_called_once()

    async def test_routes_simulate_poisoned_corpus(self):
        result = await call_tool("simulate_poisoned_corpus", {
            "dataset_path": "ds.jsonl", "corpus_path": "corpus.jsonl",
            "poisoning_strategy": "noise",
        })
        body = json.loads(result[0].text)
        assert body["status"] == "not_implemented"
