"""End-to-end integration tests for retrieval and RAG evaluation.

Uses the JSONL fixtures under ``tests/fixtures/``. The retrieval adapter
runs for real (BM25 against 12 chunks); the judge and provider runners are
mocked since we never make real API calls in tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mcp_llm_eval import engine
from mcp_llm_eval.engine import (
    run_rag_evaluation,
    run_retrieval_evaluation,
)
from mcp_llm_eval.types import ModelConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPUS_PATH = FIXTURES_DIR / "retrieval_corpus.jsonl"
DATASET_PATH = FIXTURES_DIR / "retrieval_dataset.jsonl"


# ---------------------------------------------------------------------------
# Retrieval-only end-to-end (real BM25, no mocks)
# ---------------------------------------------------------------------------


class TestRetrievalEndToEnd:
    def test_real_bm25_against_fixtures(self):
        summary = run_retrieval_evaluation(
            dataset_path=str(DATASET_PATH),
            corpus_path=str(CORPUS_PATH),
            k=5,
        )
        assert summary["total_queries"] == 4
        assert summary["total_errors"] == 0
        agg = summary["aggregate"]
        # All four entries have correct relevant chunks present in corpus;
        # BM25 should retrieve at least the most-on-topic chunk in top-5.
        assert agg["avg_recall_at_k"] > 0.5
        assert agg["avg_mrr"] > 0.0
        # Latency tracking is populated.
        assert agg["p95_retrieval_latency_ms"] >= 0.0
        # Per-query results carry retrieval metrics.
        assert len(summary["per_query"]) == 4
        for row in summary["per_query"]:
            assert "metrics" in row
            assert set(row["metrics"].keys()) == {
                "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
            }

    def test_writes_files_to_output_dir(self, tmp_path):
        summary = run_retrieval_evaluation(
            dataset_path=str(DATASET_PATH),
            corpus_path=str(CORPUS_PATH),
            k=3,
            output_dir=str(tmp_path),
        )
        assert summary["k"] == 3
        files = {f.name for f in tmp_path.glob("*.json")}
        assert any(n.endswith("_retrieval_summary.json") for n in files)
        assert any(n.endswith("_retrieval_detail.json") for n in files)
        assert "latest_retrieval_summary.json" in files


# ---------------------------------------------------------------------------
# Full RAG end-to-end (real BM25 + mocked judge + mocked provider)
# ---------------------------------------------------------------------------


class TestRagEndToEnd:
    @patch("mcp_llm_eval.engine.judge_module")
    @patch("mcp_llm_eval.engine._get_runner")
    @patch("mcp_llm_eval.engine._get_client")
    def test_full_pipeline_against_fixtures(
        self, mock_get_client, mock_get_runner, mock_judge_mod,
    ):
        mock_get_client.return_value = MagicMock()

        runner_mod = MagicMock()
        runner_mod.run.return_value = {
            "response": "Generated answer from retrieved context.",
            "input_tokens": 200,
            "output_tokens": 30,
            "stop_reason": "stop",
            "time_to_first_token_ms": 90,
            "total_latency_ms": 450,
        }
        mock_get_runner.return_value = runner_mod

        mock_judge_mod._resolve_judge_model.return_value = "gpt-4o-mini"
        mock_judge_mod.judge_context_relevance.return_value = {
            "score": 0.75,
            "reason": "on-topic",
            "raw_score": 4,
            "judge_model": "gpt-4o-mini",
        }
        mock_judge_mod.judge_citation_faithfulness.return_value = {
            "score": 1.0,
            "reason": "fully supported",
            "raw_score": 5,
            "judge_model": "gpt-4o-mini",
        }

        models = [
            ModelConfig(
                provider="openai", model="gpt-4o-mini",
                input_cost_per_mtok=0.15, output_cost_per_mtok=0.60,
            ),
        ]
        summary = run_rag_evaluation(
            dataset_path=str(DATASET_PATH),
            corpus_path=str(CORPUS_PATH),
            models=models,
            k=3,
        )

        assert summary["total_queries"] == 4
        assert summary["total_model_runs"] == 4
        assert summary["total_errors"] == 0
        assert "gpt-4o-mini" in summary["overall"]
        agg = summary["overall"]["gpt-4o-mini"]
        assert agg["runs"] == 4
        assert agg["avg_context_relevance"] == 0.75
        assert agg["avg_citation_faithfulness"] == 1.0
        # Cost computed from token usage:
        # cost_per_query = (200 * 0.15 + 30 * 0.60) / 1e6 = 0.000048
        assert agg["avg_cost_per_query"] == pytest.approx(0.000048, abs=1e-6)

    @patch("mcp_llm_eval.engine.judge_module")
    @patch("mcp_llm_eval.engine._get_runner")
    @patch("mcp_llm_eval.engine._get_client")
    def test_writes_rag_files(
        self, mock_get_client, mock_get_runner, mock_judge_mod, tmp_path,
    ):
        mock_get_client.return_value = MagicMock()
        runner_mod = MagicMock()
        runner_mod.run.return_value = {
            "response": "ans",
            "input_tokens": 100,
            "output_tokens": 20,
            "stop_reason": "stop",
            "time_to_first_token_ms": 80,
            "total_latency_ms": 300,
        }
        mock_get_runner.return_value = runner_mod
        mock_judge_mod._resolve_judge_model.return_value = "gpt-4o-mini"
        mock_judge_mod.judge_context_relevance.return_value = {
            "score": 0.5, "reason": "ok", "raw_score": 3, "judge_model": "gpt-4o-mini",
        }
        mock_judge_mod.judge_citation_faithfulness.return_value = {
            "score": 0.75, "reason": "ok", "raw_score": 4, "judge_model": "gpt-4o-mini",
        }

        models = [ModelConfig(provider="openai", model="gpt-4o-mini")]
        run_rag_evaluation(
            dataset_path=str(DATASET_PATH),
            corpus_path=str(CORPUS_PATH),
            models=models,
            k=2,
            output_dir=str(tmp_path),
        )
        files = {f.name for f in tmp_path.glob("*.json")}
        assert any(n.endswith("_rag_summary.json") for n in files)
        assert any(n.endswith("_rag_benchmark.json") for n in files)
        assert "latest_rag_summary.json" in files


# ---------------------------------------------------------------------------
# Drift comparison on real summary shapes
# ---------------------------------------------------------------------------


class TestDriftOnRealSummaries:
    def test_baseline_vs_current_no_regression(self, tmp_path):
        baseline = run_retrieval_evaluation(
            dataset_path=str(DATASET_PATH),
            corpus_path=str(CORPUS_PATH),
            k=5,
        )
        current = run_retrieval_evaluation(
            dataset_path=str(DATASET_PATH),
            corpus_path=str(CORPUS_PATH),
            k=5,
        )
        # Same corpus + same dataset → identical metrics → no regressions.
        result = engine.check_retrieval_drift(baseline, current)
        assert result["has_regressions"] is False
