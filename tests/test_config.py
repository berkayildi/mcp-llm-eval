"""Tests for config.py: YAML loading, validation, defaults, error handling."""

import os
import tempfile

import pytest

from mcp_llm_eval.config import load_config, _validate_config


# ---------------------------------------------------------------------------
# load_config — file handling
# ---------------------------------------------------------------------------


class TestLoadConfigFileHandling:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/config.yml")

    def test_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(":\n  invalid: yaml: [broken")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Invalid YAML"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_not_a_mapping(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("- item1\n- item2\n")
            f.flush()
            try:
                with pytest.raises(ValueError, match="YAML mapping"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_valid_minimal_config(self):
        content = """
models:
  - provider: openai
    model: gpt-4o-mini
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(content)
            f.flush()
            try:
                config = load_config(f.name)
                assert len(config["models"]) == 1
                assert config["models"][0]["provider"] == "openai"
            finally:
                os.unlink(f.name)

    def test_full_config(self):
        content = """
dataset: eval/dataset.json

models:
  - provider: anthropic
    model: claude-sonnet-4-6
    max_tokens: 500
    input_cost_per_mtok: 3.0
    output_cost_per_mtok: 15.0
  - provider: openai
    model: gpt-4o-mini
    max_tokens: 500
    input_cost_per_mtok: 0.15
    output_cost_per_mtok: 0.60

judge:
  provider: openai
  model: gpt-4o-mini
  temperature: 0

thresholds:
  avg_faithfulness: 0.85
  avg_relevance: 0.80
  p95_ttft_ms: 2000
  max_cost_per_query: 0.02

output_dir: eval/results

tracing:
  enabled: false
  project: my-project-eval
  endpoint: https://eu.api.smith.langchain.com
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(content)
            f.flush()
            try:
                config = load_config(f.name)
                assert config["dataset"] == "eval/dataset.json"
                assert len(config["models"]) == 2
                assert config["models"][0]["provider"] == "anthropic"
                assert config["models"][1]["model"] == "gpt-4o-mini"
                assert config["judge"]["provider"] == "openai"
                assert config["thresholds"]["avg_faithfulness"] == 0.85
                assert config["output_dir"] == "eval/results"
                assert config["tracing"]["enabled"] is False
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# _validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_missing_models(self):
        with pytest.raises(ValueError, match="at least one model"):
            _validate_config({})

    def test_models_not_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            _validate_config({"models": "not-a-list"})

    def test_model_not_mapping(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _validate_config({"models": ["not-a-dict"]})

    def test_model_missing_provider(self):
        with pytest.raises(ValueError, match="missing required field 'provider'"):
            _validate_config({"models": [{"model": "gpt-4o"}]})

    def test_model_missing_model(self):
        with pytest.raises(ValueError, match="missing required field 'model'"):
            _validate_config({"models": [{"provider": "openai"}]})

    def test_defaults_applied(self):
        config = _validate_config({
            "models": [{"provider": "openai", "model": "gpt-4o"}],
        })
        # Model defaults
        assert config["models"][0]["max_tokens"] == 500
        assert config["models"][0]["input_cost_per_mtok"] == 0.0
        assert config["models"][0]["output_cost_per_mtok"] == 0.0
        # Judge defaults
        assert config["judge"]["provider"] == "openai"
        assert config["judge"]["model"] == "gpt-4o-mini"
        assert config["judge"]["temperature"] == 0
        # Output dir default
        assert config["output_dir"] == "eval/results"
        # Tracing defaults
        assert config["tracing"]["enabled"] is False

    def test_thresholds_all_none_by_default(self):
        config = _validate_config({
            "models": [{"provider": "openai", "model": "gpt-4o"}],
        })
        assert config["thresholds"]["avg_faithfulness"] is None
        assert config["thresholds"]["avg_relevance"] is None
        assert config["thresholds"]["p95_ttft_ms"] is None
        assert config["thresholds"]["max_cost_per_query"] is None

    def test_dataset_optional(self):
        config = _validate_config({
            "models": [{"provider": "openai", "model": "gpt-4o"}],
        })
        assert config["dataset"] is None

    def test_multiple_models(self):
        config = _validate_config({
            "models": [
                {"provider": "openai", "model": "gpt-4o"},
                {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            ],
        })
        assert len(config["models"]) == 2
        assert config["models"][0]["provider"] == "openai"
        assert config["models"][1]["provider"] == "anthropic"

    def test_custom_judge(self):
        config = _validate_config({
            "models": [{"provider": "openai", "model": "gpt-4o"}],
            "judge": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "temperature": 0.5},
        })
        assert config["judge"]["provider"] == "anthropic"
        assert config["judge"]["model"] == "claude-haiku-4-5-20251001"
        assert config["judge"]["temperature"] == 0.5

    def test_tracing_enabled(self):
        config = _validate_config({
            "models": [{"provider": "openai", "model": "gpt-4o"}],
            "tracing": {"enabled": True, "project": "my-proj", "endpoint": "https://example.com"},
        })
        assert config["tracing"]["enabled"] is True
        assert config["tracing"]["project"] == "my-proj"
        assert config["tracing"]["endpoint"] == "https://example.com"

    def test_empty_models_list(self):
        with pytest.raises(ValueError, match="at least one model"):
            _validate_config({"models": []})


# ---------------------------------------------------------------------------
# v0.5.0 — corpus, retrieval, retrieval/RAG threshold keys
# ---------------------------------------------------------------------------


class TestV050ConfigKeys:
    def test_v04_config_loads_unchanged(self):
        cfg = _validate_config({
            "models": [{"provider": "openai", "model": "gpt-4o-mini"}],
            "thresholds": {"avg_faithfulness": 0.8, "avg_relevance": 0.7},
        })
        # v0.5.0 threshold keys default to None
        assert cfg["thresholds"]["avg_recall_at_k"] is None
        assert cfg["thresholds"]["avg_context_relevance"] is None
        assert cfg["thresholds"]["p95_retrieval_latency_ms"] is None
        # v0.4.x keys preserved
        assert cfg["thresholds"]["avg_faithfulness"] == 0.8
        # corpus and retrieval defaults
        assert cfg["corpus"] is None
        assert cfg["retrieval"] == {"adapter": "bm25", "k": 5}

    def test_mixed_v04_and_v05_thresholds(self):
        cfg = _validate_config({
            "models": [{"provider": "openai", "model": "gpt-4o-mini"}],
            "thresholds": {
                # v0.4.x
                "avg_faithfulness": 0.80,
                "avg_relevance": 0.75,
                "p95_ttft_ms": 500,
                "max_cost_per_query": 0.01,
                # v0.5.0 — retrieval
                "avg_recall_at_k": 0.75,
                "avg_precision_at_k": 0.50,
                "avg_mrr": 0.70,
                "avg_ndcg_at_k": 0.75,
                "p95_retrieval_latency_ms": 50,
                # v0.5.0 — RAG
                "avg_context_relevance": 0.70,
                "avg_citation_faithfulness": 0.80,
            },
        })
        t = cfg["thresholds"]
        assert t["avg_faithfulness"] == 0.80
        assert t["avg_recall_at_k"] == 0.75
        assert t["avg_context_relevance"] == 0.70
        assert t["p95_retrieval_latency_ms"] == 50

    def test_corpus_key_loaded(self):
        cfg = _validate_config({
            "models": [{"provider": "openai", "model": "gpt-4o-mini"}],
            "corpus": "eval/corpus.jsonl",
        })
        assert cfg["corpus"] == "eval/corpus.jsonl"

    def test_retrieval_block_overrides_defaults(self):
        cfg = _validate_config({
            "models": [{"provider": "openai", "model": "gpt-4o-mini"}],
            "retrieval": {"adapter": "bm25", "k": 10},
        })
        assert cfg["retrieval"] == {"adapter": "bm25", "k": 10}

    def test_threshold_config_from_dict_round_trips_v05(self):
        from mcp_llm_eval.types import ThresholdConfig
        t = ThresholdConfig.from_dict({
            "avg_faithfulness": 0.8,
            "avg_recall_at_k": 0.75,
            "avg_context_relevance": 0.70,
            "p95_retrieval_latency_ms": 50,
        })
        assert t.avg_faithfulness == 0.8
        assert t.avg_recall_at_k == 0.75
        assert t.avg_context_relevance == 0.70
        assert t.p95_retrieval_latency_ms == 50
