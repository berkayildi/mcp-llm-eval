"""Load and validate .eval-gate.yml configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """Load .eval-gate.yml config file.

    Returns a validated dict with the following structure:
    - dataset: str (path to dataset JSON)
    - models: list of model config dicts
    - judge: dict with provider, model, temperature
    - thresholds: dict with threshold values
    - output_dir: str (path to output directory)
    - tracing: dict with enabled, project, endpoint
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "pyyaml is required for config file support. "
            "Install it with: pip install pyyaml"
        ) from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Config file must be a YAML mapping")

    return _validate_config(data)


def _validate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and apply defaults to config data."""
    config: dict[str, Any] = {}

    # dataset (optional in config, can be overridden by CLI)
    config["dataset"] = data.get("dataset")

    # models (required)
    models_raw = data.get("models")
    if not models_raw:
        raise ValueError("Config must specify at least one model under 'models'")
    if not isinstance(models_raw, list):
        raise ValueError("'models' must be a list")

    models = []
    for i, m in enumerate(models_raw):
        if not isinstance(m, dict):
            raise ValueError(f"models[{i}] must be a mapping")
        if "provider" not in m:
            raise ValueError(f"models[{i}] missing required field 'provider'")
        if "model" not in m:
            raise ValueError(f"models[{i}] missing required field 'model'")
        models.append({
            "provider": m["provider"],
            "model": m["model"],
            "max_tokens": m.get("max_tokens", 500),
            "input_cost_per_mtok": m.get("input_cost_per_mtok", 0.0),
            "output_cost_per_mtok": m.get("output_cost_per_mtok", 0.0),
        })
    config["models"] = models

    # judge (optional with defaults)
    judge_raw = data.get("judge", {})
    config["judge"] = {
        "provider": judge_raw.get("provider", "openai"),
        "model": judge_raw.get("model", "gpt-4o-mini"),
        "temperature": judge_raw.get("temperature", 0),
    }

    # thresholds (optional)
    thresh_raw = data.get("thresholds", {})
    config["thresholds"] = {
        "avg_faithfulness": thresh_raw.get("avg_faithfulness"),
        "avg_relevance": thresh_raw.get("avg_relevance"),
        "p95_ttft_ms": thresh_raw.get("p95_ttft_ms"),
        "max_cost_per_query": thresh_raw.get("max_cost_per_query"),
    }

    # output_dir (optional)
    config["output_dir"] = data.get("output_dir", "eval/results")

    # tracing (optional)
    tracing_raw = data.get("tracing", {})
    config["tracing"] = {
        "enabled": tracing_raw.get("enabled", False),
        "project": tracing_raw.get("project"),
        "endpoint": tracing_raw.get("endpoint"),
    }

    return config
