"""Shared dataclasses for eval pipeline: entries, results, summaries, thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EvalEntry:
    """A single evaluation question from the dataset."""

    id: str
    category: str
    context: str
    question: str
    expected_response: str
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalEntry:
        return cls(
            id=data["id"],
            category=data["category"],
            context=data["context"],
            question=data["question"],
            expected_response=data["expected_response"],
            tags=data.get("tags", []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""

    provider: str
    model: str
    max_tokens: int = 500
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return cls(
            provider=data["provider"],
            model=data["model"],
            max_tokens=data.get("max_tokens", 500),
            input_cost_per_mtok=data.get("input_cost_per_mtok", 0.0),
            output_cost_per_mtok=data.get("output_cost_per_mtok", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    """Result of running a single eval entry against a single model."""

    eval_id: str
    category: str
    model: str
    provider: str
    response: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    time_to_first_token_ms: int | None = None
    total_latency_ms: int = 0
    cost_per_query: float = 0.0
    faithfulness_score: float | None = None
    faithfulness_reason: str | None = None
    relevance_score: float | None = None
    relevance_reason: str | None = None
    judge_model: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AggregateMetrics:
    """Aggregated metrics for a group of eval results."""

    provider: str
    model: str
    runs: int = 0
    avg_ttft_ms: float = 0.0
    avg_latency_ms: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_cost_per_query: float = 0.0
    avg_faithfulness: float | None = None
    avg_relevance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    """Summary of a complete evaluation run."""

    timestamp: str
    total_questions: int
    total_model_runs: int
    total_errors: int
    total_elapsed_sec: float
    total_estimated_cost: float
    judge_model: str | None = None
    overall: dict[str, Any] = field(default_factory=dict)
    by_category: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSummary:
        return cls(
            timestamp=data["timestamp"],
            total_questions=data["total_questions"],
            total_model_runs=data["total_model_runs"],
            total_errors=data.get("total_errors", 0),
            total_elapsed_sec=data.get("total_elapsed_sec", 0.0),
            total_estimated_cost=data.get("total_estimated_cost", 0.0),
            judge_model=data.get("judge_model"),
            overall=data.get("overall", {}),
            by_category=data.get("by_category", {}),
            results=data.get("results", []),
        )


@dataclass
class ThresholdConfig:
    """Quality gate thresholds."""

    avg_faithfulness: float | None = None
    avg_relevance: float | None = None
    p95_ttft_ms: int | None = None
    max_cost_per_query: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdConfig:
        return cls(
            avg_faithfulness=data.get("avg_faithfulness"),
            avg_relevance=data.get("avg_relevance"),
            p95_ttft_ms=data.get("p95_ttft_ms"),
            max_cost_per_query=data.get("max_cost_per_query"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricCheck:
    """Result of checking a single metric against its threshold."""

    metric: str
    threshold: float
    actual: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdResult:
    """Result of checking all thresholds."""

    overall_pass: bool
    per_metric: list[MetricCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_pass": self.overall_pass,
            "per_metric": [m.to_dict() for m in self.per_metric],
        }
