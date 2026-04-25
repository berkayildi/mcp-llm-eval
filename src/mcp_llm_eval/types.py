"""Shared dataclasses for eval pipeline: entries, results, summaries, thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PoisonedChunk:
    """Stub for Phase 2 corpus poisoning. Fields reserved so v0.5.0 datasets
    that declare poisoned_chunks deserialise cleanly; no behaviour attached."""

    chunk_id: str
    poison_type: str
    payload: str
    expected_detection: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoisonedChunk:
        return cls(
            chunk_id=data["chunk_id"],
            poison_type=data["poison_type"],
            payload=data["payload"],
            expected_detection=data.get("expected_detection", True),
        )


@dataclass
class EvalEntry:
    """A single evaluation question from the dataset."""

    id: str
    category: str
    context: str
    question: str
    expected_response: str
    tags: list[str] = field(default_factory=list)
    # v0.5.0 — optional retrieval / RAG annotations. Default None so v0.4.x
    # datasets load and serialise unchanged.
    relevant_chunk_ids: list[str] | None = None
    poisoned_chunks: list[PoisonedChunk] | None = None
    # Reserved for future graded relevance (v0.6.x):
    #   relevant_chunk_scores: dict[str, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalEntry:
        poisoned_raw = data.get("poisoned_chunks")
        return cls(
            id=data["id"],
            category=data["category"],
            context=data["context"],
            question=data["question"],
            expected_response=data["expected_response"],
            tags=data.get("tags", []),
            relevant_chunk_ids=data.get("relevant_chunk_ids"),
            poisoned_chunks=(
                [PoisonedChunk.from_dict(p) for p in poisoned_raw]
                if poisoned_raw else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Drop v0.5.0 fields when unset so v0.4.x entries serialise identically.
        if self.relevant_chunk_ids is None:
            d.pop("relevant_chunk_ids", None)
        if self.poisoned_chunks is None:
            d.pop("poisoned_chunks", None)
        return d


@dataclass
class RetrievedChunk:
    """A single chunk returned by a RetrievalAdapter."""

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievedChunk:
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            score=float(data["score"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """One query's retrieval-only output."""

    query_id: str
    query: str
    retrieved_chunk_ids: list[str]
    retrieval_latency_ms: float
    metrics: dict[str, float] = field(default_factory=dict)
    k: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalResult:
        return cls(
            query_id=data["query_id"],
            query=data["query"],
            retrieved_chunk_ids=list(data.get("retrieved_chunk_ids", [])),
            retrieval_latency_ms=float(data.get("retrieval_latency_ms", 0.0)),
            metrics=dict(data.get("metrics", {})),
            k=int(data.get("k", 0)),
            error=data.get("error"),
        )


@dataclass
class RAGResult:
    """One query's full RAG pipeline output (retrieval + generation + judge)."""

    query_id: str
    query: str
    retrieved_chunks: list[RetrievedChunk]
    retrieval_latency_ms: float
    retrieval_metrics: dict[str, float]
    model: str
    provider: str
    answer: str | None = None
    generation_metrics: dict[str, Any] = field(default_factory=dict)
    context_relevance_score: float | None = None
    context_relevance_reason: str | None = None
    citation_faithfulness_score: float | None = None
    citation_faithfulness_reason: str | None = None
    judge_model: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGResult:
        return cls(
            query_id=data["query_id"],
            query=data["query"],
            retrieved_chunks=[
                RetrievedChunk.from_dict(c) for c in data.get("retrieved_chunks", [])
            ],
            retrieval_latency_ms=float(data.get("retrieval_latency_ms", 0.0)),
            retrieval_metrics=dict(data.get("retrieval_metrics", {})),
            model=data["model"],
            provider=data["provider"],
            answer=data.get("answer"),
            generation_metrics=dict(data.get("generation_metrics", {})),
            context_relevance_score=data.get("context_relevance_score"),
            context_relevance_reason=data.get("context_relevance_reason"),
            citation_faithfulness_score=data.get("citation_faithfulness_score"),
            citation_faithfulness_reason=data.get("citation_faithfulness_reason"),
            judge_model=data.get("judge_model"),
            error=data.get("error"),
        )


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

    # v0.4.x — generation
    avg_faithfulness: float | None = None
    avg_relevance: float | None = None
    p95_ttft_ms: int | None = None
    max_cost_per_query: float | None = None
    # v0.5.0 — retrieval
    avg_recall_at_k: float | None = None
    avg_precision_at_k: float | None = None
    avg_mrr: float | None = None
    avg_ndcg_at_k: float | None = None
    p95_retrieval_latency_ms: int | None = None
    # v0.5.0 — RAG
    avg_context_relevance: float | None = None
    avg_citation_faithfulness: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdConfig:
        return cls(
            avg_faithfulness=data.get("avg_faithfulness"),
            avg_relevance=data.get("avg_relevance"),
            p95_ttft_ms=data.get("p95_ttft_ms"),
            max_cost_per_query=data.get("max_cost_per_query"),
            avg_recall_at_k=data.get("avg_recall_at_k"),
            avg_precision_at_k=data.get("avg_precision_at_k"),
            avg_mrr=data.get("avg_mrr"),
            avg_ndcg_at_k=data.get("avg_ndcg_at_k"),
            p95_retrieval_latency_ms=data.get("p95_retrieval_latency_ms"),
            avg_context_relevance=data.get("avg_context_relevance"),
            avg_citation_faithfulness=data.get("avg_citation_faithfulness"),
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
