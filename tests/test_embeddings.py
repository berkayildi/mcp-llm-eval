"""Tests for OpenAIEmbeddingAdapter and GoogleEmbeddingAdapter.

All SDK calls are mocked. No network traffic.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# numpy is in the embeddings extra; treat as required for these tests.
np = pytest.importorskip("numpy")

from mcp_llm_eval import embeddings as emb_mod  # noqa: E402
from mcp_llm_eval.embeddings import (  # noqa: E402
    GoogleEmbeddingAdapter,
    OpenAIEmbeddingAdapter,
    _corpus_hash,
)
from mcp_llm_eval.types import RetrievedChunk  # noqa: E402


SMALL_CORPUS = [
    {"chunk_id": "c1", "content": "Python is a popular programming language",
     "metadata": {"topic": "py"}},
    {"chunk_id": "c2", "content": "Poetry manages Python dependencies",
     "metadata": {"topic": "py"}},
    {"chunk_id": "c3", "content": "The James Webb telescope orbits at L2",
     "metadata": {"topic": "space"}},
    {"chunk_id": "c4", "content": "Hubble orbits Earth in low orbit",
     "metadata": {"topic": "space"}},
]


# --- Synthetic embedding helpers --------------------------------------------


def _toy_vector(text: str, dim: int = 8) -> list[float]:
    """Deterministic toy 'embedding' so 'python' queries land near 'python' chunks."""
    bag = {
        "python": [1.0, 0.0, 0.0, 0.0],
        "poetry": [0.9, 0.1, 0.0, 0.0],
        "telescope": [0.0, 1.0, 0.0, 0.0],
        "hubble": [0.0, 0.9, 0.1, 0.0],
        "webb": [0.0, 0.95, 0.05, 0.0],
        "earth": [0.0, 0.5, 0.5, 0.0],
        "orbit": [0.0, 0.5, 0.4, 0.1],
    }
    v = np.zeros(dim, dtype=np.float32)
    base = np.zeros(4, dtype=np.float32)
    for word, coords in bag.items():
        if word in text.lower():
            base += np.asarray(coords, dtype=np.float32)
    if base.sum() == 0:
        base += np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    v[:4] = base
    return v.tolist()


# --- OpenAI mock client ------------------------------------------------------


class _FakeOpenAIEmbeddings:
    def __init__(self, parent: "_FakeOpenAIClient") -> None:
        self._parent = parent

    def create(self, model: str, input: list[str]):
        self._parent.calls.append({"model": model, "n": len(input)})
        data_objs = [types.SimpleNamespace(embedding=_toy_vector(t)) for t in input]
        return types.SimpleNamespace(data=data_objs)


class _FakeOpenAIClient:
    def __init__(self, **_: object) -> None:
        self.embeddings = _FakeOpenAIEmbeddings(self)
        self.calls: list[dict] = []


def _install_openai_mock(monkeypatch: pytest.MonkeyPatch) -> _FakeOpenAIClient:
    holder: dict[str, _FakeOpenAIClient] = {}

    def fake_OpenAI(**kwargs):
        client = _FakeOpenAIClient(**kwargs)
        holder["client"] = client
        return client

    fake_module = types.SimpleNamespace(OpenAI=fake_OpenAI)
    monkeypatch.setattr(emb_mod, "_get_openai", lambda: fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return holder  # holder["client"] populated on first instantiation


# --- Google mock client ------------------------------------------------------


class _FakeGoogleModels:
    def __init__(self, parent: "_FakeGoogleClient") -> None:
        self._parent = parent

    def embed_content(self, model: str, contents: list[str]):
        self._parent.calls.append({"model": model, "n": len(contents)})
        embeds = [types.SimpleNamespace(values=_toy_vector(t)) for t in contents]
        return types.SimpleNamespace(embeddings=embeds)


class _FakeGoogleClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.models = _FakeGoogleModels(self)
        self.calls: list[dict] = []


def _install_google_mock(monkeypatch: pytest.MonkeyPatch) -> dict:
    holder: dict[str, _FakeGoogleClient] = {}

    def fake_Client(api_key=None):
        client = _FakeGoogleClient(api_key=api_key)
        holder["client"] = client
        return client

    fake_genai = types.SimpleNamespace(Client=fake_Client)
    monkeypatch.setattr(emb_mod, "_get_google_genai", lambda: fake_genai)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    return holder


# --- Tests: OpenAI adapter ---------------------------------------------------


class TestOpenAIEmbeddingAdapter:
    def test_constructs_from_corpus(self, monkeypatch, tmp_path):
        _install_openai_mock(monkeypatch)
        adapter = OpenAIEmbeddingAdapter(
            corpus=SMALL_CORPUS,
            model_name="text-embedding-3-small",
            cache_dir=str(tmp_path),
        )
        assert adapter is not None

    def test_from_jsonl(self, monkeypatch, tmp_path):
        _install_openai_mock(monkeypatch)
        corpus_path = tmp_path / "corpus.jsonl"
        corpus_path.write_text(
            "\n".join(json.dumps(c) for c in SMALL_CORPUS), encoding="utf-8",
        )
        adapter = OpenAIEmbeddingAdapter.from_jsonl(
            str(corpus_path), model_name="text-embedding-3-small",
        )
        out = adapter.retrieve("python", k=2)
        assert len(out) == 2

    def test_retrieve_returns_at_most_k(self, monkeypatch, tmp_path):
        _install_openai_mock(monkeypatch)
        adapter = OpenAIEmbeddingAdapter(
            corpus=SMALL_CORPUS, cache_dir=str(tmp_path),
        )
        for k in (1, 2, 3, 4):
            out = adapter.retrieve("python", k=k)
            assert len(out) == k
            assert all(isinstance(c, RetrievedChunk) for c in out)

    def test_retrieve_orders_by_similarity_descending(self, monkeypatch, tmp_path):
        _install_openai_mock(monkeypatch)
        adapter = OpenAIEmbeddingAdapter(
            corpus=SMALL_CORPUS, cache_dir=str(tmp_path),
        )
        out = adapter.retrieve("python language", k=4)
        scores = [c.score for c in out]
        assert scores == sorted(scores, reverse=True)
        # Top result should be a python-flavoured chunk (c1 or c2), not space.
        assert out[0].chunk_id in ("c1", "c2")

    def test_empty_query_returns_empty(self, monkeypatch, tmp_path):
        _install_openai_mock(monkeypatch)
        adapter = OpenAIEmbeddingAdapter(
            corpus=SMALL_CORPUS, cache_dir=str(tmp_path),
        )
        assert adapter.retrieve("", k=3) == []
        assert adapter.retrieve("   ", k=3) == []

    def test_cache_skips_reembed_on_second_instantiation(self, monkeypatch, tmp_path):
        holder = _install_openai_mock(monkeypatch)
        adapter1 = OpenAIEmbeddingAdapter(
            corpus=SMALL_CORPUS,
            model_name="text-embedding-3-small",
            cache_dir=str(tmp_path),
        )
        # First instantiation should have called the embeddings API at least once
        # for the corpus. Count *only* corpus-embedding calls (n == len(SMALL_CORPUS)).
        first_calls = list(holder["client"].calls)
        corpus_calls_first = [c for c in first_calls if c["n"] == len(SMALL_CORPUS)]
        assert corpus_calls_first, (
            "expected at least one corpus-embedding call on first instantiation"
        )

        # Drop the first client; new instantiation rebuilds the client.
        holder.clear()
        adapter2 = OpenAIEmbeddingAdapter(
            corpus=SMALL_CORPUS,
            model_name="text-embedding-3-small",
            cache_dir=str(tmp_path),
        )
        # Cache hit means ZERO corpus-sized calls; only single-query calls allowed.
        second_calls = holder["client"].calls
        corpus_calls_second = [c for c in second_calls if c["n"] == len(SMALL_CORPUS)]
        assert corpus_calls_second == [], (
            f"expected cache hit, but adapter re-embedded the corpus: {second_calls}"
        )
        # Sanity: both adapters return the same top-1 for the same query.
        assert adapter1.retrieve("python", 1)[0].chunk_id == \
               adapter2.retrieve("python", 1)[0].chunk_id

    def test_missing_api_key_raises(self, monkeypatch, tmp_path):
        # Install the SDK mock but clear the env var.
        fake_module = types.SimpleNamespace(OpenAI=lambda **kw: _FakeOpenAIClient(**kw))
        monkeypatch.setattr(emb_mod, "_get_openai", lambda: fake_module)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbeddingAdapter(corpus=SMALL_CORPUS, cache_dir=str(tmp_path))

    def test_duplicate_chunk_ids_rejected(self, monkeypatch, tmp_path):
        _install_openai_mock(monkeypatch)
        bad = SMALL_CORPUS + [{"chunk_id": "c1", "content": "duplicate"}]
        with pytest.raises(ValueError, match="Duplicate chunk_id"):
            OpenAIEmbeddingAdapter(corpus=bad, cache_dir=str(tmp_path))


# --- Tests: Google adapter ---------------------------------------------------


class TestGoogleEmbeddingAdapter:
    def test_constructs_from_corpus(self, monkeypatch, tmp_path):
        _install_google_mock(monkeypatch)
        adapter = GoogleEmbeddingAdapter(
            corpus=SMALL_CORPUS, cache_dir=str(tmp_path),
        )
        assert adapter is not None

    def test_retrieve_returns_at_most_k(self, monkeypatch, tmp_path):
        _install_google_mock(monkeypatch)
        adapter = GoogleEmbeddingAdapter(
            corpus=SMALL_CORPUS, cache_dir=str(tmp_path),
        )
        out = adapter.retrieve("telescope", k=2)
        assert len(out) == 2

    def test_orders_by_similarity_descending(self, monkeypatch, tmp_path):
        _install_google_mock(monkeypatch)
        adapter = GoogleEmbeddingAdapter(
            corpus=SMALL_CORPUS, cache_dir=str(tmp_path),
        )
        out = adapter.retrieve("hubble telescope", k=4)
        scores = [c.score for c in out]
        assert scores == sorted(scores, reverse=True)
        assert out[0].chunk_id in ("c3", "c4")

    def test_batches_within_size_limit(self, monkeypatch, tmp_path):
        # Build a 250-chunk corpus and verify the Google adapter splits into
        # batches of <= 100 (its _default_batch_size).
        holder = _install_google_mock(monkeypatch)
        big_corpus = [
            {"chunk_id": f"c{i}", "content": f"chunk number {i} python"}
            for i in range(250)
        ]
        GoogleEmbeddingAdapter(corpus=big_corpus, cache_dir=str(tmp_path))
        sizes = [c["n"] for c in holder["client"].calls]
        assert max(sizes) <= 100
        assert sum(sizes) == 250

    def test_empty_query_returns_empty(self, monkeypatch, tmp_path):
        _install_google_mock(monkeypatch)
        adapter = GoogleEmbeddingAdapter(
            corpus=SMALL_CORPUS, cache_dir=str(tmp_path),
        )
        assert adapter.retrieve("", k=3) == []

    def test_cache_skips_reembed(self, monkeypatch, tmp_path):
        holder = _install_google_mock(monkeypatch)
        GoogleEmbeddingAdapter(
            corpus=SMALL_CORPUS,
            model_name="gemini-embedding-001",
            cache_dir=str(tmp_path),
        )
        first_calls = list(holder["client"].calls)
        corpus_calls_first = [c for c in first_calls if c["n"] == len(SMALL_CORPUS)]
        assert corpus_calls_first

        holder.clear()
        GoogleEmbeddingAdapter(
            corpus=SMALL_CORPUS,
            model_name="gemini-embedding-001",
            cache_dir=str(tmp_path),
        )
        second_calls = holder["client"].calls
        corpus_calls_second = [c for c in second_calls if c["n"] == len(SMALL_CORPUS)]
        assert corpus_calls_second == []

    def test_missing_api_key_raises(self, monkeypatch, tmp_path):
        fake_genai = types.SimpleNamespace(Client=lambda api_key=None: _FakeGoogleClient(api_key=api_key))
        monkeypatch.setattr(emb_mod, "_get_google_genai", lambda: fake_genai)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleEmbeddingAdapter(corpus=SMALL_CORPUS, cache_dir=str(tmp_path))


# --- Tests: corpus hashing ---------------------------------------------------


class TestCorpusHash:
    def test_stable_for_same_corpus(self):
        a = _corpus_hash(SMALL_CORPUS)
        b = _corpus_hash(list(SMALL_CORPUS))
        assert a == b

    def test_changes_when_content_changes(self):
        a = _corpus_hash(SMALL_CORPUS)
        modified = list(SMALL_CORPUS)
        modified[0] = {**modified[0], "content": modified[0]["content"] + " edited"}
        b = _corpus_hash(modified)
        assert a != b


# --- Tests: engine adapter wiring -------------------------------------------


class TestBuildRetrievalAdapter:
    def test_unknown_adapter_raises(self, tmp_path):
        from mcp_llm_eval.engine import _build_retrieval_adapter
        with pytest.raises(ValueError, match="Unsupported"):
            _build_retrieval_adapter("not-real", str(tmp_path / "x"))

    def test_openai_small_dispatches_to_embeddings(self, monkeypatch, tmp_path):
        from mcp_llm_eval.engine import _build_retrieval_adapter
        _install_openai_mock(monkeypatch)
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("\n".join(json.dumps(c) for c in SMALL_CORPUS))
        a = _build_retrieval_adapter("openai-small", str(corpus))
        assert isinstance(a, OpenAIEmbeddingAdapter)
        assert a._model_name == "text-embedding-3-small"

    def test_openai_large_uses_large_model(self, monkeypatch, tmp_path):
        from mcp_llm_eval.engine import _build_retrieval_adapter
        _install_openai_mock(monkeypatch)
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("\n".join(json.dumps(c) for c in SMALL_CORPUS))
        a = _build_retrieval_adapter("openai-large", str(corpus))
        assert a._model_name == "text-embedding-3-large"

    def test_google_dispatches(self, monkeypatch, tmp_path):
        from mcp_llm_eval.engine import _build_retrieval_adapter
        _install_google_mock(monkeypatch)
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text("\n".join(json.dumps(c) for c in SMALL_CORPUS))
        a = _build_retrieval_adapter("google", str(corpus))
        assert isinstance(a, GoogleEmbeddingAdapter)
        assert a._model_name == "gemini-embedding-001"
