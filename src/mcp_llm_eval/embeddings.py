"""Embedding-based retrieval adapters: OpenAI and Google.

Both adapters implement the ``RetrievalAdapter`` protocol from ``retrieval.py``.
Tie-breaking matches BM25Adapter: descending similarity, then ascending corpus
index.

Embeddings for a given (corpus, model) pair are cached to
``{corpus_dir}/.embeddings-cache/{adapter_name}-{model}-{corpus_hash}.npz``.
The corpus hash covers chunk_id+content of every chunk, so any change to the
corpus invalidates the cache automatically.

SDK imports are lazy: ``openai`` / ``google.genai`` / ``numpy`` are only
imported when an adapter is instantiated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .types import RetrievedChunk


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _get_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "numpy not installed. Embedding adapters require it: "
            "pip install numpy"
        ) from e
    return np


def _get_openai() -> Any:
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "openai not installed. OpenAIEmbeddingAdapter requires it: "
            "pip install openai"
        ) from e
    return openai


def _get_google_genai() -> Any:
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "google-genai not installed. GoogleEmbeddingAdapter requires it: "
            "pip install google-genai"
        ) from e
    return genai


# ---------------------------------------------------------------------------
# Corpus validation + hashing — shared with BM25Adapter conventions
# ---------------------------------------------------------------------------


def _load_jsonl_corpus(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    chunks: list[dict[str, Any]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            chunk = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {lineno}: invalid JSON ({e})") from e
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Line {lineno}: expected JSON object, got {type(chunk).__name__}"
            )
        if "chunk_id" not in chunk:
            raise ValueError(f"Line {lineno}: missing chunk_id")
        if "content" not in chunk:
            raise ValueError(f"Line {lineno}: missing content")
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Line {lineno}: metadata must be a mapping")
        chunks.append(chunk)
    return chunks


def _validate_and_normalise_corpus(
    corpus: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not corpus:
        raise ValueError("corpus is empty")
    seen_ids: set[str] = set()
    out: list[dict[str, Any]] = []
    for i, chunk in enumerate(corpus):
        if not isinstance(chunk, dict):
            raise ValueError(f"corpus[{i}] must be a mapping")
        if "chunk_id" not in chunk or "content" not in chunk:
            raise ValueError(
                f"corpus[{i}] missing required keys: chunk_id and content"
            )
        cid = chunk["chunk_id"]
        if cid in seen_ids:
            raise ValueError(f"Duplicate chunk_id: {cid}")
        seen_ids.add(cid)
        out.append({
            "chunk_id": cid,
            "content": chunk["content"],
            "metadata": chunk.get("metadata", {}),
        })
    return out


def _corpus_hash(chunks: list[dict[str, Any]]) -> str:
    """Stable hash over (chunk_id, content) pairs, in original order."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(b"\x00")
        h.update(c["content"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(s: str) -> str:
    return _SAFE_NAME_RE.sub("_", s)


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _BaseEmbeddingAdapter:
    """Common embed-corpus / cache / cosine-retrieve logic.

    Subclasses implement ``_embed_batch(texts)`` and ``_adapter_name`` /
    ``_default_batch_size``.
    """

    _adapter_name: str = "embedding"
    _default_batch_size: int = 256

    def __init__(
        self,
        corpus: list[dict[str, Any]] | None = None,
        corpus_path: str | None = None,
        model_name: str = "",
        api_key: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if (corpus is None) == (corpus_path is None):
            raise ValueError(
                f"{type(self).__name__} requires exactly one of corpus or corpus_path"
            )

        if corpus_path is not None:
            corpus = _load_jsonl_corpus(corpus_path)
        assert corpus is not None
        self._chunks = _validate_and_normalise_corpus(corpus)
        self._model_name = model_name
        self._api_key = api_key

        # Cache dir: explicit override, else sibling of the corpus path, else cwd.
        if cache_dir:
            self._cache_dir = Path(cache_dir)
        elif corpus_path is not None:
            self._cache_dir = Path(corpus_path).resolve().parent / ".embeddings-cache"
        else:
            self._cache_dir = Path(".embeddings-cache").resolve()

        self._np = _get_numpy()
        self._client = self._build_client()

        chash = _corpus_hash(self._chunks)
        cache_name = (
            f"{self._adapter_name}-{_safe_filename(model_name)}-{chash}.npz"
        )
        cache_path = self._cache_dir / cache_name

        embeddings = self._load_cached(cache_path)
        if embeddings is None:
            embeddings = self._embed_corpus()
            self._save_cache(cache_path, embeddings)
        self._embeddings = embeddings  # shape (N, D), L2-normalised float32

    # -- subclass hooks --------------------------------------------------

    def _build_client(self) -> Any:
        raise NotImplementedError

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    # -- caching ---------------------------------------------------------

    def _load_cached(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            data = self._np.load(path)
            arr = data["embeddings"]
            if arr.shape[0] != len(self._chunks):
                return None
            return arr
        except Exception:
            return None

    def _save_cache(self, path: Path, embeddings: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._np.savez(path, embeddings=embeddings)

    # -- embedding -------------------------------------------------------

    def _embed_corpus(self) -> Any:
        contents = [c["content"] for c in self._chunks]
        all_vectors: list[list[float]] = []
        for i in range(0, len(contents), self._default_batch_size):
            batch = contents[i : i + self._default_batch_size]
            vectors = self._embed_batch(batch)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"{type(self).__name__}: batch returned {len(vectors)} "
                    f"vectors for {len(batch)} inputs"
                )
            all_vectors.extend(vectors)

        arr = self._np.asarray(all_vectors, dtype=self._np.float32)
        # L2-normalise so dot product == cosine similarity.
        norms = self._np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    # -- retrieve --------------------------------------------------------

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not query or not query.strip():
            return []

        qv = self._np.asarray(
            self._embed_batch([query])[0], dtype=self._np.float32
        )
        qn = self._np.linalg.norm(qv)
        if qn == 0:
            return []
        qv = qv / qn

        scores = self._embeddings @ qv  # (N,)
        # Sort: descending score, ascending index for ties.
        indexed = [(float(scores[i]), i) for i in range(len(self._chunks))]
        indexed.sort(key=lambda t: (-t[0], t[1]))
        top = indexed[:k]
        return [
            RetrievedChunk(
                chunk_id=self._chunks[i]["chunk_id"],
                content=self._chunks[i]["content"],
                score=score,
                metadata=dict(self._chunks[i]["metadata"]),
            )
            for score, i in top
        ]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIEmbeddingAdapter(_BaseEmbeddingAdapter):
    """Cosine-similarity retrieval over OpenAI text-embedding-3 models."""

    _adapter_name = "openai"
    _default_batch_size = 256  # OpenAI accepts up to ~2048 inputs per request

    def __init__(
        self,
        corpus: list[dict[str, Any]] | None = None,
        corpus_path: str | None = None,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            corpus=corpus,
            corpus_path=corpus_path,
            model_name=model_name,
            api_key=api_key,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        cache_dir: str | None = None,
    ) -> OpenAIEmbeddingAdapter:
        return cls(
            corpus_path=path,
            model_name=model_name,
            api_key=api_key,
            cache_dir=cache_dir,
        )

    def _build_client(self) -> Any:
        openai = _get_openai()
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY missing — pass api_key= or set the env var"
            )
        return openai.OpenAI(api_key=key)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


class GoogleEmbeddingAdapter(_BaseEmbeddingAdapter):
    """Cosine-similarity retrieval over Google Gemini embedding models.

    Google's embed_content batch endpoint caps at 100 inputs per request, so
    the default batch size is tuned down from the OpenAI default.
    """

    _adapter_name = "google"
    _default_batch_size = 100

    def __init__(
        self,
        corpus: list[dict[str, Any]] | None = None,
        corpus_path: str | None = None,
        model_name: str = "gemini-embedding-001",
        api_key: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(
            corpus=corpus,
            corpus_path=corpus_path,
            model_name=model_name,
            api_key=api_key,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        model_name: str = "gemini-embedding-001",
        api_key: str | None = None,
        cache_dir: str | None = None,
    ) -> GoogleEmbeddingAdapter:
        return cls(
            corpus_path=path,
            model_name=model_name,
            api_key=api_key,
            cache_dir=cache_dir,
        )

    def _build_client(self) -> Any:
        genai = _get_google_genai()
        key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY missing — pass api_key= or set the env var"
            )
        return genai.Client(api_key=key)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
        )
        return [list(e.values) for e in resp.embeddings]
