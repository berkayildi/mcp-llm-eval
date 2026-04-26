# CLAUDE.md — mcp-llm-eval

This file helps Claude Code understand the project structure and conventions for future sessions.

## Project overview

**mcp-llm-eval** is a local MCP (Model Context Protocol) server written in Python. It packages LLM evaluation gates as reusable CI/CD primitives — load a dataset, run models, score with an LLM-as-judge, and check quality thresholds, all exposed as MCP tools that AI agents can call.

Transport: **stdio** (standard input/output).

---

## Directory structure

```
mcp-llm-eval/
├── src/
│   └── mcp_llm_eval/
│       ├── __init__.py          # Package version (__version__ = "0.1.0")
│       ├── server.py            # MCP server + tool registration (6 tools) + entry point routing
│       ├── engine.py            # Eval engine: dataset loader, LLM runner, judge, threshold checker
│       ├── cli.py               # CLI argument parsing and subcommand routing (run, check, compare, comment)
│       ├── config.py            # .eval-gate.yml loader and validator
│       ├── comparison.py        # compare_runs logic: regression detection with tolerance
│       ├── formatter.py         # PR comment markdown generator
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── anthropic.py     # Streaming runner for Anthropic (messages.stream, TTFT capture)
│       │   ├── openai.py        # Streaming runner for OpenAI (chat.completions.create stream)
│       │   └── google.py        # Streaming runner for Google GenAI (generate_content_stream)
│       ├── judge.py             # LLM-as-judge scorer (faithfulness + relevance, 0-1 scale)
│       ├── retrieval.py         # RetrievalAdapter protocol + BM25Adapter
│       ├── retrieval_metrics.py # IR metrics (recall@k, precision@k, MRR, nDCG@k)
│       ├── embeddings.py        # OpenAIEmbeddingAdapter + GoogleEmbeddingAdapter (v0.7.0)
│       └── types.py             # Shared dataclasses: EvalEntry, EvalResult, RunSummary, ThresholdConfig
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── sample_dataset.json  # 3 sample eval entries (one per category: factual, reasoning, summarization)
│   ├── test_server.py           # MCP tool integration tests (6 tools)
│   ├── test_engine.py           # Eval engine unit tests
│   ├── test_providers.py        # Provider runner tests (mock API calls)
│   ├── test_judge.py            # Judge scoring tests (mock OpenAI)
│   ├── test_types.py            # Type validation tests
│   ├── test_cli.py              # CLI argument parsing, subcommand routing, exit codes
│   ├── test_config.py           # YAML loading, validation, defaults, error handling
│   ├── test_comparison.py       # Regression detection, tolerance math, edge cases
│   └── test_formatter.py        # Markdown generation, with/without comparison and thresholds
├── pyproject.toml               # Build config (hatchling), deps, pytest settings
├── Makefile                     # setup / build / start / test / clean
├── README.md                    # User-facing docs
├── CLAUDE.md                    # This file
├── LICENSE                      # MIT
├── CHANGELOG.md                 # Release Please manages this
├── release-please-config.json
├── .release-please-manifest.json
└── .github/
    └── workflows/
        └── release.yml          # Release Please + PyPI OIDC publish
```

---

## Key design decisions

1. **Lazy provider imports** — Provider SDKs (anthropic, openai, google-genai) are NOT in core dependencies. Each provider module imports its SDK lazily in `_get_client()` and raises a clear `ImportError` with install instructions if missing. Users install only what they need: `pip install mcp-llm-eval anthropic openai`.

2. **No LangChain dependency** — The project deliberately avoids LangChain. `langsmith` SDK is only used for optional tracing and is not a dependency.

3. **Single-responsibility modules** — Logic is split across focused files: `types.py` (dataclasses), `engine.py` (orchestration), `judge.py` (scoring), `providers/` (LLM runners), `server.py` (MCP tool registration). This is different from mcp-tfstate-reader's single-file approach because eval has more moving parts.

4. **Private helpers are directly importable in tests** — Test files import `_aggregate`, `_get_runner`, `load_dataset`, etc. directly. Keep helper functions at module level so tests can access them.

5. **`asyncio_mode = "auto"`** is set in `pyproject.toml`, so `@pytest.mark.asyncio` is optional.

6. **All external API calls mocked in tests** — Zero real LLM calls. Provider runners, judge scoring, and engine orchestration all use mock clients.

---

## MCP tool signatures

| Tool | Required inputs | Returns |
|------|----------------|---------|
| `run_evaluation` | `dataset_path: str`, `models: array` | JSON with per-question scores, aggregate summary, pass/fail |
| `check_thresholds` | `results_path: str`, `thresholds: object` | JSON with pass/fail per metric, overall gate status |
| `list_evaluations` | `results_dir: str` | JSON array of past runs with metadata |
| `get_evaluation` | `results_path: str` | Full JSON of a specific evaluation run |
| `compare_runs` | `baseline_path: str`, `current_path: str` | JSON with per-model metric deltas and regression flags |
| `format_pr_comment` | `summary_path: str` | Markdown string ready to post as a GitHub PR comment |

---

## CLI subcommands

The `mcp-llm-eval` entry point routes based on CLI arguments:
- No args → starts MCP stdio server (unchanged from v0.1.0)
- `run` → run full evaluation using .eval-gate.yml config
- `check` → check thresholds (exit 1 on failure, blocks PRs in CI)
- `compare` → compare two runs for regressions (exit 1 if found)
- `comment` → generate markdown PR comment

---

## Adding a new CLI subcommand

1. Add a subparser in `cli.py` `cli_main()` with `subparsers.add_parser("name", ...)`.
2. Add arguments to the subparser.
3. Add routing in the `if args.command == "name":` block.
4. Implement `_cmd_name(args)` function. Use `sys.exit(1)` for failure, `sys.exit(0)` for success.
5. Add tests in `tests/test_cli.py` that call `cli_main(["name", ...])` and assert exit codes.

---

## Adding a new provider

1. Create `src/mcp_llm_eval/providers/<name>.py` with:
   - `_get_client()` function that lazily imports the SDK and returns a client
   - `run(client, model, system_prompt, question, max_tokens)` function that returns a dict with: `response`, `input_tokens`, `output_tokens`, `stop_reason`, `time_to_first_token_ms`, `total_latency_ms`
2. Register the provider in `engine.py` `_get_runner()` and `_get_client()` functions.
3. Add tests in `tests/test_providers.py` with a mocked SDK client.

---

## Adding a new retrieval adapter

Available out of the box: `bm25` (lexical, rank_bm25), `openai-small` (text-embedding-3-small), `openai-large` (text-embedding-3-large), `google` (gemini-embedding-001). Embedding adapters share `_BaseEmbeddingAdapter` in `embeddings.py`, which handles corpus validation, cosine retrieval, and on-disk caching at `{corpus_dir}/.embeddings-cache/{name}-{model}-{hash}.npz`.

To add a new one (e.g., Cohere, OpenSearch):

1. For another embedding provider: subclass `_BaseEmbeddingAdapter` in `embeddings.py`, implement `_build_client()` and `_embed_batch(texts)`. Set `_adapter_name` and `_default_batch_size` (Google's batch limit is 100; OpenAI's is ~2048).
2. For a non-embedding store (BM25-style or external service): implement the `RetrievalAdapter` Protocol directly — a class with `retrieve(query, k) -> list[RetrievedChunk]` and a `from_jsonl(path)` classmethod.
3. Register the adapter name in `engine._build_retrieval_adapter()` and append it to `engine.SUPPORTED_RETRIEVAL_ADAPTERS`.
4. Add tests in `tests/test_embeddings.py` (or `test_retrieval.py`) with a mocked SDK; never make real API calls in tests.

---

## Adding a new metric

1. Add the metric field to `ThresholdConfig` in `types.py` (with `None` default).
2. Add threshold checking logic in `engine.py` `check_thresholds()`.
3. If it requires new data collection, update the relevant provider `run()` return dict and `EvalResult` dataclass.
4. Add the metric to the `check_thresholds` MCP tool's `inputSchema` in `server.py`.
5. Add tests in `test_engine.py` for the new threshold check.

---

## Running

```bash
make setup    # create venv + install deps
make test     # pytest tests/ -v
make start    # python -m mcp_llm_eval.server (stdio mode)
make build    # python -m build
make clean    # remove venv, dist, caches
```

---

## Output schema reference

### Summary JSON (`{timestamp}_summary.json`)

```json
{
  "timestamp": "20250101_120000",
  "total_questions": 10,
  "total_model_runs": 30,
  "total_errors": 0,
  "total_elapsed_sec": 120.5,
  "total_estimated_cost": 0.045,
  "judge_model": "gpt-4o-mini",
  "overall": {
    "model-name": {
      "provider": "openai",
      "model": "model-name",
      "runs": 10,
      "avg_ttft_ms": 150.0,
      "avg_latency_ms": 2000.0,
      "avg_input_tokens": 500.0,
      "avg_output_tokens": 200.0,
      "avg_cost_per_query": 0.0015,
      "avg_faithfulness": 0.85,
      "avg_relevance": 0.90
    }
  },
  "by_category": { ... },
  "results": [ ... ]
}
```

### Benchmark JSON (`{timestamp}_benchmark.json`)

```json
{
  "timestamp": "20250101_120000",
  "total_entries": 10,
  "models": ["model-a", "model-b"],
  "judge_model": "gpt-4o-mini",
  "total_runs": 30,
  "total_elapsed_sec": 120.5,
  "results": [
    {
      "eval_id": "entry-001",
      "category": "factual",
      "model": "model-name",
      "provider": "openai",
      "response": "...",
      "input_tokens": 500,
      "output_tokens": 200,
      "stop_reason": "stop",
      "time_to_first_token_ms": 150,
      "total_latency_ms": 2000,
      "cost_per_query": 0.0015,
      "faithfulness_score": 0.9,
      "faithfulness_reason": "...",
      "relevance_score": 0.85,
      "relevance_reason": "...",
      "judge_model": "gpt-4o-mini"
    }
  ]
}
```
