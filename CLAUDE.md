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
│       ├── server.py            # MCP server + tool registration (4 tools)
│       ├── engine.py            # Eval engine: dataset loader, LLM runner, judge, threshold checker
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── anthropic.py     # Streaming runner for Anthropic (messages.stream, TTFT capture)
│       │   ├── openai.py        # Streaming runner for OpenAI (chat.completions.create stream)
│       │   └── google.py        # Streaming runner for Google GenAI (generate_content_stream)
│       ├── judge.py             # LLM-as-judge scorer (faithfulness + relevance, 0-1 scale)
│       └── types.py             # Shared dataclasses: EvalEntry, EvalResult, RunSummary, ThresholdConfig
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── sample_dataset.json  # 3 sample eval entries (one per category: factual, reasoning, summarization)
│   ├── test_server.py           # MCP tool integration tests
│   ├── test_engine.py           # Eval engine unit tests
│   ├── test_providers.py        # Provider runner tests (mock API calls)
│   ├── test_judge.py            # Judge scoring tests (mock OpenAI)
│   └── test_types.py            # Type validation tests
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

---

## Adding a new provider

1. Create `src/mcp_llm_eval/providers/<name>.py` with:
   - `_get_client()` function that lazily imports the SDK and returns a client
   - `run(client, model, system_prompt, question, max_tokens)` function that returns a dict with: `response`, `input_tokens`, `output_tokens`, `stop_reason`, `time_to_first_token_ms`, `total_latency_ms`
2. Register the provider in `engine.py` `_get_runner()` and `_get_client()` functions.
3. Add tests in `tests/test_providers.py` with a mocked SDK client.

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
      "eval_id": "adr-001",
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
