# Changelog

## [0.5.0](https://github.com/berkayildi/mcp-llm-eval/compare/v0.4.1...v0.5.0) (2026-04-25)


### Features

* **cli:** add evaluate-retrieval and evaluate-rag subcommands ([f6133ba](https://github.com/berkayildi/mcp-llm-eval/commit/f6133ba2af13f6ca4f7b0e86be205f009e84baf8))
* **config:** extend ThresholdConfig with retrieval and RAG thresholds ([725c56e](https://github.com/berkayildi/mcp-llm-eval/commit/725c56e42be6ae2eae9d231c718d58ced6b418ec))
* **engine:** add run_rag_evaluation orchestrator and retrieval drift check ([202a741](https://github.com/berkayildi/mcp-llm-eval/commit/202a7414cc1c0b940caba8500b445ea73a95ee7a))
* **judge:** add context_relevance and citation_faithfulness judges ([8b5074b](https://github.com/berkayildi/mcp-llm-eval/commit/8b5074becf55aca1b8b02324a9b9925d6de14ca0))
* **metrics:** add retrieval_metrics module with recall, precision, MRR, nDCG ([91ac8d0](https://github.com/berkayildi/mcp-llm-eval/commit/91ac8d04dc220e3f28ae20c71dd5285c9b0f4176))
* **retrieval:** add RetrievalAdapter protocol and BM25 in-memory adapter ([7109118](https://github.com/berkayildi/mcp-llm-eval/commit/7109118d724a211200c152ea4e5c8c70e4b761ac))
* **server:** add evaluate_retrieval, evaluate_rag, drift, and poisoned-corpus MCP tools ([c8a1ab1](https://github.com/berkayildi/mcp-llm-eval/commit/c8a1ab12c92bcdd84aea7ed3fadbc14e6290591f))
* **types:** add retrieval and RAG result types, extend EvalEntry ([650fbc0](https://github.com/berkayildi/mcp-llm-eval/commit/650fbc071f0668abf70f79328ccd155272708a47))


### Documentation

* **design:** add v0.5.0 retrieval eval design spec ([5bf7e04](https://github.com/berkayildi/mcp-llm-eval/commit/5bf7e041b9678e86a305379f2de2412c45c516b5))

## [0.4.1](https://github.com/berkayildi/mcp-llm-eval/compare/v0.4.0...v0.4.1) (2026-04-19)


### Documentation

* add benchmark workflow and CI flow diagram ([e1e0900](https://github.com/berkayildi/mcp-llm-eval/commit/e1e09005c6f35d25622d8fdaa7ca43aaf189fab1))
* update readme ([46f8de5](https://github.com/berkayildi/mcp-llm-eval/commit/46f8de5fc5c97200ac6bbcf59038b13af71b4ad8))

## [0.4.0](https://github.com/berkayildi/mcp-llm-eval/compare/v0.3.0...v0.4.0) (2026-04-19)


### Features

* expand eval dataset to 9 questions, 5 models, add benchmark targets ([67c4a02](https://github.com/berkayildi/mcp-llm-eval/commit/67c4a028edaca10fe382c57c712638a1ba32f42b))


### Bug Fixes

* **providers:** disable gemini thinking for benchmark parity ([9d3e15a](https://github.com/berkayildi/mcp-llm-eval/commit/9d3e15afecbfe64c2dcf519a5398240ee2909088))

## [0.3.0](https://github.com/berkayildi/mcp-llm-eval/compare/v0.2.0...v0.3.0) (2026-04-16)


### Features

* add e2e test, CLI inline model flags, env example ([701d51c](https://github.com/berkayildi/mcp-llm-eval/commit/701d51cf6a1591e336c9b6ab32d7946e1a330717))

## [0.2.0](https://github.com/berkayildi/mcp-llm-eval/compare/v0.1.0...v0.2.0) (2026-04-12)


### Features

* add compare_runs, format_pr_comment tools, CLI mode, config loader ([b3d54ee](https://github.com/berkayildi/mcp-llm-eval/commit/b3d54ee00edb527ba488fc6314702b307419e36c))

## 0.1.0 (2026-04-12)


### Features

* initial MCP server with eval engine, 4 tools, 153 tests ([a8cb3b5](https://github.com/berkayildi/mcp-llm-eval/commit/a8cb3b52236c7f4fabbed8dac273257eeb8fc841))

## Changelog
