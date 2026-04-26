.PHONY: setup build start test clean e2e benchmark benchmark-copy benchmark-retrieval benchmark-retrieval-copy benchmark-embeddings benchmark-embeddings-copy

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install anthropic openai google-genai

build:
	$(PYTHON) -m build

start:
	$(PYTHON) -m mcp_llm_eval.server

test:
	$(VENV)/bin/pytest tests/ -v

e2e:
	@if [ -f .env ]; then set -a && . .env && set +a; fi && \
	PATH="$(VENV)/bin:$$PATH" bash eval/e2e.sh

benchmark:
	@if [ ! -f .env ]; then echo "Error: .env required with provider keys"; exit 1; fi
	@set -a && . ./.env && set +a && uvx --with anthropic --with openai --with google-genai \
		mcp-llm-eval run --config .eval-gate.yml --dataset eval/dataset.json --output-dir eval/results

benchmark-copy:
	cp eval/results/latest_summary.json ../llm-benchmarks/text-generation/eval-gates-summary.json
	cp eval/results/$$(ls -t eval/results/*_benchmark.json | head -1 | xargs basename) ../llm-benchmarks/text-generation/eval-gates-benchmark.json
	@echo "Copied to ../llm-benchmarks/text-generation/"

benchmark-retrieval:
	@if [ ! -f .env ]; then echo "Error: .env required with provider keys"; exit 1; fi
	@set -a && . ./.env && set +a && uvx --with anthropic --with openai --with google-genai \
		mcp-llm-eval evaluate-rag \
		--dataset eval/retrieval_dataset.jsonl \
		--corpus eval/retrieval_corpus.jsonl \
		--config .eval-gate.yml \
		--output-dir eval/results

benchmark-retrieval-copy:
	@if [ ! -d "../llm-benchmarks/retrieval" ]; then \
		mkdir -p ../llm-benchmarks/retrieval; \
		echo "Created ../llm-benchmarks/retrieval/"; \
	fi
	cp eval/results/latest_rag_summary.json ../llm-benchmarks/retrieval/eval-gates-rag-summary.json
	cp $$(ls -t eval/results/*_rag_benchmark.json | head -1) ../llm-benchmarks/retrieval/eval-gates-rag-benchmark.json
	@echo "Copied retrieval results to ../llm-benchmarks/retrieval/"

benchmark-embeddings:
	@if [ ! -f .env ]; then echo "Error: .env required with provider keys"; exit 1; fi
	@set -a && . ./.env && set +a && uvx --with anthropic --with openai --with google-genai --with numpy \
		mcp-llm-eval evaluate-rag-multi \
		--dataset eval/retrieval_dataset.jsonl \
		--corpus eval/retrieval_corpus.jsonl \
		--config eval/.eval-gate-embeddings.yml \
		--output-dir eval/results

benchmark-embeddings-copy:
	@if [ ! -d "../llm-benchmarks/retrieval" ]; then \
		echo "Error: ../llm-benchmarks/retrieval not found"; exit 1; \
	fi
	cp eval/results/latest_embeddings_summary.json ../llm-benchmarks/retrieval/embeddings-summary.json
	cp $$(ls -t eval/results/*_embeddings_benchmark.json | head -1) ../llm-benchmarks/retrieval/embeddings-benchmark.json
	@echo "Copied embeddings results to ../llm-benchmarks/retrieval/"

clean:
	rm -rf $(VENV) dist/ build/ *.egg-info/ src/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/
