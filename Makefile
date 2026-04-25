.PHONY: setup build start test clean e2e benchmark benchmark-copy

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

clean:
	rm -rf $(VENV) dist/ build/ *.egg-info/ src/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/
