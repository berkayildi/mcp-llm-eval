.PHONY: setup build start test clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3.10 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

build:
	$(PYTHON) -m build

start:
	$(PYTHON) -m mcp_llm_eval.server

test:
	$(VENV)/bin/pytest tests/ -v

clean:
	rm -rf $(VENV) dist/ build/ *.egg-info/ src/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/
