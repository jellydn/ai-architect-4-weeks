.PHONY: help install dev test lint format typecheck server ingest query clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/python -m pytest
UV := uv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(UV) pip install .

dev: ## Install dev dependencies
	$(UV) pip install ".[dev]"

venv: ## Create virtual environment
	$(UV) venv .venv

setup: venv dev ## Full setup (venv + deps + dev tools)
	@cp -n .env.example .env 2>/dev/null || true
	@echo "Setup complete. Edit .env with your OPENAI_API_KEY"

test: ## Run all tests
	$(PYTEST) week-1/test_rag.py -v

test-fast: ## Run tests without coverage
	$(PYTEST) week-1/test_rag.py -v --tb=short

lintsymotion-s): ## Run linter
	uvx ruff check week-1/

lint-fix: ## Run linter with auto-fix
	uvx ruff check week-1/ --fix

format: ## Format code
	uvx ruff format week-1/

typecheck: ## Run type checker
	uvx ty check week-1/

check: lint test ## Run lint + tests

server: ## Start FastAPI server
	cd week-1 && $(PYTHON) main.py

server-dev: ## Start server with reload
	cd week-1 && ../.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

ingest: ## Ingest sample document
	curl -X POST http://localhost:8000/ingest \
		-H "Content-Type: application/json" \
		-d '{"file_paths": ["../data/sample.txt"]}'

query: ## Query RAG (usage: make query Q="What is RAG?")
	curl -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"query": "$(Q)", "top_k": 3}'

health: ## Check server health
	curl http://localhost:8000/health

demo: ## Run full demo (ingest + query)
	@echo "=== Ingesting sample document ==="
	@make ingest
	@echo "\n=== Querying: What is RAG? ==="
	@make query Q="What is RAG?"

clean: ## Clean cache and temp files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f test_sample.txt 2>/dev/null || true

# Week 1 specific targets
week1-test: ## Run Week 1 ingestion test
	cd week-1 && $(PYTHON) ingestion.py

week1-check: lint test ## Verify Week 1 is complete
	@echo "\n✅ Week 1 RAG Foundation verified!"
