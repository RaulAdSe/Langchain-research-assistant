.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check clean run-api run-cli ingest eval docs build docker

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
MYPY := mypy
FLAKE8 := flake8

# Help
help: ## Show this help message
	@echo "Multi-Agent Research Assistant - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev,test,docs]"

install-all: ## Install all dependencies
	$(PIP) install -e ".[dev,test,docs]"

# Testing
test: ## Run all tests
	$(PYTEST) tests/ -v

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v -m "unit"

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v -m "integration"

test-e2e: ## Run end-to-end tests only
	$(PYTEST) tests/e2e/ -v -m "e2e"

test-fast: ## Run tests excluding slow ones
	$(PYTEST) tests/ -v -m "not slow"

test-coverage: ## Run tests with coverage report
	$(PYTEST) tests/ --cov=app --cov-report=html --cov-report=term

test-watch: ## Run tests in watch mode
	$(PYTEST) tests/ -v --tb=short -x --lf --looponfail

# Code Quality
lint: ## Run all linting tools
	$(FLAKE8) app tests
	$(BLACK) --check app tests
	$(ISORT) --check-only app tests

format: ## Format code with black and isort
	$(BLACK) app tests
	$(ISORT) app tests

type-check: ## Run type checking with mypy
	$(MYPY) app

check-all: lint type-check ## Run all code quality checks

fix: format ## Auto-fix code formatting issues

# Application Commands
run-api: ## Start the FastAPI server
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

run-api-prod: ## Start the FastAPI server in production mode
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 4

cli: ## Run CLI (use with args: make cli ARGS="ask 'What is AI?'")
	$(PYTHON) -m app.cli $(ARGS)

ask: ## Ask a question via CLI (use: make ask Q="Your question")
	$(PYTHON) -m app.cli ask "$(Q)"

# Data Management
ingest-sample: ## Ingest sample documents
	$(PYTHON) -m app.cli sample

ingest: ## Ingest documents (use: make ingest PATH="/path/to/docs")
	$(PYTHON) -m app.cli ingest "$(PATH)"

stats: ## Show knowledge base statistics
	$(PYTHON) -m app.cli stats

reset-kb: ## Reset knowledge base
	$(PYTHON) -m app.cli reset

# Evaluation
eval: ## Run evaluation on test dataset
	$(PYTHON) -m app.eval.run_eval

eval-fast: ## Run evaluation on subset (first 5 questions)
	$(PYTHON) -m app.eval.run_eval --max-questions 5

eval-full: ## Run full evaluation with detailed output
	$(PYTHON) -m app.eval.run_eval --output-dir eval_results_full

# Development
watch: ## Start API with auto-reload for development
	uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

debug: ## Start API in debug mode
	$(PYTHON) -m debugpy --listen 5678 --wait-for-client -m uvicorn app.api:app --reload

dev-setup: install-dev ## Set up development environment
	pre-commit install
	@echo "Development environment set up successfully!"

# Documentation
docs-serve: ## Serve documentation locally
	mkdocs serve

docs-build: ## Build documentation
	mkdocs build

docs-deploy: ## Deploy documentation to GitHub Pages
	mkdocs gh-deploy

# Build and Distribution
build: ## Build package
	$(PYTHON) -m build

build-docker: ## Build Docker image
	docker build -t research-assistant .

run-docker: ## Run Docker container
	docker run -p 8000:8000 --env-file .env research-assistant

# Cleaning
clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

clean-data: ## Clean generated data and caches
	rm -rf chroma_db/
	rm -rf eval_results/
	rm -rf data/sample_docs/*.txt
	rm -rf data/sample_docs/*.md

# CI/CD helpers
ci-test: ## Run tests for CI environment
	$(PYTEST) tests/ -v --tb=short --cov=app --cov-report=xml

ci-check: ## Run all checks for CI
	$(FLAKE8) app tests
	$(BLACK) --check app tests
	$(ISORT) --check-only app tests
	$(MYPY) app

pre-commit: format lint type-check test-fast ## Run pre-commit checks

# Environment management
env-create: ## Create .env file from template
	cp .env.example .env
	@echo "Created .env file. Please edit with your API keys."

env-check: ## Check environment configuration
	$(PYTHON) -c "from app.core.config import settings; settings.validate_provider_keys(); print('✅ Configuration valid')"

# Quick shortcuts
up: run-api ## Alias for run-api
test-quick: test-unit ## Alias for test-unit
qa: check-all test-fast ## Quick quality assurance check