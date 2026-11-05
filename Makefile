PY      ?= python
PIP     ?= $(PY) -m pip
SRC     := qts
TESTS   := tests
PYTEST  := $(PY) -m pytest

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-fast test-slow coverage demo demo-ml clean precommit

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev extras and git hooks
	$(PIP) install -e ".[dev]"
	pre-commit install --install-hooks || true

lint: ## Ruff lint (no fixes)
	ruff check $(SRC) $(TESTS) scripts

format: ## Ruff autofix + format (rewrites files)
	ruff check --fix $(SRC) $(TESTS) scripts
	ruff format $(SRC) $(TESTS) scripts

typecheck: ## Pyright: strict on qts/core, basic on qts
	pyright $(SRC)/core
	pyright $(SRC)

test: ## Full test suite
	$(PYTEST) -q

test-fast: ## Fast tier (excludes tests marked slow)
	$(PYTEST) -q -m "not slow"

test-slow: ## Slow tier only (walk-forward ML, full pipeline)
	$(PYTEST) -q -m "slow"

coverage: ## Fast tier with coverage report (term + coverage.xml + htmlcov/)
	$(PYTEST) -q -m "not slow" --cov=$(SRC) --cov-report=term --cov-report=xml --cov-report=html

demo: ## Phase-1 demo on the synthetic market (rank composite)
	$(PY) scripts/run_demo.py

demo-ml: ## Phase-2 demo with the walk-forward ML sleeve
	$(PY) scripts/run_demo.py --ml

precommit: ## Run all pre-commit hooks on all files
	pre-commit run --all-files

clean: ## Remove caches and build/coverage artefacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage coverage.xml htmlcov .hypothesis
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# --------------------------------------------------------------------------- workbench (api/ + ui/)
.PHONY: api-install api-dev api-test ui-dev dev

api-install: ## Editable install of the API (qts-api) with test extras
	$(PIP) install -e "api[dev]"

api-dev: ## API on :8000 with autoreload (QTS_RUNS_ROOT defaults to ./runs)
	$(PY) -m uvicorn qts_api.main:app --reload --port 8000 --reload-dir api/qts_api

api-test: ## API test suite (builds one tiny real run per session, ~45 s)
	cd api && $(PYTEST) -q

ui-dev: ## Vite dev server on :5173 (proxies /api to :8000)
	npm --prefix ui run dev

dev: ## API + UI dev servers together (Ctrl-C stops both)
	@trap 'kill 0' INT TERM EXIT; \
	$(MAKE) --no-print-directory api-dev & \
	$(MAKE) --no-print-directory ui-dev & \
	wait
