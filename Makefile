.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python
BENCH := $(VENV)/bin/document-ocr-bench
DATASET := benchmarks/document-ocr

.PHONY: help venv install test samples providers run report docker docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv (Python 3.12+)
	python3.12 -m venv $(VENV) || python3 -m venv $(VENV)

install: venv ## Install the harness editable with tesseract + dev extras
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e "packages/document-ocr-benchmarks[tesseract,dev]"

test: ## Run the test suite
	$(PY) -m pytest packages/document-ocr-benchmarks/tests

samples: ## Generate the synthetic Nigerian dataset
	$(BENCH) gen-samples --out $(DATASET)

providers: ## List provider candidates + capabilities + license posture
	$(BENCH) providers

run: ## Run the benchmark (PROVIDERS overrides which to run)
	$(BENCH) run --providers $(or $(PROVIDERS),tesseract)

report: ## Rebuild the report from the latest results dir
	$(BENCH) report $$(ls -dt $(DATASET)/results/*/ | head -1)

docker: ## Build the harness Docker image
	docker build -t innovantics/document-ocr-bench:0.1.0 .

docker-run: ## Run the benchmark inside Docker against the mounted dataset
	docker run --rm -v "$$PWD/benchmarks:/app/benchmarks" \
		innovantics/document-ocr-bench:0.1.0 run --providers $(or $(PROVIDERS),tesseract)

clean: ## Remove benchmark results and caches
	rm -rf $(DATASET)/results/* .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
