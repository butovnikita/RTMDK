# RTMDK Pipeline v8.3+ — Makefile
.PHONY: help install test pipeline-test benchmark load-test docker-up docker-down diagnose lint format

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements-prod.txt -r requirements-dev.txt

test: ## Run all tests
	pytest tests/ -v --tb=short

pipeline-test: ## Run pipeline-specific tests only
	pytest tests/test_pipeline_*.py -v --tb=short

pipeline-fast: ## Run pipeline tests in parallel
	pytest tests/test_pipeline_*.py -v --tb=short -n auto

benchmark: ## Run pipeline vs legacy benchmark
	python scripts/bench_pipeline_ab.py --queries 100 --nodes 500

benchmark-production: ## Run production benchmark on all datasets
	python scripts/bench_pipeline_production.py --all-datasets --output benchmarks/

load-test: ## Load test pipeline endpoints (requires running server)
	python scripts/load_test_pipeline.py --endpoint query_pipeline --rps 10 --duration 30
	python scripts/load_test_pipeline.py --endpoint stream --rps 5 --duration 30
	python scripts/load_test_pipeline.py --endpoint health --rps 50 --duration 10

diagnose: ## Run pipeline diagnostics
	python -m rtmdk pipeline-diagnose --preset local

lint: ## Run linters
	flake8 rtmdk tests --max-line-length=120
	mypy rtmdk --config-file mypy.ini

format: ## Format code
	black rtmdk tests
	isort rtmdk tests

docker-up: ## Start production stack with Prometheus + Grafana
	docker-compose -f docker-compose.prod.yml up -d

docker-down: ## Stop production stack
	docker-compose -f docker-compose.prod.yml down -v

docker-logs: ## View RTMDK logs
	docker-compose -f docker-compose.prod.yml logs -f rtmdk

pipeline-health: ## Check pipeline health (requires running server)
	curl -s http://localhost:8080/v1/memory/pipeline/health | python -m json.tool

pipeline-metrics: ## View pipeline metrics (requires running server)
	curl -s http://localhost:8080/v1/memory/pipeline/metrics | python -m json.tool

pipeline-prometheus: ## View pipeline Prometheus metrics (requires running server)
	curl -s http://localhost:8080/v1/memory/pipeline/prometheus

clean: ## Clean caches and temp files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	rm -rf .pytest_cache .mypy_cache
