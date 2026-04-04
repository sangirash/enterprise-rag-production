.PHONY: install dev test lint format build up down clean eval

install:
	pip install -e ".[dev]"

dev:
	uvicorn src.rag.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

build:
	docker build -t enterprise-rag:latest .

up:
	docker compose up -d

down:
	docker compose down

clean:
	docker compose down -v
	find . -type d -name __pycache__ | xargs rm -rf
	rm -rf .coverage htmlcov/ dist/ *.egg-info

eval:
	python scripts/run_eval.py
