.PHONY: dev up down test lint format migrate

dev:
	uvicorn neuralake.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f api

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/neuralake/

format:
	ruff format src/ tests/

migrate:
	alembic upgrade head

seed:
	python scripts/seed_data.py

mcp:
	python -m neuralake.mcp.server
