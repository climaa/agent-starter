# OFFLINE=1 → fake LLM + local hashing embeddings (no credentials, no network).
OFFLINE ?= 0
ifeq ($(OFFLINE),1)
export LLM_PROVIDER := fake
export EMBEDDINGS_PROVIDER := local
endif

.PHONY: setup dev test lint fmt ingest demo eval docker-up docker-down

setup:            ## install deps (incl. dev) and create .env if missing
	uv sync --all-extras
	@test -f .env || cp .env.example .env

dev: ingest       ## run the API with reload on :8000
	uv run uvicorn agent_starter.api.main:app --reload --port 8000

test:             ## unit + e2e tests (always offline: fakes only)
	uv run pytest -q

lint:             ## ruff (lint + format check) and mypy --strict on src
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff check --fix .
	uv run ruff format .

ingest:           ## build the vector index from ./data
	uv run agent-ingest

demo: ingest      ## run the scripted demo questions (or: make demo Q="your question")
	uv run agent-demo $(if $(Q),"$(Q)",)

eval: ingest      ## retrieval hit-rate + faithfulness table
	uv run agent-eval --min-hit-rate 0.8

docker-up:
	docker compose up --build

docker-down:
	docker compose down
