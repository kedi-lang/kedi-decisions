UV ?= uv

.PHONY: format format-check lint ty basedpyright check tests coverage build prod

format:
	$(UV) run --group dev ruff check --fix .
	$(UV) run --group dev ruff format .

format-check:
	$(UV) run --group dev ruff format --check .

lint:
	$(UV) run --group dev ruff check .

ty:
	$(UV) run --group dev ty check src packages/kedi-typesafe/src packages/kedi-laya/src

basedpyright:
	$(UV) run --group dev basedpyright

check: lint format-check ty basedpyright

tests:
	$(UV) run --group dev pytest -m "not live"

coverage:
	$(UV) run --group dev pytest -m "not live" --cov --cov-branch

build:
	$(UV) build --all-packages

prod: check coverage build
