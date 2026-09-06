.PHONY: help venv install install-a install-b install-c test lint fmt companions clean

help:
	@echo "make install      core deps + dev tools (CPU, fast)"
	@echo "make install-a    Part A extras (torch, gymnasium, SB3)"
	@echo "make install-b    Part B extras (transformers, TRL, PEFT) — GPU pod"
	@echo "make install-c    Part C extras (hftbacktest, numba, polars)"
	@echo "make test         run the smoke tests"
	@echo "make lint / fmt   ruff check / ruff format"
	@echo "make companions   clone the pinned companion forks into ./companions"

venv:
	uv venv --python 3.11

install:
	uv pip install -e ".[dev]"

install-a:
	uv pip install -e ".[part-a,dev]"

install-b:
	uv pip install -e ".[part-b,dev]"

install-c:
	uv pip install -e ".[part-c,dev]"

test:
	python -m pytest

lint:
	ruff check .

fmt:
	ruff format .

companions:
	bash scripts/clone_companions.sh

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info
