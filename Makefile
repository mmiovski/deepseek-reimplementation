.PHONY: check test lint format type clean

check: format lint test type

test:
	python -m pytest

lint:
	python -m ruff check .

format:
	python -m black --check .

type:
	python -m mypy deepseek_reimpl

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
