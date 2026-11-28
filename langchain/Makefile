.PHONY: test integration_tests lint format

test:
	uv run --locked pytest tests/unit_tests

integration_tests:
	uv run pytest tests/integration_tests

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked --group examples mypy --strict langchain_aicheckerhumanizer tests examples

format:
	uv run ruff format .
	uv run ruff check --fix .
