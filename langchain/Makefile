.PHONY: test integration_tests lint format

test:
	uv run pytest --disable-socket --allow-unix-socket tests/unit_tests

integration_tests:
	uv run pytest tests/integration_tests

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run --group examples mypy --strict langchain_aicheckerhumanizer tests examples

format:
	uv run ruff format .
	uv run ruff check --fix .
