.PHONY: test test-model lint typecheck run image

test:
	uv run pytest --cov

test-model:  ## integration test against the real HHEM weights (downloads ~420 MB once)
	uv run --extra hhem pytest -m model --no-cov

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

run:  ## serve the API locally (needs the model extra)
	uv run --extra hhem uvicorn api.main:app

image:  ## build the container image (CPU-only torch; weights download at start)
	docker build -t plumb:dev .
