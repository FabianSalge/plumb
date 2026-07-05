.PHONY: test test-model lint typecheck run image kind-up deploy e2e

KIND_CLUSTER = plumb
# Digest-pinned default node image of the kind release in use (v0.32.0) —
# the kind project guarantees node images only for the release they ship with.
KIND_NODE_IMAGE = kindest/node:v1.36.1@sha256:3489c7674813ba5d8b1a9977baea8a6e553784dab7b84759d1014dbd78f7ebd5

test:
	uv run pytest --cov

test-model:  ## integration test against the real LettuceDetect weights (downloads ~1.2 GB once)
	uv run --extra model pytest -m model --no-cov

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

run:  ## serve the API locally (needs the model extra)
	uv run --extra model uvicorn api.main:app

image:  ## build the container image (CPU-only torch; weights download at start)
	docker build -t plumb:dev .

kind-up:  ## create the local kind cluster
	kind create cluster --name $(KIND_CLUSTER) --image $(KIND_NODE_IMAGE)

deploy: image  ## build the image, load it into kind, and install the chart
	kind load docker-image plumb:dev --name $(KIND_CLUSTER)
	helm upgrade --install plumb charts/plumb \
		--kube-context kind-$(KIND_CLUSTER) --wait --timeout 10m

e2e:  ## golden verify request against the chart in kind (run after `make deploy`)
	@set -e; \
	kubectl --context kind-$(KIND_CLUSTER) port-forward svc/plumb 8000:8000 >/dev/null & \
	pf=$$!; trap 'kill $$pf 2>/dev/null' EXIT; \
	for _ in $$(seq 30); do \
		curl -fsS -o /dev/null http://127.0.0.1:8000/healthz 2>/dev/null && break; sleep 1; \
	done; \
	PLUMB_URL=http://127.0.0.1:8000 uv run pytest -m e2e --no-cov
