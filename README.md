# Plumb

[![ci](https://github.com/FabianSalge/plumb/actions/workflows/ci.yml/badge.svg)](https://github.com/FabianSalge/plumb/actions/workflows/ci.yml)

The open-source, self-hostable groundedness gate — one calibrated verifier that checks AI-generated answers against your own knowledge base and gates both CI and production on the result, without data leaving your cluster.

> **Status:** early development — a first `/v1/verify` endpoint exists and deploys to Kubernetes via the Helm chart.

Work is planned and tracked publicly via [milestones](https://github.com/FabianSalge/plumb/milestones) and [issues](https://github.com/FabianSalge/plumb/issues).

## Try it locally

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). The first run downloads the scoring model (~420 MB).

```sh
make run
```

```sh
curl -s localhost:8000/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The capital of France is Paris.",
    "context": ["Paris is the capital of France."],
    "mode": "fast"
  }'
```

```json
{
  "claims": [
    {"text": "The capital of France is Paris.", "verdict": "supported", "score": 0.9, "evidence_index": 0}
  ],
  "gate": "pass",
  "engine_version": "0.1.0",
  "config_version": "0.1.0"
}
```

The whole input is treated as one claim and checked against the caller-provided evidence with a single grounding signal (HHEM-2.1-open, pinned by revision in [config/verifier.yaml](config/verifier.yaml)). Verdicts are `supported`/`unsupported` only — `contradicted` arrives with the NLI signal. Claim decomposition, retrieval, tenancy, and calibration are on the [roadmap](https://github.com/FabianSalge/plumb/milestones).

### Container

```sh
make image   # build plumb:dev
docker run --rm -p 8000:8000 plumb:dev
```

The image runs as a non-root user with CPU-only torch. Model weights are not baked in: the container downloads them on first start (~420 MB, cached under `HF_HOME`), and `/readyz` returns 503 until the model is loaded.

## Deploy with Helm

The chart in [charts/plumb](charts/plumb) is the product's front door: a probed, resource-bounded Deployment with the verifier config injected from values.

```sh
helm install plumb charts/plumb
```

Highlights of [values.yaml](charts/plumb/values.yaml):

- `verifier.*` — the signal model pin and threshold, rendered into a ConfigMap the API loads via `PLUMB_CONFIG`. Changing the threshold is a values change and a rollout, never an image rebuild.
- `networkPolicy.enabled` (default `true`) — egress is default-deny except DNS and TCP 443, the one hole that lets the pod fetch the pinned model weights on first start. On clusters that mirror or pre-bake weights, set `networkPolicy.allowModelDownload=false` to close it and run fully sovereign.
- `image.repository` / `image.tag` — no registry is assumed; bring your own or use the kind flow below.

To see it live on a local [kind](https://kind.sigs.k8s.io) cluster (requires docker, kind, and helm):

```sh
make kind-up   # create the cluster
make deploy    # build the image, load it into kind, install the chart
make e2e       # golden verify request against the deployed chart
```

CI runs this same flow on every PR: kind cluster, chart install, one golden
verify request asserting a `supported` verdict.

## Development

```sh
pre-commit install --hook-type commit-msg --hook-type pre-commit
make test        # pytest with the coverage floor
make lint        # ruff check + format check
make typecheck   # mypy (strict)
make test-model  # integration test against the real HHEM weights
```
