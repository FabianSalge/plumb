# Plumb

[![ci](https://github.com/FabianSalge/plumb/actions/workflows/ci.yml/badge.svg)](https://github.com/FabianSalge/plumb/actions/workflows/ci.yml)

The open-source, self-hostable groundedness gate — one calibrated verifier that checks AI-generated answers against your own knowledge base and gates both CI and production on the result, without data leaving your cluster.

> **Status:** early development — the tracer bullet works end to end: `/v1/verify` checks answers against provided context with a benchmarked LettuceDetect groundedness signal, returns calibrated confidences, and deploys to Kubernetes via the Helm chart.

Work is planned and tracked publicly via [milestones](https://github.com/FabianSalge/plumb/milestones) and [issues](https://github.com/FabianSalge/plumb/issues); progress is journaled in [docs/devlog/](docs/devlog/).

## Try it locally

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). The first run downloads the scoring model (~1.2 GB).

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
    {"text": "The capital of France is Paris.", "start": 0, "end": 31, "verdict": "supported", "confidence": 0.986, "spans": []}
  ],
  "gate": "pass",
  "engine_version": "0.2.0",
  "config_version": "0.5.0"
}
```

The input is decomposed into claims — one verbatim sentence each — and scored in a single whole-answer pass against the union of the caller-provided evidence with one grounding signal ([LettuceDetect v2](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base), pinned by revision in [config/verifier.yaml](config/verifier.yaml) — see ADR-0006 for the selection benchmark). Each claim carries answer-relative `start`/`end` (Unicode code-point offsets into the request `text`, with `text == request.text[start:end]`); text with no sentence boundary yields one whole-text claim (ADR-0009). Unsupported regions of a claim come back as `spans` — `start`/`end` code-point offsets into that claim's `text` plus the flagged substring (ADR-0007). Spans localize the problem, they are not the verdict's proof: the span-flagging threshold is a separate configured knob from the verdict threshold, so an `unsupported` claim with zero spans is possible, and a `supported` one can still carry spans. Decomposition refines attribution without moving the gate — the whole-answer verdict equals the conjunction over per-claim verdicts at the same threshold. Verdicts are `supported`/`unsupported` only — `contradicted` arrives with the NLI signal. Retrieval and tenancy are on the [roadmap](https://github.com/FabianSalge/plumb/milestones).

`confidence` is a calibrated probability, not a raw model score: among claims the engine scores `c`, about a fraction `c` are fully supported by the supplied passages, as measured on RAGTruth-style RAG traffic — out-of-domain calibration error is published alongside in [evals/RESULTS.md](evals/RESULTS.md) (ADR-0008). The verdict thresholds this confidence, the threshold lives in versioned config, and the Platt calibration artifact ([config/calibration/](config/calibration/)) is bound to the exact model revision, inference mode, and claim unit it was fitted against — the engine refuses to start with a mismatched calibrator rather than serve an uncalibrated number.

### Container

```sh
make image   # build plumb:dev
docker run --rm -p 8000:8000 plumb:dev
```

The image runs as a non-root user with CPU-only torch. Model weights are not baked in: the container downloads them on first start (~1.2 GB from a single Hub repo, no remote code, cached under `HF_HOME`), and `/readyz` returns 503 until the model is loaded.

## Deploy with Helm

The chart in [charts/plumb](charts/plumb) is the product's front door: a probed, resource-bounded Deployment with the verifier config injected from values.

```sh
helm install plumb charts/plumb
```

Highlights of [values.yaml](charts/plumb/values.yaml):

- `verifier.*` — the signal model pin, thresholds, and calibration artifact reference, rendered into a ConfigMap the API loads via `PLUMB_CONFIG`. Changing the threshold is a values change and a rollout, never an image rebuild; swapping the model demands a refitted calibration artifact or the pod refuses to become ready.
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
make test-model  # integration test against the real LettuceDetect weights
```

## Contributing

Every change starts as a GitHub issue; [docs/workflow.md](docs/workflow.md)
describes the loop. Contributors sign a one-time CLA before their first PR
merges — [CONTRIBUTING.md](CONTRIBUTING.md) has the policy and how to sign.
