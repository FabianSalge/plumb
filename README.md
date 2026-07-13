# Plumb

[![ci](https://github.com/FabianSalge/plumb/actions/workflows/ci.yml/badge.svg)](https://github.com/FabianSalge/plumb/actions/workflows/ci.yml)

The open-source, self-hostable groundedness gate — one calibrated verifier that checks AI-generated answers against your own knowledge base and gates both CI and production on the result, without data leaving your cluster.

> **Status:** early development — `/v1/verify` checks answers in two modes: `fast` against caller-provided context, and `thorough` against your own knowledge base via per-claim retrieval from a read-only store connection. Both score with a benchmarked LettuceDetect groundedness signal, return calibrated confidences, and deploy to Kubernetes via the Helm chart.

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
  "config_version": "0.6.0"
}
```

The input is decomposed into claims — one verbatim sentence each — and scored in a single whole-answer pass against the union of the caller-provided evidence with one grounding signal ([LettuceDetect v2](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base), pinned by revision in [config/verifier.yaml](config/verifier.yaml) — see ADR-0006 for the selection benchmark). Each claim carries answer-relative `start`/`end` (Unicode code-point offsets into the request `text`, with `text == request.text[start:end]`); text with no sentence boundary yields one whole-text claim (ADR-0009). Unsupported regions of a claim come back as `spans` — `start`/`end` code-point offsets into that claim's `text`, the flagged substring, and a calibrated `confidence` (ADR-0007). Spans localize the problem, they are not the verdict's proof: the span-flagging threshold is a separate configured knob from the verdict threshold, so an `unsupported` claim with zero spans is possible, and a `supported` one can still carry spans. Decomposition refines attribution without moving the gate — the whole-answer verdict equals the conjunction over per-claim verdicts at the same threshold. Verdicts are `supported`/`unsupported` only — `contradicted` arrives with the NLI signal. Request-level multi-tenancy is on the [roadmap](https://github.com/FabianSalge/plumb/milestones).

`confidence` is a calibrated probability, not a raw model score: among claims the engine scores `c`, about a fraction `c` are fully supported by the supplied passages, as measured on RAGTruth-style RAG traffic — out-of-domain calibration error is published alongside in [evals/RESULTS.md](evals/RESULTS.md) (ADR-0008). A span's `confidence` reads in the opposite direction — a span exists because it was flagged, so the number is the calibrated probability that the flagged region is genuinely unsupported; span-level reliability is measured on RAGTruth's human span annotations, published in the same place. The verdict thresholds the claim confidence, the thresholds live in versioned config, and the Platt calibration artifact ([config/calibration/](config/calibration/)) is bound to the exact model revision, inference mode, claim unit, and span-flagging threshold it was fitted against — the engine refuses to start with a mismatched calibrator rather than serve an uncalibrated number.

### Thorough mode: verify against your knowledge base

With a tenant store configured (see the Helm values below), `mode: "thorough"` drops the `context` requirement and retrieves the evidence itself:

```sh
curl -s localhost:8000/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is Paris.", "mode": "thorough"}'
```

Each verbatim-sentence claim becomes one retrieval query, expanded deterministically with neighboring answer sentences so pronouns carry their antecedents — no rewriting model involved (ADR-0010). Queries run store-side lexical recall plus Plumb's own cross-encoder reranker, read-only against your existing store (ADR-0002); the first supported store is Postgres full-text search over the table your RAG stack already keeps next to its pgvector embeddings. Results pool into one evidence set — every claim's top chunk is guaranteed a slot, remaining window budget fills by rerank score, truncation is logged, never silent — and the whole answer is scored in exactly the same single pass as fast mode, so the calibration artifact and gate semantics carry over unchanged. Caller-provided `context` is optional and, when present, joins the pool ahead of retrieved chunks.

Each claim in a thorough response carries `evidence`: the chunks its query retrieved that made the scoring window, with source and chunk identity, retrieval rank, and the store's snapshot identity where one exists. **This is retrieval provenance, not support attribution** — "retrieved for this claim", not "this passage proves it"; joint inference cannot name a supporting passage (ADR-0007). The calibrated confidences were fitted on provided-context traffic; the calibration error on retrieved evidence is measured and published separately in [evals/RESULTS.md](evals/RESULTS.md) rather than assumed equal.

Onboarding needs one thing from your platform team: a read-only database role over the chunk table. The verifier additionally forces its session read-only, and a store error fails the request loudly (502) — never a verdict on partial evidence.

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
- `store.*` (default disabled — fast-only) — the tenant store connection for thorough mode: `store.dsnSecret` names an existing Secret holding the read-only DSN (the credential never appears in chart values or the ConfigMap), plus `table`/`idColumn`/`textColumn`, optional `sourceColumn`/`snapshotColumn` for provenance, and the FTS `regconfig`. Enabling the store opens exactly one extra egress hole (`store.egressPort`, default 5432). The pod probes the connection and schema at startup and fails loudly on mismatch. Store-enabled pods also download the pinned reranker (~2.3 GB) on first start — raise `resources` and `startupProbe.failureThreshold` accordingly (see [tests/e2e/values-thorough.yaml](tests/e2e/values-thorough.yaml) for a working sizing).
- `verifier.retrieval.*` — the versioned thorough-mode knobs: query expansion window, recall depth, per-claim window quota, pool token budget, and the reranker pin.
- `image.repository` / `image.tag` — no registry is assumed; bring your own or use the kind flow below.

To see it live on a local [kind](https://kind.sigs.k8s.io) cluster (requires docker, kind, and helm):

```sh
make kind-up          # create the cluster
make deploy           # build the image, load it into kind, install the chart
make e2e              # golden verify request against the deployed chart
make deploy-thorough  # same, plus a seeded Postgres store and thorough mode enabled
make e2e-thorough     # fast + thorough goldens, asserting evidence provenance
```

CI runs this same flow on every PR that touches code, chart, or config:
kind cluster, chart install, one golden verify request asserting a
`supported` verdict. Prose-only PRs (docs, specs, markdown) skip the heavy
jobs; lint and secret scanning run regardless.

## Development

```sh
pre-commit install --hook-type commit-msg --hook-type pre-commit
make test           # pytest with the coverage floor
make lint           # ruff check + format check
make typecheck      # mypy (strict)
make test-model     # integration tests against the real model weights
make test-postgres  # store adapter tests against a disposable Postgres (needs Docker)
```

## Contributing

Every change starts as a GitHub issue; [docs/workflow.md](docs/workflow.md)
describes the loop. Contributors sign a one-time CLA before their first PR
merges — [CONTRIBUTING.md](CONTRIBUTING.md) has the policy and how to sign.
