# Design: add-helm-chart

## Context

The image (#9, PR #22) runs the API as UID 10001, serves on 8000, bakes `config/` in, and downloads ~420 MB of HHEM weights from Hugging Face at first start; `/readyz` stays non-200 until the model is loaded. The API reads its config path from `PLUMB_CONFIG` (default `config/verifier.yaml`). CI builds `plumb:${sha}` but pushes to no registry, so kind must be fed the image via `kind load docker-image`.

## Goals / Non-Goals

**Goals:**
- `helm install` of a probed, resource-bounded, egress-locked API — the tracer bullet live on kind via two make targets.
- Verifier config injected from values, never baked in.
- Chart lint gating in CI.

**Non-Goals:**
- The CI e2e smoke-test job (issue #11 — the make targets it will call are built here).
- Ingress, TLS, HPA, PodDisruptionBudget, multi-replica semantics — nothing forced before there's a user for it.
- GPU scheduling, weight pre-baking, persistent weight cache (revisit with vLLM).

## Decisions

**ConfigMap + `PLUMB_CONFIG` over baked config or env-var soup.** The chart renders `verifier.yaml` from values into a ConfigMap, mounts it, and points `PLUMB_CONFIG` at it. One mechanism, already supported by the API, keeps the whole config versioned and diffable in values; per-field env vars would fragment the config's `version` stamp. Defaults duplicate `config/verifier.yaml` deliberately — the chart is self-contained and the duplication is guarded by a chart test comparing the rendered default against the repo file.

**Default-deny egress with two named holes, on by default.** `policyTypes: [Egress]`, allowing only DNS (53 to kube-dns) and, behind `networkPolicy.allowModelDownload` (default `true`), TCP 443 for the weight fetch. Alternative — shipping the policy disabled — was rejected: the sovereignty claim has to be the default posture, not an opt-in. The 443 hole is the one honest concession to the download-at-start decision in the Dockerfile; clusters that mirror weights set it `false` and get full deny. No ingress rules in v0: ingress policy without a real consumer topology would be guesswork.

**startupProbe absorbs the download; steady-state probes stay tight.** A startupProbe on `/readyz` (30 × 10 s ≈ 5 min headroom) covers the first-start download; liveness (`/healthz`) and readiness (`/readyz`) then run with short periods. The alternative — a readiness probe with a huge `failureThreshold` — would also delay detection of steady-state hangs.

**kind targets wrap the existing image flow.** `make kind-up` creates a named cluster (`plumb`); `make deploy` runs `make image`, `kind load docker-image plumb:dev`, `helm upgrade --install` with `pullPolicy: Never` semantics via values. No registry appears anywhere — matches CI's push-less build.

**CI: one additive `chart` job.** `helm lint` + `ct lint` (chart-testing) pinned by action version, running on the same PR pipeline. No existing gate is touched.

**Resources sized from measurement, not folklore.** The spike recorded HHEM's runtime RSS; requests/limits in default values come from that (~2 Gi limit, requests below), and the values comment cites the source so the next model (#18) re-derives rather than inherits.

## Risks / Trade-offs

- [Weight download flakes in kind e2e] → startupProbe headroom plus `helm upgrade --install --wait` in `make deploy`; the download is a pinned revision, so content is stable even if the network isn't.
- [Values-duplicated verifier config drifts from `config/verifier.yaml`] → chart test asserts rendered-default equality with the repo file; drift fails CI.
- [Default 443 egress hole reads as "default-deny in name only"] → the hole is a single named toggle with a comment stating exactly why it exists and how to close it; docs say the same.
- [kind version skew across contributor machines] → make targets pin the kind node image tag.

## Open Questions

- None blocking. Whether weights should ship in an initContainer/PVC instead of the 443 hole is deferred to the vLLM/model decision (#18) fallout.
