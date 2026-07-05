# Proposal: add-helm-chart

## Why

The chart is the product's front door: Plumb's pitch is a groundedness gate that runs in the tenant's cluster, and until `helm install` works that claim is prose, not code. Issue #9 delivered the image; issue #10 makes the tracer bullet live on Kubernetes — installable, probed, resource-bounded, and egress-locked from the first release.

## What Changes

- New `charts/plumb/` Helm chart deploying the API as a single Deployment with liveness (`/healthz`) and readiness (`/readyz`) probes, resource requests/limits, and a ClusterIP Service.
- Chart values expose image repository/tag, replica count, resources, and the verifier config (threshold, model pin) — config stays versioned and injected, never baked in.
- Default-deny egress NetworkPolicy template shipped and enabled by default, with an explicit, documented exception for DNS and HTTPS so the pod can fetch pinned model weights at first start — the sovereignty claim in code, with its one honest hole visible in values.
- `make kind-up` and `make deploy` targets: create a local kind cluster, build and load the image, install the chart.
- CI gains `helm lint` + chart-testing (`ct lint`) on chart changes.
- E2E proof: `curl` against the kind service returns a correct verdict for a claim with inline evidence (the CI smoke-test job itself is issue #11, out of scope here).

## Capabilities

### New Capabilities

- `helm-deploy`: the deployment contract of the chart — what an operator can rely on when installing Plumb: probes wired to the API's health endpoints, resource bounds, values-driven image and verifier config, default-deny egress posture.

### Modified Capabilities

<!-- none — verify-api's HTTP contract is unchanged; the chart consumes it -->

## Impact

- New `charts/plumb/` directory (templates, values, chart metadata).
- `Makefile`: new `kind-up` and `deploy` targets; CLAUDE.md command list updated to match.
- `.github/workflows/ci.yml`: new chart lint job (additive — no existing gate weakened).
- No API or engine code changes; depends on the image from #9 (merged as PR #22).
- Readiness must tolerate the ~420 MB first-start weight download — probe timing is a chart concern, not an API change.
