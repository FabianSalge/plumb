# Tasks: add-helm-chart

## 1. Checks first

- [x] 1.1 Add chart render tests (pytest shelling out to `helm template`): deployment carries liveness/readiness/startup probes and resource requests/limits; Service is ClusterIP on the API port — failing while the chart doesn't exist
- [x] 1.2 Add render test: ConfigMap rendered with default values equals `config/verifier.yaml`, and a threshold override lands in the rendered config
- [x] 1.3 Add render tests for the NetworkPolicy: default values allow DNS + 443 only; `allowModelDownload: false` renders DNS-only egress; `networkPolicy.enabled: false` renders no policy

## 2. Chart

- [x] 2.1 Scaffold `charts/plumb/` (Chart.yaml, values.yaml, _helpers.tpl, NOTES.txt) with values for image repository/tag/pullPolicy, replicas, resources, verifier config, networkPolicy toggles
- [x] 2.2 Deployment template: probes per design (startupProbe absorbing the weight download), resources, ConfigMap mount + `PLUMB_CONFIG`, checksum annotation so config changes roll pods
- [x] 2.3 Service (ClusterIP) and verifier ConfigMap templates
- [x] 2.4 NetworkPolicy template: default-deny egress, DNS allowance, 443 behind `allowModelDownload`
- [x] 2.5 All render tests from group 1 pass; `helm lint charts/plumb` clean

## 3. kind targets

- [x] 3.1 `make kind-up`: create the `plumb` kind cluster with a pinned node image
- [x] 3.2 `make deploy`: `make image`, `kind load docker-image`, `helm upgrade --install --wait`
- [ ] 3.3 Verify the tracer bullet live: `curl` the in-cluster Service with a supported claim + inline evidence, confirm verdict, `engine_version`, `config_version`

## 4. CI

- [x] 4.1 Add chart job to `.github/workflows/ci.yml`: `helm lint` + `ct lint`, pinned action versions, additive only

## 5. Docs

- [x] 5.1 README: install-from-chart section (helm install, values highlights, sovereign mode note)
- [x] 5.2 CLAUDE.md command list: add `kind-up`/`deploy`, drop the "arrive with the Helm work" note
