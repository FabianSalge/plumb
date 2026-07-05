# helm-deploy Specification

## Purpose
TBD - created by archiving change add-helm-chart. Update Purpose after archive.
## Requirements
### Requirement: Chart installs a probed, bounded API deployment
The chart SHALL deploy the API as a Deployment whose pods carry a liveness probe on `GET /healthz`, a readiness probe on `GET /readyz`, and resource requests and limits, and SHALL expose the pods through a ClusterIP Service. Probe timing SHALL tolerate the first-start model weight download (~420 MB) without the pod being killed or flapping.

#### Scenario: Fresh install becomes ready
- **WHEN** the chart is installed with default values into a cluster that can reach the model host
- **THEN** the pod is not restarted during the weight download, and the Service routes traffic only after `/readyz` returns 200

#### Scenario: Hung process is restarted
- **WHEN** a running pod stops answering `/healthz`
- **THEN** the kubelet restarts the container

### Requirement: Image and verifier config are values-driven
The chart SHALL expose the image repository and tag as values, and SHALL render the verifier config (config version, signal model name, pinned revision, threshold) from values into a ConfigMap that the pod loads via `PLUMB_CONFIG`. Default values SHALL match the repository's `config/verifier.yaml`. Changing the threshold or model pin SHALL require only a values change and rollout, never an image rebuild.

#### Scenario: Threshold override
- **WHEN** the chart is upgraded with a different `verifier.signals.groundedness.threshold` value
- **THEN** the new pod serves verdicts using the new threshold and reports the values-declared `config_version`

#### Scenario: Config rendered from values
- **WHEN** the chart is templated with default values
- **THEN** the rendered ConfigMap content equals the repository's `config/verifier.yaml`

### Requirement: Default-deny egress NetworkPolicy
The chart SHALL ship a NetworkPolicy template, enabled by default, that denies all pod egress except DNS resolution and HTTPS for the pinned model weight download. The weight-download exception SHALL be independently switchable off in values for clusters that pre-bake or mirror weights, restoring full default-deny.

#### Scenario: Egress denied by default
- **WHEN** the chart is installed with default values
- **THEN** the pod cannot open connections other than DNS and HTTPS

#### Scenario: Sovereign mode
- **WHEN** the weight-download exception is disabled in values
- **THEN** the rendered policy allows DNS only, and all other egress is denied

### Requirement: Local kind deployment
The repository SHALL provide `make kind-up` to create a local kind cluster and `make deploy` to build the image, load it into kind, and install the chart, such that a verification request against the in-cluster Service returns a correct verdict.

#### Scenario: Tracer bullet live on kind
- **WHEN** `make kind-up && make deploy` completes and `POST /v1/verify` is sent to the Service with a claim and inline evidence that supports it
- **THEN** the response verdict is `supported` and carries `engine_version` and `config_version`

### Requirement: Chart linting in CI
CI SHALL lint the chart (`helm lint` and chart-testing lint) and fail the pipeline on lint errors.

#### Scenario: Broken template blocks the pipeline
- **WHEN** a change introduces a chart template that fails lint
- **THEN** the CI pipeline fails
