# helm-deploy (delta)

## ADDED Requirements

### Requirement: Chart carries the tenant store connection
The chart SHALL expose the tenant store connection as values: enabled flag (default off —
fast-only deployments stay the default), DSN sourced from a Kubernetes Secret reference
(never a plain value), table name, id/text columns, optional source and snapshot columns,
and the FTS regconfig. The connection SHALL be rendered into the pod as deployment
configuration, versioned with the chart values like the rest — it is deployment identity,
not engine behavior, and does not bump `config_version`. When the store is enabled, the
pod SHALL validate the connection and configured table/columns at startup and fail loudly
on mismatch.

#### Scenario: Fast-only default
- **WHEN** the chart is installed with default values
- **THEN** no store connection is rendered and thorough requests are rejected as fast-only

#### Scenario: Store misconfiguration fails startup
- **WHEN** the store is enabled but the DSN, table, or columns are wrong
- **THEN** the pod fails startup with an error naming the store problem instead of serving

#### Scenario: Credentials come from a Secret
- **WHEN** the store is enabled
- **THEN** the DSN reaches the pod only through a Secret reference, never rendered inline in the ConfigMap

## MODIFIED Requirements

### Requirement: Default-deny egress NetworkPolicy
The chart SHALL ship a NetworkPolicy template, enabled by default, that denies all pod
egress except DNS resolution and HTTPS for the pinned model weight download. The
weight-download exception SHALL be independently switchable off in values for clusters
that pre-bake or mirror weights, restoring full default-deny. When the tenant store is
enabled, the policy SHALL additionally allow egress to the configured store endpoint and
nothing else — enabling the store MUST NOT widen egress beyond it.

#### Scenario: Egress denied by default
- **WHEN** the chart is installed with default values
- **THEN** the pod cannot open connections other than DNS and HTTPS

#### Scenario: Sovereign mode
- **WHEN** the weight-download exception is disabled in values
- **THEN** the rendered policy allows DNS only, and all other egress is denied

#### Scenario: Store egress is scoped
- **WHEN** the tenant store is enabled in values
- **THEN** the rendered policy allows egress to the store's configured endpoint, and all other non-DNS, non-weight-download egress remains denied
