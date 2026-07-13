# evidence-retrieval (delta)

## ADDED Requirements

### Requirement: Per-claim queries by deterministic context expansion
In thorough mode the system SHALL build one retrieval query per claim, where the query is
the answer's leading sentence, up to `expansion_window` (versioned config) sentences
immediately preceding the claim, and the claim's verbatim sentence — deduplicated
preserving order and joined with single spaces. Query construction SHALL use no model and
no rewriting. Queries are ephemeral: they MUST NOT appear in the response, and the claim
text returned to the caller keeps ADR-0009's substring invariant untouched. Identical
sentences SHALL share one query and its results within a request.

#### Scenario: Query carries the antecedent's sentence
- **WHEN** a claim is a sentence with a dangling pronoun and `expansion_window` ≥ 1
- **THEN** its query contains the preceding sentence's text ahead of the claim's verbatim text

#### Scenario: Expansion is pinned by golden tests
- **WHEN** the same answer text and `expansion_window` are given
- **THEN** the constructed query strings are byte-identical across runs, and a change to windowing or joining shows up as a golden-test diff

#### Scenario: Leading claim expands without duplication
- **WHEN** the claim is the answer's leading sentence
- **THEN** its query is the claim text once, not repeated

### Requirement: Recall-then-rerank through a read-only store adapter
Retrieval SHALL run each query through a store adapter implementing a recall interface
(`recall(query, k)` returning chunks) and rerank the recalled chunks with the engine's own
cross-encoder, per ADR-0002. Adapters SHALL be read-only against the tenant's store. Each
recalled chunk SHALL carry text, source identity, chunk identity, and — only where the
store exposes one — a snapshot identity; the system MUST NOT invent snapshot identities.
The first adapter is Postgres full-text search: rank by `ts_rank_cd` against
`websearch_to_tsquery` over a configured table, id column, and text column, with the FTS
regconfig configurable. A store connection, query, or schema error SHALL fail the request
loudly with context; an erroring store MUST NOT degrade to an empty result.

#### Scenario: Postgres recall is read-only
- **WHEN** the Postgres adapter opens a session
- **THEN** the session is set read-only, so even a misconfigured role cannot write to the tenant's store

#### Scenario: Store failure is loud
- **WHEN** the store is unreachable or the configured table/columns do not match
- **THEN** the verification request fails with an error naming the store problem, never an empty-evidence verdict

#### Scenario: Reranker is pinned and swappable
- **WHEN** the retrieval config is loaded
- **THEN** it names the reranker model and pinned revision, and a reranker swap is a config-version bump, never a silent change

### Requirement: Pooling with dedupe, caller precedence, per-claim quota, global fill
Results SHALL pool across claims into one evidence set: deduplicated by chunk identity;
caller-provided `context` passages (when present) enter first labeled with caller
provenance and are never displaced by retrieved chunks; every claim's top-reranked chunk
is guaranteed `per_claim_quota` slots; remaining budget fills by global rerank score.
The pool budget (`pool_budget_tokens`) SHALL be counted in scoring-tokenizer tokens of
the rendered passages, reserving room for the prompt template and the answer, so pooling —
never the scorer's silent window — decides what is scored. Budget, quota, recall depth,
and expansion window SHALL live in versioned config. Every chunk dropped at any pooling
step SHALL be logged with its identity and reason; pool truncation MUST NOT be silent.

#### Scenario: Minority claim keeps its slot
- **WHEN** the pooled candidates exceed the budget and one claim's chunks all rank low globally
- **THEN** that claim's top-reranked chunk is still in the scored pool via its guaranteed quota slot

#### Scenario: Truncation is logged
- **WHEN** candidates are dropped because the budget is exhausted
- **THEN** a structured log line records each dropped chunk's identity and the reason

#### Scenario: Caller passages survive pooling
- **WHEN** a thorough request carries `context` passages
- **THEN** those passages are in the scored pool labeled as caller-provided, ahead of any retrieved chunk

### Requirement: Pooled evidence is scored in fast mode's single joint pass
The pooled evidence SHALL be scored by exactly the scoring path fast mode uses — same
prompt rendering, one joint forward pass over the whole answer, segment-after-score
per-claim reduction, same calibration artifact. Thorough mode changes what fills the
passage list, never the scoring semantics; no calibration refit is triggered (ADR-0008
bindings unchanged). The gate-parity property SHALL hold in thorough mode over a fixed
pool: the decomposed gate equals the whole-text gate at the same threshold.

#### Scenario: Gate parity over a fixed pool
- **WHEN** the same answer and the same pooled evidence are scored decomposed and as a single whole-text claim at the same threshold
- **THEN** the two gate decisions are identical

#### Scenario: Scorer truncation indicates a budgeting bug
- **WHEN** the scorer reports its own window truncation in thorough mode
- **THEN** the event is logged as an error condition — the pool budget should have prevented it
