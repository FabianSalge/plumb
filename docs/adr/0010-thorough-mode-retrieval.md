# ADR-0010: Thorough-mode retrieval — per-claim queries, pooled evidence, one scoring pass

2026-07-12 · Status: Accepted

## Context

The one-liner promises verification against a tenant's knowledge base;
`/v1/verify` still checks only caller-provided passages. Retrieval is the
step between them, and ADR-0003 already named where it lives — thorough
mode — without defining it. The decisions since then bound this one
tightly. ADR-0002 fixed the retrieval mechanics: store-side lexical/sparse
recall plus our own reranker, read-only against the tenant's store.
ADR-0009 fixed the claim as a verbatim sentence and proved gate parity
from a single whole-answer forward pass. ADR-0007 fixed that pass as joint
all-passages inference, punted window budgeting to "retrieval's problem",
and promised per-claim chunk provenance to this ADR. ADR-0008 bound the
calibration artifact to the inference mode, with a forced refit for
anything that moves the raw score distribution. And #47 recorded the trap
waiting at the intersection: a verbatim sentence claim can carry a
dangling pronoun, and the moment each claim becomes a query, that pronoun
retrieves weak or wrong evidence. Options considered:

- **Per-claim queries, per-claim scoring contexts** — each claim scored in
  its own forward pass against its own evidence. The cleanest attribution
  story, but it is M forward passes, it needs coreference resolved in the
  *scoring* input rather than just the query, it abandons the one-pass
  argument gate parity rests on, and it changes the inference mode —
  forcing a calibration refit and giving each mode its own scoring
  semantics, exactly what ADR-0003 promised never happens.
- **Per-answer retrieval, then attribute** — one query from the whole
  answer, retrieve once, score jointly. Sidesteps #47 entirely, but a
  single query over a multi-fact answer dilutes lexical recall — the
  claim about the second topic never surfaces its evidence — and there is
  no per-claim provenance left to return.
- **Per-claim queries with model-based decontextualization** — rewrite
  each claim into a standalone query with a coreference model or LLM.
  Buys resolution with a new model dependency, its latency, and
  nondeterministic queries as a fresh failure surface: the grounds on
  which ADR-0009 rejected model rewrite, applied one step earlier.
- **Per-claim queries with deterministic context expansion, pooled into
  one scoring pass** — the query is the claim plus neighboring answer
  sentences; results pool into a single evidence set scored exactly like
  fast mode. Chosen.

## Decision

Thorough mode retrieves per claim and scores jointly. Each claim (the
verbatim sentence of ADR-0009) becomes one retrieval query, expanded
deterministically with a context window from the answer itself — the
preceding sentence(s) plus the answer's leading sentence, window size in
versioned config — so a dangling pronoun's query carries its antecedent's
content words. Queries are ephemeral: they never appear in the response,
so the claim text and ADR-0009's substring invariant are untouched — this
is expansion, not rewriting, and it resolves #47 with no new model. Each
query runs ADR-0002's recall-then-rerank; results are pooled across
claims, deduplicated, and packed into the model window by quota — every
claim's top-ranked chunk is guaranteed a slot, then remaining space fills
by global rerank score, budget and quota in versioned config. One joint
forward pass (ADR-0007) scores the whole answer against the pooled
evidence, segment-after-score cuts per-claim scores (ADR-0009): mode
changes what fills the context window, never the scoring semantics.

The mode's promise: the answer is checked against the tenant's knowledge
base, not just whatever the caller supplied — closing the context-native
blind spot ADR-0003 documented as fast mode's known cost.

In the contract, `mode: "thorough"` becomes accepted; `context` becomes
optional in that mode and, when present, joins the pool labeled with
caller provenance. Each claim gains `evidence` — references to the chunks
its query retrieved that made the scoring window, carrying source and
chunk identity, retrieval rank, and whatever snapshot identity the store
exposes (the verdict-pinning seed). The latency contract is p95 ≤ 10 s on
stated hardware — a target until measured, published in
`evals/RESULTS.md`, and covering the full signal stack: NLI and
self-consistency spend from this budget when they land, rather than
renegotiating it. The tenant-isolation boundary for v1 is deployment
topology: one deployment serves one tenant's knowledge base, the store
connection is read-only versioned deployment config, and any evidence
cache lives inside that boundary. Request-level multi-tenancy is
explicitly deferred to its own ADR. Thorough-mode v1 is retrieval plus
the groundedness signal; implementation is its own issue.

## Consequences

- The scoring semantics stay mode-invariant, so the calibration artifact,
  the gate-parity property, and the claim unit all carry over unchanged —
  no refit is triggered by ADR-0008's bindings. The honest caveat: the
  calibrator was fitted on provided-context traffic, and retrieved
  evidence is a different passage distribution; thorough-mode ECE gets
  measured in the evals rather than assumed equal.
- `evidence` is retrieval provenance, not support attribution — "retrieved
  for this claim", not "supports this claim". Joint inference cannot name
  a supporting passage without the N passes ADR-0007 retired, and the docs
  must say so plainly.
- Deterministic query expansion is a bet that neighboring sentences carry
  the antecedent. Where they don't, recall degrades toward the per-answer
  baseline and the reranker cannot recover what recall never surfaced
  (ADR-0002's known cost). If the evals show expansion recall
  insufficient, the recorded fallback is per-answer-retrieve-then-
  attribute via a superseding ADR — not a coreference model by default.
- The window quota prices long answers honestly: many claims leave each
  claim little beyond its guaranteed slot. Pool truncation is logged, not
  silent — the same treatment ADR-0007 gave passage truncation.
- #47 closes with this decision; the constraint it carried is now load-
  bearing design.
- The API breaks once more, pre-1.0: thorough mode accepted, `context`
  optional there, `evidence` added. The verify-api spec changes ride the
  implementation issue, and fast mode's contract does not move.
- A 10 s budget shared with future signals means retrieval cannot spend
  it all; how much each signal gets is priced when it lands, but the
  ceiling is fixed now.
- Deployment-as-tenant-boundary keeps isolation trivial to review and
  defers the hard tenancy questions; the cost is that nothing in the
  request path carries a tenant identity yet, and the future
  multi-tenancy ADR inherits that migration.
