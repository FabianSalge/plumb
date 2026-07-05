# ADR-0002: Embedding compatibility — sparse recall from the tenant's store, our own reranker

2026-07-05 · Status: Accepted

## Context

Plumb's Tier 2 integration reads a tenant's existing vector store or search
index (pgvector, Qdrant, OpenSearch) and runs retrieval per atomic claim
over their full corpus. The trap is dense search: the vectors in the
tenant's store were produced by *their* embedding model, and querying them
requires embedding the query with that same model — which Plumb may not
serve, may not be able to serve (proprietary APIs), and cannot be expected
to for every tenant.

Options considered:

1. Require access to the tenant's embedding model (their API key or an
   in-cluster copy). Maximum recall reuse, but it makes onboarding depend on
   the one component we can't standardize, and for API-embedded corpora it
   reintroduces the data egress the product exists to avoid.
2. Use the store's lexical/sparse search for recall and apply our own
   cross-encoder reranker for precision. Works read-only against any store
   with text search, with no embedding-model coupling.
3. Maintain our own parallel dense index in Plumb's embedding space.
   Restores full hybrid recall, but costs ingest, storage, and index-drift
   management for every tenant.

## Decision

Default to option 2: store-side lexical/sparse recall plus Plumb's own
reranker. Offer option 3 — a slim parallel dense index in Plumb's own
embedding space — as an opt-in for tenants who want full hybrid recall.
Never require option 1.

## Consequences

- Onboarding stays read-only against the tenant's store, which keeps the
  platform team's security review trivial and preserves the zero-egress
  property.
- Recall in the default mode is bounded by the quality of the store's
  lexical search; the reranker restores precision but cannot recover
  documents lexical recall never surfaced. This is the honest cost, and the
  eval harness must measure it rather than hide it.
- The opt-in dense index buys back hybrid recall at the price of an ingest
  pipeline, extra storage, and a second index that can drift from the
  store of record; verdict pinning has to account for its version too.
- Retrieval code is written against a recall-then-rerank interface from the
  start, so the two modes differ in configuration, not architecture.
