# ADR-0003: Tiered verification — two named modes with latency contracts

2026-07-05 · Status: Accepted

## Context

Full verification — decomposition, fresh per-claim retrieval, multiple
signals, self-consistency sampling — costs seconds. That is fine for CI and
batch audit and awkward for gating a live answer a user is waiting on. One
pipeline with one vague latency number would either make the runtime gate
unusable or quietly degrade thoroughness to hit a deadline; both hide the
trade-off instead of pricing it.

## Decision

Verification runs in two named modes, each with an explicit latency
contract:

- **Fast** — decomposition-light, single groundedness signal, checked
  against caller-provided or cached context. Contract: sub-second p95.
  This is the mode for gating live answers.
- **Thorough** — the full pipeline: decomposition, per-claim retrieval
  against the tenant KB, all signals, self-consistency. Contract: seconds,
  not sub-second. This is the mode for CI gates, batch audit, and async
  re-verification of fast-mode passes.

The caller selects the mode explicitly; the API never silently downgrades.
Latency budgets live in versioned config, and measured p50/p95 per mode on
stated hardware are published in `evals/RESULTS.md`. Until measured, the
contracts above are targets and are labelled as such.

## Consequences

- The trade-off is a documented product concept instead of an internal
  compromise: callers know what a fast-mode pass does and does not claim.
- Fast mode inherits the context-native blind spot — an answer faithful to
  bad context passes — which is why thorough-mode async re-verification of
  fast passes exists. The docs state this plainly.
- Both modes are paths through one engine with shared signals, config, and
  thresholds (ADR-0005); mode changes the work done per claim, never the
  scoring semantics.
- Published numbers become a regression surface: once the eval gate is
  wired into CI, a change that blows a latency budget fails the build like
  an accuracy regression does.
