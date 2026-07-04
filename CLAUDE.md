# Plumb — agent constitution

## What this is
The open-source, self-hostable groundedness gate: one engine that scores
PRs and production answers with the same calibrated verifier, against a
tenant's own knowledge base, in-cluster. Scope and roadmap live in the
GitHub issues and milestones — there is no plan doc in the repo.

## Layout
- api/        HTTP surface (contracts in openspec/specs/)
- engine/     verification pipeline (decomposition, retrieval, signals,
              aggregation, calibration, gate)
- charts/     Helm chart — the product's front door
- deploy/     Terraform for ephemeral demo/GPU infra
- evals/      benchmark harness, golden sets, RESULTS.md
- docs/adr/   architecture decisions — read before proposing design changes
- docs/       workflow.md (how development runs), playbooks/, devlog/

## Workflow (non-negotiable)
1. Work happens against a GitHub issue with acceptance criteria.
2. Features go through OpenSpec: propose → review → apply → verify → archive.
   Chores and spikes skip the spec stage.
3. Checks come first: failing tests exist before implementation.
4. One issue = one branch = one PR.
5. Never merge without human review. Never weaken a CI gate to pass it.

## Protected zones — plan first, never modify silently
- engine/calibration/ and engine/aggregation/ (interview-critical math)
- gate semantics and API response contracts
- anything touching tenant isolation
- CI workflow files and coverage/threshold configs
- docs/adr/ (append-only; decisions change via a new ADR)

## Conventions
- Commits: conventional, one sentence max, no bodies, no trailers.
  Never add AI attribution or a Co-Authored-By line.
- All prose (commits, PRs, issues, docs, comments) is written in the
  author's voice, never the AI's.
- No dead code, no speculative abstractions, no unexplained dependencies.
- Errors: fail loudly with context; no silent fallbacks — this product's
  job is honesty about uncertainty.
- Config is versioned and injected; no magic constants in the engine.
- Structured JSON logs; every request carries a request ID.
- Docs are code: user-visible changes update README/docs in the same PR.

## Definition of done
Acceptance criteria checked; tests green locally and in CI; lint/typecheck
clean; OpenSpec change verified + archived; docs updated; PR description
says how it was verified.

## Commands
- pre-commit install --hook-type commit-msg --hook-type pre-commit  (once per clone)
- pre-commit run --all-files
- make targets (test/lint/run/kind-up/deploy/e2e) arrive with the language
  decision (#7) — keep this list current; agents rely on it.
