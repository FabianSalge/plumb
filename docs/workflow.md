# Development workflow

How changes get made in this repo.

## The loop

1. **Issue first.** Every change starts as a GitHub issue with acceptance criteria. The criteria are the spec; the PR that closes the issue must check every box.
2. **One issue = one branch = one PR.** Branches look like `type/N-short-name`; commits are conventional, one sentence max.
3. **Spec before code (features only).** Feature work goes through OpenSpec: a change under `openspec/changes/<name>/` (proposal, design, spec deltas, tasks) is written and reviewed before implementation starts. When the work merges, the change is archived and its deltas fold into `openspec/specs/` — the living system spec. Chores skip the spec stage. Spikes are timeboxed and their code is discarded; notes under `docs/spikes/` are the only artifact.
4. **Tests first.** Failing tests exist before implementation. CI must be green; a gate is never weakened to pass it.
5. **Review, then merge.** I review every PR line by line and merge only what I can explain cold.

## Working with the agent

I develop with Claude Code under the rules in `CLAUDE.md`:

- One session per issue, anchored on `CLAUDE.md` and the issue text; each new issue gets a fresh session.
- Substantive PR feedback goes as inline review comments on GitHub; the agent addresses them on the same branch. Trivial fixes can go through chat.
- Trust-critical code (calibration, aggregation, gate semantics, tenant isolation) I drive by hand. No agent code merges unreviewed.

## Board

The public project board tracks the current milestone: Backlog → This week → In review → Done. Cards move by hand: In review when the PR opens, Done on merge.
