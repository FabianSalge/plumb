# Contributing to Plumb

Development runs the loop described in [docs/workflow.md](docs/workflow.md):
every change starts as a GitHub issue with acceptance criteria, one issue =
one branch = one PR, tests exist before implementation, and nothing merges
without review. Read that first. This file covers what is specific to
external contributions — above all, licensing.

## Licensing policy

Plumb is Apache-2.0 and stays Apache-2.0. On top of that, every contributor
signs a one-time Contributor License Agreement (CLA) before their first
change merges.

**Why a CLA.** Without one, copyright ends up distributed across every
contributor, and any future licensing decision — relicensing, dual
licensing, donating the project to a foundation — needs each contributor's
individual sign-off. One signature per contributor, collected up front,
keeps those doors open. The CLA is a license, not an assignment: you keep
ownership of your contributions and can use them however you like elsewhere.

**Why not a DCO.** A DCO (`Signed-off-by` on every commit) only attests
provenance; it grants no rights beyond the inbound license, so on its own it
would not solve the problem above. Stacking a DCO on top of the CLA adds
per-commit friction without adding rights — the CLA's representations
already cover provenance. So: CLA only, no DCO.

**Why signatures live in this repo.** The usual tooling was the weaker
option at the time of this decision: the cla-assistant GitHub Action was
archived in March 2026, and the hosted cla-assistant.io service — dormant
since mid-2024 — would put signatures on a third-party server. Instead the
ledger is a versioned file in this repo and your signature is your own
commit: auditable in git history, no bot, no external service. That also
matches what Plumb is — self-hostable, no hidden dependencies.

## How to sign

Individuals:

1. Read the [Individual CLA](.github/cla/individual-cla.md).
2. In the same pull request as your first contribution, add an entry for
   yourself to [`.github/cla/signatures.json`](.github/cla/signatures.json):

```json
{
  "type": "individual",
  "github": "your-github-username",
  "name": "Your Full Name",
  "claVersion": "1.0",
  "date": "YYYY-MM-DD"
}
```

Companies: someone authorized to bind the company reads the
[Entity CLA](.github/cla/entity-cla.md) and adds an entry listing the
GitHub usernames allowed to contribute on the company's behalf. Updating
that list later is a pull request against the same entry.

```json
{
  "type": "entity",
  "entity": "Company Name Inc.",
  "github": "github-username-of-the-signer",
  "name": "Full Name of the Authorized Signer",
  "authorized": ["employee-username-1", "employee-username-2"],
  "claVersion": "1.0",
  "date": "YYYY-MM-DD"
}
```

Committing the entry under your GitHub account is the signature — there is
nothing else to do.

## Enforcement

The `cla` check ([.github/workflows/cla.yml](.github/workflows/cla.yml))
runs on every pull request and fails unless the author appears in
`signatures.json` (directly or in an entity's `authorized` list). The check
is required for merging into `main`. The steward and dependency bots are
exempt.
