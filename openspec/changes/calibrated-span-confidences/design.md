# Design — calibrated span confidences

## Context

ADR-0007 fixed the span shape: claim-relative positions, no confidence field until
calibration produces one worth shipping. Calibration v0 (#32, ADR-0008) shipped the
claim path: a Platt map `sigmoid(a · logit(s) + b)` fitted on held-out RAGTruth,
carried by a versioned artifact (schema 1) whose bindings — model, revision, inference
mode, claim unit — the engine validates at startup, refusing to serve on mismatch.

The span-side raw signal already exists: `engine/decomposition/reduction.py` merges
contiguous flagged tokens into spans and records each span's maximum token risk, which
today goes to structured logs only (`Span.confidence`, a raw hallucination probability,
never the API). RAGTruth's human span annotations — already used by #45 to derive
sentence labels (a sentence is supported iff it overlaps no annotated span) — can label
engine-produced spans the same way, which is what makes a span-level reliability
measurement possible.

The issue's core honesty constraint: whether the claim-level calibrator transfers to
span-level scores is a question this change must *answer with data*, not assume.

## Goals / Non-Goals

**Goals:**

- Each span in the `/v1/verify` response carries a calibrated `confidence`.
- Span-level reliability evidence in `evals/RESULTS.md`, whichever way the transfer
  check goes.
- The artifact carries the span calibration under the same refusal discipline as the
  claim path, including the one binding unique to spans: the flagging threshold.

**Non-Goals:**

- Span geometry and the span-flagging threshold value (fixed by ADR-0007; the threshold
  becomes a *binding*, not a different number).
- Out-of-domain span calibration: LLM-AggreFact carries claim labels, not span
  annotations, so span-level OOD error is not measurable with data in hand. The
  artifact and RESULTS.md state that absence explicitly instead of inventing a number.
- Per-tenant refit, thorough mode, retrieval.

## Decisions

### 1. Span confidence means P(the flagged region is genuinely unsupported)

A span exists *because* the engine flagged it; the number a caller needs is how sure
the engine is about the flag. Docs-plain: *among spans the engine flags with confidence
c, about a fraction c genuinely mark unsupported content, as measured on RAGTruth-style
RAG traffic.* The alternative — reusing the claim direction, P(supported) — would ship
0.2-means-probably-hallucinated next to a claim confidence where 0.2 means the same
thing, inviting double-negation misreads on the one field that only appears on flagged
regions. Monotone in the span's raw max token risk, strictly inside (0, 1), and the raw
risk stays in structured logs, never the response — same rules as the claim path.

### 2. One arithmetic path: the span map reuses `platt_confidence`

A span's raw support-analog is `1 − r` (r = the span's max token risk). The served
value is `1 − platt_confidence(1 − r, a_span, b_span)` — algebraically
`sigmoid(a_span · logit(r) − b_span)`, so it is exactly the claim map's family in the
risk direction. Fit-time and serve-time arithmetic stay the same engine code, the ε
clamp included, exactly as the claim path does it.

### 3. Transfer check with a pre-registered decision rule

ADR-0008 rejected "let the last run pick the winner"; the same discipline applies here,
so the rule is fixed before any fitting:

- Fit a span-level Platt map on spans derived from the fit population (the ~2,100
  RAGTruth test responses outside the seed-18 slice, scored exactly as the serve path,
  spans at the configured flagging threshold, labeled by overlap with the human span
  annotations).
- Evaluate both candidates on the held-out seed-18 slice's spans: the transferred claim
  coefficients and the span-fitted coefficients, span-level ECE each.
- **Ship the transferred claim coefficients iff their held-out span ECE is within 0.01
  absolute of the span-fitted map's; otherwise ship the span fit.** Ties go to transfer
  because fewer fitted parameters is the smaller trust surface.
- Both numbers, the reliability tables, and which rule fired land in
  `evals/RESULTS.md` regardless of outcome.

The artifact shape does not depend on the outcome: the span section always records the
coefficients actually served (possibly numerically equal to the claim pair) plus their
provenance, so the serve path never special-cases "transferred".

### 4. Span labels: any overlap with an annotated span

An engine-produced span is labeled unsupported iff it overlaps ≥1 character of a
RAGTruth-annotated hallucination span — the mirror of the #45 sentence convention
(supported iff overlapping no annotated span). Partial-overlap counts are reported in
RESULTS.md so the convention's sharpness is visible, but only the one pre-registered
convention feeds the fit and the ECE.

### 5. Artifact schema 2; schema 1 is refused

The artifact grows a `span` section — method, coefficients, provenance (transfer or
fit, with the fit population's identity and hash, and the flagging threshold the spans
were derived at), and in-domain metrics with an explicit statement that span-level
out-of-domain error is unmeasured and why. `KNOWN_SCHEMAS` becomes `{2}`: a schema-1
artifact has no span calibration, and an engine that ships span confidences must refuse
it rather than serve spans uncalibrated — there is no degraded mode, per the existing
refusal requirement. The new artifact ships with a verifier config-version bump, never
an in-place edit.

### 6. The span-flagging threshold becomes a binding

The span population is *conditioned on the threshold*: a map fitted on spans flagged at
0.5 has never seen the spans a 0.3 config would produce. So the artifact's span section
records the threshold it was fitted at, and startup validation compares it against the
running config's span-flagging threshold alongside the existing four bindings —
mismatch names the field with expected and found values and refuses to serve. A
threshold change therefore forces a span refit by construction, exactly as a model swap
forces a claim refit.

### 7. Internal naming stops overloading `confidence`

`reduction.Span.confidence` currently holds the raw max token risk. With a calibrated
number entering the picture, the raw field is renamed (`raw_risk`) so no code path can
confuse the log-only raw value with the served calibrated one; the API schema's span
`confidence` is only ever the calibrated value.

## Risks / Trade-offs

- **Sparse span population** — flagged spans are much rarer than sentences, and bins
  may be thin → report per-bin counts and a bootstrap CI on the held-out span ECE;
  if the seed-18 slice yields too few spans for a stable ECE, that is itself a
  published finding and the transfer rule falls back to shipping the span fit only if
  its CI separates from transfer's.
- **Label noise from the any-overlap convention** (a long engine span grazing a short
  annotation counts as unsupported) → the convention is pre-registered and its
  partial-overlap rate published; refining it is future work with its own evidence.
- **No span-level OOD number** → stated plainly in the artifact and RESULTS.md rather
  than proxied by the claim-level OOD figure; acquiring span-annotated OOD data is out
  of scope.
- **Schema-1 refusal couples deploy** — an engine at this change's version cannot start
  against the old artifact → the new artifact and config-version bump ship in the same
  change, and the refusal error names the schema versions involved.
- **Fit population skew** — in-domain flagged spans may be heavily unsupported-skewed,
  leaving the map poorly constrained on the supported side → the reliability table
  makes this visible; Platt's two parameters keep the extrapolation tame and monotone.

## Migration Plan

One breaking-free, additive API change (a new span field), one coupled config change:
engine + schema-2 artifact + config version bump land together; rollback is the
previous engine with the previous config. No data migration.

## Open Questions

None blocking; the 0.01 ECE transfer margin (Decision 3) is the reviewable knob.
