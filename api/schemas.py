"""Request/response shapes for POST /v1/verify."""

from pydantic import BaseModel, Field

from engine.gate import GateDecision, Verdict


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1)
    # Required non-empty in fast mode (enforced in the handler, where the mode
    # is known); optional in thorough mode, where it joins the evidence pool
    # with caller provenance.
    context: list[str] | None = None
    mode: str


class EvidenceResult(BaseModel):
    """Retrieval provenance: this chunk was retrieved for the claim's query and
    made the scoring window — "retrieved for", never "supports" (joint inference
    cannot name a supporting passage, ADR-0007/0010)."""

    source_id: str
    chunk_id: str
    # Claim-local rerank rank, 1-based.
    rank: int
    # Store snapshot identity where the store exposes one — never invented.
    snapshot_id: str | None = None


class SpanResult(BaseModel):
    # Unicode code-point offsets into the claim's `text`.
    start: int
    end: int
    text: str
    # Calibrated probability that this flagged region is genuinely unsupported by
    # the union of the passages — strictly inside (0, 1), never the raw token risk,
    # which stays in structured logs. Note the direction: higher means more surely
    # unsupported, the opposite of the claim-level confidence.
    confidence: float


class ClaimResult(BaseModel):
    text: str
    # Answer-relative Unicode code-point offsets of this claim in the request `text`,
    # with `text == request.text[start:end]`. Span offsets stay claim-relative.
    start: int
    end: int
    # `contradicted` is deliberately absent from the vocabulary until the NLI signal lands.
    verdict: Verdict
    # Calibrated probability that the claim is fully supported by the union of the
    # passages (ADR-0008) — strictly inside (0, 1), never the raw model score. The
    # raw support stays in structured logs.
    confidence: float
    # Unsupported regions of the claim. Localization, not the verdict's proof:
    # the span-flagging threshold is a separate knob from the verdict threshold,
    # so an unsupported claim with zero spans is legal.
    spans: list[SpanResult]
    # Thorough mode only (fast-mode responses never carry it): retrieval
    # provenance for this claim, in claim-local rank order.
    evidence: list[EvidenceResult] | None = None


class VerifyResponse(BaseModel):
    claims: list[ClaimResult]
    gate: GateDecision
    engine_version: str
    config_version: str
