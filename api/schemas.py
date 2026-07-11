"""Request/response shapes for POST /v1/verify."""

from pydantic import BaseModel, Field

from engine.verdict import GateDecision, Verdict


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1)
    context: list[str] = Field(min_length=1)
    mode: str


class SpanResult(BaseModel):
    # Unicode code-point offsets into the claim's `text`. No confidence field
    # until calibration (#32) produces one worth shipping.
    start: int
    end: int
    text: str


class ClaimResult(BaseModel):
    text: str
    # Answer-relative Unicode code-point offsets of this claim in the request `text`,
    # with `text == request.text[start:end]`. Span offsets stay claim-relative.
    start: int
    end: int
    # `contradicted` is deliberately absent from the vocabulary until the NLI signal lands.
    verdict: Verdict
    score: float
    # Unsupported regions of the claim. Localization, not the verdict's proof:
    # the span-flagging threshold is a separate knob from the verdict threshold,
    # so an unsupported claim with zero spans is legal.
    spans: list[SpanResult]


class VerifyResponse(BaseModel):
    claims: list[ClaimResult]
    gate: GateDecision
    engine_version: str
    config_version: str
