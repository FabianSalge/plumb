"""Request/response shapes for POST /v1/verify."""

from pydantic import BaseModel, Field

from engine.verdict import GateDecision, Verdict


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1)
    context: list[str] = Field(min_length=1)
    mode: str


class ClaimResult(BaseModel):
    text: str
    # `contradicted` is deliberately absent from the vocabulary until the NLI signal lands.
    verdict: Verdict
    score: float
    evidence_index: int


class VerifyResponse(BaseModel):
    claims: list[ClaimResult]
    gate: GateDecision
    engine_version: str
    config_version: str
