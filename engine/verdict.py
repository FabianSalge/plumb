"""Verdict mapping and the conjunctive gate decision."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["supported", "unsupported"]
GateDecision = Literal["pass", "block"]


@dataclass(frozen=True)
class ClaimVerdict:
    text: str
    verdict: Verdict
    score: float


def judge_claim(text: str, score: float, threshold: float) -> ClaimVerdict:
    verdict: Verdict = "supported" if score >= threshold else "unsupported"
    return ClaimVerdict(text=text, verdict=verdict, score=score)


def gate_decision(claims: Sequence[ClaimVerdict]) -> GateDecision:
    return "pass" if all(claim.verdict == "supported" for claim in claims) else "block"
