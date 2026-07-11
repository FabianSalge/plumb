"""Verdict mapping and the conjunctive gate decision.

Since ADR-0008 the number judged is the calibrated confidence, never the raw
support score — the threshold lives in confidence space.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["supported", "unsupported"]
GateDecision = Literal["pass", "block"]


@dataclass(frozen=True)
class ClaimVerdict:
    text: str
    verdict: Verdict
    confidence: float


def judge_claim(text: str, confidence: float, threshold: float) -> ClaimVerdict:
    verdict: Verdict = "supported" if confidence >= threshold else "unsupported"
    return ClaimVerdict(text=text, verdict=verdict, confidence=confidence)


def gate_decision(claims: Sequence[ClaimVerdict]) -> GateDecision:
    return "pass" if all(claim.verdict == "supported" for claim in claims) else "block"
