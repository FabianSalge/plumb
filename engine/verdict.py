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
    evidence_index: int


def judge_claim(text: str, scores: Sequence[float], threshold: float) -> ClaimVerdict:
    if not scores:
        raise ValueError("cannot judge a claim with no evidence scores")
    best_index = max(range(len(scores)), key=lambda i: scores[i])
    best = scores[best_index]
    verdict: Verdict = "supported" if best >= threshold else "unsupported"
    return ClaimVerdict(text=text, verdict=verdict, score=best, evidence_index=best_index)


def gate_decision(claims: Sequence[ClaimVerdict]) -> GateDecision:
    return "pass" if all(claim.verdict == "supported" for claim in claims) else "block"
