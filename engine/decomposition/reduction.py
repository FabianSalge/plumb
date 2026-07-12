"""Segment-after-score reduction (ADR-0009): score one claim off the whole-answer
forward pass — support = 1 − the maximum token risk overlapping the claim's range,
with spans clipped to the claim and kept claim-relative. Because segmentation is a
total partition and a boundary-straddling token counts toward every claim it
overlaps, the maximum over per-claim risks equals the whole-answer maximum, so the
gate's decision boundary provably does not move."""

import logging
from dataclasses import asdict, dataclass

from engine.decomposition.segmentation import Claim, DecompositionError, segment
from engine.signals import TokenScores

logger = logging.getLogger("plumb.engine.decomposition.reduction")


@dataclass(frozen=True)
class Span:
    """An unsupported region of a claim; `start`/`end` are claim-relative code-point
    offsets into the claim text. `raw_risk` is the raw maximum token probability —
    named raw so it cannot be mistaken for the calibrated confidence the API layer
    derives from it; the raw value itself stays in structured logs."""

    start: int
    end: int
    text: str
    raw_risk: float


@dataclass(frozen=True)
class ScoredClaim:
    """One claim with its union support and the spans flagged as unsupported."""

    text: str
    start: int
    end: int
    support: float
    spans: list[Span]


def reduce_claim(claim: Claim, scores: TokenScores, span_threshold: float) -> ScoredClaim:
    """Score one claim off the whole-answer pass: support = 1 − max risk over the
    tokens overlapping the claim, spans clipped to the claim and kept claim-relative."""
    risks: list[float] = []
    overlapping: list[tuple[float, tuple[int, int]]] = []
    for prob, (ts, te) in zip(scores.probs, scores.offsets, strict=True):
        if ts == te:  # zero-width special token — covers no answer characters
            continue
        if ts < claim.end and te > claim.start:  # overlaps the claim's range
            risks.append(prob)
            rel_start = max(ts, claim.start) - claim.start
            rel_end = min(te, claim.end) - claim.start
            overlapping.append((prob, (rel_start, rel_end)))

    support = 1.0 - max(risks) if risks else 1.0
    if not 0.0 <= support <= 1.0:
        raise DecompositionError(f"claim support {support} outside [0, 1]")

    spans = _merge_spans(claim.text, overlapping, span_threshold)
    if spans:
        logger.info(
            "claim tokens flagged as unsupported",
            extra={"spans": [asdict(span) for span in spans]},
        )
    return ScoredClaim(
        text=claim.text, start=claim.start, end=claim.end, support=support, spans=spans
    )


def decompose(text: str, scores: TokenScores, span_threshold: float) -> list[ScoredClaim]:
    """Segment the answer, then reduce each claim off the single whole-answer pass."""
    return [reduce_claim(claim, scores, span_threshold) for claim in segment(text)]


def _merge_spans(
    claim_text: str, tokens: list[tuple[float, tuple[int, int]]], threshold: float
) -> list[Span]:
    """Merge contiguous flagged tokens into claim-relative character spans."""
    open_span: dict[str, float] | None = None
    closed: list[tuple[int, int, float]] = []

    def close() -> None:
        nonlocal open_span
        if open_span is not None:
            closed.append((int(open_span["start"]), int(open_span["end"]), open_span["conf"]))
            open_span = None

    for prob, (start, end) in tokens:
        if prob >= threshold:
            if open_span is None:
                open_span = {"start": start, "end": end, "conf": prob}
            else:
                open_span["end"] = end
                open_span["conf"] = max(open_span["conf"], prob)
        else:
            close()
    close()
    return [Span(start=s, end=e, text=claim_text[s:e], raw_risk=c) for s, e, c in closed]
