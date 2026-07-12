"""Span-level labelling on RAGTruth span annotations (issue #40).

`labeled_spans` mirrors `bench.sentence.scored_sentences` one level down: score a
response exactly as `/v1/verify` does (one whole-answer pass, the engine's own
segmenter and reduction), take the engine's flagged spans, re-base them to answer
coordinates, and label each span unsupported iff it overlaps at least one
character of a human-annotated hallucination span — the mirror of the sentence
convention. `partial` marks the label convention's fuzzy edge: an unsupported
span not fully covered by the annotations."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from bench.data import Example
from engine.decomposition import decompose
from engine.signals import Scorer


@dataclass(frozen=True)
class LabeledSpan:
    """One engine-flagged span in answer coordinates with its human label."""

    example_id: str
    start: int
    end: int
    raw_risk: float
    unsupported: int  # 1 iff the span overlaps any annotated hallucination span
    partial: bool  # unsupported but not fully covered by the annotations


def span_unsupported(start: int, end: int, annotations: Sequence[tuple[int, int]]) -> bool:
    """True iff the span range [start, end) overlaps any annotated span."""
    return any(a_start < end and a_end > start for a_start, a_end in annotations)


def span_fully_covered(start: int, end: int, annotations: Sequence[tuple[int, int]]) -> bool:
    """True iff every character of [start, end) lies inside some annotated span."""
    position = start
    for a_start, a_end in sorted(annotations):
        if a_end <= position:
            continue
        if a_start > position:
            return False
        position = a_end
        if position >= end:
            return True
    return position >= end


def labeled_spans(example: Example, scorer: Scorer, span_threshold: float) -> Iterator[LabeledSpan]:
    """Yield every engine-flagged span of one response, answer-relative and labeled
    against the example's annotated hallucination spans."""
    scores = scorer.score(example.response, [example.context])
    for claim in decompose(example.response, scores, span_threshold):
        for span in claim.spans:
            start = claim.start + span.start
            end = claim.start + span.end
            unsupported = span_unsupported(start, end, example.spans)
            yield LabeledSpan(
                example_id=example.id,
                start=start,
                end=end,
                raw_risk=span.raw_risk,
                unsupported=int(unsupported),
                partial=unsupported and not span_fully_covered(start, end, example.spans),
            )
