"""Sentence-level discrimination on RAGTruth span annotations (issue #45, ADR-0009).

Segment each response with the engine's own segmenter, score it in one
whole-answer pass through the shipping scorer, and reduce per sentence. A
sentence is hallucinated iff its character range overlaps an annotated span;
discrimination is AUROC over sentence risk = 1 − support.
"""

from collections.abc import Iterator, Sequence

from bench.data import Example
from engine.decomposition import decompose
from engine.scoring import Scorer


def sentence_hallucinated(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    """True iff the sentence range [start, end) overlaps any annotated span."""
    return any(span_start < end and span_end > start for span_start, span_end in spans)


def sentence_scores(
    example: Example, scorer: Scorer, span_threshold: float
) -> Iterator[tuple[int, float]]:
    """Yield (label, risk) per sentence of one response — one whole-answer pass,
    segmented by the engine's segmenter, risk = 1 − per-sentence support."""
    scores = scorer.score(example.response, [example.context])
    for claim in decompose(example.response, scores, span_threshold):
        label = int(sentence_hallucinated(claim.start, claim.end, example.spans))
        yield label, 1.0 - claim.support
