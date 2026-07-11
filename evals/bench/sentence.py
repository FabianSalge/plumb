"""Sentence-level scoring on RAGTruth span annotations (issue #45, ADR-0009).

`scored_sentences` is the one sentence scoring loop in the harness: segment a
response with the engine's own segmenter, score it in one whole-answer pass
through the shipping scorer, reduce per sentence, and label each sentence
hallucinated iff its character range overlaps an annotated span. Everything
downstream (benchmark AUROC, calibration fits) is a view over its output.
"""

from collections.abc import Iterator, Sequence

from bench.data import Example
from engine.decomposition import ScoredClaim, decompose
from engine.signals import Scorer


def sentence_hallucinated(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    """True iff the sentence range [start, end) overlaps any annotated span."""
    return any(span_start < end and span_end > start for span_start, span_end in spans)


def scored_sentences(
    example: Example, scorer: Scorer, span_threshold: float
) -> Iterator[tuple[ScoredClaim, int]]:
    """Yield (claim, label) per sentence of one response — one whole-answer pass,
    segmented by the engine's segmenter; label 1 = the sentence is hallucinated."""
    scores = scorer.score(example.response, [example.context])
    for claim in decompose(example.response, scores, span_threshold):
        yield claim, int(sentence_hallucinated(claim.start, claim.end, example.spans))


def sentence_scores(
    example: Example, scorer: Scorer, span_threshold: float
) -> Iterator[tuple[int, float]]:
    """(label, risk) view over `scored_sentences`: risk = 1 − per-sentence support."""
    for claim, label in scored_sentences(example, scorer, span_threshold):
        yield label, 1.0 - claim.support
