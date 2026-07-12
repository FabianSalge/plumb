"""Sentence-level NLI labelling on RAGTruth span annotations (issue #60).

The NLI slot exists to tell *refuted-by-evidence* apart from *merely
unsupported*, so each sentence gets a three-way class from the annotated span
kinds: `conflict` if it overlaps any conflict span, else `baseless` if it
overlaps any baseless span, else `supported`. Segmentation is the engine's own
segmenter — the same sentence unit the groundedness benchmarks score.
"""

from collections.abc import Iterator, Sequence

from bench.adapters.nli import NliProbs
from bench.data import Example
from engine.decomposition import Claim, segment

CONFLICT = "conflict"
BASELESS = "baseless"
SUPPORTED = "supported"


def sentence_nli_class(
    start: int,
    end: int,
    spans: Sequence[tuple[int, int]],
    kinds: Sequence[str],
) -> str:
    """Three-way label for the sentence range [start, end); conflict wins when
    spans of both kinds overlap, because a refuted sentence is refuted."""
    overlapping = {
        kind
        for (span_start, span_end), kind in zip(spans, kinds, strict=True)
        if span_start < end and span_end > start
    }
    if CONFLICT in overlapping:
        return CONFLICT
    if BASELESS in overlapping:
        return BASELESS
    return SUPPORTED


def nli_sentence_classes(example: Example) -> Iterator[tuple[Claim, str]]:
    """Yield (claim, class) per sentence of one response, segmented by the
    engine's segmenter and labelled from the annotated span kinds."""
    for claim in segment(example.response):
        yield claim, sentence_nli_class(claim.start, claim.end, example.spans, example.span_kinds)


def predicted_contradicted(probs: NliProbs) -> bool:
    """The argmax verdict a `contradicted` gate outcome would read off this signal."""
    return probs.contradiction > max(probs.entailment, probs.neutral)


def contradiction_pairs(rows: Sequence[tuple[str, NliProbs]]) -> tuple[list[int], list[float]]:
    """(labels, scores) for the headline metric: among hallucinated sentences,
    does P(contradiction) rank refuted (conflict) above merely unsupported
    (baseless)? Label 1 = conflict."""
    hallucinated = [(cls, probs) for cls, probs in rows if cls != SUPPORTED]
    labels = [int(cls == CONFLICT) for cls, _ in hallucinated]
    scores = [probs.contradiction for _, probs in hallucinated]
    return labels, scores


def hallucination_pairs(rows: Sequence[tuple[str, NliProbs]]) -> tuple[list[int], list[float]]:
    """(labels, risks) for comparability with the groundedness sentence benchmark:
    label 1 = hallucinated (either kind), risk = 1 − P(entailment)."""
    labels = [int(cls != SUPPORTED) for cls, _ in rows]
    risks = [1.0 - probs.entailment for _, probs in rows]
    return labels, risks
