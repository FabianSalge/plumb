"""Unit tests for sentence decomposition and segment-after-score reduction
(ADR-0009, openspec/specs/groundedness-scoring)."""

import logging
import random

import pytest

from engine.decomposition import (
    Claim,
    DecompositionError,
    ScoredClaim,
    decompose,
    reduce_claim,
    segment,
)
from engine.scoring import TokenScores
from engine.verdict import gate_decision, judge_claim


def token_scores(*tokens: tuple[float, int, int]) -> TokenScores:
    """Build whole-answer TokenScores from (prob, start, end) triples."""
    return TokenScores(probs=[p for p, _, _ in tokens], offsets=[(s, e) for _, s, e in tokens])


# --- Segmentation golden cases -------------------------------------------------
#
# The hand-rolled segmenter is only as good as these goldens (ADR-0009). Each
# pins one named hard case; the whole point of owning the rules is that the
# expected splits live here, verbatim.

GOLDENS: list[tuple[str, str, list[str]]] = [
    (
        "abbreviations and initialisms do not split",
        "Dr. Smith studied at M.I.T. and later moved abroad. He never returned.",
        [
            "Dr. Smith studied at M.I.T. and later moved abroad. ",
            "He never returned.",
        ],
    ),
    (
        "numbered list items are separate claims",
        "Steps:\n1. Preheat the oven.\n2. Bake the cake.",
        ["Steps:\n", "1. Preheat the oven.\n", "2. Bake the cake."],
    ),
    (
        "fenced code blocks stay atomic",
        "Run this:\n```\nx = 1. y = 2.\n```\nDone.",
        ["Run this:\n", "```\nx = 1. y = 2.\n```\n", "Done."],
    ),
    (
        "trailing sentence without terminal punctuation runs to the end",
        "First fact. Second fact without period",
        ["First fact. ", "Second fact without period"],
    ),
    (
        "a single sentence is one claim",
        "Just one sentence.",
        ["Just one sentence."],
    ),
    (
        "text with no boundary is one whole-text claim",
        "The sky is blue",
        ["The sky is blue"],
    ),
]


@pytest.mark.parametrize("name, text, expected", GOLDENS, ids=[g[0] for g in GOLDENS])
def test_segmentation_goldens(name, text, expected):
    assert [claim.text for claim in segment(text)] == expected


@pytest.mark.parametrize("text", [g[1] for g in GOLDENS], ids=[g[0] for g in GOLDENS])
def test_segmentation_is_a_total_partition(text):
    """Claims tile the text with no gaps and the substring invariant holds — the
    two properties every token's risk relies on to land in some claim."""
    claims = segment(text)
    assert "".join(claim.text for claim in claims) == text
    for claim in claims:
        assert text[claim.start : claim.end] == claim.text
    # offsets are contiguous
    assert claims[0].start == 0
    assert claims[-1].end == len(text)
    for prev, nxt in zip(claims, claims[1:], strict=False):
        assert prev.end == nxt.start


def test_no_boundary_yields_one_whole_text_claim():
    claims = segment("the sky is blue")
    assert claims == [Claim(text="the sky is blue", start=0, end=15)]


def test_invariant_violation_fails_loud():
    """A partition that leaves a gap or whose claim text disagrees with its offsets
    must never escape — the substring invariant is enforced, not trusted."""
    from engine.decomposition import _validate_partition

    _validate_partition("abc", [Claim("ab", 0, 2), Claim("c", 2, 3)])  # well-formed
    with pytest.raises(DecompositionError):
        _validate_partition("abc", [Claim("ab", 0, 2)])  # gap: does not reach the end
    with pytest.raises(DecompositionError):
        _validate_partition("abc", [Claim("xx", 0, 2), Claim("c", 2, 3)])  # text mismatch


# --- Per-claim reduction -------------------------------------------------------


def test_support_is_one_minus_max_overlapping_token_risk():
    claim = Claim("Paris is small.", 0, 15)
    scores = token_scores((0.1, 0, 5), (0.9, 6, 8), (0.95, 9, 14), (0.2, 14, 15))
    assert reduce_claim(claim, scores, span_threshold=0.5).support == pytest.approx(0.05)


def test_only_tokens_overlapping_the_claim_count():
    """A token entirely outside the claim's character range is ignored, so a
    later claim's risk does not leak into an earlier one."""
    claim = Claim("AB", 0, 2)
    scores = token_scores((0.1, 0, 2), (0.99, 5, 9))
    assert reduce_claim(claim, scores, span_threshold=0.5).support == pytest.approx(0.9)


def test_boundary_straddling_token_counts_for_both_claims():
    """A token whose range crosses a claim boundary counts toward every claim it
    overlaps, so no token risk is dropped at the seam (ADR-0009)."""
    scores = token_scores((0.8, 0, 2))  # straddles [0,1) and [1,2)
    left = reduce_claim(Claim("A", 0, 1), scores, span_threshold=0.5)
    right = reduce_claim(Claim("B", 1, 2), scores, span_threshold=0.5)
    assert left.support == pytest.approx(0.2)
    assert right.support == pytest.approx(0.2)


def test_zero_width_special_tokens_are_excluded():
    """Special tokens (zero-length offsets) cover no answer characters and must
    not enter any claim's reduction."""
    claim = Claim("Paris", 0, 5)
    scores = token_scores((0.99, 0, 0), (0.1, 0, 5))
    assert reduce_claim(claim, scores, span_threshold=0.5).support == pytest.approx(0.9)


def test_out_of_range_support_fails_loud():
    claim = Claim("x", 0, 1)
    with pytest.raises(DecompositionError):
        reduce_claim(claim, token_scores((1.5, 0, 1)), span_threshold=0.5)
    with pytest.raises(DecompositionError):
        reduce_claim(claim, token_scores((-0.5, 0, 1)), span_threshold=0.5)


# --- Per-claim spans -----------------------------------------------------------


def test_flagged_tokens_become_claim_relative_spans():
    claim = Claim("Paris is small.", 0, 15)
    scores = token_scores((0.1, 0, 5), (0.9, 6, 8), (0.95, 9, 14), (0.2, 14, 15))
    spans = reduce_claim(claim, scores, span_threshold=0.5).spans
    assert [(s.start, s.end, s.text) for s in spans] == [(6, 14, "is small")]


def test_spans_are_clipped_and_rebased_to_the_claim():
    """A flagged token straddling the claim's left boundary is clipped to the
    claim and its offsets are claim-relative, so span.text slices claim.text."""
    claim = Claim("small tail", 10, 20)
    scores = token_scores((0.9, 8, 15))  # answer chars 8..15, claim starts at 10
    span = reduce_claim(claim, scores, span_threshold=0.5).spans[0]
    assert (span.start, span.end) == (0, 5)
    assert span.text == claim.text[span.start : span.end] == "small"


def test_span_threshold_is_injected_not_hardcoded():
    claim = Claim("hello world", 0, 11)
    scores = token_scores((0.6, 0, 5), (0.2, 5, 11))
    assert reduce_claim(claim, scores, span_threshold=0.5).spans
    assert not reduce_claim(claim, scores, span_threshold=0.7).spans


def test_spans_logged_with_confidences_response_carries_positions_only(caplog):
    claim = Claim("Paris is small.", 0, 15)
    scores = token_scores((0.1, 0, 5), (0.9, 6, 8), (0.95, 9, 14), (0.2, 14, 15))
    with caplog.at_level(logging.INFO, logger="plumb.engine.decomposition"):
        result = reduce_claim(claim, scores, span_threshold=0.5)
    records = [r for r in caplog.records if hasattr(r, "spans")]
    assert len(records) == 1
    assert records[0].spans == [{"start": 6, "end": 14, "text": "is small", "confidence": 0.95}]
    # The returned span object carries the confidence for the API layer to drop.
    assert result.spans[0].confidence == 0.95


# --- Decompose orchestration ---------------------------------------------------


def test_decompose_segments_then_reduces_each_claim():
    text = "The sky is blue. Grass is green."
    # one token per character, so per-claim maxima are easy to reason about
    scores = token_scores(*[(0.1, i, i + 1) for i in range(len(text))])
    claims = decompose(text, scores, span_threshold=0.5)
    assert [c.text for c in claims] == ["The sky is blue. ", "Grass is green."]
    assert [(c.start, c.end) for c in claims] == [(0, 17), (17, 32)]
    assert all(isinstance(c, ScoredClaim) for c in claims)


# --- Gate parity (the property ADR-0009 stakes the design on) ------------------


def _random_scores(rng: random.Random, length: int) -> TokenScores:
    """Tokens tiling [0, length) with random widths and risks, plus a trailing
    zero-width special token — the shape a real forward pass produces."""
    tokens: list[tuple[float, int, int]] = []
    pos = 0
    while pos < length:
        width = rng.randint(1, 3)
        end = min(length, pos + width)
        tokens.append((round(rng.random(), 3), pos, end))
        pos = end
    tokens.append((round(rng.random(), 3), 0, 0))  # special token
    return token_scores(*tokens)


def _random_partition(rng: random.Random, length: int) -> list[Claim]:
    cuts = sorted(rng.sample(range(1, length), k=rng.randint(0, min(4, length - 1))))
    bounds = [0, *cuts, length]
    return [Claim("x" * (b - a), a, b) for a, b in zip(bounds, bounds[1:], strict=False)]


@pytest.mark.parametrize("seed", range(50))
def test_gate_parity_reduction_is_min_over_claims(seed):
    """min(per-claim support) == whole-answer support, for any total partition —
    the identity the gate's decision boundary rests on."""
    rng = random.Random(seed)
    length = rng.randint(2, 40)
    scores = _random_scores(rng, length)
    claims = _random_partition(rng, length)
    whole = reduce_claim(Claim("x" * length, 0, length), scores, span_threshold=0.5)
    per_claim = [reduce_claim(c, scores, span_threshold=0.5) for c in claims]
    assert min(c.support for c in per_claim) == pytest.approx(whole.support)


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("text", [g[1] for g in GOLDENS], ids=[g[0] for g in GOLDENS])
def test_gate_parity_decomposed_equals_whole_text(seed, text):
    """The decomposed gate equals the whole-text gate at the same threshold, over
    the real segmenter and random token risks (AC: gate-parity property test)."""
    rng = random.Random(seed)
    scores = _random_scores(rng, len(text))
    threshold = round(rng.random(), 3)

    decomposed = [
        judge_claim(c.text, c.support, threshold)
        for c in decompose(text, scores, span_threshold=0.5)
    ]
    whole = reduce_claim(Claim(text, 0, len(text)), scores, span_threshold=0.5)
    whole_verdict = judge_claim(text, whole.support, threshold)

    assert gate_decision(decomposed) == gate_decision([whole_verdict])
