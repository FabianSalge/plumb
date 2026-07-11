"""Unit tests for sentence segmentation (ADR-0009, openspec/specs/groundedness-scoring)."""

import pytest

from engine.decomposition import Claim, DecompositionError, segment

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
        "indented list markers still split, indentation joins the prior claim",
        "Items:\n  1. First\n  2. Second",
        ["Items:\n  ", "1. First\n  ", "2. Second"],
    ),
    (
        "exclamation and question marks end sentences",
        "Really? Yes! Done.",
        ["Really? ", "Yes! ", "Done."],
    ),
    (
        "a blank line breaks a paragraph even without terminal punctuation",
        "First line\n\nSecond line",
        ["First line\n\n", "Second line"],
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


def test_empty_text_yields_one_empty_claim():
    """The API forbids empty text, but the engine must degrade to the whole-text
    floor rather than crash for an empty answer."""
    assert segment("") == [Claim(text="", start=0, end=0)]


def test_invariant_violation_fails_loud():
    """A partition that leaves a gap or whose claim text disagrees with its offsets
    must never escape — the substring invariant is enforced, not trusted."""
    from engine.decomposition.segmentation import _validate_partition

    _validate_partition("abc", [Claim("ab", 0, 2), Claim("c", 2, 3)])  # well-formed
    with pytest.raises(DecompositionError):
        _validate_partition("abc", [Claim("ab", 0, 2)])  # gap: does not reach the end
    with pytest.raises(DecompositionError):
        _validate_partition("abc", [Claim("a", 0, 1), Claim("c", 2, 3)])  # mid gap
    with pytest.raises(DecompositionError):
        _validate_partition("abc", [Claim("xx", 0, 2), Claim("c", 2, 3)])  # text mismatch
