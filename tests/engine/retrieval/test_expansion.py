"""Golden tests pinning deterministic query expansion (ADR-0010).

These pin exact query strings: a change to windowing, ordering, or joining is
a visible diff here plus a config-version bump, never a silent retrieval shift.
"""

from engine.decomposition import segment
from engine.retrieval import expand_queries

ANSWER = (
    "The Eiffel Tower is in Paris. "
    "It was completed in 1889. "
    "It is 330 metres tall. "
    "The summit holds a private apartment."
)
SENTENCES = [claim.text for claim in segment(ANSWER)]


def test_segmentation_fixture_matches_expectation():
    """The goldens below assume this exact segmentation."""
    assert [s.strip() for s in SENTENCES] == [
        "The Eiffel Tower is in Paris.",
        "It was completed in 1889.",
        "It is 330 metres tall.",
        "The summit holds a private apartment.",
    ]


def test_window_zero_is_leading_sentence_plus_claim():
    assert expand_queries(SENTENCES, window=0) == [
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower is in Paris. It was completed in 1889.",
        "The Eiffel Tower is in Paris. It is 330 metres tall.",
        "The Eiffel Tower is in Paris. The summit holds a private apartment.",
    ]


def test_window_one_adds_the_preceding_sentence():
    assert expand_queries(SENTENCES, window=1) == [
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower is in Paris. It was completed in 1889.",
        "The Eiffel Tower is in Paris. It was completed in 1889. It is 330 metres tall.",
        "The Eiffel Tower is in Paris. It is 330 metres tall. "
        "The summit holds a private apartment.",
    ]


def test_window_two_keeps_answer_order_oldest_first():
    assert expand_queries(SENTENCES, window=2)[3] == (
        "The Eiffel Tower is in Paris. It was completed in 1889. "
        "It is 330 metres tall. The summit holds a private apartment."
    )


def test_window_larger_than_answer_is_the_whole_prefix():
    assert expand_queries(SENTENCES, window=10)[3] == (
        "The Eiffel Tower is in Paris. It was completed in 1889. "
        "It is 330 metres tall. The summit holds a private apartment."
    )


def test_leading_claim_is_not_duplicated():
    assert expand_queries(SENTENCES, window=3)[0] == "The Eiffel Tower is in Paris."


def test_duplicate_sentences_deduplicate_preserving_order():
    assert expand_queries(["Yes.", "Yes.", "No."], window=2) == [
        "Yes.",
        "Yes.",
        "Yes. No.",
    ]


def test_components_are_stripped_and_single_space_joined():
    sentences = ["  First sentence.  ", "\nSecond one.\t"]
    assert expand_queries(sentences, window=1) == [
        "First sentence.",
        "First sentence. Second one.",
    ]


def test_single_sentence_answer():
    assert expand_queries(["Only sentence."], window=2) == ["Only sentence."]
