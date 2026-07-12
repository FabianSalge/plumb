"""Unit tests for NLI sentence labelling and adapter plumbing — pure logic, no model."""

import pytest
from bench.adapters.nli import label_indices
from bench.data import DataError, Example, hallucination_spans, span_kind
from bench.nli import (
    BASELESS,
    CONFLICT,
    SUPPORTED,
    contradiction_pairs,
    hallucination_pairs,
    nli_sentence_classes,
    sentence_nli_class,
)

# --- span kinds (bench.data) ----------------------------------------------------


def test_span_kind_maps_all_four_ragtruth_label_types():
    assert span_kind("Evident Conflict") == CONFLICT
    assert span_kind("Subtle Conflict") == CONFLICT
    assert span_kind("Evident Baseless Info") == BASELESS
    assert span_kind("Subtle Baseless Info") == BASELESS


def test_span_kind_refuses_unknown_label_type():
    with pytest.raises(DataError):
        span_kind("Novel Category")


def test_hallucination_spans_carry_their_kind():
    raw = (
        '[{"start": 0, "end": 5, "label_type": "Evident Conflict"},'
        ' {"start": 7, "end": 9, "label_type": "Subtle Baseless Info"}]'
    )
    assert hallucination_spans(raw) == ((0, 5, CONFLICT), (7, 9, BASELESS))


def test_hallucination_spans_missing_label_type_fails_loudly():
    with pytest.raises(DataError):
        hallucination_spans('[{"start": 0, "end": 5}]')


# --- sentence labelling (bench.nli) ----------------------------------------------


def test_sentence_overlapping_a_conflict_span_is_contradicted():
    assert sentence_nli_class(0, 15, ((10, 20),), (CONFLICT,)) == CONFLICT


def test_sentence_overlapping_only_baseless_spans_is_baseless():
    assert sentence_nli_class(0, 15, ((10, 20),), (BASELESS,)) == BASELESS


def test_conflict_wins_when_both_kinds_overlap():
    spans = ((0, 8), (10, 20))
    kinds = (BASELESS, CONFLICT)
    assert sentence_nli_class(0, 15, spans, kinds) == CONFLICT


def test_sentence_touching_no_span_is_supported():
    assert sentence_nli_class(0, 10, ((10, 20),), (CONFLICT,)) == SUPPORTED
    assert sentence_nli_class(0, 10, (), ()) == SUPPORTED


def test_nli_sentence_classes_segments_and_labels():
    text = "Paris is nice. It has 50m people."
    second_start = len("Paris is nice.") + 1
    example = Example(
        id="x",
        task_type="QA",
        query="",
        context="doc",
        response=text,
        hallucinated=True,
        spans=((second_start, len(text)),),
        span_kinds=(CONFLICT,),
    )
    labelled = [(claim.text.strip(), cls) for claim, cls in nli_sentence_classes(example)]
    assert labelled == [("Paris is nice.", SUPPORTED), ("It has 50m people.", CONFLICT)]


# --- metric views (bench.nli) -----------------------------------------------------


def _probs(entailment: float, neutral: float, contradiction: float):
    from bench.adapters.nli import NliProbs

    return NliProbs(
        entailment=entailment, neutral=neutral, contradiction=contradiction, truncated=False
    )


def test_contradiction_pairs_cover_only_hallucinated_sentences():
    rows = [
        (CONFLICT, _probs(0.1, 0.2, 0.7)),
        (BASELESS, _probs(0.2, 0.7, 0.1)),
        (SUPPORTED, _probs(0.9, 0.05, 0.05)),
    ]
    labels, scores = contradiction_pairs(rows)
    assert labels == [1, 0]
    assert scores == [0.7, 0.1]


def test_predicted_contradicted_is_the_argmax_verdict():
    from bench.nli import predicted_contradicted

    assert predicted_contradicted(_probs(0.1, 0.2, 0.7))
    assert not predicted_contradicted(_probs(0.2, 0.7, 0.1))
    assert not predicted_contradicted(_probs(0.7, 0.2, 0.1))


def test_hallucination_pairs_score_risk_as_one_minus_entailment():
    rows = [
        (CONFLICT, _probs(0.1, 0.2, 0.7)),
        (SUPPORTED, _probs(0.9, 0.05, 0.05)),
    ]
    labels, risks = hallucination_pairs(rows)
    assert labels == [1, 0]
    assert risks == [pytest.approx(0.9), pytest.approx(0.1)]


# --- label index resolution (bench.adapters.nli) ----------------------------------


def test_label_indices_resolves_tasksource_order():
    assert label_indices({0: "entailment", 1: "neutral", 2: "contradiction"}) == (0, 1, 2)


def test_label_indices_resolves_crossencoder_order_and_case():
    assert label_indices({0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}) == (1, 2, 0)


def test_label_indices_refuses_non_nli_heads():
    with pytest.raises(ValueError):
        label_indices({0: "positive", 1: "negative"})
    with pytest.raises(ValueError):
        label_indices({0: "entailment", 1: "neutral"})
