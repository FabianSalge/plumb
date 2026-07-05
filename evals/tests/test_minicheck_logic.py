"""Unit tests for the MiniCheck flan-t5 chunking/aggregation logic.

Mirrors the reference implementation in Liyan06/MiniCheck (inference.py):
sentence-based greedy chunks of <= chunk_size words, support per response =
min over response sentences of (max over doc chunks).
"""

from bench.adapters.minicheck_flan_t5 import aggregate_support, chunk_sentences


def test_chunks_respect_word_budget():
    sentences = ["one two three.", "four five six.", "seven eight nine."]
    chunks = chunk_sentences(sentences, chunk_size=6)
    assert chunks == ["one two three. four five six.", "seven eight nine."]


def test_oversized_sentence_gets_own_chunk():
    sentences = ["a b c d e f g h.", "short one."]
    chunks = chunk_sentences(sentences, chunk_size=4)
    assert chunks == ["a b c d e f g h.", "short one."]


def test_empty_sentences_yield_single_empty_chunk():
    assert chunk_sentences([], chunk_size=500) == [""]


def test_aggregate_support_min_over_sentences_max_over_chunks():
    # rows = response sentences, cols = doc chunks
    probs = [
        [0.2, 0.9],  # sentence supported by second chunk
        [0.4, 0.3],  # sentence unsupported everywhere
    ]
    assert aggregate_support(probs) == 0.4


def test_aggregate_support_single_cell():
    assert aggregate_support([[0.7]]) == 0.7
