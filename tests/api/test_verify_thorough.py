"""Contract tests for thorough mode: retrieval-backed verification (ADR-0010)."""

import logging

import pytest

from engine.retrieval import Chunk, StoreError
from tests.engine.signals.fakes import char_scores

ANSWER = "The sky is blue. Grass is green."

SKY = Chunk(
    text="The sky appears blue from Rayleigh scattering.",
    source_id="wiki/sky",
    chunk_id="s1",
    snapshot_id="v1",
)
MISC = Chunk(text="An unrelated passage.", source_id="wiki/misc", chunk_id="m1")
GRASS = Chunk(
    text="Grass is green because of chlorophyll.",
    source_id="wiki/grass",
    chunk_id="g1",
    snapshot_id="v1",
)


class FakeStore:
    """Maps the first matching needle to canned chunks; records queries."""

    def __init__(self, results: dict[str, list[Chunk]]) -> None:
        self._results = results
        self.queries: list[str] = []

    def recall(self, query: str, k: int) -> list[Chunk]:
        self.queries.append(query)
        for needle, chunks in self._results.items():
            if needle in query:
                return chunks[:k]
        return []


class FailingStore:
    def recall(self, query: str, k: int) -> list[Chunk]:
        raise StoreError("store unreachable at db:5432")


class FakeReranker:
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [
            float(sum(word.lower().strip(".") in passage.lower() for word in query.split()))
            for query, passage in pairs
        ]


def default_store() -> FakeStore:
    # Ordered: the grass needle first, since claim 1's expanded query also
    # contains claim 0's text.
    return FakeStore({"Grass": [GRASS], "sky": [SKY, MISC]})


def thorough_client(make_client, scores=None, **kwargs):
    defaults = {"retrieval": True, "store": default_store(), "reranker": FakeReranker()}
    defaults.update(kwargs)
    return make_client(scores or char_scores(ANSWER), **defaults)


def verify_thorough(client, *, text: str = ANSWER, **body):
    return client.post("/v1/verify", json={"text": text, "mode": "thorough", **body})


def test_thorough_mode_without_context_verifies_against_the_store(make_client):
    resp = verify_thorough(thorough_client(make_client))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["claims"]) == 2
    assert body["engine_version"] and body["config_version"]


def test_claims_carry_retrieval_provenance(make_client):
    body = verify_thorough(thorough_client(make_client)).json()
    sky_refs = body["claims"][0]["evidence"]
    assert [(ref["source_id"], ref["chunk_id"], ref["rank"]) for ref in sky_refs] == [
        ("wiki/sky", "s1", 1),
        ("wiki/misc", "m1", 2),
    ]
    grass_refs = body["claims"][1]["evidence"]
    assert [(ref["source_id"], ref["chunk_id"], ref["rank"]) for ref in grass_refs] == [
        ("wiki/grass", "g1", 1),
    ]


def test_snapshot_identity_present_only_where_the_store_exposes_one(make_client):
    body = verify_thorough(thorough_client(make_client)).json()
    sky_refs = body["claims"][0]["evidence"]
    assert sky_refs[0]["snapshot_id"] == "v1"
    assert "snapshot_id" not in sky_refs[1], "absent snapshot identity must not be invented"


def test_queries_are_expanded_with_the_configured_window(make_client):
    store = default_store()
    verify_thorough(thorough_client(make_client, store=store))
    assert store.queries == [
        "The sky is blue.",
        "The sky is blue. Grass is green.",
    ]


def test_caller_context_joins_the_pool_but_not_the_evidence(make_client):
    empty_store = FakeStore({})
    resp = verify_thorough(
        thorough_client(make_client, store=empty_store),
        context=["The sky is blue and grass is green."],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert all(claim["evidence"] == [] for claim in body["claims"]), (
        "caller passages are the caller's own — never listed as retrieved evidence"
    )


def test_fast_mode_responses_carry_no_evidence(make_client):
    client = thorough_client(make_client)
    resp = client.post(
        "/v1/verify",
        json={"text": ANSWER, "context": ["The sky is blue."], "mode": "fast"},
    )
    assert resp.status_code == 200
    assert all("evidence" not in claim for claim in resp.json()["claims"])


def test_unknown_mode_is_rejected_naming_both_modes(make_client):
    resp = thorough_client(make_client).post("/v1/verify", json={"text": ANSWER, "mode": "turbo"})
    assert resp.status_code == 400
    assert "fast" in resp.text and "thorough" in resp.text


def test_thorough_on_a_deployment_without_a_store_is_400_fast_only(make_client):
    client = make_client(char_scores(ANSWER), retrieval=True, store=None)
    resp = verify_thorough(client)
    assert resp.status_code == 400
    assert "fast" in resp.text


def test_thorough_without_retrieval_config_is_400_fast_only(make_client):
    client = make_client(
        char_scores(ANSWER), retrieval=False, store=default_store(), reranker=FakeReranker()
    )
    resp = verify_thorough(client)
    assert resp.status_code == 400
    assert "fast" in resp.text


def test_store_failure_is_502_with_no_verdict(make_client):
    resp = verify_thorough(thorough_client(make_client, store=FailingStore()))
    assert resp.status_code == 502
    assert "store" in resp.text
    assert "claims" not in resp.json()


def test_no_evidence_at_all_fails_loudly(make_client):
    resp = verify_thorough(thorough_client(make_client, store=FakeStore({})))
    assert resp.status_code == 422
    assert "evidence" in resp.text


def test_context_optional_in_thorough_but_required_in_fast(make_client):
    client = thorough_client(make_client)
    assert verify_thorough(client).status_code == 200
    fast = client.post("/v1/verify", json={"text": ANSWER, "mode": "fast"})
    assert fast.status_code == 400
    assert "context" in fast.text


@pytest.mark.parametrize(
    ("flag", "expected_gate"),
    [
        (None, "pass"),
        ((0, 3), "block"),
        ((20, 25), "block"),
        ((0, len(ANSWER)), "block"),
    ],
)
def test_gate_parity_over_a_fixed_pool(make_client, flag, expected_gate):
    """Over the same pooled evidence, the decomposed gate equals the whole-text
    gate: the whole-answer risk is the max over per-claim risks, so with the
    identity calibration a flagged region blocks either way."""
    scores = char_scores(ANSWER, flag=flag)
    body = verify_thorough(thorough_client(make_client, scores=scores)).json()
    whole_text_support = 1.0 - max(scores.probs)
    whole_text_gate = "pass" if whole_text_support >= 0.5 else "block"
    assert body["gate"] == expected_gate
    assert body["gate"] == whole_text_gate


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_scorer_truncation_in_thorough_mode_is_an_error_log(make_client):
    from engine.signals import TokenScores

    scores = char_scores(ANSWER)
    truncated = TokenScores(probs=scores.probs, offsets=scores.offsets, truncated=True)
    client = thorough_client(make_client, scores=truncated)
    # setup_logging (run by create_app) resets root handlers, so capture on the
    # api logger directly rather than through caplog.
    handler = _ListHandler()
    api_logger = logging.getLogger("plumb.api")
    api_logger.addHandler(handler)
    try:
        resp = verify_thorough(client)
    finally:
        api_logger.removeHandler(handler)
    assert resp.status_code == 200
    assert any(
        "truncat" in record.getMessage() and record.levelno >= logging.ERROR
        for record in handler.records
    ), "scorer-window truncation in thorough mode indicates a budgeting bug and must log as error"
