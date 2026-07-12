"""Fast-mode latency bench logic (#36): request shape, summary, server sanity."""

import pytest
from bench.data import Example
from bench.latency import LatencyError, ambient_load, check_identity, summarize, verify_payload


def example(**overrides) -> Example:
    fields = {
        "id": "qa-1",
        "task_type": "QA",
        "query": "What is the capital of France?",
        "context": "Paris is the capital of France.",
        "response": "The capital of France is Paris.",
        "hallucinated": False,
    }
    fields.update(overrides)
    return Example(**fields)


class TestVerifyPayload:
    def test_whole_answer_fast_mode(self):
        payload = verify_payload(example())
        assert payload == {
            "text": "The capital of France is Paris.",
            "context": ["Paris is the capital of France."],
            "mode": "fast",
        }


class TestSummarize:
    def test_percentiles_against_contract(self):
        millis = [float(v) for v in range(100, 1100, 10)]  # 100..1090, 100 values
        summary = summarize(millis, contract_ms=1000.0)
        assert summary["n"] == 100
        assert summary["p50_ms"] == 595.0
        assert summary["p95_ms"] == 1050.0
        assert summary["contract_ms"] == 1000.0
        assert summary["meets_contract"] is False

    def test_meets_contract_when_p95_under(self):
        summary = summarize([200.0, 300.0, 400.0], contract_ms=1000.0)
        assert summary["meets_contract"] is True

    def test_empty_run_fails_loudly(self):
        with pytest.raises(LatencyError):
            summarize([], contract_ms=1000.0)


class TestAmbientLoad:
    # #59: #36's p95 was inflated by concurrent machine load the results JSON
    # never recorded. Every latency run now stamps the load averages found
    # before its first request, so a contaminated measurement reads as one.
    def test_reports_the_three_load_averages(self):
        load = ambient_load()
        assert set(load) == {"1m", "5m", "15m"}
        for value in load.values():
            assert isinstance(value, float)
            assert value >= 0.0


class TestCheckIdentity:
    def test_first_response_sets_identity(self):
        body = {"claims": [{"text": "x"}], "engine_version": "0.1.0", "config_version": "0.5.0"}
        assert check_identity(body, expected=None) == ("0.1.0", "0.5.0")

    def test_consistent_identity_passes(self):
        body = {"claims": [{"text": "x"}], "engine_version": "0.1.0", "config_version": "0.5.0"}
        assert check_identity(body, expected=("0.1.0", "0.5.0")) == ("0.1.0", "0.5.0")

    def test_identity_drift_fails_loudly(self):
        body = {"claims": [{"text": "x"}], "engine_version": "0.1.0", "config_version": "0.6.0"}
        with pytest.raises(LatencyError, match="0.6.0"):
            check_identity(body, expected=("0.1.0", "0.5.0"))

    def test_empty_claims_fails_loudly(self):
        body = {"claims": [], "engine_version": "0.1.0", "config_version": "0.5.0"}
        with pytest.raises(LatencyError, match="claims"):
            check_identity(body, expected=None)

    def test_missing_fields_fail_loudly(self):
        with pytest.raises(LatencyError):
            check_identity({"claims": [{"text": "x"}]}, expected=None)
