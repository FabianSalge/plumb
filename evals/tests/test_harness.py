import json

from bench.harness import DEFAULT_CONFIG, environment, percentile, write_results


def test_percentile_p50_of_odd_list_is_the_median():
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0


def test_percentile_p95_of_hundred_values():
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 95) == 96.0


def test_percentile_single_value():
    assert percentile([7.0], 95) == 7.0


def test_environment_stamps_platform_and_libraries():
    env = environment()
    assert set(env) >= {"platform", "python", "torch", "transformers"}


def test_default_config_points_at_the_repo_verifier_config():
    assert DEFAULT_CONFIG.name == "verifier.yaml"
    assert DEFAULT_CONFIG.parent.name == "config"


def test_write_results_creates_parents_and_ends_with_newline(tmp_path):
    out = tmp_path / "nested" / "result.json"
    write_results(out, {"a": 1})
    text = out.read_text()
    assert text.endswith("\n")
    assert json.loads(text) == {"a": 1}
