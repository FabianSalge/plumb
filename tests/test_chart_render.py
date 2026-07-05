"""Render tests for charts/plumb: the deployment contract, checked via `helm template`.

These shell out to helm (fails loudly if it is not installed) and assert on the
rendered manifests — no cluster involved.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[1]
CHART = REPO_ROOT / "charts" / "plumb"
VERIFIER_CONFIG = REPO_ROOT / "config" / "verifier.yaml"


def render(*set_args: str, values: dict | None = None) -> dict[str, list[dict]]:
    """helm template with optional overrides, returning manifests grouped by kind.

    Overrides go through --set, or through a values file (`values`) for types
    --set cannot express — helm parses floats on the CLI as strings.
    """
    cmd = ["helm", "template", "plumb", str(CHART)]
    for arg in set_args:
        cmd += ["--set", arg]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as override:
        if values is not None:
            yaml.safe_dump(values, override)
            override.flush()
            cmd += ["-f", override.name]
        completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise AssertionError(f"helm template failed:\n{completed.stderr}")
    manifests: dict[str, list[dict]] = {}
    for doc in yaml.safe_load_all(completed.stdout):
        if doc:
            manifests.setdefault(doc["kind"], []).append(doc)
    return manifests


def only(manifests: dict[str, list[dict]], kind: str) -> dict:
    assert kind in manifests, f"no {kind} rendered; got {sorted(manifests)}"
    assert len(manifests[kind]) == 1, f"expected exactly one {kind}"
    return manifests[kind][0]


@pytest.fixture(scope="module")
def default_render() -> dict[str, list[dict]]:
    return render()


def container(deployment: dict) -> dict:
    containers = deployment["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 1
    return containers[0]


class TestDeployment:
    def test_probes(self, default_render):
        c = container(only(default_render, "Deployment"))
        assert c["livenessProbe"]["httpGet"]["path"] == "/healthz"
        assert c["readinessProbe"]["httpGet"]["path"] == "/readyz"
        startup = c["startupProbe"]
        assert startup["httpGet"]["path"] == "/readyz"
        headroom = startup["failureThreshold"] * startup["periodSeconds"]
        assert headroom >= 300, "startupProbe must absorb the ~420 MB weight download"

    def test_resources_bounded(self, default_render):
        resources = container(only(default_render, "Deployment"))["resources"]
        assert resources["requests"], "resource requests must be set"
        assert resources["limits"], "resource limits must be set"

    def test_config_injected_via_plumb_config(self, default_render):
        deployment = only(default_render, "Deployment")
        c = container(deployment)
        env = {e["name"]: e["value"] for e in c["env"]}
        mounts = {m["mountPath"] for m in c["volumeMounts"]}
        assert any(env["PLUMB_CONFIG"].startswith(m) for m in mounts), (
            "PLUMB_CONFIG must point inside the config mount"
        )
        annotations = deployment["spec"]["template"]["metadata"]["annotations"]
        assert any("checksum" in key for key in annotations), (
            "config changes must roll pods via a checksum annotation"
        )


class TestService:
    def test_cluster_ip_on_api_port(self, default_render):
        service = only(default_render, "Service")
        assert service["spec"]["type"] == "ClusterIP"
        assert service["spec"]["ports"][0]["targetPort"] == 8000


class TestVerifierConfig:
    def test_default_render_equals_repo_config(self, default_render):
        configmap = only(default_render, "ConfigMap")
        (rendered,) = configmap["data"].values()
        assert yaml.safe_load(rendered) == yaml.safe_load(VERIFIER_CONFIG.read_text()), (
            "chart default verifier config drifted from config/verifier.yaml"
        )

    def test_threshold_override_lands(self):
        override = {"verifier": {"signals": {"groundedness": {"threshold": 0.7}}}}
        manifests = render(values=override)
        (rendered,) = only(manifests, "ConfigMap")["data"].values()
        assert yaml.safe_load(rendered)["signals"]["groundedness"]["threshold"] == 0.7


def egress_ports(policy: dict) -> set[int]:
    return {port["port"] for rule in policy["spec"]["egress"] for port in rule.get("ports", [])}


class TestNetworkPolicy:
    def test_default_deny_except_dns_and_weights(self, default_render):
        policy = only(default_render, "NetworkPolicy")
        assert policy["spec"]["policyTypes"] == ["Egress"]
        assert egress_ports(policy) == {53, 443}

    def test_sovereign_mode_is_dns_only(self):
        manifests = render("networkPolicy.allowModelDownload=false")
        assert egress_ports(only(manifests, "NetworkPolicy")) == {53}

    def test_disabled_renders_no_policy(self):
        manifests = render("networkPolicy.enabled=false")
        assert "NetworkPolicy" not in manifests
