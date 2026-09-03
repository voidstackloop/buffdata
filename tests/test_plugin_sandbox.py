import os
import shutil
import time
from pathlib import Path

import pytest

from buffdata import plugins
from buffdata.models.schemas import DatasetItem
from buffdata.security import sandbox as sandbox_module
from buffdata.security.policy import ExecutionContext, SecurityError, SecurityPolicy, execution_context
from buffdata.security.sandbox import PluginSandbox, build_sandboxed_recognizer


@pytest.fixture(scope="module")
def fixture_plugin_path(tmp_path_factory):
    """A real, hand-built .dist-info next to the diagnostic plugin module, so a real spawned
    subprocess's own importlib.metadata.entry_points() lookup finds it -- no pip install, and
    nothing added to the shared dev venv (which would otherwise break every other test that
    doesn't approve this plugin, since an installed-but-unapproved plugin fails closed)."""
    root = tmp_path_factory.mktemp("diagnostic-plugin-dist")
    shutil.copy(Path(__file__).parent / "fixtures" / "diagnostic_plugin.py", root / "diagnostic_plugin.py")
    dist_info = root / "buffdata_diagnostic_plugin-0.0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: buffdata-diagnostic-plugin\nVersion: 0.0.1\n")
    (dist_info / "entry_points.txt").write_text(
        "[buffdata.validators]\n"
        "diagnostic = diagnostic_plugin:diagnostic_validator\n\n"
        "[buffdata.pii_recognizers]\n"
        "diagnostic = diagnostic_plugin:build_diagnostic_recognizer\n")
    return root


@pytest.fixture(autouse=True)
def _reset_plugin_singletons():
    """plugins._sandbox is a process-global singleton; leaking one (possibly poisoned) across
    tests would make failures depend on test order."""
    yield
    if plugins._sandbox is not None:
        plugins._sandbox.close()
    plugins._sandbox = None
    plugins._SANDBOX_ACTIVE = False
    plugins._validator_cache = plugins._pii_recognizer_cache = plugins._approval_key = None


def item(text="ordinary text"):
    return DatasetItem.from_dict({"id": "1", "text": text})


def make_sandbox(fixture_plugin_path, *, groups=("buffdata.validators:diagnostic",), subprocess_seconds=120):
    policy = SecurityPolicy(approved_plugins=list(groups), subprocess_seconds=subprocess_seconds)
    return PluginSandbox(policy, extra_env={"PYTHONPATH": str(fixture_plugin_path)})


# --- Direct PluginSandbox tests: the production env allowlist is untouched here, proving the
# real default (not a widened test-only one) actually isolates the child. ---

def test_separate_process_and_normal_result(fixture_plugin_path):
    with make_sandbox(fixture_plugin_path) as sandbox:
        response = sandbox.call("validate", {"item": item("nothing special").to_dict()})
        assert response == {"errors": []}
        assert sandbox._process.pid != os.getpid()
        assert sandbox._process.poll() is None


def test_plugin_raise_becomes_security_error(fixture_plugin_path):
    with make_sandbox(fixture_plugin_path) as sandbox:
        with pytest.raises(SecurityError, match="Approved validator plugin failed"):
            sandbox.call("validate", {"item": item("DIAG:RAISE").to_dict()})


def test_secret_env_var_is_absent_from_the_child(fixture_plugin_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-cross-the-boundary")
    with make_sandbox(fixture_plugin_path) as sandbox:
        response = sandbox.call("validate", {"item": item("DIAG:ENV").to_dict()})
        seen = response["errors"][0]
        assert "ANTHROPIC_API_KEY" not in seen
        assert "PATH" in seen  # the allowlist does let ordinary interpreter plumbing through


def test_network_is_blocked_inside_the_sandbox(fixture_plugin_path):
    with make_sandbox(fixture_plugin_path) as sandbox:
        response = sandbox.call("validate", {"item": item("DIAG:NETWORK").to_dict()})
        assert response["errors"][0].startswith("network blocked:")


def test_call_times_out_and_kills_the_child(fixture_plugin_path):
    with make_sandbox(fixture_plugin_path, subprocess_seconds=1) as sandbox:
        with pytest.raises(SecurityError, match="timed out"):
            sandbox.call("validate", {"item": item("DIAG:SLEEP").to_dict()})
        process = sandbox._process
    assert process is None  # poisoned: _poison_locked() already cleared it
    with pytest.raises(SecurityError, match="unavailable"):
        sandbox.call("validate", {"item": item().to_dict()})


def test_crash_fails_closed_without_auto_respawn(fixture_plugin_path):
    with make_sandbox(fixture_plugin_path) as sandbox:
        with pytest.raises(SecurityError):
            sandbox.call("validate", {"item": item("DIAG:CRASH").to_dict()})
        with pytest.raises(SecurityError, match="unavailable"):
            sandbox.call("validate", {"item": item().to_dict()})
    # A fresh instance is unaffected -- poison state is per-instance, not global.
    with make_sandbox(fixture_plugin_path) as fresh:
        assert fresh.call("validate", {"item": item().to_dict()}) == {"errors": []}


@pytest.mark.parametrize("format_kwargs", [
    {"text": "raw format item"},
    {"instruction": "do the thing", "output": "done"},
    {"messages": [{"role": "user", "content": "hi"}]},
    {"chosen": "a", "rejected": "b"},
])
def test_dataset_item_round_trips_every_format_through_the_boundary(fixture_plugin_path, format_kwargs):
    original = DatasetItem.from_dict({"id": "1", **format_kwargs})
    with make_sandbox(fixture_plugin_path) as sandbox:
        response = sandbox.call("validate", {"item": original.to_dict()})
    assert response == {"errors": []}  # the plugin ran at all, against a reconstructed item

    # And the same item round-tripped through JSON produces the same classification text the
    # in-process plugin would have seen.
    import json
    reconstructed = DatasetItem.from_dict(json.loads(json.dumps(original.to_dict())))
    assert reconstructed.get_classification_text() == original.get_classification_text()
    assert reconstructed.format == original.format


# --- python_executable/worker_argv generalization (used by the presidio-anonymizer sandbox) ---

def test_bad_python_executable_poisons_on_first_call_not_every_call():
    sandbox = PluginSandbox(SecurityPolicy(approved_plugins=[]), python_executable="/no/such/interpreter")
    with pytest.raises(SecurityError):
        sandbox.call("validate", {"item": {"id": "1", "text": "x"}})
    assert sandbox._poisoned
    # A second call must reuse the poisoned state (raise immediately), not re-attempt the
    # same broken spawn -- the whole point of moving _start() inside the poison-wrapping try.
    with pytest.raises(SecurityError, match="unavailable"):
        sandbox.call("validate", {"item": {"id": "1", "text": "x"}})


def test_worker_argv_overrides_the_default_module_invocation(fixture_plugin_path, tmp_path):
    # Prove worker_argv is actually honored: point it at a trivial standalone script instead
    # of buffdata.security.sandbox_worker, and confirm that script -- not the real worker --
    # is what runs.
    script = tmp_path / "echo_worker.py"
    script.write_text(
        "import json, struct, sys\n"
        "def read_frame(s):\n"
        "    h = s.read(4)\n"
        "    n = struct.unpack('>I', h)[0]\n"
        "    return json.loads(s.read(n))\n"
        "def write_frame(s, obj):\n"
        "    b = json.dumps(obj).encode()\n"
        "    s.write(struct.pack('>I', len(b)))\n"
        "    s.write(b)\n"
        "    s.flush()\n"
        "read_frame(sys.stdin.buffer)\n"  # init handshake, discarded
        "while True:\n"
        "    try:\n"
        "        req = read_frame(sys.stdin.buffer)\n"
        "    except Exception:\n"
        "        break\n"
        "    write_frame(sys.stdout.buffer, {'id': req['id'], 'ok': True, 'result': {'echo': req['kind']}})\n"
    )
    import sys
    sandbox = PluginSandbox(SecurityPolicy(), python_executable=sys.executable, worker_argv=[str(script)])
    assert sandbox.call("anything", {}) == {"echo": "anything"}


# --- PII recognizer construction/analysis proxy ---

def test_pii_recognizer_construction_and_analyze_round_trip(fixture_plugin_path):
    with make_sandbox(fixture_plugin_path, groups=["buffdata.pii_recognizers:diagnostic"]) as sandbox:
        response = sandbox.call("pii_construct", {})
        [metadata] = response["recognizers"]
        assert metadata["name"] == "diagnostic-pii"
        assert metadata["supported_entities"] == ["DIAGNOSTIC"]

        proxy = build_sandboxed_recognizer(sandbox, metadata)
        assert proxy.name == "diagnostic-pii"
        results = proxy.analyze("this text has a DIAG-HIT marker in it", ["DIAGNOSTIC"], nlp_artifacts=None)
        assert len(results) == 1 and results[0].entity_type == "DIAGNOSTIC"


# --- Routing through the real buffdata.plugins API ---

def test_run_validator_plugins_routes_through_the_sandbox_when_approved(fixture_plugin_path, monkeypatch):
    monkeypatch.setattr(sandbox_module, "_ENV_ALLOWLIST", sandbox_module._ENV_ALLOWLIST | {"PYTHONPATH"})
    monkeypatch.setenv("PYTHONPATH", str(fixture_plugin_path))
    context = ExecutionContext(policy=SecurityPolicy(approved_plugins=["buffdata.validators:diagnostic"]))
    with execution_context(context):
        assert plugins.run_validator_plugins(item("nothing special")) == []
        assert plugins._sandbox is not None  # actually went through the sandbox, not a stub
        with pytest.raises(SecurityError):
            plugins.run_validator_plugins(item("DIAG:RAISE"))


def test_zero_plugins_never_constructs_a_sandbox():
    with execution_context(ExecutionContext(policy=SecurityPolicy())):
        assert plugins.run_validator_plugins(item()) == []
    assert plugins._sandbox is None


def test_recursion_guard_runs_the_real_loader_inside_the_worker(monkeypatch):
    # Simulates being inside sandbox_worker.py: _SANDBOX_ACTIVE=True must make the real
    # loader run directly rather than trying to spawn a sandbox of its own.
    plugins._SANDBOX_ACTIVE = True

    def explode(*args, **kwargs):
        raise AssertionError("must not construct a sandbox while _SANDBOX_ACTIVE is True")

    monkeypatch.setattr(plugins, "_get_sandbox", explode)
    with execution_context(ExecutionContext(policy=SecurityPolicy(approved_plugins=["buffdata.validators:diagnostic"]))):
        # No plugin actually installed in-process here; the point is only that it never
        # reaches _get_sandbox -- it should fail closed on the missing plugin instead.
        with pytest.raises(SecurityError, match="missing"):
            plugins.run_validator_plugins(item())
