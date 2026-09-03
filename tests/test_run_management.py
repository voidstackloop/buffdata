import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import json
from pathlib import Path
import time

import pytest

from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.models.formats import read_dataset, write_dataset
from buffdata.models.schemas import DatasetItem, PipelineConfig
from buffdata.runs.models import RunSpec, RunConflict
from buffdata.runs.runner import run_local
from buffdata.runs.service import RunService

CONFIG = dict(quality_mode="off", network_policy="strict", accuracy_contract="strict", scrub_pii=False)


@pytest.fixture
def manager(tmp_path):
    service = RunService(tmp_path / "managed")
    service.store.ensure_project("local", {"members": {"local": "administrator"}, "policy": {}})
    source = tmp_path / "source.jsonl"
    write_dataset([DatasetItem.from_dict({"id": str(i), "text": "identical sample " + str(i % 4), "label": i % 2})
                   for i in range(12)], source)
    dataset = service.register("local", source)
    return service, RunSpec(dataset_id=dataset["id"], configuration=CONFIG), source


def test_managed_execution_matches_unchanged_pipeline(manager, tmp_path):
    service, spec, source = manager
    baseline = asyncio.run(OptimizationPipeline(PipelineConfig(**CONFIG)).run_file(source, tmp_path / "baseline.jsonl"))
    run = service.submit(spec)
    result = run_local(service, "local", run["id"])
    assert result["status"] == "succeeded", result
    assert service.artifact("local", run["id"], "output").read_bytes() == (tmp_path / "baseline.jsonl").read_bytes()
    assert [x.to_dict() for x in read_dataset(service.artifact("local", run["id"], "output"))] == [x.to_dict() for x in baseline.accepted]
    assert service.verify("local", run["id"])["verified"]
    comparison = service.compare("local", run["id"])
    assert comparison["original"] == comparison["generated"]
    assert comparison["accuracy"] is None
    assert [e["kind"] for e in service.store.run_events("local", run["id"])] == ["run.queued", "run.started", "run.succeeded"]


def test_idempotency_and_exclusive_project_claim(manager):
    service, spec, _ = manager
    with ThreadPoolExecutor(4) as pool:
        ids = list(pool.map(lambda _: service.submit(spec, key="request")["id"], range(4)))
    assert len(set(ids)) == 1
    changed = spec.model_copy(update={"output_format": "csv"})
    with pytest.raises(RunConflict):
        service.submit(changed, key="request")
    service.submit(spec)
    with ThreadPoolExecutor(4) as pool:
        claimed = list(pool.map(lambda _: service.store.claim("local"), range(4)))
    assert sum(value is not None for value in claimed) == 1


def test_cancel_wins_publication_race(manager):
    service, spec, _ = manager
    run = service.submit(spec)
    claim = service.store.claim("local")
    service.store.cancel("local", run["id"], "local")
    result = service.store.finish("local", run["id"], claim["lease"], "succeeded", {"invalid": "manifest"})
    assert result["status"] == "cancelled" and result["manifest"] is None
    with pytest.raises(RunConflict):
        service.verify("local", run["id"])


def test_resume_fails_closed_on_environment_drift(manager, monkeypatch):
    service, spec, _ = manager
    run = service.submit(spec)
    service.store.cancel("local", run["id"], "local")
    monkeypatch.setattr("buffdata.runs.service.runtime_identity", lambda _: {"changed": True})
    with pytest.raises(RunConflict, match="identical"):
        service.resume("local", run["id"])


def test_snapshot_changes_are_detected(manager):
    service, spec, _ = manager
    path = service.dataset_path("local", spec.dataset_id)
    path.chmod(0o600)
    path.write_text("changed")
    with pytest.raises(RunConflict, match="integrity"):
        service.submit(spec)


def test_cancelled_run_can_resume_without_overwriting_attempt(manager):
    service, spec, _ = manager
    run = service.submit(spec)
    service.store.cancel("local", run["id"], "local")
    resumed = service.resume("local", run["id"])
    assert resumed["id"] != run["id"]
    assert resumed["parent_run_id"] == run["id"]
    assert service.store.get("local", run["id"])["status"] == "cancelled"
    assert run_local(service, "local", resumed["id"])["status"] == "succeeded"


def test_arbitrary_config_and_paths_rejected():
    with pytest.raises(ValueError):
        RunSpec(dataset_id="../../etc/passwd")
    with pytest.raises(ValueError):
        RunSpec(dataset_id="a" * 32, configuration={"command": "bash"})
    with pytest.raises(ValueError):
        RunSpec(dataset_id="a" * 32, command="bash")


def test_runtime_fingerprint_ignores_vendor_sys_path_injection(tmp_path, monkeypatch):
    from buffdata.runs.service import runtime_identity
    from buffdata.security.policy import SecurityPolicy
    baseline = runtime_identity(SecurityPolicy())
    info = tmp_path / "vendored_extra-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: vendored-extra\nVersion: 1.0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert runtime_identity(SecurityPolicy()) == baseline


def test_runtime_fingerprint_dotenv_path_is_explicit_not_ambient(monkeypatch):
    """Regression test for a live-testing find: python-dotenv's own find_dotenv() resolves its
    search root off how the *calling process itself* was invoked (a bare `python -c` snippet
    searches from the CWD; a real .py file or `-m` module searches from wherever the first
    real call frame lives) -- so relying on engine/client.py's ambient load_dotenv() made the
    provider_environment_sha256 component below nondeterministic across process kinds (a
    submission CLI script vs. an `-m` executor subprocess vs. a server under uvicorn), which
    could falsely trip "Worker implementation differs from submission" for a run whose
    environment never actually changed. Fixed by resolving the .env path explicitly off the
    installed package's own location instead of relying on ambient discovery."""
    import buffdata
    from buffdata.runs.service import runtime_identity
    from buffdata.security.policy import SecurityPolicy
    calls = []
    monkeypatch.setattr("dotenv.load_dotenv", lambda path: calls.append(path))
    runtime_identity(SecurityPolicy())
    assert calls == [Path(buffdata.__file__).parent.parent / ".env"]


def test_replayed_provider_requests_and_outputs_are_identical(manager, tmp_path, monkeypatch):
    from tests.test_adaptive_pipeline import PipelineFakeClient
    from buffdata.runs.executor import execute
    service, spec, source = manager
    configuration = dict(quality_mode="sampled", scrub_pii=False, provider="anthropic", model="fake-balanced")
    expected_client = PipelineFakeClient()
    baseline = asyncio.run(OptimizationPipeline(PipelineConfig(**configuration), client=expected_client)
                           .run_file(source, tmp_path / "replayed.jsonl"))
    run = service.submit(RunSpec(dataset_id=spec.dataset_id, configuration=configuration))
    claim = service.store.claim("local")
    directory, _ = service.prepare(claim)
    replay = PipelineFakeClient()
    monkeypatch.setattr("buffdata.engine.pipeline.create_llm_client", lambda **kwargs: replay)
    monkeypatch.setattr("buffdata.security.network.install_worker_network_guard", lambda context: None)
    execute(directory)
    assert replay.prompts == expected_client.prompts
    assert replay.prompts, "This must actually compare provider requests"
    assert (directory / "optimized.jsonl").read_bytes() == (tmp_path / "replayed.jsonl").read_bytes()


def test_disk_failure_never_publishes(manager, monkeypatch):
    service, spec, _ = manager
    run = service.submit(spec)
    def full(*args):
        raise OSError("No space left on device")
    monkeypatch.setattr("buffdata.runs.service.private_json", full)
    with pytest.raises(OSError):
        run_local(service, "local", run["id"])
    assert service.store.get("local", run["id"])["manifest"] is None
    assert service.store.get("local", run["id"])["status"] == "failed"


def test_corrupt_checkpoint_fails_without_publication(manager):
    service, spec, _ = manager
    run = service.submit(spec)
    claim = service.store.claim("local")
    directory, _ = service.prepare(claim)
    (directory / ".optimized.checkpoint.json").write_text("not-json")
    service.store.finish("local", run["id"], claim["lease"], "failed")
    resumed = service.resume("local", run["id"])
    result = run_local(service, "local", resumed["id"])
    assert result["status"] == "failed" and result["manifest"] is None


def test_explicit_parent_run_lineage_walks_both_directions(manager):
    service, spec, _ = manager
    first = service.submit(spec)
    assert run_local(service, "local", first["id"])["status"] == "succeeded"
    second = service.submit(spec, parent=first["id"])
    assert second["parent_run_id"] == first["id"]
    assert run_local(service, "local", second["id"])["status"] == "succeeded"

    forward = service.lineage("local", second["id"])
    assert [a["run_id"] for a in forward["ancestors"]] == [first["id"]]
    assert forward["descendants"] == []

    backward = service.lineage("local", first["id"])
    assert backward["ancestors"] == []
    assert [d["run_id"] for d in backward["descendants"]] == [second["id"]]


def test_submit_rejects_unknown_parent_run(manager):
    service, spec, _ = manager
    with pytest.raises(KeyError):
        service.submit(spec, parent="a" * 32)


def test_resume_lineage_is_queryable_through_the_same_api(manager):
    service, spec, _ = manager
    run = service.submit(spec)
    service.store.cancel("local", run["id"], "local")
    resumed = service.resume("local", run["id"])
    lineage = service.lineage("local", resumed["id"])
    assert [a["run_id"] for a in lineage["ancestors"]] == [run["id"]]


def test_webhook_registration_validates_url_and_events(manager):
    service, spec, _ = manager
    with pytest.raises(ValueError, match="HTTPS"):
        service.register_webhook("local", "http://93.184.216.34/hook", ["run.succeeded"], "local")
    with pytest.raises(ValueError, match="event"):
        service.register_webhook("local", "https://93.184.216.34/hook", ["not.a.real.event"], "local")
    from buffdata.security.policy import SecurityError
    with pytest.raises(SecurityError):
        service.register_webhook("local", "https://127.0.0.1/hook", ["run.succeeded"], "local")


def test_webhook_delivers_signed_payload_on_completion_and_masks_secret(manager, monkeypatch):
    service, spec, _ = manager
    hook = service.register_webhook("local", "https://93.184.216.34/hook", ["run.succeeded", "run.failed"], "local")
    assert "secret" in hook
    listed = service.list_webhooks("local")
    assert listed == [{k: v for k, v in hook.items() if k != "secret"}]

    calls = []

    class FakeResponse:
        status_code = 200

    monkeypatch.setattr(service.__class__, "_send_webhook",
        staticmethod(lambda url, body, headers: calls.append((url, body, headers)) or FakeResponse()))
    run = service.submit(spec)
    result = run_local(service, "local", run["id"])
    assert result["status"] == "succeeded"

    assert len(calls) == 1
    url, body, headers = calls[0]
    assert url == hook["url"]
    payload = json.loads(body)
    assert payload["event"] == "run.succeeded" and payload["run_id"] == run["id"] and payload["project_id"] == "local"
    expected_signature = "sha256=" + hmac.new(hook["secret"].encode(), body, hashlib.sha256).hexdigest()
    assert headers["X-BuffData-Signature"] == expected_signature

    kinds = [e["kind"] for e in service.store.run_events("local", run["id"])]
    assert "webhook.delivered" in kinds


def test_webhook_delivery_failure_retries_then_records_event_without_failing_the_run(manager, monkeypatch):
    service, spec, _ = manager
    hook = service.register_webhook("local", "https://93.184.216.34/hook", ["run.succeeded"], "local")
    monkeypatch.setattr("buffdata.runs.service.time.sleep", lambda seconds: None)
    calls = []

    def always_fails(url, body, headers):
        calls.append(1)
        raise ConnectionError("refused")

    monkeypatch.setattr(service.__class__, "_send_webhook", staticmethod(always_fails))
    run = service.submit(spec)
    result = run_local(service, "local", run["id"])
    assert result["status"] == "succeeded"
    assert len(calls) == 3
    events = service.store.run_events("local", run["id"])
    failure = next(e for e in events if e["kind"] == "webhook.delivery_failed")
    assert failure["webhook_id"] == hook["id"] and failure["error"]


def test_deleted_webhook_receives_no_further_deliveries(manager, monkeypatch):
    service, spec, _ = manager
    hook = service.register_webhook("local", "https://93.184.216.34/hook", ["run.succeeded"], "local")
    calls = []
    monkeypatch.setattr(service.__class__, "_send_webhook",
        staticmethod(lambda url, body, headers: calls.append(1) or type("R", (), {"status_code": 200})()))
    service.delete_webhook("local", hook["id"], "local")
    run = service.submit(spec)
    run_local(service, "local", run["id"])
    assert calls == []


def test_stale_recovery_rechecks_heartbeat_under_lock(manager, monkeypatch):
    from buffdata.runs.store import runs
    service, spec, _ = manager
    run = service.submit(spec)
    claim = service.store.claim("local")
    with service.store.transaction() as conn:
        conn.execute(runs.update().where(runs.c.id == run["id"]).values(heartbeat=time.time() - 180))
    finish = service.store.finish
    def renewed(*args, **kwargs):
        service.store.heartbeat("local", run["id"], claim["lease"])
        return finish(*args, **kwargs)
    monkeypatch.setattr(service.store, "finish", renewed)
    with pytest.raises(RunConflict, match="renewed"):
        service.store.interrupt_stale("local", run["id"])
    assert service.store.get("local", run["id"])["status"] == "running"
