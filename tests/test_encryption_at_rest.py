import base64
import os

import pytest

from buffdata.models.formats import write_dataset
from buffdata.models.schemas import DatasetItem
from buffdata.runs.models import RunSpec
from buffdata.runs.runner import run_local
from buffdata.runs.service import RunService
from buffdata.security.keys import LocalKeyManager, create_key_manager, decrypt_bytes, encrypt_file_in_place, is_encrypted

CONFIG = dict(quality_mode="off", network_policy="strict", accuracy_contract="strict", scrub_pii=False)


# --- LocalKeyManager / create_key_manager ------------------------------------------------

def test_local_key_manager_wrap_unwrap_round_trip():
    manager = LocalKeyManager(os.urandom(32))
    data_key = os.urandom(32)
    wrapped = manager.wrap(data_key)
    assert wrapped != data_key
    assert manager.unwrap(wrapped) == data_key


def test_wrong_length_master_key_fails_at_construction_not_lazily():
    with pytest.raises(ValueError, match="32 bytes"):
        LocalKeyManager(b"too-short")


def test_create_key_manager_makes_zero_resolver_calls_when_unconfigured(monkeypatch):
    monkeypatch.delenv("BUFFDATA_ARTIFACT_MASTER_KEY_SECRET", raising=False)

    def explode(*args, **kwargs):
        raise AssertionError("must not touch the secret resolver when the feature is off")
    monkeypatch.setattr("buffdata.engine.secrets.get_default_secret_resolver", explode)

    assert create_key_manager() is None


def test_create_key_manager_resolves_and_validates_when_configured(monkeypatch):
    key = base64.b64encode(os.urandom(32)).decode()
    monkeypatch.setenv("BUFFDATA_ARTIFACT_MASTER_KEY_SECRET", "TEST_MASTER_KEY")
    monkeypatch.setenv("TEST_MASTER_KEY", key)
    manager = create_key_manager()
    assert isinstance(manager, LocalKeyManager)


# --- file envelope -------------------------------------------------------------------------

def test_encrypt_file_in_place_round_trips(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_bytes(b'{"id":"1","text":"hello"}\n')
    data_key = os.urandom(32)
    encrypt_file_in_place(path, data_key)
    raw = path.read_bytes()
    assert is_encrypted(raw)
    assert is_encrypted(path)
    assert decrypt_bytes(raw, data_key) == b'{"id":"1","text":"hello"}\n'
    # Wrong key fails loudly (AES-GCM auth tag mismatch), not silently.
    with pytest.raises(Exception):
        decrypt_bytes(raw, os.urandom(32))


def test_is_encrypted_false_for_plain_files(tmp_path):
    path = tmp_path / "plain.jsonl"
    path.write_bytes(b'{"id":"1"}\n')
    assert not is_encrypted(path)
    assert not is_encrypted(path.read_bytes())


# --- get_or_create_data_key -----------------------------------------------------------------

@pytest.fixture
def keyed_manager(tmp_path):
    service = RunService(tmp_path / "managed")
    service.key_manager = LocalKeyManager(os.urandom(32))
    service.store.ensure_project("local", {"members": {"local": "administrator"}, "policy": {}})
    return service


def test_get_or_create_data_key_is_stable_across_calls(keyed_manager):
    first = keyed_manager.store.get_or_create_data_key("local", keyed_manager.key_manager)
    second = keyed_manager.store.get_or_create_data_key("local", keyed_manager.key_manager)
    assert first == second


def test_concurrent_data_key_creation_never_produces_two_keys(keyed_manager):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        keys = list(pool.map(lambda _: keyed_manager.store.get_or_create_data_key(
            "local", keyed_manager.key_manager), range(8)))
    assert len(set(keys)) == 1


def test_concurrent_project_update_does_not_disturb_the_stored_key(keyed_manager):
    """Regression test for the race the design review caught: wrapped_data_key lives in its
    own table, immune to update_project()'s unlocked read-modify-write on managed_projects.body."""
    key_before = keyed_manager.store.get_or_create_data_key("local", keyed_manager.key_manager)
    body = keyed_manager.store.project("local")
    body["policy"] = {"execution_seconds": 999}
    keyed_manager.store.update_project("local", {k: v for k, v in body.items() if k != "id"}, "admin")
    key_after = keyed_manager.store.get_or_create_data_key("local", keyed_manager.key_manager)
    assert key_before == key_after
    assert keyed_manager.store.project("local")["policy"]["execution_seconds"] == 999


# --- end-to-end managed run -------------------------------------------------------------

@pytest.fixture
def encrypted_manager(tmp_path):
    service = RunService(tmp_path / "managed")
    service.key_manager = LocalKeyManager(os.urandom(32))
    service.store.ensure_project("local", {"members": {"local": "administrator"}, "policy": {}})
    source = tmp_path / "source.jsonl"
    write_dataset([DatasetItem.from_dict({"id": str(i), "text": "identical sample " + str(i % 4), "label": i % 2})
                   for i in range(12)], source)
    dataset = service.register("local", source)
    return service, RunSpec(dataset_id=dataset["id"], configuration=CONFIG)


def test_managed_run_encrypts_output_rejected_report(encrypted_manager):
    service, spec = encrypted_manager
    run = service.submit(spec)
    result = run_local(service, "local", run["id"])
    assert result["status"] == "succeeded", result

    for name in ("output", "rejected", "report"):
        path = service.artifact("local", run["id"], name)
        assert is_encrypted(path), f"{name} was not encrypted at rest"

    # verify()/manifest hashing needs no awareness of encryption -- it already just hashes
    # whatever bytes are on disk.
    assert service.verify("local", run["id"])["verified"]


def test_compare_and_artifact_bytes_transparently_decrypt(encrypted_manager):
    service, spec = encrypted_manager
    run = service.submit(spec)
    run_local(service, "local", run["id"])

    comparison = service.compare("local", run["id"])
    assert comparison["original"] == comparison["generated"]

    content = service.artifact_bytes("local", run["id"], "output")
    assert b'"id"' in content  # real plaintext dataset content, not ciphertext


def test_repeated_compare_leaves_no_plaintext_files_behind(encrypted_manager):
    """Regression test for the design review's tempfile-leak finding: compare() must not
    accumulate plaintext copies under the run directory across repeated calls."""
    service, spec = encrypted_manager
    run = service.submit(spec)
    run_local(service, "local", run["id"])
    directory = service.run_directory("local", run["id"])
    before = sorted(p.relative_to(directory) for p in directory.rglob("*") if p.is_file())

    for _ in range(10):
        service.compare("local", run["id"])

    after = sorted(p.relative_to(directory) for p in directory.rglob("*") if p.is_file())
    assert before == after


def test_data_key_never_persisted_to_disk(encrypted_manager):
    """Regression test for the design review's persisted-plaintext-key finding: the raw DEK
    must never appear anywhere under the run directory after a run completes."""
    service, spec = encrypted_manager
    run = service.submit(spec)
    run_local(service, "local", run["id"])

    data_key = service.store.get_or_create_data_key("local", service.key_manager)
    directory = service.run_directory("local", run["id"])
    for path in directory.rglob("*"):
        if path.is_file():
            assert data_key not in path.read_bytes(), f"raw data key found in {path}"


def test_unconfigured_run_is_completely_unaffected(tmp_path):
    """No key manager at all -- byte-for-byte the same behavior as before this feature existed."""
    service = RunService(tmp_path / "managed")
    assert service.key_manager is None
    service.store.ensure_project("local", {"members": {"local": "administrator"}, "policy": {}})
    source = tmp_path / "source.jsonl"
    write_dataset([DatasetItem.from_dict({"id": "1", "text": "a plain unencrypted row", "label": 0})], source)
    dataset = service.register("local", source)
    run = service.submit(RunSpec(dataset_id=dataset["id"], configuration=CONFIG))
    result = run_local(service, "local", run["id"])
    assert result["status"] == "succeeded"
    for name in ("output", "rejected", "report"):
        assert not is_encrypted(service.artifact("local", run["id"], name))
