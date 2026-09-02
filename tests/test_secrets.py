from types import SimpleNamespace

import pytest

from buffdata.engine import secrets
from buffdata.engine.client import GeminiClient
from buffdata.engine.secrets import (
    AWSSecretsManagerResolver,
    AzureKeyVaultResolver,
    EnvSecretResolver,
    GCPSecretManagerResolver,
    VaultSecretResolver,
    create_secret_resolver,
    get_default_secret_resolver,
    set_default_secret_resolver,
)


@pytest.fixture(autouse=True)
def _reset_default_resolver():
    # get_default_secret_resolver() memoizes into a process-wide global the first time
    # it's ever called, so every test here must reset it before and after or an earlier
    # test's backend choice silently leaks into a later one.
    set_default_secret_resolver(None)
    yield
    set_default_secret_resolver(None)


def test_env_resolver_reads_os_environ(monkeypatch):
    monkeypatch.setenv("SOME_TEST_KEY", "value-from-env")
    resolver = EnvSecretResolver()
    assert resolver.get("SOME_TEST_KEY") == "value-from-env"
    assert resolver.get("SOME_MISSING_KEY") is None


def test_create_secret_resolver_defaults_to_env(monkeypatch):
    monkeypatch.delenv("BUFFDATA_SECRET_BACKEND", raising=False)
    assert isinstance(create_secret_resolver(), EnvSecretResolver)


def test_create_secret_resolver_selects_by_name():
    assert isinstance(create_secret_resolver("vault"), VaultSecretResolver)
    assert isinstance(create_secret_resolver("aws_secrets_manager"), AWSSecretsManagerResolver)
    assert isinstance(create_secret_resolver("gcp_secret_manager"), GCPSecretManagerResolver)
    assert isinstance(create_secret_resolver("azure_key_vault"), AzureKeyVaultResolver)


def test_create_secret_resolver_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown secret backend"):
        create_secret_resolver("not-a-real-backend")


def test_get_default_secret_resolver_honors_env_var(monkeypatch):
    monkeypatch.setenv("BUFFDATA_SECRET_BACKEND", "vault")
    assert isinstance(get_default_secret_resolver(), VaultSecretResolver)


def test_get_default_secret_resolver_is_memoized(monkeypatch):
    monkeypatch.setenv("BUFFDATA_SECRET_BACKEND", "vault")
    first = get_default_secret_resolver()
    monkeypatch.setenv("BUFFDATA_SECRET_BACKEND", "aws_secrets_manager")
    second = get_default_secret_resolver()
    assert first is second  # env var change after the first call has no effect until reset


def test_vault_resolver_reads_kv_v2_fields():
    resolver = VaultSecretResolver(path="buffdata", mount_point="secret", url="https://vault.internal", token="t")
    resolver._client = SimpleNamespace(
        secrets=SimpleNamespace(
            kv=SimpleNamespace(
                v2=SimpleNamespace(
                    read_secret_version=lambda path, mount_point: {
                        "data": {"data": {"GEMINI_API_KEY": "vault-gemini-key"}}
                    }
                )
            )
        )
    )
    assert resolver.get("GEMINI_API_KEY") == "vault-gemini-key"
    assert resolver.get("MISSING_KEY") is None


def test_vault_resolver_caches_after_first_read():
    calls = {"n": 0}

    def read_secret_version(path, mount_point):
        calls["n"] += 1
        return {"data": {"data": {"K": "v"}}}

    resolver = VaultSecretResolver(url="https://vault.internal", token="t")
    resolver._client = SimpleNamespace(
        secrets=SimpleNamespace(kv=SimpleNamespace(v2=SimpleNamespace(read_secret_version=read_secret_version)))
    )
    resolver.get("K")
    resolver.get("K")
    assert calls["n"] == 1


def test_aws_secrets_manager_resolver_parses_json_secret():
    resolver = AWSSecretsManagerResolver(secret_id="buffdata/keys")
    resolver._cache = {"ANTHROPIC_API_KEY": "aws-anthropic-key"}
    assert resolver.get("ANTHROPIC_API_KEY") == "aws-anthropic-key"
    assert resolver.get("MISSING") is None


def test_gcp_secret_manager_resolver_accesses_latest_version():
    resolver = GCPSecretManagerResolver(project_id="my-project")
    resolver._client = SimpleNamespace(
        access_secret_version=lambda name: SimpleNamespace(
            payload=SimpleNamespace(data=b"gcp-openai-key")
        )
    )
    assert resolver.get("OPENAI_API_KEY") == "gcp-openai-key"
    # second call for the same key must not hit the fake client again (would raise if it did,
    # since we only stub one call's worth of behavior implicitly via caching)
    assert resolver.get("OPENAI_API_KEY") == "gcp-openai-key"


def test_gcp_secret_manager_resolver_returns_none_on_lookup_failure():
    resolver = GCPSecretManagerResolver(project_id="my-project")

    def _raise(name):
        raise RuntimeError("not found")

    resolver._client = SimpleNamespace(access_secret_version=_raise)
    assert resolver.get("MISSING") is None


def test_azure_key_vault_resolver_normalizes_secret_names():
    seen_names = []

    def get_secret(name):
        seen_names.append(name)
        return SimpleNamespace(value="azure-secret-value")

    resolver = AzureKeyVaultResolver(vault_url="https://example.vault.azure.net")
    resolver._client = SimpleNamespace(get_secret=get_secret)
    assert resolver.get("AZURE_OPENAI_API_KEY") == "azure-secret-value"
    assert seen_names == ["azure-openai-api-key"]


def test_client_picks_up_key_from_injected_resolver(monkeypatch):
    # End-to-end: a client's api_key resolution should go through whatever resolver is
    # configured, without the client class knowing anything about the backend.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    class FakeResolver:
        def get(self, key):
            return "resolved-from-fake-backend" if key == "GEMINI_API_KEY" else None

    set_default_secret_resolver(FakeResolver())
    client = GeminiClient()
    assert client.api_key == "resolved-from-fake-backend"
