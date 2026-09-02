import pytest
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.engine.secrets import EnvSecretResolver, KeyringSecretResolver

keyring = pytest.importorskip("keyring")
from keyrings.alt.file import PlaintextKeyring  # noqa: E402


@pytest.fixture
def isolated_keyring(tmp_path, monkeypatch):
    # A real keyring backend (not the OS-native one, which isn't available in a headless
    # CI/dev environment) -- genuine set/get/delete round trips against a real file, just
    # not the platform-native credential store this uses in production. Isolated to a
    # tmp_path file so tests never touch the real dev machine's keyring.
    backend = PlaintextKeyring()
    backend.file_path = str(tmp_path / "keyring_test.cfg")
    previous = keyring.get_keyring()
    keyring.set_keyring(backend)
    monkeypatch.setenv("BUFFDATA_KEYRING_SERVICE", "buffdata-test")
    try:
        yield backend
    finally:
        keyring.set_keyring(previous)


# --- KeyringSecretResolver ------------------------------------------------------------

def test_keyring_resolver_round_trips_a_real_stored_value(isolated_keyring):
    keyring.set_password("buffdata-test", "GEMINI_API_KEY", "sk-real-value")
    resolver = KeyringSecretResolver(service_name="buffdata-test")

    assert resolver.get("GEMINI_API_KEY") == "sk-real-value"


def test_keyring_resolver_returns_none_for_unset_key(isolated_keyring):
    resolver = KeyringSecretResolver(service_name="buffdata-test")
    assert resolver.get("NEVER_SET_KEY") is None


def test_keyring_resolver_degrades_to_none_when_keyring_raises(monkeypatch):
    # KeyringSecretResolver.get() imports keyring lazily (inside the method), so it always
    # gets the module object already in sys.modules -- patching that object's function
    # directly is what a real "the OS backend is broken" failure looks like from here.
    def _broken_get_password(service, key):
        raise RuntimeError("no D-Bus session available")

    monkeypatch.setattr(keyring, "get_password", _broken_get_password)

    resolver = KeyringSecretResolver(service_name="buffdata-test")
    assert resolver.get("GEMINI_API_KEY") is None  # never raises


def test_keyring_resolver_uses_env_override_for_service_name(monkeypatch):
    monkeypatch.setenv("BUFFDATA_KEYRING_SERVICE", "custom-service")
    resolver = KeyringSecretResolver()
    assert resolver.service_name == "custom-service"


# --- EnvSecretResolver: env wins, keyring is the fallback ------------------------------

def test_env_resolver_prefers_environment_variable_over_keyring(isolated_keyring, monkeypatch):
    keyring.set_password("buffdata-test", "GEMINI_API_KEY", "from-keyring")
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")

    assert EnvSecretResolver().get("GEMINI_API_KEY") == "from-env"


def test_env_resolver_falls_back_to_keyring_when_env_unset(isolated_keyring, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    keyring.set_password("buffdata-test", "GEMINI_API_KEY", "from-keyring")

    assert EnvSecretResolver().get("GEMINI_API_KEY") == "from-keyring"


def test_env_resolver_returns_none_when_neither_is_set(isolated_keyring, monkeypatch):
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    assert EnvSecretResolver().get("SOME_UNSET_KEY") is None


# --- CLI: buffdata auth set / remove / status -------------------------------------------

def test_cli_auth_set_stores_a_real_value_in_the_keyring(isolated_keyring):
    result = CliRunner().invoke(app, ["auth", "set", "GEMINI_API_KEY"], input="sk-typed-value\nsk-typed-value\n")

    assert result.exit_code == 0, result.output
    assert "Stored GEMINI_API_KEY" in result.output
    # The typed value never appears in the command's own output (only a masked prompt).
    assert "sk-typed-value" not in result.output
    assert keyring.get_password("buffdata-test", "GEMINI_API_KEY") == "sk-typed-value"


def test_cli_auth_set_requires_matching_confirmation(isolated_keyring):
    result = CliRunner().invoke(
        app, ["auth", "set", "GEMINI_API_KEY"], input="sk-one\nsk-different\n"
    )
    assert result.exit_code != 0
    assert keyring.get_password("buffdata-test", "GEMINI_API_KEY") is None


def test_cli_auth_remove_deletes_a_stored_value(isolated_keyring):
    keyring.set_password("buffdata-test", "GEMINI_API_KEY", "sk-value")

    result = CliRunner().invoke(app, ["auth", "remove", "GEMINI_API_KEY"])

    assert result.exit_code == 0, result.output
    assert "Removed GEMINI_API_KEY" in result.output
    assert keyring.get_password("buffdata-test", "GEMINI_API_KEY") is None


def test_cli_auth_remove_reports_cleanly_when_nothing_was_set(isolated_keyring):
    result = CliRunner().invoke(app, ["auth", "remove", "NEVER_SET_KEY"])
    assert result.exit_code == 0, result.output
    assert "was not set" in result.output


def test_cli_auth_status_shows_source_without_ever_printing_the_value(isolated_keyring, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    keyring.set_password("buffdata-test", "GEMINI_API_KEY", "sk-gemini-secret-value")

    result = CliRunner().invoke(app, ["auth", "status"])

    assert result.exit_code == 0, result.output
    assert "sk-ant-secret-value" not in result.output
    assert "sk-gemini-secret-value" not in result.output
    assert "GEMINI_API_KEY" in result.output
    assert "ANTHROPIC_API_KEY" in result.output
    assert "OPENAI_API_KEY" in result.output


# --- End-to-end: a key stored via `buffdata auth set` is picked up by client construction

def test_key_stored_via_cli_auth_set_is_resolved_by_create_llm_client(isolated_keyring, monkeypatch):
    from buffdata.engine.client import GeminiClient, create_llm_client
    from buffdata.engine.secrets import set_default_secret_resolver

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    set_default_secret_resolver(None)  # force EnvSecretResolver to be rebuilt fresh

    set_result = CliRunner().invoke(app, ["auth", "set", "GEMINI_API_KEY"], input="sk-e2e-value\nsk-e2e-value\n")
    assert set_result.exit_code == 0, set_result.output

    client = create_llm_client("gemini", model="gemini-3.7-flash")
    assert isinstance(client, GeminiClient)
    assert client.api_key == "sk-e2e-value"

    set_default_secret_resolver(None)  # don't leak this resolver into later tests
