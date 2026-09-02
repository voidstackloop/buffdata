"""Pluggable secret resolution for API keys and credentials.

Every LLM client in engine/client.py resolves its API key through whatever
SecretResolver is configured, instead of calling os.getenv directly. The
default (EnvSecretResolver) makes this a no-op for every existing setup --
nothing changes until BUFFDATA_SECRET_BACKEND is set to something else, except
that it now also checks the OS keyring (see KeyringSecretResolver) when an
environment variable isn't set, so a value stored with `buffdata auth set`
just works. Beyond the default, an enterprise deployment can point BuffData at
Vault, AWS Secrets Manager, GCP Secret Manager, or Azure Key Vault without any
code changes above this layer -- only environment configuration.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class SecretResolver(Protocol):
    def get(self, key: str) -> Optional[str]: ...


class KeyringSecretResolver:
    """Reads secrets from the OS-native credential store -- Windows Credential Manager,
    macOS Keychain, or Linux Secret Service/KWallet -- via the `keyring` package. The one
    backend that needs no server, no cloud account, and no infrastructure to already exist
    to be useful: exactly the gap for a `pip install`ed CLI running on someone's own
    machine, where Vault/AWS/GCP/Azure secret managers all assume infrastructure that
    plainly isn't there. Populated with `buffdata auth set <NAME>`, which prompts for the
    value with hidden input and never writes it to any file.
    """

    SERVICE_NAME = "buffdata"

    def __init__(self, service_name: Optional[str] = None):
        self.service_name = service_name or os.getenv("BUFFDATA_KEYRING_SERVICE", self.SERVICE_NAME)

    def get(self, key: str) -> Optional[str]:
        try:
            import keyring
        except ImportError:
            return None
        try:
            return keyring.get_password(self.service_name, key)
        except Exception:
            # A missing/misconfigured OS backend (headless Linux with no Secret Service,
            # a locked keychain, ...) must degrade to "not found" for this one lookup, not
            # crash every secret resolution on a machine that simply has none configured.
            return None


class EnvSecretResolver:
    """Default resolver: environment variables (including .env, already loaded via
    engine/client.py's load_dotenv()), falling back to the OS keyring
    (KeyringSecretResolver) for any key not found in the environment. This is the only
    resolver that falls back to anything -- every other backend below is explicit and
    deliberately does not -- because the fallback exists specifically to make the
    unconfigured default case work with zero setup: `buffdata auth set GEMINI_API_KEY`
    followed immediately by any command that needs it, no BUFFDATA_SECRET_BACKEND change
    required. An environment variable, when set, always wins over the keyring.
    """

    def __init__(self):
        self._keyring = KeyringSecretResolver()

    def get(self, key: str) -> Optional[str]:
        value = os.getenv(key)
        if value:
            return value
        return self._keyring.get(key)


class VaultSecretResolver:
    """Resolves secrets from one HashiCorp Vault KV v2 secret document. Every BuffData
    secret name (GEMINI_API_KEY, ANTHROPIC_API_KEY, ...) is looked up as a field inside
    that single document, so one Vault secret can hold every provider's credentials.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        mount_point: Optional[str] = None,
        url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.path = path or os.getenv("BUFFDATA_VAULT_PATH", "buffdata")
        self.mount_point = mount_point or os.getenv("BUFFDATA_VAULT_MOUNT", "secret")
        self.url = url or os.getenv("VAULT_ADDR")
        self.token = token or os.getenv("VAULT_TOKEN")
        self._client = None
        self._cache: Optional[dict[str, str]] = None

    @property
    def client(self):
        if self._client is None:
            try:
                import hvac
            except ImportError as exc:
                raise RuntimeError(
                    "Install hvac (pip install buffdata[enterprise]) to use the vault secret backend."
                ) from exc
            if not self.url or not self.token:
                raise RuntimeError("VAULT_ADDR and VAULT_TOKEN are required for the vault secret backend.")
            self._client = hvac.Client(url=self.url, token=self.token)
        return self._client

    def get(self, key: str) -> Optional[str]:
        if self._cache is None:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=self.path, mount_point=self.mount_point,
            )
            self._cache = dict(response["data"]["data"])
        return self._cache.get(key)


class AWSSecretsManagerResolver:
    """Resolves secrets from one AWS Secrets Manager secret holding a JSON object, one
    field per BuffData secret name -- e.g. {"GEMINI_API_KEY": "...", "ANTHROPIC_API_KEY": "..."}.
    """

    def __init__(self, secret_id: Optional[str] = None, region: Optional[str] = None):
        self.secret_id = secret_id or os.getenv("BUFFDATA_AWS_SECRET_ID")
        self.region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        self._cache: Optional[dict[str, str]] = None

    def get(self, key: str) -> Optional[str]:
        if self._cache is None:
            if not self.secret_id:
                raise RuntimeError("BUFFDATA_AWS_SECRET_ID is required for the aws_secrets_manager backend.")
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "Install boto3 (pip install buffdata[enterprise]) to use the aws_secrets_manager backend."
                ) from exc
            client = boto3.client("secretsmanager", region_name=self.region)
            response = client.get_secret_value(SecretId=self.secret_id)
            self._cache = json.loads(response["SecretString"])
        return self._cache.get(key)


class GCPSecretManagerResolver:
    """Resolves each BuffData secret name as its own GCP Secret Manager secret (a secret
    literally named GEMINI_API_KEY, etc.), latest version.
    """

    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
        self._client = None
        self._cache: dict[str, Optional[str]] = {}

    @property
    def client(self):
        if self._client is None:
            try:
                from google.cloud import secretmanager
            except ImportError as exc:
                raise RuntimeError(
                    "Install google-cloud-secret-manager (pip install buffdata[enterprise]) "
                    "to use the gcp_secret_manager backend."
                ) from exc
            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            return self._cache[key]
        if not self.project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for the gcp_secret_manager backend.")
        name = f"projects/{self.project_id}/secrets/{key}/versions/latest"
        try:
            response = self.client.access_secret_version(name=name)
            value = response.payload.data.decode("utf-8")
        except Exception:
            value = None
        self._cache[key] = value
        return value


class AzureKeyVaultResolver:
    """Resolves each BuffData secret name as its own Key Vault secret. Key Vault secret
    names cannot contain underscores, so GEMINI_API_KEY is looked up as gemini-api-key.
    """

    def __init__(self, vault_url: Optional[str] = None):
        self.vault_url = vault_url or os.getenv("AZURE_KEY_VAULT_URL")
        self._client = None
        self._cache: dict[str, Optional[str]] = {}

    @property
    def client(self):
        if self._client is None:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient
            except ImportError as exc:
                raise RuntimeError(
                    "Install azure-identity and azure-keyvault-secrets (pip install "
                    "buffdata[enterprise]) to use the azure_key_vault backend."
                ) from exc
            if not self.vault_url:
                raise RuntimeError("AZURE_KEY_VAULT_URL is required for the azure_key_vault backend.")
            self._client = SecretClient(vault_url=self.vault_url, credential=DefaultAzureCredential())
        return self._client

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            return self._cache[key]
        secret_name = key.lower().replace("_", "-")
        try:
            value = self.client.get_secret(secret_name).value
        except Exception:
            value = None
        self._cache[key] = value
        return value


_RESOLVERS = {
    "env": EnvSecretResolver,
    "keyring": KeyringSecretResolver,
    "vault": VaultSecretResolver,
    "aws_secrets_manager": AWSSecretsManagerResolver,
    "gcp_secret_manager": GCPSecretManagerResolver,
    "azure_key_vault": AzureKeyVaultResolver,
}

_default_resolver: Optional[SecretResolver] = None


def create_secret_resolver(backend: Optional[str] = None) -> SecretResolver:
    """Build a resolver for the given backend name (or BUFFDATA_SECRET_BACKEND, or "env")."""
    selected = (backend or os.getenv("BUFFDATA_SECRET_BACKEND") or "env").lower()
    try:
        resolver_cls = _RESOLVERS[selected]
    except KeyError as exc:
        choices = ", ".join(_RESOLVERS)
        raise ValueError(f"Unknown secret backend '{selected}'. Choose one of: {choices}.") from exc
    return resolver_cls()


def get_default_secret_resolver() -> SecretResolver:
    """Process-wide default resolver, built once (lazily) from BUFFDATA_SECRET_BACKEND."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = create_secret_resolver()
    return _default_resolver


def set_default_secret_resolver(resolver: Optional[SecretResolver]) -> None:
    """Override the process-wide default resolver, or reset it (pass None) so the next
    get_default_secret_resolver() call rebuilds it from BUFFDATA_SECRET_BACKEND. Mainly for
    tests and for programmatic SDK use that wants to inject a resolver explicitly rather
    than through environment variables.
    """
    global _default_resolver
    _default_resolver = resolver
