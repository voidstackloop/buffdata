"""Envelope encryption for managed-run artifacts (output/rejected/report -- see
docs/system-design-roadmap.md #3.7). Opt-in, team-deployment-scoped: unless
BUFFDATA_ARTIFACT_MASTER_KEY_SECRET names a secret, create_key_manager() returns None and
every code path that touches this module is a complete no-op. This protects data at rest from
anyone with only filesystem/backup access (a stolen disk, a leaked backup) -- not from the
running API/worker processes, which already have full filesystem access to every project's
artifacts by design (see docs/team-deployment.md).

Not a real KMS wrap/unwrap API: the "master key" is fetched as a plain secret value via the
existing buffdata.engine.secrets.SecretResolver abstraction (whatever backend -- env, vault,
aws_secrets_manager, ... -- an operator already has configured), then used to wrap/unwrap a
per-project data key locally with AES-GCM. Real, disclosed trade-off: key material passes
through this process's memory, unlike a true HSM-backed KMS that never reveals it.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"BFEK1"
_NONCE_BYTES = 12


class KeyManager(Protocol):
    def wrap(self, plaintext_key: bytes) -> bytes: ...
    def unwrap(self, wrapped_key: bytes) -> bytes: ...


class LocalKeyManager:
    def __init__(self, master_key: bytes):
        if len(master_key) != 32:
            raise ValueError("Master key must decode to exactly 32 bytes (AES-256)")
        self._master_key = master_key

    def wrap(self, plaintext_key: bytes) -> bytes:
        nonce = os.urandom(_NONCE_BYTES)
        return nonce + AESGCM(self._master_key).encrypt(nonce, plaintext_key, None)

    def unwrap(self, wrapped_key: bytes) -> bytes:
        nonce, ciphertext = wrapped_key[:_NONCE_BYTES], wrapped_key[_NONCE_BYTES:]
        return AESGCM(self._master_key).decrypt(nonce, ciphertext, None)


def create_key_manager() -> "KeyManager | None":
    # Explicit opt-in gate BEFORE touching any secret resolver -- an unconfigured deployment
    # (the default) makes zero resolver calls. Deliberately not gated on BUFFDATA_SECRET_BACKEND,
    # which may already be vault/aws_secrets_manager/etc. for unrelated provider-key reasons;
    # implicitly riding that would mean a real network call (and startup-failure risk) at
    # every RunService construction for deployments that never asked for this feature.
    secret_name = os.getenv("BUFFDATA_ARTIFACT_MASTER_KEY_SECRET")
    if not secret_name:
        return None
    from buffdata.engine.secrets import get_default_secret_resolver
    from buffdata.security.policy import remember_secret
    value = get_default_secret_resolver().get(secret_name)
    if not value:
        raise RuntimeError(f"Configured master key secret {secret_name!r} did not resolve to a value")
    remember_secret(value)
    return LocalKeyManager(base64.b64decode(value))


def encrypt_file_in_place(path: Path, data_key: bytes) -> None:
    import tempfile
    plaintext = path.read_bytes()
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(data_key).encrypt(nonce, plaintext, None)
    fd, temporary = tempfile.mkstemp(prefix=".encrypt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(MAGIC + nonce + ciphertext)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def decrypt_bytes(blob: bytes, data_key: bytes) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("Not a recognized encrypted artifact")
    offset = len(MAGIC)
    nonce, ciphertext = blob[offset:offset + _NONCE_BYTES], blob[offset + _NONCE_BYTES:]
    return AESGCM(data_key).decrypt(nonce, ciphertext, None)


def is_encrypted(blob_or_path) -> bool:
    if isinstance(blob_or_path, (bytes, bytearray)):
        return bytes(blob_or_path[:len(MAGIC)]) == MAGIC
    with open(blob_or_path, "rb") as handle:
        return handle.read(len(MAGIC)) == MAGIC
