"""Execution-boundary policy. This module never edits dataset records."""
from __future__ import annotations

import contextvars
import gzip
import ipaddress
import json
import logging
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field


class SecurityError(ValueError):
    pass


class SecurityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved_plugins: list[str] = Field(default_factory=list)
    allowed_hosts: list[str] = Field(default_factory=lambda: [
        "generativelanguage.googleapis.com", "api.openai.com", "api.anthropic.com",
        "huggingface.co", "context7.com",
    ])
    local_endpoints: list[str] = Field(default_factory=list)
    allowed_cloud_prefixes: list[str] = Field(default_factory=list)
    read_roots: list[str] = Field(default_factory=list)
    write_roots: list[str] = Field(default_factory=list)
    max_upload_bytes: int = Field(1024**3, gt=0)
    max_expanded_bytes: int = Field(8 * 1024**3, gt=0)
    max_record_bytes: int = Field(16 * 1024**2, gt=0)
    execution_seconds: int = Field(21600, gt=0)
    subprocess_seconds: int = Field(120, gt=0)
    presidio_anonymizer_python: str | None = None


class ExecutionContext(BaseModel):
    actor: str = "local"
    project_id: str = "local"
    network: str = "unrestricted"
    policy: SecurityPolicy = Field(default_factory=SecurityPolicy)


_current = contextvars.ContextVar("buffdata_execution", default=None)


def current_context() -> ExecutionContext | None:
    return _current.get()


@contextmanager
def execution_context(context: ExecutionContext):
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


def approved_plugins() -> set[str]:
    ctx = current_context()
    if ctx:
        return set(ctx.policy.approved_plugins)
    return {x.strip() for x in os.getenv("BUFFDATA_APPROVED_PLUGINS", "").split(",") if x.strip()}


def presidio_anonymizer_python() -> str | None:
    """Path to a separate Python interpreter with presidio-anonymizer + its older-pinned
    cryptography installed -- unset by default (regex-only PII redaction fallback applies
    whenever presidio-anonymizer isn't importable in-process either). Same precedence as
    approved_plugins(): an execution-context policy first, a bare env var only when there's
    no context at all."""
    ctx = current_context()
    if ctx:
        return ctx.policy.presidio_anonymizer_python
    return os.getenv("BUFFDATA_PRESIDIO_ANONYMIZER_PYTHON") or None


def contained_path(path: str | Path, root: str | Path | None = None) -> Path:
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise SecurityError("Symbolic links are not accepted at an artifact boundary")
    resolved = path.resolve()
    if root is not None and not resolved.is_relative_to(Path(root).resolve()):
        raise SecurityError("Path is outside the authorized project directory")
    return resolved


def check_path(path: str | Path, *, write: bool = False) -> Path:
    ctx = current_context()
    result = contained_path(path)
    roots = (ctx.policy.write_roots if write else ctx.policy.read_roots) if ctx else []
    if roots and not any(result.is_relative_to(Path(root).resolve()) for root in roots):
        raise SecurityError("Path is outside the execution policy roots")
    return result


def check_input(path: str | Path, policy: SecurityPolicy | None = None) -> Path:
    path = check_path(path)
    policy = policy or (current_context().policy if current_context() else SecurityPolicy())
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    total = 0
    for file in files:
        contained_path(file, path if path.is_dir() else path.parent)
        if file.is_dir():
            continue
        if not file.is_file():
            raise SecurityError("Input must contain only regular files")
        if file.name.endswith((".jsonl", ".ndjson", ".txt", ".jsonl.gz", ".ndjson.gz", ".txt.gz")):
            opener = gzip.open if file.name.endswith(".gz") else open
            with opener(file, "rb") as rows:
                expanded = 0
                while line := rows.readline(policy.max_record_bytes + 1):
                    expanded += len(line)
                    if len(line) > policy.max_record_bytes:
                        raise SecurityError("Input record exceeds the configured size limit")
                    if expanded > policy.max_expanded_bytes:
                        raise SecurityError("Decompressed input exceeds the configured limit")
        size = file.stat().st_size
        if size > policy.max_upload_bytes:
            raise SecurityError("Input exceeds the configured file size limit")
        if file.name.endswith(".gz"):
            with gzip.open(file, "rb") as handle:
                while block := handle.read(1024**2):
                    total += len(block)
                    if total > policy.max_expanded_bytes:
                        raise SecurityError("Decompressed input exceeds the configured limit")
        else:
            total += size
        if total > policy.max_expanded_bytes:
            raise SecurityError("Input exceeds the configured expanded size limit")
    return path


def check_network_url(url: str, *, resolve: bool = True, ordinary_import: bool = False) -> list[str]:
    ctx = current_context()
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise SecurityError("Only credential-free HTTP(S) URLs are accepted")
    if ctx and ctx.network == "strict":
        raise SecurityError("Strict execution forbids network access")
    host = parsed.hostname.lower().rstrip(".")
    origin = f"{parsed.scheme}://{host}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}"
    local = False
    if ctx and not ordinary_import:
        local = any(origin == _origin(endpoint) for endpoint in ctx.policy.local_endpoints)
        if ctx.network == "local" and not local:
            raise SecurityError("Local endpoint requires explicit policy approval")
        if not local and host not in ctx.policy.allowed_hosts:
            raise SecurityError("Destination is not in the execution allowlist")
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        if not resolve:
            return []
        try:
            addresses = list({row[4][0] for row in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)})
        except OSError as exc:
            raise SecurityError("Destination could not be resolved") from exc
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses) and not local:
        raise SecurityError("Non-public destination is forbidden")
    return addresses


def _origin(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{(p.hostname or '').lower()}:{p.port or (443 if p.scheme == 'https' else 80)}"


def check_cloud(url: str):
    ctx = current_context()
    if ctx and (ctx.network != "unrestricted" or not any(
        url.startswith(prefix.rstrip("/") + "/") for prefix in ctx.policy.allowed_cloud_prefixes
    )):
        raise SecurityError("Cloud destination is not approved for this execution")


def private_directory(path: Path):
    contained_path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        path.chmod(0o700)


def private_json(path: Path, data):
    import tempfile
    private_directory(path.parent)
    contained_path(path)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


_secrets: set[str] = set()


def remember_secret(value):
    if value:
        _secrets.add(str(value))
    return value


def sanitize(message: object) -> str:
    from buffdata.security.validator import SecretMasker
    text = str(message)
    for secret in sorted(_secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return SecretMasker.mask(text)


class SecretFilter(logging.Filter):
    def filter(self, record):
        record.msg, record.args = sanitize(record.getMessage()), ()
        if record.exc_info:
            record.exc_info = None
            record.exc_text = "Exception details omitted from operational logs"
        return True
