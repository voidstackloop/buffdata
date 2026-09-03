import gzip
import json
import socket
import time

import jwt
import pytest

from buffdata.security.policy import (ExecutionContext, SecurityPolicy, SecurityError,
    check_input, check_network_url, contained_path, execution_context, remember_secret, sanitize)
from buffdata.report.generator import ReportGenerator
from buffdata.models.schemas import DatasetItem
from buffdata.governance.oidc import verify_bearer_token, OIDCVerificationError
from tests.test_oidc import keypair, _config, ISSUER, AUDIENCE, KID


@pytest.mark.parametrize("claim", ["exp", "iss", "aud", "sub"])
def test_required_claims_cannot_be_omitted(keypair, claim):
    private, jwks = keypair
    claims = {"exp": int(time.time()) + 60, "iss": ISSUER, "aud": AUDIENCE, "sub": "alice"}
    claims.pop(claim)
    token = jwt.encode(claims, private, algorithm="RS256", headers={"kid": KID})
    with pytest.raises(OIDCVerificationError):
        verify_bearer_token(token, config=_config(jwks))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1", "http://169.254.169.254", "http://[::1]", "http://user:pass@example.com"])
def test_reject_unsafe_urls(url):
    with pytest.raises(SecurityError):
        check_network_url(url)


def test_private_dns_resolution_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 80))])
    with pytest.raises(SecurityError):
        check_network_url("https://public-looking.example")


def test_strict_policy_applies_to_integrations():
    with execution_context(ExecutionContext(network="strict")):
        with pytest.raises(SecurityError):
            check_network_url("https://context7.com")


def test_symlinks_and_decompression_limit(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("[]")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(SecurityError):
        contained_path(link)
    archive = tmp_path / "data.json.gz"
    archive.write_bytes(gzip.compress(b"x" * 10000))
    with pytest.raises(SecurityError, match="Decompressed"):
        check_input(archive, SecurityPolicy(max_expanded_bytes=50))


def test_html_report_escapes_record_content(tmp_path):
    path = ReportGenerator.generate_html_report([DatasetItem.from_dict({"id": "<img src=x onerror=alert(1)>",
        "instruction": "</pre><script>alert(1)</script>", "output": "<iframe src=x>"})], tmp_path / "report.html", provider="<script>")
    body = path.read_text()
    assert "<script>" not in body and "<iframe" not in body
    assert "&lt;script&gt;" in body


def test_exact_secret_and_bearer_masking():
    remember_secret("unusual-secret-value")
    assert "unusual-secret-value" not in sanitize("error unusual-secret-value")
    assert "secretbearer" not in sanitize("Bearer secretbearer")


def test_plugin_approval_precedes_import(monkeypatch):
    from buffdata import plugins
    class Entry:
        name = "untrusted"
        def load(self):
            raise AssertionError("must never import")
    monkeypatch.setattr(plugins, "entry_points", lambda **kw: [Entry()])
    with execution_context(ExecutionContext()):
        with pytest.raises(SecurityError, match="requires approval"):
            plugins.load_validator_plugins(refresh=True)


@pytest.mark.asyncio
async def test_redirect_to_metadata_is_rejected_before_second_request(monkeypatch):
    import aiohttp
    from buffdata.security.network import fetch_public
    calls = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    class Response:
        status = 302
        headers = {"Location": "http://169.254.169.254/latest/meta-data"}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
    class Session:
        def __init__(self, **kwargs):
            self.connector = kwargs["connector"]
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            await self.connector.close()
        def get(self, url, **kwargs):
            assert kwargs["allow_redirects"] is False
            calls.append(url)
            return Response()
    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    with pytest.raises(SecurityError):
        await fetch_public("http://public.example")
    assert calls == ["http://public.example"]


@pytest.mark.asyncio
async def test_sdk_exception_cannot_be_copied_into_dataset_metadata():
    from buffdata.security.provider import SafeProvider
    from buffdata.engine.client import ProviderError
    class SDK:
        async def generate_structured_async(self, **kwargs):
            raise RuntimeError("credential-in-SDK-error")
    with pytest.raises(ProviderError) as error:
        await SafeProvider(SDK()).generate_structured_async()
    assert "credential-in-SDK-error" not in str(error.value)
