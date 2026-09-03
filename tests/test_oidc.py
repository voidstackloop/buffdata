import base64
import json
import time

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from buffdata.governance.oidc import (
    OIDCConfig,
    OIDCVerificationError,
    load_oidc_config,
    verify_bearer_token,
)
from buffdata.governance import oidc as oidc_module


ISSUER = "https://issuer.example.com"
AUDIENCE = "buffdata"
KID = "test-key-1"


def _b64url_uint(value: int) -> str:
    length = (value.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(value.to_bytes(length, "big")).rstrip(b"=").decode("ascii")


@pytest.fixture(scope="module")
def keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": KID,
        "use": "sig",
        "alg": "RS256",
        "n": _b64url_uint(numbers.n),
        "e": _b64url_uint(numbers.e),
    }
    return private_pem, {"keys": [jwk]}


def _make_token(private_pem, *, claims_override=None, kid=KID, algorithm="RS256", key=None):
    now = int(time.time())
    claims = {
        "sub": "alice@example.com",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 300,
    }
    if claims_override:
        claims.update(claims_override)
    headers = {"kid": kid} if kid is not None else {}
    return jwt.encode(claims, key or private_pem, algorithm=algorithm, headers=headers)


def _config(jwks, **overrides):
    return OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks=jwks, **overrides)


# --- happy path ---------------------------------------------------------------------------

def test_verify_bearer_token_accepts_a_correctly_signed_token(keypair):
    private_pem, jwks = keypair
    token = _make_token(private_pem)

    identity = verify_bearer_token(token, config=_config(jwks))

    assert identity.actor == "alice@example.com"
    assert identity.claims["iss"] == ISSUER


def test_verify_bearer_token_uses_the_configured_actor_claim(keypair):
    private_pem, jwks = keypair
    token = _make_token(private_pem, claims_override={"email": "bob@example.com"})

    identity = verify_bearer_token(token, config=_config(jwks, actor_claim="email"))

    assert identity.actor == "bob@example.com"


# --- rejection: tampering / forgery -------------------------------------------------------

def test_verify_bearer_token_rejects_alg_none_forgery(keypair):
    _private_pem, jwks = keypair
    forged = jwt.encode({"sub": "mallory", "iss": ISSUER, "aud": AUDIENCE}, key="", algorithm="none")

    with pytest.raises(OIDCVerificationError):
        verify_bearer_token(forged, config=_config(jwks))


def test_verify_bearer_token_rejects_hs256_key_confusion_using_the_public_key_as_secret(keypair):
    import hashlib
    import hmac as hmac_module

    private_pem, jwks = keypair
    # Forge an HS256 token "signed" using the RSA public key material as an HMAC secret --
    # the classic RS256->HS256 downgrade attack. This must be rejected because algorithm
    # selection comes from *our* JWKS entry (alg: RS256), never from the attacker's header.
    # Crafted by hand (not via jwt.encode) because PyJWT's own encoder refuses to use PEM
    # material as an HMAC secret -- a real attacker forging raw bytes has no such guardrail.
    private_key = serialization.load_pem_private_key(private_pem, password=None)
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "kid": KID, "typ": "JWT"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "mallory", "iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300}).encode()
    ).rstrip(b"=")
    signing_input = header + b"." + payload
    signature = base64.urlsafe_b64encode(hmac_module.new(public_bytes, signing_input, hashlib.sha256).digest()).rstrip(b"=")
    forged = (signing_input + b"." + signature).decode("ascii")

    with pytest.raises(OIDCVerificationError):
        verify_bearer_token(forged, config=_config(jwks))


def test_verify_bearer_token_rejects_signature_from_a_different_key(keypair):
    _private_pem, jwks = keypair
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other_pem = other_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    token = _make_token(other_pem)  # signed by a key not in the JWKS

    with pytest.raises(OIDCVerificationError):
        verify_bearer_token(token, config=_config(jwks))


def test_verify_bearer_token_rejects_unknown_kid(keypair):
    private_pem, jwks = keypair
    token = _make_token(private_pem, kid="some-other-key")

    with pytest.raises(OIDCVerificationError, match="No JWKS key found"):
        verify_bearer_token(token, config=_config(jwks))


def test_verify_bearer_token_rejects_malformed_token(keypair):
    _private_pem, jwks = keypair
    with pytest.raises(OIDCVerificationError, match="Malformed bearer token"):
        verify_bearer_token("not-a-jwt-at-all", config=_config(jwks))


# --- rejection: claim mismatches ------------------------------------------------------------

def test_verify_bearer_token_rejects_expired_token(keypair):
    private_pem, jwks = keypair
    token = _make_token(private_pem, claims_override={"exp": int(time.time()) - 10})

    with pytest.raises(OIDCVerificationError):
        verify_bearer_token(token, config=_config(jwks))


def test_verify_bearer_token_rejects_wrong_audience(keypair):
    private_pem, jwks = keypair
    token = _make_token(private_pem, claims_override={"aud": "someone-elses-app"})

    with pytest.raises(OIDCVerificationError):
        verify_bearer_token(token, config=_config(jwks))


def test_verify_bearer_token_rejects_wrong_issuer(keypair):
    private_pem, jwks = keypair
    token = _make_token(private_pem, claims_override={"iss": "https://not-the-real-issuer.example.com"})

    with pytest.raises(OIDCVerificationError):
        verify_bearer_token(token, config=_config(jwks))


def test_verify_bearer_token_rejects_missing_actor_claim(keypair):
    private_pem, jwks = keypair
    now = int(time.time())
    token = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 300},  # no "sub"
        private_pem,
        algorithm="RS256",
        headers={"kid": KID},
    )

    with pytest.raises(OIDCVerificationError, match="sub"):
        verify_bearer_token(token, config=_config(jwks))


# --- config loading ---------------------------------------------------------------------

def test_load_oidc_config_from_yaml(tmp_path):
    path = tmp_path / "oidc.yaml"
    path.write_text(
        yaml.dump({"issuer": ISSUER, "audience": AUDIENCE, "jwks_url": "https://issuer.example.com/jwks.json"}),
        encoding="utf-8",
    )

    config = load_oidc_config(path)

    assert config.issuer == ISSUER
    assert config.jwks_url == "https://issuer.example.com/jwks.json"
    assert config.actor_claim == "sub"  # default


def test_config_requires_either_static_jwks_or_jwks_url(keypair):
    private_pem, _jwks = keypair
    token = _make_token(private_pem)
    config = OIDCConfig(issuer=ISSUER, audience=AUDIENCE)  # neither jwks nor jwks_url set

    with pytest.raises(OIDCVerificationError, match="jwks"):
        verify_bearer_token(token, config=config)


# --- JWKS fetching + caching (mocked HTTP, no real network) ------------------------------

def test_jwks_url_is_fetched_and_cached(monkeypatch, keypair):
    private_pem, jwks = keypair
    token = _make_token(private_pem)
    fetch_calls = []

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            fetch_calls.append(1)
            return json.dumps(jwks).encode("utf-8")

    monkeypatch.setattr(oidc_module, "_fetch_jwks", lambda url: _FakeResponse())
    oidc_module._jwks_cache.clear()

    config = OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks_url="https://issuer.example.com/jwks.json", jwks_cache_seconds=300)

    verify_bearer_token(token, config=config)
    verify_bearer_token(token, config=config)

    assert len(fetch_calls) == 1  # second call served from cache, not refetched
    oidc_module._jwks_cache.clear()


def test_jwks_url_is_refetched_after_ttl_expires(monkeypatch, keypair):
    private_pem, jwks = keypair
    token = _make_token(private_pem)
    fetch_calls = []

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            fetch_calls.append(1)
            return json.dumps(jwks).encode("utf-8")

    monkeypatch.setattr(oidc_module, "_fetch_jwks", lambda url: _FakeResponse())
    oidc_module._jwks_cache.clear()

    config = OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks_url="https://issuer.example.com/jwks.json", jwks_cache_seconds=1)

    verify_bearer_token(token, config=config)
    url = config.jwks_url
    timestamp, cached = oidc_module._jwks_cache._entries[url]
    oidc_module._jwks_cache._entries[url] = (timestamp - 2, cached)
    oidc_module._jwks_cache._attempts[url] -= 2
    verify_bearer_token(token, config=config)

    assert len(fetch_calls) == 2
    oidc_module._jwks_cache.clear()


def test_jwks_fetch_failure_raises_a_clear_error(monkeypatch, keypair):
    private_pem, _jwks = keypair
    token = _make_token(private_pem)

    def _boom(url, timeout=10):
        raise OSError("connection refused")

    monkeypatch.setattr(oidc_module, "_fetch_jwks", _boom)
    oidc_module._jwks_cache.clear()

    config = OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks_url="https://issuer.example.com/jwks.json")
    with pytest.raises(OIDCVerificationError, match="Could not fetch JWKS"):
        verify_bearer_token(token, config=config)
    oidc_module._jwks_cache.clear()


def test_oidc_algorithms_must_be_configured(keypair):
    private, jwks = keypair
    other = json.loads(json.dumps(jwks))
    other["keys"][0]["alg"] = "RS512"
    token = _make_token(private, algorithm="RS512")
    with pytest.raises(OIDCVerificationError, match="not configured"):
        verify_bearer_token(token, config=_config(other))
    assert verify_bearer_token(token, config=_config(other, algorithms=["RS512"])).actor
    with pytest.raises(ValueError):
        _config(jwks, algorithms=["HS256"])


def test_jwks_rotation_is_rate_limited(monkeypatch, keypair):
    import io
    private, jwks = keypair
    rotated = json.loads(json.dumps(jwks))
    rotated["keys"][0]["kid"] = "rotated"
    calls = []
    def fetch(url):
        calls.append(url)
        return io.BytesIO(json.dumps(jwks if len(calls) == 1 else rotated).encode())
    monkeypatch.setattr(oidc_module, "_fetch_jwks", fetch)
    oidc_module._jwks_cache.clear()
    config = OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks_url=ISSUER + "/keys")
    verify_bearer_token(_make_token(private), config=config)
    for _ in range(5):
        with pytest.raises(OIDCVerificationError):
            verify_bearer_token(_make_token(private, kid="random"), config=config)
    assert len(calls) == 1
    oidc_module._jwks_cache._attempts[config.jwks_url] -= 6
    assert verify_bearer_token(_make_token(private, kid="rotated"), config=config).actor
    assert len(calls) == 2
    oidc_module._jwks_cache.clear()


def test_jwks_rejects_redirects_and_unsafe_urls():
    with pytest.raises(OIDCVerificationError, match="redirects"):
        oidc_module._NoJWKSRedirect().redirect_request(None, None, 302, "", {}, "http://169.254.169.254")
    for url in ("http://example.com/keys", "https://user:secret@example.com/keys", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            OIDCConfig(issuer=ISSUER, audience=AUDIENCE, jwks_url=url)
