"""OIDC bearer-token verification: the authentication half access.py deliberately left out.

access.py answers "is actor X allowed to do Y" against a policy file, with no opinion on
how X was established -- that's what this module is for. Point it at a real identity
provider's JWKS endpoint (Okta, Auth0, Azure AD, Google Workspace, or any standards-
compliant OIDC provider all publish one at /.well-known/jwks.json) and it verifies a
bearer token's signature, expiry, issuer, and audience, then extracts an actor identity
that access.check_permission can be run against -- the same split real systems use (an
IdP authenticates; an authorization layer decides what the resulting identity can do).

Security-critical detail: the algorithm used to verify a token is taken only from *our
own* fetched JWKS entry (or a static one supplied by the deployer), never from the
untrusted token's own header. That's what stops the classic "alg: none" / RS256-to-HS256
downgrade attacks -- an attacker controls the token they send, not the JWKS this module
already trusts.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    import jwt

# JWKS keys are for asymmetric signature verification only. Restricting to this safelist
# (even though jwt.decode's `algorithms=` already prevents algorithm-confusion attacks on
# its own) means a malformed or malicious JWKS document can't smuggle in something like
# HS256 by way of a key's "alg" field.
_ALLOWED_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "PS256", "PS384", "PS512"})


class OIDCVerificationError(RuntimeError):
    """Raised when a bearer token fails signature, expiry, issuer, or audience checks, or
    when its algorithm/key can't be resolved against the configured JWKS."""


class OIDCIdentity(BaseModel):
    actor: str
    claims: dict[str, Any]


class OIDCConfig(BaseModel):
    issuer: str
    audience: str
    jwks_url: Optional[str] = None
    # Static JWKS document, mainly for tests and air-gapped/offline verification. Prefer
    # jwks_url in production so key rotation on the IdP's side is picked up automatically.
    jwks: Optional[dict[str, Any]] = None
    actor_claim: str = "sub"
    algorithms: list[str] = Field(default_factory=lambda: ["RS256"], min_length=1)
    jwks_cache_seconds: int = Field(300, ge=1, le=3600)
    jwks_refresh_seconds: int = Field(5, ge=1, le=60)
    leeway_seconds: float = Field(0, ge=0, le=60)

    @field_validator("algorithms")
    @classmethod
    def signing_algorithms(cls, value):
        if not set(value) <= _ALLOWED_ALGORITHMS:
            raise ValueError("Only explicitly approved asymmetric signing algorithms are supported")
        return value

    @field_validator("jwks_url")
    @classmethod
    def jwks_transport(cls, value):
        if value is not None:
            p = urlsplit(value)
            if p.scheme != "https" or not p.hostname or p.username or p.password or p.fragment:
                raise ValueError("JWKS requires a credential-free HTTPS URL")
        return value

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "OIDCConfig":
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls(**data)


def load_oidc_config(path: Union[str, Path]) -> OIDCConfig:
    return OIDCConfig.from_yaml(path)


class _JWKSCache:
    """Process-local cache keyed by URL, so a CLI invocation checking several commands'
    worth of permissions (or a long-lived server embedding this module) doesn't refetch
    the JWKS on every single token."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, dict[str, Any]]] = {}
        self._attempts: dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, url: str, ttl_seconds: int, *, refresh=False, refresh_seconds=5) -> dict[str, Any]:
        # Single-flight fetches, including failed fetches and unknown-key requests. An
        # attacker sending random kids cannot turn each API request into an IdP fetch.
        with self._lock:
            now = time.monotonic()
            cached = self._entries.get(url)
            if not refresh and cached is not None and now - cached[0] < ttl_seconds:
                return cached[1]
            if now - self._attempts.get(url, float("-inf")) < min(ttl_seconds, refresh_seconds):
                if cached is not None and now - cached[0] < ttl_seconds:
                    return cached[1]
                raise OIDCVerificationError("Could not fetch JWKS: refresh cooldown is active")
            self._attempts[url] = now
            try:
                with _fetch_jwks(url) as response:
                    body = response.read(1024 * 1024 + 1)
                    if len(body) > 1024 * 1024:
                        raise ValueError("JWKS exceeds size limit")
                    data = json.loads(body.decode("utf-8"))
                    if not isinstance(data, dict) or not isinstance(data.get("keys"), list) or not all(
                        isinstance(key, dict) for key in data["keys"]):
                        raise ValueError("Invalid JWKS shape")
            except (OSError, ValueError):
                raise OIDCVerificationError("Could not fetch JWKS from the configured identity provider") from None
            self._entries[url] = (now, data)
            return data

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._attempts.clear()


class _NoJWKSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise OIDCVerificationError("JWKS redirects are forbidden; configure the final HTTPS endpoint")


def _fetch_jwks(url):
    # Administrator-configured IdP may be private. Never follow it to another trust domain.
    return urllib.request.build_opener(_NoJWKSRedirect()).open(url, timeout=10)


_jwks_cache = _JWKSCache()


def _resolve_jwks(config: OIDCConfig) -> dict[str, Any]:
    if config.jwks is not None:
        return config.jwks
    if config.jwks_url is None:
        raise OIDCVerificationError("OIDC config must set either 'jwks' (static) or 'jwks_url'.")
    return _jwks_cache.get(config.jwks_url, config.jwks_cache_seconds, refresh_seconds=config.jwks_refresh_seconds)


def verify_bearer_token(token: str, *, config: OIDCConfig) -> OIDCIdentity:
    """Verify `token` against `config` and return the identity it proves. Raises
    OIDCVerificationError on any failure -- expired, wrong issuer/audience, bad
    signature, unknown key, disallowed algorithm, or missing actor claim. Never returns
    a partially-trusted result; every failure mode raises."""
    try:
        import jwt
    except ImportError as exc:
        raise OIDCVerificationError(
            "Install PyJWT (pip install buffdata[enterprise]) to verify OIDC bearer tokens."
        ) from exc

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise OIDCVerificationError(f"Malformed bearer token: {exc}") from exc

    kid = header.get("kid")
    if header.get("alg") not in config.algorithms:
        raise OIDCVerificationError("Token signing algorithm is not configured")
    jwks = _resolve_jwks(config)
    candidates = [key for key in jwks.get("keys", []) if kid is None or key.get("kid") == kid]
    if not candidates and config.jwks is None and config.jwks_url:
        jwks = _jwks_cache.get(config.jwks_url, config.jwks_cache_seconds,
            refresh=True, refresh_seconds=config.jwks_refresh_seconds)
        candidates = [key for key in jwks.get("keys", []) if kid is None or key.get("kid") == kid]
    if not candidates:
        raise OIDCVerificationError(f"No JWKS key found for kid={kid!r} (issuer {config.issuer!r}).")

    last_error: Optional[Exception] = None
    for jwk in candidates:
        algorithm = jwk.get("alg", "RS256")
        if algorithm not in config.algorithms or jwk.get("use", "sig") != "sig" or "verify" not in jwk.get("key_ops", ["verify"]):
            last_error = OIDCVerificationError(f"JWKS key algorithm {algorithm!r} is not an allowed signing algorithm.")
            continue
        try:
            public_key = jwt.PyJWK.from_json(json.dumps(jwk), algorithm=algorithm)
            claims = jwt.decode(
                token,
                key=public_key,
                algorithms=[algorithm],
                audience=config.audience,
                issuer=config.issuer,
                leeway=config.leeway_seconds,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as exc:
            last_error = exc
            continue
        actor = claims.get(config.actor_claim)
        if not isinstance(actor, str) or not actor.strip():
            raise OIDCVerificationError(
                f"Verified token has no usable '{config.actor_claim}' claim to use as the actor identity."
            )
        return OIDCIdentity(actor=str(actor), claims=claims)

    raise OIDCVerificationError(f"Bearer token failed verification: {last_error}")
