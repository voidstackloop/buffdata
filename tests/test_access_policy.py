import base64
import json
import time

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.governance import (
    PermissionDeniedError,
    Policy,
    check_permission,
)
from buffdata.models.formats import write_dataset
from buffdata.models.schemas import DatasetItem

OIDC_ISSUER = "https://issuer.example.com"
OIDC_AUDIENCE = "buffdata"
OIDC_KID = "cli-test-key"


def _b64url_uint(value: int) -> str:
    length = (value.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(value.to_bytes(length, "big")).rstrip(b"=").decode("ascii")


def _oidc_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    numbers = private_key.public_key().public_numbers()
    jwks = {"keys": [{
        "kty": "RSA", "kid": OIDC_KID, "use": "sig", "alg": "RS256",
        "n": _b64url_uint(numbers.n), "e": _b64url_uint(numbers.e),
    }]}
    return private_pem, jwks


def _oidc_token(private_pem, *, sub="bob", exp_delta=300):
    now = int(time.time())
    return jwt.encode(
        {"sub": sub, "iss": OIDC_ISSUER, "aud": OIDC_AUDIENCE, "iat": now, "exp": now + exp_delta},
        private_pem,
        algorithm="RS256",
        headers={"kid": OIDC_KID},
    )


def _write_oidc_config(path, jwks):
    path.write_text(
        yaml.safe_dump({"issuer": OIDC_ISSUER, "audience": OIDC_AUDIENCE, "jwks": jwks}),
        encoding="utf-8",
    )
    return path


def _labeled_rows(n: int) -> list[DatasetItem]:
    return [DatasetItem.from_dict({"text": f"row {i}", "label": i % 2}) for i in range(n)]


def _write_policy(path, roles=None, actors=None, name="test-policy"):
    path.write_text(yaml.safe_dump({
        "name": name,
        "roles": roles or {"admin": {"permissions": ["run", "use_external_providers"]}},
        "actors": actors or {"alice": "admin"},
    }), encoding="utf-8")
    return path


# --- Policy / check_permission --------------------------------------------------------

def test_policy_loads_from_yaml(tmp_path):
    path = _write_policy(tmp_path / "policy.yaml")
    policy = Policy.from_yaml(path)
    assert policy.name == "test-policy"
    assert policy.has_permission("alice", "run") is True


def test_check_permission_allows_when_role_grants_it(tmp_path):
    policy = Policy.from_yaml(_write_policy(tmp_path / "policy.yaml"))
    check_permission(policy, "alice", "run")  # must not raise


def test_check_permission_denies_missing_permission(tmp_path):
    policy = Policy.from_yaml(_write_policy(
        tmp_path / "policy.yaml",
        roles={"restricted": {"permissions": ["run"]}},
        actors={"bob": "restricted"},
    ))
    check_permission(policy, "bob", "run")  # has this one
    with pytest.raises(PermissionDeniedError, match="does not have permission 'use_external_providers'"):
        check_permission(policy, "bob", "use_external_providers")


def test_check_permission_denies_unknown_actor(tmp_path):
    policy = Policy.from_yaml(_write_policy(tmp_path / "policy.yaml"))
    with pytest.raises(PermissionDeniedError, match="not a recognized actor"):
        check_permission(policy, "mallory", "run")


def test_check_permission_denies_undefined_role_fails_closed(tmp_path):
    # actor points at a role name that doesn't exist in roles: -- must deny, not crash or
    # silently allow.
    policy = Policy.from_yaml(_write_policy(
        tmp_path / "policy.yaml",
        roles={"admin": {"permissions": ["run"]}},
        actors={"carol": "role-that-does-not-exist"},
    ))
    with pytest.raises(PermissionDeniedError, match="isn't defined in policy"):
        check_permission(policy, "carol", "run")


def test_empty_policy_denies_everyone():
    policy = Policy(name="empty")
    with pytest.raises(PermissionDeniedError):
        check_permission(policy, "anyone", "run")


# --- CLI enforcement ---------------------------------------------------------------------

def test_cli_requires_both_actor_and_policy_together(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output), "--actor", "alice",
    ])
    assert result.exit_code != 0
    assert "must be given together" in str(result.output) + str(result.exception)


def test_cli_denies_actor_without_run_permission_before_any_work(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"
    policy_file = _write_policy(
        tmp_path / "policy.yaml",
        roles={"none": {"permissions": []}},
        actors={"bob": "none"},
    )

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output),
        "--actor", "bob", "--policy", str(policy_file), "--network-policy", "strict",
    ])

    assert result.exit_code != 0
    assert result.exception is not None
    assert "does not have permission 'run'" in str(result.exception)
    assert not output.exists()  # blocked before any work happened


def test_cli_allows_run_only_actor_under_strict_network_policy(tmp_path):
    # An actor with only "run" (no "use_external_providers") must be allowed through when
    # network_policy=strict, since strict never reaches an external provider at all.
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"
    policy_file = _write_policy(
        tmp_path / "policy.yaml",
        roles={"restricted": {"permissions": ["run"]}},
        actors={"bob": "restricted"},
    )

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output),
        "--actor", "bob", "--policy", str(policy_file), "--network-policy", "strict",
    ])

    # Enforcement passes; the run then fails for an entirely different, expected reason
    # (NetworkForbiddenClient blocking the actual scoring call) -- not a permission denial.
    assert result.exit_code != 0
    assert "does not have permission" not in str(result.exception)
    assert "Network policy 'strict' blocked" in str(result.exception)


def test_cli_denies_restricted_actor_requesting_external_providers(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"
    policy_file = _write_policy(
        tmp_path / "policy.yaml",
        roles={"restricted": {"permissions": ["run"]}},
        actors={"bob": "restricted"},
    )

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output),
        "--actor", "bob", "--policy", str(policy_file),
        # no --network-policy -> defaults to "unrestricted", which bob isn't allowed to use
    ])

    assert result.exit_code != 0
    assert "does not have permission 'use_external_providers'" in str(result.exception)
    assert not output.exists()


# --- CLI enforcement: OIDC-verified identity --------------------------------------------

def test_cli_accepts_a_verified_bearer_token_as_identity(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"
    private_pem, jwks = _oidc_keypair()
    token = _oidc_token(private_pem, sub="bob")
    oidc_config = _write_oidc_config(tmp_path / "oidc.yaml", jwks)
    policy_file = _write_policy(
        tmp_path / "policy.yaml",
        roles={"restricted": {"permissions": ["run"]}},
        actors={"bob": "restricted"},
    )

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output),
        "--bearer-token", token, "--oidc-config", str(oidc_config),
        "--policy", str(policy_file), "--network-policy", "strict",
    ])

    # Enforcement passes on the *verified* identity ("bob" from the token's sub claim);
    # the run then fails for the same expected, unrelated reason as the --actor tests.
    assert result.exit_code != 0
    assert "does not have permission" not in str(result.exception)
    assert "Bearer token verification failed" not in str(result.exception)
    assert "Network policy 'strict' blocked" in str(result.exception)


def test_cli_denies_verified_actor_without_run_permission(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"
    private_pem, jwks = _oidc_keypair()
    token = _oidc_token(private_pem, sub="bob")
    oidc_config = _write_oidc_config(tmp_path / "oidc.yaml", jwks)
    policy_file = _write_policy(
        tmp_path / "policy.yaml",
        roles={"none": {"permissions": []}},
        actors={"bob": "none"},
    )

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output),
        "--bearer-token", token, "--oidc-config", str(oidc_config),
        "--policy", str(policy_file), "--network-policy", "strict",
    ])

    assert result.exit_code != 0
    assert "does not have permission 'run'" in str(result.exception)
    assert not output.exists()


def test_cli_rejects_an_invalid_bearer_token_before_touching_the_policy(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"
    _private_pem, jwks = _oidc_keypair()
    other_private_pem, _other_jwks = _oidc_keypair()
    # Signed by a key that isn't in the configured JWKS.
    forged_token = _oidc_token(other_private_pem, sub="bob")
    oidc_config = _write_oidc_config(tmp_path / "oidc.yaml", jwks)
    policy_file = _write_policy(
        tmp_path / "policy.yaml",
        roles={"admin": {"permissions": ["run", "use_external_providers"]}},
        actors={"bob": "admin"},
    )

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output),
        "--bearer-token", forged_token, "--oidc-config", str(oidc_config),
        "--policy", str(policy_file),
    ])

    assert result.exit_code != 0
    assert "Bearer token verification failed" in str(result.output) + str(result.exception)
    assert not output.exists()


def test_cli_rejects_actor_mismatched_with_the_verified_bearer_token(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"
    private_pem, jwks = _oidc_keypair()
    token = _oidc_token(private_pem, sub="bob")
    oidc_config = _write_oidc_config(tmp_path / "oidc.yaml", jwks)
    policy_file = _write_policy(
        tmp_path / "policy.yaml",
        roles={"admin": {"permissions": ["run", "use_external_providers"]}},
        actors={"alice": "admin", "bob": "admin"},
    )

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output),
        "--actor", "alice",
        "--bearer-token", token, "--oidc-config", str(oidc_config),
        "--policy", str(policy_file),
    ])

    assert result.exit_code != 0
    assert "does not match the identity verified" in str(result.output) + str(result.exception)
    assert not output.exists()


def test_cli_requires_bearer_token_and_oidc_config_together(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output), "--bearer-token", "some-token",
    ])

    assert result.exit_code != 0
    assert "--bearer-token and --oidc-config must be given together" in str(result.output) + str(result.exception)


def test_cli_without_actor_or_policy_is_unaffected(tmp_path):
    # Default (no --actor/--policy at all) behavior must be completely untouched: this
    # should fail for the *same* reason it always would (strict blocks the network call),
    # never a permission error, since no policy was ever consulted.
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"

    result = CliRunner().invoke(app, [
        "score", str(source), "-o", str(output), "--network-policy", "strict",
    ])
    assert result.exit_code != 0
    assert "Network policy 'strict' blocked" in str(result.exception)


def test_cli_optimize_enforces_policy_via_resolved_config_network_policy(tmp_path):
    # optimize_cmd doesn't call create_llm_client directly -- it goes through PipelineConfig
    # -- so this exercises the separate enforcement call wired in after config construction.
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(4), source)
    output = tmp_path / "out.jsonl"
    policy_file = _write_policy(
        tmp_path / "policy.yaml",
        roles={"restricted": {"permissions": ["run"]}},
        actors={"bob": "restricted"},
    )

    result = CliRunner().invoke(app, [
        "optimize", str(source), "-o", str(output),
        "--actor", "bob", "--policy", str(policy_file),
        "--quality-mode", "off",
        # no --network-policy override -> PipelineConfig defaults to "unrestricted"
    ])

    assert result.exit_code != 0
    assert "does not have permission 'use_external_providers'" in str(result.exception)
    assert not output.exists()
