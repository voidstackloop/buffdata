# Governance

Five independent, all opt-in pieces: who's allowed to run what (RBAC), proving who's
actually asking (SSO), a durable record of what ran (audit log), a machine-checkable
bar a run must clear (Data Contracts), and a manifest of what's installed (SBOM). None
of these change default behavior until you turn them on.

## Access policy (RBAC)

[`buffdata/governance/access.py`](../buffdata/governance/access.py) answers "is this
actor allowed to do this" against a policy file -- deliberately authorization only, not
authentication (see [OIDC](#oidc-bearer-token-verification) below for that half).
Fails closed: an actor missing from the policy, or assigned a role that isn't defined,
or a role missing the needed permission, is denied -- never silently allowed because
the policy file has a gap.

```bash
buffdata score data.jsonl -o scored.jsonl --actor bob --policy examples/access_policy.yaml
```

`--actor` and `--policy` must be given together or not at all -- omitting both leaves
every command's default behavior completely unaffected; this is opt-in, never a silent
default. Two permissions are checked: `run` (may invoke buffdata at all) and
`use_external_providers` (may use anything other than `--network-policy strict`). See
[`examples/access_policy.yaml`](../examples/access_policy.yaml) for the full worked
example and every role/actor field.

```yaml
policy_version: "1"
name: "default-policy"
roles:
  restricted:
    permissions: ["run"]                              # strict network policy only
  operator:
    permissions: ["run", "use_external_providers"]
actors:
  bob: operator
```

## OIDC bearer-token verification

[`buffdata/governance/oidc.py`](../buffdata/governance/oidc.py) is the authentication
half access.py leaves out: verifies a real bearer token's signature, expiry, issuer,
and audience against your identity provider's JWKS (Okta, Auth0, Azure AD, Google
Workspace, or any standards-compliant OIDC provider) before trusting any of its claims.

```bash
buffdata score data.jsonl -o scored.jsonl \
  --bearer-token "$OIDC_TOKEN" --oidc-config examples/oidc_config.yaml \
  --policy examples/access_policy.yaml
```

The verified identity (from the configured claim, default `sub`) is what gets checked
against `--policy` -- not a trusted `--actor` string. Passing both requires them to
match, or the run is refused rather than silently preferring one. See
[`examples/oidc_config.yaml`](../examples/oidc_config.yaml) for the config shape
(`issuer`, `audience`, `jwks_url`, `actor_claim`, `jwks_cache_seconds`).

**Security-critical detail**: the signing algorithm used to verify a token is taken
only from your own fetched JWKS entry, never from the untrusted token's own header --
this is what stops the classic `alg: none` / RS256-to-HS256 downgrade attacks. Verified
in [`tests/test_oidc.py`](../tests/test_oidc.py) against both attacks directly, plus
expiry, wrong issuer/audience, wrong signing key, and unknown `kid`, using real signed
JWTs and a real JWKS -- no live IdP needed for any of it.

Requires `pip install -e ".[enterprise]"` for `pyjwt[crypto]`; the package imports
cleanly and `--actor`-only usage is completely unaffected without it installed.

## Audit log

[`buffdata/governance/audit_store.py`](../buffdata/governance/audit_store.py): a
durable, queryable, SQLite-backed (by default) log of completed runs.

```bash
buffdata audit record run.report.json --command optimize
buffdata audit query --command optimize --provider gemini --since 2026-01-01
buffdata audit usage-report --pricing my_rates.yaml   # never guesses at $ pricing itself
```

`--db` (or `$BUFFDATA_AUDIT_DB`) selects the database, default `buffdata_audit.db`.
`audit record --contract` also checks the run against a Data Contract and records the
pass/fail alongside it. `usage-report` aggregates token usage across recorded runs and
only estimates cost when you supply `--pricing` -- a YAML/JSON map of
`{"provider/model": {input_per_1k, output_per_1k}}` -- so no dollar figure is ever
fabricated. `AuditStore` is a `Protocol`, so a Postgres/BigQuery-backed implementation
can be swapped in later without touching call sites.

## Data Contracts

[`buffdata/governance/contract.py`](../buffdata/governance/contract.py): a
versioned, YAML-owned bar a completed run's `report.json` must clear -- fast,
deterministic, no provider credentials needed, built for a CI gate.

```bash
buffdata contract check examples/data_contract.yaml --report output.report.json
echo $?  # 0 = every requirement met, 1 = printed violations
```

```yaml
contract_version: "1"
name: "customer-support-tickets"
requirements:
  min_relative_accuracy_gain: 0.10    # needs --accuracy-gate too, unless report.json already carries one
  required_pii_policy: identifiers
  required_network_policy: strict
  max_rejected_fraction: 0.20
```

See [`examples/data_contract.yaml`](../examples/data_contract.yaml) for the annotated
version. Composes directly into a build pipeline:
`buffdata contract check ... || exit 1`.

## SBOM

[`buffdata/governance/sbom.py`](../buffdata/governance/sbom.py): a CycloneDX 1.5
Software Bill of Materials of packages actually installed in the current environment
(via `importlib.metadata`, not a static requirements-file guess) -- for a vendor
security review or a compliance package.

```bash
buffdata sbom -o sbom.json
```
