# Internal-team deployment with Compose

Install `buffdata[server]` to run the API locally, or build the supplied Compose image.
The dashboard uses OIDC; there is no default password, development authentication bypass,
or public signup. API calls accept verified bearer tokens. Browser login uses authorization
code + PKCE, a one-time state/nonce, server-side sessions, HttpOnly Secure cookies, and CSRF
checks. Put an HTTPS reverse proxy in front of the loopback-bound ingress port (8000).
The credential-free nginx ingress forwards only to the internal API and does not expose
worker-control routes. API/worker containers remain on internal networks; the gateway
has a separate edge network for host port publication. Do not attach workers to that network.

## Configure

1. Copy `deploy/server.example.yaml` to ignored `deploy/team/server.yaml`. Configure the
   real issuer, audience, JWKS, authorization/token endpoints, browser client ID, public URL,
   and administrator's immutable OIDC subject. Register `PUBLIC_URL/callback` with the IdP.
2. Create private `deploy/team/secrets`, `deploy/team/data/team`, and `deploy/team/models`
   directories owned by the configured container UID/GID (default 1000). Secret files must
   be readable only by their owner. Do not commit any of these files.
3. Supply `postgres_password`, `database_url` (a `postgresql+psycopg://buffdata:...@postgres/buffdata`
   URL), `worker_token` (at least 32 random bytes, encoded), and `oidc_client_secret` files.
   Set the SHA-256 of the stripped worker token as `worker_token_sha256` in server YAML.
4. Supply `provider_secrets.json`, a JSON object containing only the provider API-key names
   needed by this project. `{}` is valid for offline runs. Only workers mount this file;
   the API receives no provider credentials and workers receive no database password.
5. Add the actual IdP hostname to `deploy/squid.conf`. Keep its provider allowlist aligned
   with project policy. Prewarm model assets into the read-only model cache before running
   local embedding operations; runtime model downloads are disabled by default.
6. For full Presidio PII redaction (not just the regex-only fallback -- see
   [configuration.md](configuration.md#privacy)), set `policy.presidio_anonymizer_python:
   /opt/venv-presidio/bin/python` in server YAML. `deploy/Dockerfile.team` already builds
   that isolated venv (see [dependency-release-blocker.md](dependency-release-blocker.md));
   this just points the worker at it. Omit it to accept regex-only redaction.

```bash
docker compose -f deploy/compose.yaml config --quiet
docker compose -f deploy/compose.yaml build api
docker compose -f deploy/compose.yaml up -d
```

The API and worker must use the same built image because submissions bind source/dependency
identity. Add a dedicated worker, token, and project-only volume mount for each project. Never
mount the entire artifact root into a worker. The API alone has the aggregate artifact mount.
Workers have no container-engine socket and no database-network membership. Both control and
database networks are internal. Worker outbound traffic must use the allowlisting proxy;
the separate ingress gateway has no dataset mounts or application credentials.

## Worker pools

A worker container is still one project by default (above). For an organization with many
mostly-idle projects, a single worker's token can instead be authorized for *several*
projects -- a pool -- sharing a claim queue across them rather than running one idle
container per project.

Pooling is **always an explicit, matching declaration**, never an accidental consequence of
two projects ending up with the same token hash: give every pool member project the *same*
`worker_token_sha256` **and** the same `worker_pool: <name>` in server YAML. A shared hash
without a matching `worker_pool` on every member fails `buffdata`'s own startup validation --
pooling can't happen by copy-paste accident.

```yaml
projects:
  alpha:
    worker_token_sha256: "..."   # same value on every pool member
    worker_pool: teamx
  beta:
    worker_token_sha256: "..."   # same value as alpha's
    worker_pool: teamx           # same name as alpha's
```

Two things a pooled worker needs that a single-project one doesn't:

- **An aggregate volume mount** covering every pool member's project directory (the same
  shape the `api` service already uses, `./team/data:/data`), not a single project-scoped
  subpath. This is the real trade-off, stated plainly: **a pooled worker's container has
  filesystem access to every pool member's data, not just one.** The worker *process's* own
  code stays scoped to one project per claimed run (via the same `contained_path()` calls
  that already scope every dataset/run-directory access), but that's an application-level
  guarantee, not a second isolation layer underneath the container boundary -- there is no
  chroot, separate OS user, or mount namespace on the optimizer subprocess. Only pool projects
  that are comfortable sharing that container's filesystem access with each other.
- **Provider credentials resolved per project.** `provider_secrets.json` may stay the
  existing flat shape (`{"GEMINI_API_KEY": "..."}`) if every pool member intentionally shares
  one provider account, or become nested per project
  (`{"alpha": {"GEMINI_API_KEY": "..."}, "beta": {"OPENAI_API_KEY": "..."}}`) so pool members
  on different provider accounts don't share -- or leak -- each other's keys. The worker
  resolves and applies the right set fresh for every claimed run, clearing the previous run's
  keys first.

`BUFFDATA_WORKER_PROJECT` is no longer required or used by the worker process -- a pooled
worker's project is decided per claimed run, not fixed at container start.

**Not built here, explicitly deferred**: per-project run concurrency is unchanged by
pooling -- a project still gets at most one active run at a time, pool or not -- so nothing
about the existing single-active-run guarantee needs revisiting. A live Compose/container
smoke test of the aggregate-mount trade-off above (`deploy/smoke_test.py` currently asserts
the *opposite*, that a worker can't see another project's directory, for the non-pooled
case) has not been added; this has only been verified at the application layer
(`tests/test_team_server.py`), not against a real container's filesystem boundary.

## Rust worker (optional)

[`rust-worker/`](../rust-worker) is a from-scratch Rust reimplementation of the worker
container's claim loop and process supervisor (what `buffdata/server/worker.py` does today) --
same `/internal/worker/{claim,heartbeat,finish}` wire protocol, same per-project provider
secret isolation, same path-containment and process-group kill guarantees (see
[`rust-worker/README.md`](../rust-worker/README.md) for the full behavioral comparison). It
still shells out to `python -m buffdata.runs.executor` for the actual optimization work --
this replaces the poll-loop process, not the Python ML stack.

`deploy/Dockerfile.team` builds it in an additive `rust-worker-builder` stage and copies the
binary to `/usr/local/bin/buffdata-worker` in the final image; building the image is
behaviorally unchanged unless you opt in. To use it, change the `worker` service's `command:`
in `deploy/compose.yaml` from `[python, -m, buffdata.server.worker]` to `[buffdata-worker]` --
everything else (secrets, mounts, network policy, `BUFFDATA_ARTIFACT_ROOT`, proxy env vars)
stays the same. One addition: `BUFFDATA_PYTHON` (defaults to `python3` on `PATH`) tells it
which interpreter to invoke for `buffdata.runs.executor`, since a Rust binary has no
`sys.executable` of its own to fall back on the way the Python worker does.

Tested this session with `cargo test` (19 tests, including a real process-group kill that
also reaps a grandchild process) and a real end-to-end run: a live BuffData API server plus
this binary as the worker, claiming and executing real managed runs across a two-project
pool exactly like the Python worker's own live test. **Not** verified inside an actual Docker
build in this session (no Docker access) -- confirm the `rust-worker-builder` stage actually
builds, and re-run the promotion checks below, before using it in place of the Python worker
in a real deployment.

## Encryption at rest for managed artifacts

Off by default, additive to the existing file-permission protections
([security.md](security.md)) — nothing changes until `BUFFDATA_ARTIFACT_MASTER_KEY_SECRET`
is set. Covers the three published artifacts a successful run produces
(`output`/`rejected`/`report`, and `report_html` if present) — **not** checkpoints, which stay
plaintext + owner-only permissions as today; encrypting them would mean threading key material
into `buffdata/engine/pipeline.py`, used by every CLI command, not just managed runs, a
separate and larger change.

**What this actually protects against**: anyone with *only* filesystem or backup access (a
stolen disk, a leaked backup, a misconfigured volume mount) — not the running API or worker
processes themselves, which already have full filesystem access to every project's artifacts
by design (the API's aggregate artifact mount, above). Both may decrypt when they legitimately
need content for an existing feature: the worker to publish a run's artifacts, the API for
`compare`/artifact downloads.

Configure by setting `BUFFDATA_ARTIFACT_MASTER_KEY_SECRET` to the *name* of a secret in
whatever `SecretResolver` backend is already configured (`env`, `vault`,
`aws_secrets_manager`, `gcp_secret_manager`, `azure_key_vault` — see
[providers.md](providers.md#secret-backends)) — not the key material itself. That secret's
value must be a base64-encoded 32-byte key (AES-256). Each project gets its own randomly
generated data key on first use, wrapped by this master key and stored in its own database
table (never in the same row `PUT /api/v1/projects/{id}` read-modify-writes). The wrapped
key never leaves the database; the unwrapped data key is never written to disk anywhere —
it's handed to a run's worker subprocess only via that subprocess's own environment (the
claim response, then `supervise()`), and is gone when that process exits.

Not a real KMS wrap/unwrap integration — master-key material passes through the API/worker
processes' own memory to do the AES-GCM wrap/unwrap locally, unlike a true HSM-backed KMS
that never reveals key material. A real cloud-KMS-backed `KeyManager` implementation
(`buffdata/security/keys.py`) is a possible future addition, not attempted here since it
can't be tested against live cloud credentials in the environment this was built in.

## Reproducible isolated smoke test

```bash
docker build -t buffdata-team:local -f deploy/Dockerfile.team .
python deploy/smoke_test.py
```

Run under Linux/WSL with Docker access and `httpx`, `PyJWT`, `cryptography`, and `PyYAML`
installed. The harness reuses the production Compose definition with unique project names,
two synthetic projects, ephemeral loopback ports, temporary TLS certificates and a synthetic
OIDC issuer. It makes no paid provider calls and never loads real `.env` or team secret files.
The synthetic issuer deliberately auto-authorizes a test subject; it is never included in
the production image/configuration and must not be deployed as an identity provider.

Tests cover real PostgreSQL concurrent submissions/claims, 2,048 original/generated rows,
TLS/PKCE/JWKS login and CSRF/logout, project boundaries, a real worker kill and stale-lease
recovery, proxy/network controls, restart recovery, and quiescent backup/restore into a separate
database and copied artifact tree. Default cleanup removes only the randomly named test
containers/networks/volume; the private temporary directory retains results and backups.
`--keep` retains that synthetic stack for debugging; `--image` selects a locally built image.
The CI smoke workflow uploads only `results.json`, never temporary credentials or backups.

See the [dependency release blocker](dependency-release-blocker.md): passing deployment
tests is not production/security approval while that audit gate remains failing.

OIDC signing algorithms are explicitly configured (`algorithms: [RS256]` by default).
Configure another supported asymmetric algorithm before migrating an IdP that uses it.
JWKS redirects are rejected: configure the final HTTPS endpoint. Unknown signing keys trigger
a rate-limited refresh (`jwks_refresh_seconds`, default 5); repeated failures cannot force a
new fetch on every request. No stale JWKS document is accepted after its configured expiry.

## Remote CLI and endpoints

```bash
buffdata runs start train.csv --config pipeline.yaml --server https://buffdata.example.com \
  --project team --token-file /private/buffdata-access-token
buffdata runs list --server https://buffdata.example.com --project team --token-file /private/buffdata-access-token
```

The CLI uploads bytes and submits a registered dataset ID. API submission cannot set shell
commands, arbitrary filesystem paths, images, worker environment variables, or credentials.
Use `Idempotency-Key` for `POST /api/v1/runs`. The API includes project listing/membership
updates, bounded dataset uploads/listing/deletion, storage usage, run list/detail, events,
comparison, verification, artifact downloads, cancellation, and explicit resume.

Browser upload/download uses the existing single-file dataset formats. Local SDK execution
also supports Hugging Face directories; browser directory upload/download is not implemented.
Set `report_html` in the unchanged pipeline configuration for standalone escaped reports.

## Required promotion checks

- Complete the image build and run `pip check` plus dependency/secret scanning.
- Sign in through the real IdP; verify logout, expiry, CSRF, and revoked membership behavior.
- Submit two runs concurrently and verify only one worker claims a project at a time.
- Verify another project's token cannot access a run, dataset, event, or artifact.
- Cancel during execution, stop/restart a worker, and resume explicitly; confirm no partial
  output is advertised as successful and the prior attempt remains available in history.
- From the worker, verify direct internet and metadata access fail while approved provider
  calls through the proxy work. Confirm the database is absent, and confirm the project
  volume mount covers exactly the intended project (or, for a pooled worker, exactly its
  declared pool members) and nothing else.
- Back up PostgreSQL with `pg_dump` and the artifact tree while submissions/workers are
  paused; restore to a separate deployment and verify all manifests. Keep encrypted backups
  under operator-managed retention. Secret backups must follow the organization's key policy.

Rollback is additive: stop server/worker services, retain metadata and artifact volumes, and
continue using the legacy CLI. Do not use `docker compose down -v` to roll back. There is no
automatic retention deletion and no downgrade migration that drops run history.

## Limits

This is a single-organization deployment, not a sandbox for hostile plugin authors or database
administrators. Same-project worker code is trusted. TLS ingress, filesystem encryption,
IdP setup, secret rotation, and backup operation are deployment responsibilities. The database
container's bootstrap account is not a substitute for a separately provisioned least-privilege
production database role. Do not call this deployment production-validated until the checks
above have actually run in the target environment.
