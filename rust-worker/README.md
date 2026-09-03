# buffdata-worker (Rust)

A drop-in replacement for [`buffdata/server/worker.py`](../buffdata/server/worker.py): the
claim loop and process supervisor for [team-deployment](../docs/team-deployment.md)'s managed
runs. This binary owns none of the actual dataset-optimization logic -- it still spawns
`python -m buffdata.runs.executor <directory>` to do that work, exactly like the Python
worker does. All this replaces is the *worker process itself*: polling
`/internal/worker/claim`, applying per-project provider secrets, supervising the executor
subprocess (heartbeat, cancellation, deadline, kill-on-the-way-out), and reporting
`/internal/worker/{run_id}/finish` -- rewritten as a small, dependency-light binary instead
of a Python process holding its own copy of every ML dependency just to run a poll loop.

Same wire contract as the Python worker (see `buffdata/server/app.py`'s
`/internal/worker/{claim,heartbeat,finish}` handlers) -- a server config does not need to
know or care which implementation is polling it. A claimed run's data key (encryption at
rest, opt-in -- see [`buffdata/security/keys.py`](../buffdata/security/keys.py)) never
touches this process's own persistent environment or disk; it is forwarded straight into the
executor subprocess's environment for that one run only, matching the Python worker's own
guarantee.

## Build

```bash
cargo build --release
```

Produces `target/release/buffdata-worker` (~2.5MB, stripped). Requires a Rust toolchain
(1.75+; developed and tested against 1.89) -- nothing else. `cargo test` runs the full suite,
including real subprocess spawn/kill/process-group tests (Unix only for the process-group
tests; the rest are cross-platform).

## Configure (environment variables)

Identical names to the Python worker, plus one Rust-specific addition:

| Variable | Required | Meaning |
|---|---|---|
| `BUFFDATA_SERVER_URL` | yes | Base URL of the BuffData API, e.g. `https://buffdata.internal:8000` |
| `BUFFDATA_WORKER_TOKEN_FILE` | yes | Path to a file containing the raw worker bearer token |
| `BUFFDATA_ARTIFACT_ROOT` | no (default `/data`) | Root of the mounted artifact tree |
| `BUFFDATA_PROVIDER_SECRETS_FILE` | no | Path to the provider-secrets JSON (flat or per-project nested shape -- see team-deployment.md#worker-pools) |
| `BUFFDATA_PYTHON` | no (default `python3` on `PATH`) | The Python interpreter to invoke for `buffdata.runs.executor`. **Not present in the Python worker**, which always re-execs via its own `sys.executable`; a Rust binary has no such interpreter of its own, so this must point at the same environment `buffdata[server]` is installed into. |

## What's intentionally identical to the Python worker

- **Claim loop**: `POST /internal/worker/claim` every 2s when idle; on a control-plane or
  preflight error, back off 5s and retry (the claimed run, if any, is deliberately left for
  the server's own stale-lease recovery -- no cleverness added here beyond what
  `worker.py` already does).
- **Provider secrets, per project**: `secrets_for_project()` -- flat shape shared across every
  pool member, or a per-project nested shape so pooled projects on different provider
  accounts never see each other's keys. Every known secret key is cleared before this run's
  are applied, so a previous claimed run's credentials (a different pooled project) can never
  leak into this one's subprocess environment.
- **Path containment**: the claimed run's `run_id` is validated against the same
  `[a-f0-9]{32}` shape `buffdata/runs/service.py::run_directory()` enforces, and the resulting
  directory is checked against the same no-symlink, stays-inside-root rule as
  `buffdata/security/policy.py::contained_path()`, before anything touches it.
- **Supervision**: 500ms heartbeat/poll cadence, SIGTERM-then-SIGKILL-after-30s process*group*
  kill (not just the direct child -- a grandchild the optimizer subprocess spawns is killed
  too), and the executor's stdout/stderr are never captured (`Stdio::null()` for both) --
  provider SDK exceptions or dataset excerpts can contain secrets, and this worker has no
  more business logging them than the Python one did.
- **Egress isolation**: the worker's own control-plane HTTP client ignores
  `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` (`httpx`'s `trust_env=False`, mirrored here via
  `.no_proxy()`) and never follows redirects -- only the executor *subprocess* inherits those
  proxy variables, to actually reach providers through the egress allowlist.

## What's different

- **No manifest parsing.** The Python worker's shared `supervise()` helper (in
  `buffdata/runs/runner.py`) also serves the local-CLI path, which needs the parsed manifest
  object to call `verify_manifest()` itself. The *server*-attached worker never used that
  return value (`status, _ = supervise(...)`) -- the API reads `candidate-manifest.json`
  itself from the shared mount when `/finish` is called. This binary reflects that: it never
  reads or parses the manifest at all.
- **No implicit interpreter discovery.** See `BUFFDATA_PYTHON` above.

## Deploying it

This binary still needs the *same* Python environment the API/worker image already has
installed (it invokes `buffdata.runs.executor` as a subprocess) -- it is not a replacement for
that environment, only for the poll-loop process that drives it. See
[`deploy/Dockerfile.team`](../deploy/Dockerfile.team)'s `rust-worker-builder` stage and
[`docs/team-deployment.md`](../docs/team-deployment.md#rust-worker-optional) for how to build
it into the existing team image and swap it in via Compose.
