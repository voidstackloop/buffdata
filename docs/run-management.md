# Secure run management

BuffData's managed executor wraps the existing optimizer; it does not change stage order,
prompts, model defaults, scoring/refinement, deduplication, classification, PII rules, or
accuracy-gate criteria. Its policy can reject a run. It never silently switches settings
or truncates a dataset to fit a quota.

## Local CLI

```bash
buffdata runs start input.parquet --config pipeline.yaml
buffdata runs list
buffdata runs show RUN_ID
buffdata runs compare RUN_ID
buffdata runs verify RUN_ID
buffdata runs cancel RUN_ID
buffdata runs resume RUN_ID
buffdata runs lineage RUN_ID
buffdata runs webhooks add|list|remove
```

Use `--root /private/run-storage` or `BUFFDATA_RUNS_HOME` to select storage (default
`~/.local/share/buffdata/runs`). `--output-format` accepts the existing dataset formats.
For the existing accuracy gate, add `--validation-file holdout.jsonl --require-positive-gain`.
Ordinary `optimize`, `pipeline`, and individual-stage commands remain available.

The typed SDK entry points are `buffdata.runs.RunSpec`, `RunManifest`, `RunStatus`,
`SecurityPolicy`, `buffdata.runs.service.RunService`, and `buffdata.runs.runner.run_local`.
Create a project with `service.store.ensure_project(...)` before using it through the SDK.
Authorization of an embedding application's callers remains that application's responsibility.
The server performs verified project authorization before accessing the service.

## State, provenance, and recovery

Submission creates a queued run with immutable registered input hashes. A transaction claims
at most one running attempt per project. A supervisor enforces the deadline, checks
cancellation, and stops its subprocess on control-plane failure. The optimizer writes into
a unique attempt directory. The control plane verifies all required artifact hashes before
publishing a successful manifest; downloads require a successful run and re-verification.
Cancellation wins over a racing completion. Failed/cancelled artifacts are not published.

`resume` creates a new attempt, preserving the original attempt and events. Inputs, resolved
configuration, source digest, installed dependency versions, plugin inventory, and policy
must match. A matching checkpoint is copied into the new attempt. A still-live worker
cannot be resumed; a heartbeat older than 120 seconds can be explicitly interrupted/fenced.
There is no automatic retry of billable work. Provider calls interrupted before checkpointing
can be repeated and charged again. Exactly-once provider execution is not promised.

The original input for accuracy evaluation is parsed independently from the objects passed
to the optimizer. Comparisons retain duplicate multiplicities and display measured accuracy
only when an evaluation actually exists. Content deltas are not inferred causal lineage;
field comparisons use matching unique IDs and do not guess after row deletions.

Local metadata is SQLite; team metadata uses PostgreSQL. Tables are additive (`managed_*`).
Events have no application update/delete API. This is not tamper-proof against database
administrators or the OS account controlling local files. Existing `audit record` imports
still work and explicitly label imported reports rather than reconstructing execution events.

## Lineage and webhooks

`resume` has always recorded `parent_run_id` on the new attempt. `runs start --parent-run
RUN_ID` (or `parent_run_id` on `POST /api/v1/runs`) records the same field for an ordinary,
non-resume submission, asserting "this run was derived from that one" -- like run
comparisons above, this is caller-asserted provenance, never inferred from content. An
unknown parent (wrong ID, wrong project) is rejected before the run is created.

```bash
buffdata runs lineage RUN_ID
```

Walks `parent_run_id` in both directions: ancestors (exact, however deep) and descendants
(a scan of the project's most recent 1000 runs -- the same bound `runs list` and dataset
deletion's reference check already use, so a descendant older than that window won't
surface). `GET /api/v1/runs/{id}/lineage` is the server equivalent.

```bash
buffdata runs webhooks add https://example.com/hook --event run.succeeded --event run.failed
buffdata runs webhooks list
buffdata runs webhooks remove WEBHOOK_ID
```

A project administrator (`POST /api/v1/projects/{id}/webhooks` requires the `administrator`
role, same as `PUT .../projects/{id}`) registers a URL and the terminal events it should
receive (`run.succeeded`, `run.failed`, `run.cancelled`, `run.interrupted`). Registration
requires HTTPS and a publicly-routable address -- the same check any other BuffData-initiated
egress target gets, so a loopback, link-local, or RFC1918 address (a cloud metadata endpoint,
for instance) is rejected up front. That check runs once, at registration; it does not
re-pin the address on every later delivery -- DNS-rebinding-resistant re-resolution at
dispatch time is a known gap, not a promise.

Delivery is best-effort and at-least-once, fired after a run has already reached its terminal
state so a slow or dead endpoint can never delay or fail the run itself -- backgrounded on the
server, and recorded either way (`webhook.delivered` / `webhook.delivery_failed`) as a run
event. The payload (`event`, `run_id`, `project_id`, `status`, `occurred_at`) is signed with
HMAC-SHA256 over a per-webhook secret returned once, at creation
(`X-BuffData-Signature: sha256=...`); `runs webhooks list` never shows it again.

## Security policies

```yaml
approved_plugins: []
allowed_hosts: [generativelanguage.googleapis.com, api.openai.com, api.anthropic.com]
local_endpoints: []
allowed_cloud_prefixes: []
read_roots: []
write_roots: []
max_upload_bytes: 1073741824
max_expanded_bytes: 8589934592
max_record_bytes: 16777216
execution_seconds: 21600
subprocess_seconds: 120
```

Managed runs use their project policy. For legacy commands, put `--security-policy policy.yaml`
before the command name, or set `BUFFDATA_SECURITY_POLICY`. Optional verified CLI authorization
uses global `--identity-token-file`, `--identity-oidc-config`, and `--identity-policy` together;
`-` as token file reads stdin. Existing `--actor` remains a local trust assertion, not server
authentication. A policy file controlled by the same local caller is not a privilege boundary.

Installed validator/PII plugins now require explicit approval before import, using
`buffdata.validators:NAME` or `buffdata.pii_recognizers:NAME`. Legacy callers can set
`BUFFDATA_APPROVED_PLUGINS` to a comma-separated list. Missing, broken, or raising approved
plugins fail closed; they are never silently skipped. Review a plugin's code before approval.
No optimizer plugin is installed or enabled by this release.

An approved plugin's own code no longer runs in the main process. As soon as any plugin is
approved, `buffdata/security/sandbox.py` transparently routes it into a long-lived subprocess
with no network access (the same audit-hook guard `network_policy=strict` uses, unconditional
here), an environment allowlist excluding every secret, and no filesystem access beyond what
Python/Presidio need to import -- the approval gate, caching, and fail-closed behavior in
`buffdata/plugins.py` are otherwise unchanged. This is defense in depth for trusted Python
dependencies, the same class of protection the network guard's own docstring claims for
itself, not a native-code sandbox -- container-level isolation (see
[team-deployment.md](team-deployment.md)) is still the boundary for that. A PII-recognizer
plugin that overrides `analyze()` to consume spaCy `nlp_artifacts` directly will see `None`
under the sandbox; the common `PatternRecognizer`-shaped case (regex/deny-list matching) is
unaffected. Presidio's own built-in engine is unrelated to this and keeps running in-process.

Managed subprocesses apply a socket audit guard, including third-party model downloads.
`strict` forbids job network access; the remote CLI/API control transport is separate and
does require a network. Use the local CLI for an end-to-end offline workflow. Local endpoints
need explicit approval. Public URL fetching checks DNS and every redirect using pinned
addresses and bounded reads. Native-code attacks are not contained by a Python audit hook;
the team deployment also isolates project volumes and restricts container egress -- stated
precisely: that isolation is a **container/volume-mount boundary**, not a second layer
underneath it. There is no chroot, separate OS user, or mount namespace on the optimizer
subprocess `runner.supervise()` spawns; anything that doesn't go through BuffData's own
`contained_path()` calls has exactly the filesystem access the container's mounts give it.
A [pooled worker](team-deployment.md#worker-pools) deliberately widens that mount to cover
more than one project, which is exactly this boundary, punched open by design for the
projects that opt into it -- not a second guarantee sitting behind it.

Logs do not contain subprocess stdout/stderr or raw dataset excerpts. Provider failures produce
sanitized error categories rather than copies of SDK exception text. HTML reports escape
content; the API serves datasets/reports as downloads, never executable previews. Exact known
credentials and common credential patterns are redacted from operational messages only.

## Team deployment and current validation boundaries

See [Compose deployment](team-deployment.md). The release is for one trusted organization,
not public multi-tenant hosting. No dataset is automatically deleted. The storage endpoint
reports usage; explicit administrator deletion requires matching confirmation and refuses
datasets referenced by run history. Referenced originals are retained for verification/resume.

The CPU-image dependency closure is version-pinned in `deploy/requirements-cpu.lock`,
derived from the tested environment with locked build tools; base images are digest-pinned.
The lock is not yet artifact-hash-pinned. The [dependency audit is currently blocked](dependency-release-blocker.md)
by an upstream PII-package constraint; no PII behavior has been changed to bypass it. Production
promotion requires a completed image build, dependency audit, real OIDC sign-in, PostgreSQL
concurrency/backup-restore checks, and the Compose isolation smoke test; unit tests do not
substitute for these deployment checks.
