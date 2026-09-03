# System design: next phase — new features, security, performance

Status: proposal, not yet built. This document designs the next increment on top of
the system described in [architecture.md](architecture.md), [security.md](security.md),
[governance.md](governance.md), [run-management.md](run-management.md),
[team-deployment.md](team-deployment.md), and [scaling.md](scaling.md). It does not
re-describe what those already cover; it starts from the gaps those documents
themselves disclose and from reading the current implementation
(`buffdata/engine/pipeline.py`, `buffdata/optimizers/dedup.py`,
`buffdata/runs/*`, `deploy/compose.yaml`).

## 1. Requirements gathering

### What's already true (baseline, not in scope to rebuild)

- Single-process, 7-stage pipeline (`validate → pii → profile → dedup →
  score_refine → filter → classify`), checkpointed per stage.
- Provider-neutral LLM client with a `network_policy` kill switch
  (`unrestricted` / `local` / `strict`).
- Managed run control plane: immutable input snapshots, verified artifact
  manifests, cancel/resume, RBAC (`governance/access.py`), OIDC auth
  (`governance/oidc.py`), audit log, Data Contracts, SBOM.
- Team deployment: Compose stack, OIDC dashboard, egress-proxied workers, one
  worker container per project, PostgreSQL-backed metadata.
- Horizontal scale today is **batch-only**: `buffdata shard` + an external
  orchestrator or Ray run N independent single-process pipeline invocations
  and `buffdata merge` recombines them. The managed server itself does not
  scale this way.

### New functional requirements (this proposal)

1. **Dataset lineage & versioning** — track parent→child relationships across
   `optimize`/`generate`/`resume` runs so a team can answer "what dataset did
   this model train on, and what produced it" without re-deriving it from
   `report.json` files by hand.
2. **Event notifications** — a run reaching `succeeded`/`failed`/`cancelled`
   should be able to push a webhook, not just be polled via `runs show`.
3. **Autoscaled worker fleet** — a project should be able to run more than one
   concurrent job without an operator manually adding a dedicated worker
   container per project (current model, per team-deployment.md).
4. **Streaming-capable dedup/scoring for very large single files** — today
   `Deduplicator` methods take `List[DatasetItem]` in memory; very large
   single-file inputs (not sharded) have no bounded-memory path.
5. **Plugin sandboxing** — approved validator/PII plugins (governance.md's
   plugin-approval gate) still run in-process today; a malicious *approved*
   plugin has full process access.

### Non-functional requirements

- **Security**: close the currently-blocked dependency audit
  ([dependency-release-blocker.md](dependency-release-blocker.md)), add
  encryption at rest for managed artifacts (explicitly out of scope in
  security.md for the local CLI, but the team deployment handles regulated
  data and doesn't have this yet), and contain plugin/worker blast radius
  beyond process-level socket auditing.
- **Performance**: bound peak memory for the dedup/scoring stages
  independent of `--concurrency`/`--max-rpm` tuning, and let a project's
  throughput scale with worker count instead of being capped at one
  in-flight run per project.
- **Constraints**: this is still a single-organization deployment (explicit,
  repeated limit in team-deployment.md) — nothing here proposes public
  multi-tenant hosting. Postgres is the only supported team metadata store.
  No change to stage order, prompts, or accuracy-contract semantics is in
  scope — those are the product's core guarantee and stay untouched.

## 2. High-level design

```
                              ┌─────────────────────────────┐
                              │   OIDC-authenticated API      │
                              │   (buffdata/server/app.py)    │
                              └───────────────┬───────────────┘
                                              │ enqueue (existing: managed_runs table)
                                              ▼
                              ┌─────────────────────────────┐
                              │ PostgreSQL (managed_* tables) │  ← existing
                              │ + NEW: managed_run_lineage    │
                              │ + NEW: managed_webhooks       │
                              └───────────────┬───────────────┘
                                              │ claim (existing SELECT ... FOR UPDATE SKIP LOCKED)
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
            worker replica 1          worker replica 2          worker replica N   ← NEW: was 1/project
        (buffdata.server.worker)  (buffdata.server.worker)  (buffdata.server.worker)
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                              ┌─────────────────────────────┐
                              │ OptimizationPipeline (per-run  │  ← existing, unchanged
                              │ subprocess, unchanged stages)  │
                              │  dedup/score_refine: NEW       │
                              │  bounded-memory batch mode      │
                              └───────────────┬───────────────┘
                                              │ on terminal status
                                              ▼
                              ┌─────────────────────────────┐
                              │ NEW: webhook dispatcher        │
                              │ (at-least-once, HMAC-signed)   │
                              └─────────────────────────────┘
```

Three additive pieces, each independent and individually revertible:

- **Worker fleet + claim fairness** (performance) — turn "one worker
  container per project" into a shared pool that claims across projects,
  with a per-project concurrent-run cap enforced in the claim transaction
  (not in infrastructure) so one project can't starve another.
- **Lineage + webhooks** (features) — pure additive tables and a dispatcher
  process; zero changes to the optimizer or existing manifest verification.
- **Bounded-memory dedup batches + plugin sandbox** (performance + security)
  — internal to `OptimizationPipeline`/`Deduplicator` and the managed
  subprocess launcher; no API or schema change.

## 3. Deep dive

### 3.1 New feature: dataset lineage

**Status: implemented.** `runs lineage RUN_ID` / `GET /api/v1/runs/{id}/lineage` and an
explicit `--parent-run` / `parent_run_id` on submission — see
[run-management.md](run-management.md#lineage-and-webhooks). Descendant discovery uses the
same 1000-run scan bound described below, not a dedicated `managed_run_lineage` table (kept
the schema additive-only rather than adding an indexed join table for this pass).

`managed_run_lineage(run_id, parent_run_id, relation)` — `relation` is
`"resume"` (already implicit today via `resume`'s attempt history, just not
queryable across runs) or `"derived_from"` (new: an explicit `--parent-run
RUN_ID` flag on `runs start`, recorded alongside the immutable input hash
that already exists). `runs show RUN_ID --lineage` walks the table both
directions. This is deliberately *not* automatic content-based lineage
inference — run-management.md is explicit that "content deltas are not
inferred causal lineage"; this feature only records lineage a caller
asserts, the same trust model `--parent-run` implies.

API: `GET /api/v1/runs/{id}/lineage`. SDK: `RunService.lineage(run_id)`.

### 3.2 New feature: webhooks

**Status: implemented**, with one scope narrowing — see
[run-management.md](run-management.md#lineage-and-webhooks): delivery is backgrounded on the
server and synchronous-but-non-blocking on the local CLI (not a separate dispatcher process
reading a `NOTIFY` queue), and retry is fixed at 3 attempts with a short fixed backoff rather
than jittered dead-lettering — both still record `webhook.delivered`/`webhook.delivery_failed`
as run events either way.

`managed_webhooks(project_id, url, secret, events[])` registered via
`POST /api/v1/projects/{id}/webhooks` (RBAC-gated, same `use_external_providers`-
style permission model as provider calls, since a webhook URL is itself an
egress target). On a run reaching a terminal state, the dispatcher:

- Signs the payload (`run_id`, `status`, `manifest_hash`, timestamp) with
  HMAC-SHA256 using the registered secret — same verification story OIDC
  already establishes for inbound auth, mirrored for outbound.
- Retries with jittered backoff, dead-letters after N attempts into an
  existing `managed_events`-style row so `runs show` still surfaces delivery
  failure without a separate monitoring system.
- Runs as its own process reading a `NOTIFY`/polling queue off the
  run-completion transaction — not inline in the worker, so a slow or dead
  webhook endpoint can never delay run completion or hold a worker slot.

Egress for webhook URLs must go through the same allowlisting proxy pattern
already used for provider calls (`allowed_hosts` in the security policy),
not a bare outbound call from inside the trusted network segment —
otherwise a registered webhook becomes an SSRF pivot into the internal
network the workers/API already sit on.

### 3.3 Performance: worker fleet + fair claiming

**Status: implemented**, narrower and hardened relative to the original proposal below --
see [team-deployment.md](team-deployment.md#worker-pools). A design-review pass (before
implementation) surfaced real issues the first draft missed, all fixed before anything was
built: pooling is now an *explicit* `worker_pool` declaration required on every project that
shares a worker token (startup-validated, never an accident of two hashes matching), the
claim scan order is randomized rather than fixed/sorted (a fixed order provably starves
whichever project sorts last under sustained backlog -- proven with a concrete trace, not
just asserted), the claim-preflight-failure handler binds to the actually-claimed project's
own `run["project_id"]` rather than any scan-loop state, and provider credentials are
resolved per claimed run (flat-shared or nested-per-project `provider_secrets.json`) with a
clear-then-set sequence, so pooled projects on different provider accounts can't leak
credentials into each other's runs. `RunStore.claim()`'s "at most one active run per
project" invariant is completely untouched -- confirmed as the reason §3.8's proposed
`max_concurrent_runs` cap turned out to be unnecessary here, see below. No live
Compose/container test of the resulting aggregate-mount trade-off was added -- see
team-deployment.md's own "not built here" note.

Today: `deploy/compose.yaml` defines exactly one `worker` service; team-deployment.md
says to "add a dedicated worker, token, and project-only volume mount for
*each* project." That means concurrency is capped at 1 running attempt per
project by construction (one worker, one project mount) and scaling out
means an operator hand-adds Compose services.

Proposed: keep the per-project volume/token isolation (that's a real
security boundary, not incidental), but let `docker compose up --scale
worker=N` (or a Helm `replicas` field — `deploy/helm/buffdata` already
exists) run N interchangeable workers that each mount *all* approved
project volumes read-write-scoped by project ID at claim time, rather than
one static mount per container. The claim transaction (already `SELECT ...
FOR UPDATE SKIP LOCKED` per run-management.md's "at most one running attempt
per project") gets one added clause: skip a project that already has a
running attempt claimed by *any* worker, so the existing single-attempt-
per-project guarantee is preserved while N workers share the queue across
projects.

Trade-off this creates and accepts: a worker process now has access to
every approved project's volume mount, not just one. That's a real
reduction from the current per-project container isolation. Mitigation:
each worker subprocess (the existing per-run subprocess launch, unchanged)
is `chroot`/bind-mount-scoped to only the claimed project's directory at
launch time, so the *container* has broad mounts but no single *run* ever
sees more than its own project's data — the isolation boundary moves from
container to subprocess, which is a weaker guarantee and should be called
out explicitly to whoever signs off on this deployment, not silently
assumed equivalent.

### 3.4 Performance: bounded-memory dedup/score batches

**Status: narrower than proposed below, done.** Building this surfaced a real correction to
the premise: `OptimizationPipeline.run()` (`buffdata/engine/pipeline.py`) reads the whole
input into one in-memory `List[DatasetItem]` and passes that *same* full list through all 7
stages, checkpointing the full accepted/rejected lists after each one — windowing just
`Deduplicator`'s internal per-item loop, as first proposed below, would not have reduced
peak memory at all, since the caller already materializes the full list before dedup runs.
True pipeline streaming would mean restructuring every stage, not just dedup — confirmed
out of scope, and not attempted.

What shipped instead, scoped down deliberately: `deduplicate_minhash` stored one growing
`Set[str]` of raw shingle text per kept item — for long texts this accumulator (held for the
whole file) was the dominant real memory cost of the stage. It now stores 64-bit hash
fingerprints (`xxhash`, already used for exact dedup in the same module) instead of the raw
strings — same Jaccard intersection/union computation, same threshold, same decisions,
verified against the original string-based implementation as a reference oracle across
parametrized thresholds and randomized text (`tests/test_optimizers.py`). Not opt-in — unlike
a windowing change, this doesn't alter processing order or the accuracy-contract-proven
whole-file behavior, only the in-memory representation of an intermediate computation.
`deduplicate_exact`'s hash set was already compact; `deduplicate_semantic_local`'s harder
ANN-index rework (below) is still untouched.

`Deduplicator.deduplicate_exact/minhash/semantic_local` (`buffdata/optimizers/dedup.py`)
take the full `List[DatasetItem]` — fine for sharded/batch use (scaling.md's
model), but a single large unsharded file has no bounded-memory path through
the managed pipeline today.

Proposed: an opt-in `PipelineConfig.dedup_batch_size` that chunks the
validated/PII-clean stream into fixed-size windows for exact and MinHash
dedup (both are order-independent set operations, so windowed processing
with a persisted hash/shingle index across windows preserves identical
results to the current whole-file pass) and streams accepted rows to
output incrementally instead of holding the full accepted list until the
end. Semantic dedup (`deduplicate_semantic_local`, sentence-transformers)
is the harder case — it needs an approximate-nearest-neighbor index (e.g.
an on-disk FAISS index) rather than an in-memory O(n²) or exhaustive
compare, and is scoped as a **separate, larger follow-up** rather than
folded into this batch-size change, since it changes an algorithm, not just
memory layout — flagging it here rather than under-scoping it silently.

Default stays "load it all" (current behavior, unchanged) — this is opt-in
because the accuracy-contract guarantees in architecture.md are proven
against the current whole-file behavior, and this doc does not propose
re-proving them under windowing without that work actually happening first.

### 3.5 Security: close the dependency-audit blocker

**Status: implemented**, narrower than the isolation workstream sketched below — see
[dependency-release-blocker.md](dependency-release-blocker.md)'s updated status. Verified
directly before building anything: `presidio-analyzer` (detection) has no `cryptography`
dependency at all -- only `presidio-anonymizer` (redaction) does, so only *that* package
needed to move into an isolated venv. `presidio-analyzer` stays in the main environment
completely unaffected, including third-party PII-recognizer plugins (§3.6), which subclass
it. The wire boundary that actually shipped carries `analyzer_results` (spans: entity type,
start, end, score) *in* and redacted text out — wider than this section's original "entity
counts out" sketch, since the isolated venv has to do the actual substitution, not just
report what it found. A real structural blocker surfaced during design and is why the worker
side isn't literally the plugin sandbox's `sandbox_worker.py`: `buffdata/__init__.py` eagerly
imports most of the base dependency closure, so *any* `buffdata.*` import inside a worker
process would defeat an intentionally minimal isolated venv. Fix:
`buffdata/security/anonymizer_worker.py` is a standalone script with zero `buffdata` imports,
run by file path rather than `-m`, sharing only the *parent-side* `PluginSandbox` class (now
accepting a configurable interpreter and worker command) with the plugin sandbox -- one
utility managing two worker scripts, not the "one bespoke sandbox" the recommendation below
first imagined, but avoiding two separate subprocess-lifecycle/wire-protocol implementations,
which was the actual goal. Packaging changes (pyproject.toml, both lock files, a new
`Dockerfile.team` build stage) were verified against a real `docker build` of
`Dockerfile.team` -- `pip check` passed cleanly with `cryptography==50.0.1` in the main
closure, and containers run from the built image confirmed the isolated-venv redaction path
actually works (see dependency-release-blocker.md's updated status) -- Docker turned out to
be available in this environment after all, stronger verification than initially expected. A
`pip-audit` advisory scan against the built image's locks is still outstanding.

[dependency-release-blocker.md](dependency-release-blocker.md) is a real,
currently-open release blocker: `cryptography==48.0.1` has three open
advisories, but `presidio-anonymizer==2.2.364` pins `cryptography<49.0.0`,
so the fix and the PII dependency conflict. Two real paths, not a false
choice:

- **Vendor-track**: open/track the upstream Presidio issue for a
  `cryptography>=49` compatible release; this is the clean fix but is
  outside this repo's control and has no committed timeline.
  it's genuinely blocked.
- **Isolation workstream** (what dependency-release-blocker.md calls out as
  the alternative, explicitly not yet authorized): run the PII/Presidio
  dependency closure in a separate process/venv from the rest of the
  pipeline, communicating over a narrow serialized interface (the redacted
  text in, entity counts out — nothing else Presidio touches needs to cross
  that boundary). That decouples Presidio's `cryptography` pin from the
  rest of BuffData's dependency closure, unblocking the `cryptography>=49`
  upgrade everywhere else immediately.

Recommendation: start the isolation workstream now rather than wait on
upstream — it also directly improves the plugin-sandboxing story in §3.6
below, since a subprocess boundary for Presidio is the same mechanism a
plugin sandbox needs, not two separate pieces of work.

### 3.6 Security: plugin execution sandbox

**Status: implemented** for third-party validator/PII-recognizer plugins — see
[run-management.md](run-management.md#lineage-and-webhooks) and
`buffdata/security/sandbox.py`/`sandbox_worker.py`. Two differences from the proposal below,
both decided during implementation: no seccomp-bpf profile (OS-level sandboxing was out of
scope for this pass — network denial is the existing Python-level audit-hook guard, reused
unconditionally, which the guard's own docstring already frames as "defense in depth for
trusted Python dependencies, not a native-code sandbox"), and PII recognizer plugins are
sandboxed for both construction *and* every `analyze()` call (not just loading) — Presidio's
`RecognizerRegistry.add_recognizer()` requires a real local object, so a lightweight proxy in
the main process forwards each `analyze()` call to the resident recognizer in the sandbox.
Presidio's own built-in engine is unaffected (§3.5 above).

Governance.md's plugin-approval gate (`buffdata.validators:NAME`,
`BUFFDATA_APPROVED_PLUGINS`) stops an *unapproved* plugin from loading, but
an approved plugin still runs with full in-process access — the same
process holding decrypted secrets (post-resolution, in memory) and the
PII-bearing pre-redaction checkpoint.

Proposed: run approved third-party validator/PII plugins in a subprocess
with a restricted syscall surface (Linux: a `seccomp-bpf` profile denying
network syscalls entirely — plugins should never need network access, that's
what `network_policy` already gates for the core pipeline — plus no
filesystem access outside a scratch directory), communicating over a
length-prefixed stdin/stdout protocol carrying only the record text and the
plugin's structured verdict. This reuses exactly the isolation pattern
proposed for Presidio in §3.5, so both should share one subprocess-sandbox
utility (`buffdata/security/sandbox.py`, new) rather than two bespoke
implementations. Team-deployment.md already establishes that "native-code
attacks are not contained by a Python audit hook" and that container-level
isolation is the real boundary for the team deployment — this brings that
same reasoning down to the single-process CLI/library case, where there's
currently no boundary at all below the plugin-approval gate.

### 3.7 Security: encryption at rest for managed artifacts

**Status: implemented**, narrower than proposed below and with two structural fixes a
design-review pass required before anything was built — see
[team-deployment.md](team-deployment.md#encryption-at-rest-for-managed-artifacts). Scope
correction, confirmed with the user: covers `output`/`rejected`/`report` only, not
checkpoints (those are written/read entirely inside `buffdata/engine/pipeline.py`, used by
every CLI command — encrypting them means threading key material far outside the managed-run
boundary, deferred separately). "Decrypted only inside the worker subprocess" doesn't hold —
`RunService.compare()` and the artifact-download endpoint read content from the control plane
too — corrected to "decrypted by whichever trusted BuffData process legitimately needs it,"
since both already have full filesystem access to every project's artifacts by design; the
real protection is against filesystem/backup-only access, not the running application. Master
key is not a live KMS wrap/unwrap integration (untestable here without cloud credentials) —
reuses the existing `SecretResolver` abstraction to fetch key material, then wraps/unwraps
locally with AES-GCM, a disclosed trade-off against a true HSM-backed KMS.

The first draft of this design was unsafe, not just imprecise: `compare()` would have leaked
a fresh plaintext tempfile on every call with no cleanup path at all, and the plan to hand a
run's plaintext data key to its worker via `execution.json` (so the worker, which has no
DB/KMS access, doesn't need to fetch it) never got cleaned up — the raw key would have sat on
disk forever, next to the ciphertext it unlocks, defeating the feature against exactly the
adversary it exists for. Both closed structurally: `compare()`'s decryption happens inside a
`tempfile.TemporaryDirectory()` whose cleanup is a real Python context-manager guarantee (runs
on exceptions too, unlike a Starlette `BackgroundTask`, confirmed by reading Starlette's own
source during review); the data key is never written to a file at all, only ever passed
through a claim-response body and a subprocess's own environment, both gone when the run
ends. A third finding — the wrapped key living in the same loosely-guarded
`managed_projects.body` blob `PUT /api/v1/projects/{id}` already read-modify-writes unlocked
— is why it has its own additive table (`managed_project_keys`) instead.

Security.md deliberately does *not* encrypt local CLI files at rest
(reasoned: no key-management story for a `pip install`ed tool, file
permissions are the standard mitigation, same as `~/.ssh`). That reasoning
does not carry over unchanged to the **team deployment** — it already has a
KMS-adjacent boundary in `provider_secrets.json` handling and a real
operator, so "no key to hang encryption off of" isn't true there. Proposed,
team-deployment only: envelope-encrypt attempt-directory artifacts
(checkpoints, rejected rows, reports — the same set security.md's
`restrict_to_owner()` already flags as sensitive) with a per-project data
key wrapped by a KMS/Vault master key, decrypted only inside the worker
subprocess for the duration of a run. This is additive to file permissions,
not a replacement, and explicitly scoped to the managed/team deployment —
the local single-user CLI story in security.md is unchanged and its
reasoning still holds there.

### 3.8 Performance: claim-fairness and provider-call batching

**Status: split.** Claim-fairness shipped as part of §3.3 (random scan order across a pool,
not the fixed order this section originally didn't specify) -- and the
`max_concurrent_runs` cap below turned out to be unneeded: it only guards against a
per-project concurrency *increase*, and §3.3's implementation deliberately left
`RunStore.claim()`'s one-active-run-per-project invariant completely untouched, so the burst
scenario this bullet describes can't occur. Provider-call batching is still open, untouched,
a genuinely separate subsystem (rate limiting, not claiming).

Two smaller items that fall out of the fleet change in §3.3:

- ~~**Per-project run cap independent of worker count**~~: turned out to be unnecessary --
  see status note above.
- **Provider request batching**: `AsyncRateLimiter` already bounds
  concurrency/RPM per run; with N workers now potentially running
  concurrently against the same provider account, RPM limits need to be
  tracked at the **project or account level**, not just per-run-process,
  or N concurrent runs against one Gemini/OpenAI key will each independently
  think they have the full RPM budget. Proposed: a shared token-bucket
  counter in Postgres (or Redis if one gets added — not currently a
  dependency, so Postgres advisory locks are the lower-dependency-cost
  option) that `AsyncRateLimiter` checks in addition to its existing
  in-process bucket.

## 4. Scale and reliability

- **Load estimate**: no current production traffic data exists (this is
  explicitly "not yet production-validated" per team-deployment.md's own
  required-promotion-checks list) — so size the initial worker fleet for
  the smoke-test workload (2 synthetic projects, per `deploy/smoke_test.py`)
  times a small safety factor (e.g. start at 3 workers), and instrument
  queue depth/claim latency (existing `observability.py` OTel/Prometheus
  hooks) before tuning further, rather than guessing a number this document
  can't actually justify.
- **Failover**: worker loss is already handled (stale-heartbeat fencing at
  120s, per run-management.md) and gets strictly better with N workers
  instead of 1 — a dead worker no longer stalls its project's queue behind
  operator intervention, another worker just claims the next run once the
  heartbeat fences. No change needed to the fencing logic itself.
- **Postgres**: still a single instance (team-deployment.md's backup
  procedure is a manual, paused-traffic `pg_dump`). At N-worker scale, add
  a managed read replica for `runs list`/`runs show`/dashboard reads (all
  read-only queries) so claim-transaction write throughput on the primary
  isn't competing with dashboard polling — this is a pure read/write split,
  no schema change.
- **Monitoring/alerting**: existing OTel spans + Prometheus counters
  (`buffdata/observability.py`) already exist per-run; add fleet-level
  gauges — queue depth, claim-to-start latency, per-project running-count
  vs. cap — since those are new failure modes introduced by §3.3, not
  covered by today's per-run metrics.

## 5. Trade-off analysis

| Decision | Alternative considered | Why this choice |
|---|---|---|
| Shared worker pool with subprocess-level project scoping (§3.3) | Keep 1 container = 1 project, scale by adding more containers | Container-per-project is a stronger isolation boundary but doesn't scale operationally — this trades some isolation strength for real horizontal scaling, and says so explicitly rather than presenting it as a free win |
| Lineage is caller-asserted (`--parent-run`), not content-inferred (§3.1) | Auto-infer lineage from input-hash overlap | run-management.md already rejects inferred lineage for comparisons ("content deltas are not inferred causal lineage"); staying consistent with that stance beats a heuristic that will be wrong sometimes |
| Windowed exact/MinHash dedup now, semantic dedup deferred (§3.4) | Build an ANN index for semantic dedup in the same change | Semantic dedup's memory fix is an algorithm change (exhaustive → ANN), not a batching change — bundling it risks re-litigating the accuracy-contract proofs for a much bigger change than "process in chunks" |
| Presidio-anonymizer sandbox (§3.5) shares `PluginSandbox` with the plugin sandbox (§3.6), not the whole worker | One worker script for both, or two fully independent sandboxes | `buffdata/__init__.py`'s eager imports rule out a shared `buffdata.*`-importing worker for the isolated venv; sharing the parent-side subprocess-lifecycle/wire-protocol class while keeping two minimal, purpose-specific worker scripts avoids duplicating that machinery without forcing the isolated venv to import anything it doesn't need |
| Encryption at rest scoped to team deployment only, not local CLI (§3.7) | Add it everywhere for consistency | security.md's reasoning against local encryption (no real key-management story for a `pip install`ed tool) still holds; the team deployment already has an operator and a secrets story, so the same objection doesn't apply there |
| Provider RPM tracked at project level via Postgres advisory locks (§3.8) | Add Redis for a proper token-bucket service | Redis is a new infra dependency the project doesn't have today; Postgres is already the trust boundary for everything else in the managed deployment, so reusing it avoids widening the deployment's dependency footprint for this |

## What this document deliberately does not do

- Does not touch stage order, prompts, scoring/refinement/classification
  logic, or the accuracy contract — those are the product, not the
  infrastructure around it, and out of scope by the same principle
  run-management.md states for the existing managed executor.
- Does not propose public multi-tenant hosting — every item here stays
  inside team-deployment.md's stated "single trusted organization" limit.
- Does not claim any of this is built. Everything above is a proposal to
  review before implementation, sized so each of §3.1–§3.8 could land as
  an independent, revertible change.
