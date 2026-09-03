# Security

The new [managed execution boundary](run-management.md) adds plugin approval, bounded
input/network handling, escaped reports, verified manifests, and fail-closed provider
exceptions. [Team deployment](team-deployment.md) adds OIDC project authorization and worker
containment. The sections below describe the pre-existing local credential/file protections;
the managed-run directory is additionally private from creation.

Two independent protections, both on by default with zero configuration: where API
keys live, and who on the local machine can read the files BuffData writes.

## API keys: the OS keyring, not a plaintext file

```bash
buffdata auth set GEMINI_API_KEY
# Value for GEMINI_API_KEY: [hidden input]
# Repeat for confirmation: [hidden input]
# Stored GEMINI_API_KEY in the OS keyring.
```

Stores the key in the platform-native credential store -- Windows Credential Manager,
macOS Keychain, or Linux Secret Service/KWallet, via the `keyring` package (a base
dependency, not an extra: this is the default story for every `pip install buffdata`
user, not an enterprise add-on). The prompt uses hidden input with a confirmation
step, and the value is never written to any file, never echoed to the terminal, and
never lands in shell history the way `export GEMINI_API_KEY=sk-...` would.

Every command picks it up automatically afterward -- no `.env` file, no
`BUFFDATA_SECRET_BACKEND` change. `EnvSecretResolver`, the default backend
([`buffdata/engine/secrets.py`](../buffdata/engine/secrets.py)), checks the
environment variable first (so CI/scripted use with real env vars is completely
unaffected) and falls back to the OS keyring only when that's unset. It's the one
backend that behaves this way; every other backend (`vault`,
`aws_secrets_manager`, `gcp_secret_manager`, `azure_key_vault` -- see
[providers.md](providers.md#secret-backends)) is explicit and doesn't fall back to
anything, since those are deliberate infrastructure choices, not a smoothing default.

```bash
buffdata auth status    # which known secrets resolve, and from where -- values never shown
buffdata auth remove GEMINI_API_KEY
```

```
          Secret resolution status
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Name                      ┃ Resolves from ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ GEMINI_API_KEY            │ not set       │
│ GOOGLE_API_KEY            │ not set       │
│ OPENAI_API_KEY            │ environment   │
│ ANTHROPIC_API_KEY         │ not set       │
│ AZURE_OPENAI_API_KEY      │ not set       │
│ OPENAI_COMPATIBLE_API_KEY │ not set       │
│ OLLAMA_API_KEY            │ not set       │
│ LMSTUDIO_API_KEY          │ not set       │
│ VLLM_API_KEY              │ not set       │
│ LLAMACPP_API_KEY          │ not set       │
└───────────────────────────┴───────────────┘
```

Set `BUFFDATA_KEYRING_SERVICE` to use a different keyring namespace than the default
`buffdata` (multiple installs/profiles on one machine, for instance).

Verified for real in
[`tests/test_auth_keyring.py`](../tests/test_auth_keyring.py) -- genuine
`keyring.set_password`/`get_password`/`delete_password` round trips (via a real,
file-backed `keyrings.alt` backend in the test environment, since no OS-native
credential store is available in headless CI; the platform-native ones are what
production actually uses), the full `buffdata auth set/remove/status` CLI flow, and an
end-to-end test confirming a key stored via `auth set` is what `create_llm_client`
actually resolves and uses -- no gap between "stored" and "used."

## Files at rest: owner-only permissions

[`buffdata/security/permissions.py`](../buffdata/security/permissions.py)'s
`restrict_to_owner()` sets `0600` (owner read/write only) on every locally-written file
that can carry sensitive content, immediately after writing it:

- **Pipeline checkpoints** (`.{output}.checkpoint.json`) -- can briefly hold
  pre-PII-redaction text, since the checkpoint after the `validate` stage is written
  before the `pii` stage runs (see [architecture.md](architecture.md)).
- **`report.json`** -- record counts, provider/model, dataset-revealing file paths.
- **The audit database** (`buffdata_audit.db` by default) -- every recorded run across
  a team, queryable by anyone who can read the file.

A no-op on Windows (`os.chmod` doesn't express meaningful ACLs there, and NTFS's
per-user-profile isolation already covers the common single-user-machine case) and
never fails the write it's protecting -- this is defense in depth for a shared POSIX
machine, not something the rest of the pipeline depends on for correctness.

## What this doesn't cover

- **Encryption at rest** for these same files was deliberately not built: without a
  passphrase or key-management story for a `pip install`ed CLI to draw from, an
  encryption key would have to be hardcoded (pointless) or become yet another secret
  the user has to manage (worse than the file-permission protection it would replace).
  File permissions are the standard mitigation for this exact scenario --
  `~/.ssh/id_rsa` and `~/.aws/credentials` use the same approach, not encryption. That
  reasoning doesn't carry over to the team deployment, which already has a real operator and
  a secrets story -- see [team-deployment.md](team-deployment.md#encryption-at-rest-for-managed-artifacts)
  for the opt-in encryption-at-rest option that exists there instead. This local-CLI story
  is otherwise unchanged.
- **The final output dataset** (`optimized.jsonl` and its `.rejected.jsonl`) is *not*
  permission-restricted -- it's the deliverable you explicitly asked BuffData to
  produce for downstream use (training, sharing, `buffdata push` to a Hub), and
  defaulting it to owner-only would just be friction for the common case of a build
  pipeline or teammate reading it under a different local user.
- Redacting secrets from third-party SDK exception messages (if `openai`/`anthropic`/
  `google-genai` ever included a raw key in an error string) is outside BuffData's
  control -- checked during this work and none of buffdata's own code paths do this
  (no client's `__repr__`, exception message, or anything written to `report.json`/the
  audit DB ever includes `self.api_key`), but a third-party library's own error
  formatting isn't something this codebase can guarantee.
