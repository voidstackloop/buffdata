# Deployment dependency audit: presidio-anonymizer isolation implemented

**Status**: the isolation workstream this doc originally called for is now built --
`presidio-anonymizer` (the package pinning `cryptography<49.0.0`, the actual blocker below)
runs in a separate, minimal virtualenv instead of the main environment, communicating over a
subprocess RPC boundary (`buffdata/security/anonymizer_worker.py`, launched via the
generalized `buffdata/security/sandbox.py:PluginSandbox`, activated by
`SecurityPolicy.presidio_anonymizer_python` / `BUFFDATA_PRESIDIO_ANONYMIZER_PYTHON`). Verified
directly: `presidio-analyzer` (detection, what `PIIScrubber`'s NLP-based PII finding and
third-party PII-recognizer plugins actually depend on) has **no** `cryptography` dependency
at all, so only `presidio-anonymizer` (redaction) needed to move -- detection stays in-process,
unchanged.

`deploy/requirements-cpu.lock` no longer includes `presidio-anonymizer` and its `cryptography`
pin has moved to `50.0.1`, clearing all three advisories below. A new
`deploy/requirements-presidio.lock` pins `presidio-anonymizer` + `cryptography<49.0.0` alone
for the isolated venv (`deploy/Dockerfile.team` builds it into `/opt/venv-presidio`, no
`buffdata` install needed there at all -- `anonymizer_worker.py` is a standalone script,
invoked by file path, that never imports anything under `buffdata.*`).

**Verified with a real `deploy/Dockerfile.team` build**, not just hand-authored: both lock
files were produced by running `deploy/lock_dependencies.py` (now accepting `--seed main` or
`--seed presidio-anonymizer`) against real venvs, then a full `docker build -f
deploy/Dockerfile.team .` was actually run end to end -- `pip install --no-deps -r
deploy/requirements-cpu.lock` followed by `pip check` reported "No broken requirements
found" (the exact gate this doc has required from the start) with `cryptography==50.0.1` in
the closure, and the new `/opt/venv-presidio` build stage installed
`deploy/requirements-presidio.lock` cleanly. Running containers from that built image then
confirmed the actual behavior: `PIIScrubber` with `BUFFDATA_PRESIDIO_ANONYMIZER_PYTHON`
unset falls back to regex-only redaction (`presidio_anonymizer` genuinely isn't in the main
image); with it set to `/opt/venv-presidio/bin/python`, full presidio detection *and*
redaction work through the real subprocess boundary -- including NLP-only entity types with
no regex equivalent (`PERSON`, `LOCATION`), not just the identifiers the regex fallback
covers, proving the isolated venv is actually doing the anonymization, not silently
no-op'ing.

**The advisory scan itself was also run for real**, not just described: `python
deploy/audit_dependencies.py --output ...` against the updated `requirements-cpu.lock`
reports **"No known vulnerabilities found"** -- the advisory this doc opened with is
actually cleared. The same command against `deploy/requirements-presidio.lock` reports the
same four known entries (three advisories, one duplicated) against `cryptography==48.0.1`,
exactly as expected and by design -- contained to the isolated venv, not suppressed.

**Still not done**: wheel-hash pinning, as already noted -- unrelated to this workstream.

## The original advisories (still the reason the isolated venv is pinned old)

The 2026-09-03 `pip-audit` check of `deploy/requirements-cpu.lock` reported three
distinct advisories against `cryptography==48.0.1` (four scanner entries because one
advisory is duplicated):

| Advisory | Affected library operation | Reported fixed version |
|---|---|---|
| [GHSA-m2h6-j472-rp4c](https://github.com/pyca/cryptography/security/advisories/GHSA-m2h6-j472-rp4c) | X.509 name-constraint verification | 49.0.0 |
| [GHSA-jwv3-5hgf-82ww](https://github.com/pyca/cryptography/security/advisories/GHSA-jwv3-5hgf-82ww) | X.509 chain verification resource amplification | 49.0.0 |
| [GHSA-g6cj-pr64-35w5](https://github.com/pyca/cryptography/security/advisories/GHSA-g6cj-pr64-35w5) | PKCS#7 decryption oracle | 50.0.0 |

An attempted lock update to 50.0.1 cleared the advisory check, but the actual image
build correctly failed `pip check`: `presidio-anonymizer==2.2.364`'s
[published requirements](https://pypi.org/pypi/presidio-anonymizer/2.2.364/json)
require `cryptography>=48.0.1,<49.0.0`. Neither dependency metadata nor the scanner
was overridden to manufacture a passing result -- the isolation workstream above is the
actual fix, not a suppression. The isolated venv still carries `cryptography==48.0.1`
(and therefore these same three advisories) deliberately and by design: it's a minimal,
narrow environment running exactly one thing (`AnonymizerEngine.anonymize()`, text
substitution, no X.509/PKCS#7 operations reachable from BuffData's own call path into it),
not a general-purpose environment exposed the way the main one is.

## Reproduce the advisory check, either lock

```bash
python deploy/audit_dependencies.py --output /private/dependency-audit-main.json
python deploy/audit_dependencies.py --lock deploy/requirements-presidio.lock \
  --output /private/dependency-audit-presidio.json
```

The command downloads `pip-audit` into a temporary environment and queries advisory
metadata. The main lock should now come back clean; the presidio lock will still show the
same three advisories against `cryptography==48.0.1` by design (see above) -- a clean
scan there was never the goal, containment was. The deployment locks include exact
versions but are not yet wheel-hash locked; a clean advisory check alone would not be a
complete software-supply-chain audit.
