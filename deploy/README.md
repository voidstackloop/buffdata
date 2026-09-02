# Deployment artifacts

Three pieces, meant to be used together:

1. **`Dockerfile`** -- builds the `buffdata` CLI into a runtime image (CPU-only torch,
   the `en_core_web_lg` spacy model PII scrubbing needs since `scrub_pii` defaults to
   `True`).
2. **`terraform/aws`** -- provisions an S3 bucket, a Secrets Manager secret, and an
   IAM role scoped to just those two, trusted by one Kubernetes ServiceAccount (IRSA).
3. **`helm/buffdata`** -- runs the image as a Job or CronJob, wired to the role from (2)
   via `serviceAccount.annotations`, reading/writing the bucket from (2) directly (no
   PVC needed for the cloud-URL-capable commands).

## Honesty about verification

Every file here was written against the real package -- `pyproject.toml`'s dependency
list and console-script entry point, the actual CLI commands and their cloud-URL
support (`buffdata/cli/main.py`), the actual `SecretResolver` backends and env var names
(`buffdata/engine/secrets.py`, `buffdata/engine/client.py`), the actual spacy model
`PIIScrubber` resolves to. That's different from having been run.

None of the three could be executed in the environment this was built in: no reachable
Docker daemon (Docker Desktop's WSL integration isn't enabled here), no `helm` binary,
no `terraform` binary, and no AWS credentials to apply against regardless. Before relying
on any of this:

- `docker build -f deploy/Dockerfile .` and run `buffdata --help` / a real command
  against the resulting image
- `helm template ./deploy/helm/buffdata` and read the rendered manifests
- `terraform init && terraform validate` in `deploy/terraform/aws`, then review a real
  `terraform plan`

Every unit and pipeline-integration test in this repo (`pytest -q`, 268 tests as of this
writing) has been run for real -- that discipline stops at the deployment layer only
because it requires infrastructure this environment doesn't have, not because it was
skipped.
