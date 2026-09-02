# Deployment

Three pieces in [`deploy/`](../deploy/), meant to be used together -- the full detail
lives beside each one rather than duplicated here:

| Artifact | What it does | Doc |
|---|---|---|
| [`deploy/Dockerfile`](../deploy/Dockerfile) | Builds the `buffdata` CLI into a runtime image | -- |
| [`deploy/terraform/aws`](../deploy/terraform/aws/) | Provisions an S3 bucket, a Secrets Manager secret, and an IAM role scoped to just those two, trusted by one Kubernetes ServiceAccount (IRSA) | [`deploy/terraform/aws/README.md`](../deploy/terraform/aws/README.md) |
| [`deploy/helm/buffdata`](../deploy/helm/buffdata/) | Runs the image as a Kubernetes `Job` or `CronJob`, wired to the Terraform role, reading/writing the bucket directly | [`deploy/helm/buffdata/README.md`](../deploy/helm/buffdata/README.md) |

## Status: written correctly, not run

Every file was written against the real package -- `pyproject.toml`'s dependencies and
console-script entry point, the actual CLI commands and their cloud-URL support (see
[providers.md](providers.md#cloud-dataset-storage)), the actual secret-backend names
and env vars (see [providers.md](providers.md#secret-backends)), the actual spacy
model `PIIScrubber` resolves to. That's different from having been executed.

None of the three could be run in the environment they were built in: no reachable
Docker daemon, no `helm`/`terraform` binaries, and no AWS credentials to apply against
regardless. Before relying on any of it:

```bash
docker build -f deploy/Dockerfile .          # then run buffdata --help / a real command against it
helm template ./deploy/helm/buffdata          # read the rendered manifests
cd deploy/terraform/aws && terraform init && terraform validate   # then review a real `terraform plan`
```

One real bug this exact gap already caught: the Helm chart originally documented and
defaulted to hyphenated secret-backend names (`aws-secrets-manager`) that don't match
`buffdata/engine/secrets.py`'s actual underscored names (`aws_secrets_manager`) --
setting one of those Helm values would have failed at runtime with an "unknown secret
backend" error. Fixed once caught (cross-referencing this doc against the real code is
what surfaced it), but it's the concrete reason "written correctly" and "verified"
aren't being used interchangeably above.

## What each piece assumes

- **Dockerfile**: installs the CPU-only torch wheel (not the default CUDA one, which
  would roughly triple image size for a binary the accuracy gate/local dedup never
  need in a container) and downloads the `en_core_web_lg` spacy model PII scrubbing
  needs -- not optional, since `scrub_pii` defaults to `True`.
- **Terraform**: assumes an EKS cluster (and its OIDC provider) already exists; it
  doesn't create one, only trusts it.
- **Helm chart**: defaults to running `buffdata score` directly against `s3://` URLs
  (no PVC needed, using the cloud-storage support most single-stage commands have).
  Running the full `buffdata pipeline` orchestrator instead needs a PVC, since
  `pipeline` doesn't accept cloud URLs -- see the chart's own `values.yaml` comments
  for exactly where to wire that in.

Every unit and pipeline-integration test in this repo (`pytest -q`) has been run for
real. That discipline stops at the deployment layer only because it requires
infrastructure this environment doesn't have, not because it was skipped -- see
[`deploy/README.md`](../deploy/README.md) for the same point made once, at the source.
