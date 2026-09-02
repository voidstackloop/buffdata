# buffdata Helm chart

Runs `buffdata` as a Kubernetes `Job` (one-shot) or `CronJob` (recurring) against a
dataset in cloud storage or a mounted volume.

**Status: written against the real CLI (`buffdata/cli/main.py`) and schema
(`buffdata/models/schemas.py`), not verified with `helm lint`/`helm template`.** No
`helm` binary was available in the environment this was built in. Render it and read the
output before trusting it in a cluster:

```bash
helm template buffdata-run ./deploy/helm/buffdata --values my-values.yaml
```

## What it actually does

- **Default (`values.yaml` as shipped)**: runs `buffdata score s3://.../input.jsonl -o
  s3://.../scored.jsonl`, reading and writing S3 directly -- no PVC, no data-staging
  step. This works because `score` (along with `refine`, `evolve`, `dpo`, `dedup`,
  `augment`, `scrub`, `classify`, `filter-ppl`) has native cloud-storage support via
  `buffdata/models/formats.py`'s fsspec integration.
- **`buffdata pipeline` (the full 7-stage orchestrator)**: deliberately rejects cloud
  URLs -- its checkpoint/fingerprint/sidecar-artifact paths assume a local filesystem.
  To run it, mount a PVC at `/data`, stage `input.jsonl` onto it yourself (an
  `initContainer`, a sync job, whatever your storage backend supports), set
  `pipelineConfig.enabled: true`, and point `command` at
  `["pipeline", "/config/pipeline.yaml", "-i", "/data/input.jsonl", "-o", "/data/output.jsonl"]`.
  This chart does not wire that PVC/staging step for you -- it's genuinely dependent on
  which storage class/backend your cluster has, so it's left as the one place you'll
  need to fill in.

## Secrets

`secretBackend: env` (the default) puts provider API keys in a plain Kubernetes
`Secret`, which also lands in `helm get values` / release history -- fine for a quick
trial, not for production. Switch to `aws_secrets_manager`, `vault`,
`gcp_secret_manager`, or `azure_key_vault` (all implemented in
`buffdata/engine/secrets.py`) and bind the pod's identity (IRSA via
`serviceAccount.annotations` on EKS, Workload Identity on GKE, etc.) to a role scoped to
just that one secret -- see `deploy/terraform/aws` for a matching IAM role example.

## Observability

Set `observability: true` in `pipelineConfig.yaml` (only takes effect when running the
`pipeline` orchestrator) to get a `<output>.metrics.prom` sidecar file next to the
output dataset -- point a Prometheus `node_exporter` textfile collector at wherever that
lands (the PVC, if you've wired one) to scrape it.
