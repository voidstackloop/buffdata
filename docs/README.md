# BuffData documentation

The top-level [README](../README.md) is the quick start and the proof-it-works
benchmark evidence. This folder is the reference layer underneath it: how the system
is built, every command and config field, and the enterprise-facing pieces (auth,
governance, observability, deployment, scaling) that don't fit in a quick start.

Written for three readers:

- **Someone integrating BuffData** into a pipeline or CI system → start with
  [cli-reference.md](cli-reference.md) and [configuration.md](configuration.md).
- **Someone standing up team infrastructure** around it (auth, secrets, a shared LLM
  server, a Kubernetes deployment) → [providers.md](providers.md),
  [governance.md](governance.md), [deployment.md](deployment.md).
- **Someone changing BuffData's own code** → [architecture.md](architecture.md) first.

## Contents

| Doc | What's in it |
|---|---|
| [architecture.md](architecture.md) | The 7-stage pipeline, the provider-neutral client contract, data flow, where each guarantee (strict/local network policy, accuracy contracts) actually lives in the code |
| [cli-reference.md](cli-reference.md) | Every command, grouped by what it's for, with real examples |
| [configuration.md](configuration.md) | Every `PipelineConfig` field: type, default, what it controls, and the YAML equivalent |
| [security.md](security.md) | API keys in the OS keyring instead of a plaintext file (`buffdata auth`), and owner-only file permissions on checkpoints/reports/the audit DB |
| [providers.md](providers.md) | Cloud providers, local LLM servers (Ollama/LM Studio/vLLM/llama.cpp), secret backends (Vault/AWS/GCP/Azure), cloud dataset storage (S3/GCS/ADLS) |
| [governance.md](governance.md) | Access policy (RBAC), OIDC/SSO bearer-token verification, the durable audit log, Data Contracts, SBOM generation |
| [observability.md](observability.md) | OpenTelemetry tracing and Prometheus metrics per pipeline run |
| [scaling.md](scaling.md) | Dataset sharding for orchestrator-level parallelism (Airflow/Dagster/k8s), and the Ray-based in-process alternative |
| [deployment.md](deployment.md) | Docker image, Helm chart, Terraform module -- what's verified and what isn't |

## What's *not* duplicated here

- Benchmark numbers and methodology live in the [README](../README.md#verified-benchmark-results)
  and `benchmarks/` -- this folder links to them rather than re-copying tables that
  would drift out of sync.
- Full flag-by-flag `--help` output for every command isn't reproduced verbatim here;
  [cli-reference.md](cli-reference.md) covers what each command is for and its most
  load-bearing flags, and points at `buffdata <command> --help` for the complete,
  always-current list.
- Deployment artifacts' own detailed usage notes live beside the files themselves in
  [`deploy/`](../deploy/README.md); [deployment.md](deployment.md) is the map, not a copy.
