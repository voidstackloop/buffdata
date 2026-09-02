# Architecture

## What BuffData actually does

`buffdata optimize` (and the YAML-driven `buffdata pipeline`) run a dataset through a
fixed sequence of stages, each of which can accept, reject, or transform a record:

```
input file ──▶ validate ──▶ pii ──▶ profile ──▶ dedup ──▶ score_refine ──▶ filter ──▶ classify ──▶ accepted.jsonl
                  │           │                    │                         │
                  ▼           ▼                    ▼                         ▼
              rejected.jsonl (every dropped/quarantined row, with a reason, from any stage)
```

This is `STAGES` in [`buffdata/engine/pipeline.py`](../buffdata/engine/pipeline.py):
`("validate", "pii", "profile", "dedup", "score_refine", "filter", "classify")`. Every
stage is resumable independently -- `OptimizationPipeline` checkpoints after each one
(`.{output}.checkpoint.json`), so an interrupted run picks back up at the next
incomplete stage rather than restarting from row zero.

| Stage | What it does | Ever calls a provider? |
|---|---|---|
| `validate` | Structural checks (schema, required fields, non-empty text) | No |
| `pii` | Presidio-based local PII detection/redaction | No -- runs before anything reaches a provider |
| `profile` | Infers task type (classification/chat/DPO/...) from a redacted sample, when not already labeled | Only for unlabeled data |
| `dedup` | Exact (hash), lexical (MinHash), local-semantic (sentence-transformers), or provider-embedding dedup | Only for `dedup_method=semantic` |
| `score_refine` | LLM quality scoring (`llm`/`sampled` modes) and targeted refinement of low scorers | Only when `quality_mode != "off"` |
| `filter` | Drops rows below `filter_min_score`, when scoring produced a real per-row score | No (acts on scores already computed) |
| `classify` | Assigns/verifies closed-set labels for classification data | Only when classification is needed and not already resolved |

Two guarantees sit on top of this, both enforced structurally rather than by
convention -- see [governance.md](governance.md) and [providers.md](providers.md) for
the full detail:

- **`accuracy_contract=strict`**: disables labeled-text redaction, refinement, quality
  filtering, relabeling, and all deduplication for already-labeled rows, so every valid
  labeled row survives with byte-identical text and labels. This is what the
  README's accuracy-recovery numbers are proven against.
- **`network_policy`**: `unrestricted` (default), `local` (calls allowed, but only to
  an endpoint you control on a loopback/private address), or `strict` (the LLM client
  is swapped for `NetworkForbiddenClient`, which raises on every method -- a stage
  that would need a remote call fails immediately and clearly instead of the run
  silently reaching a network).

## The provider-neutral client contract

Every optimizer stage that needs an LLM calls the same four async methods --
`generate_text_async`, `generate_structured_async`, `embed_texts_async`, and their sync
counterparts -- regardless of which provider is behind them. `create_llm_client()` in
[`buffdata/engine/client.py`](../buffdata/engine/client.py) is the one place that
decides which concrete class to construct:

```
                    create_llm_client(provider, model, base_url, network_policy)
                                          │
        ┌─────────────┬──────────────────┼──────────────────┬─────────────────┐
        ▼             ▼                  ▼                  ▼                 ▼
  GeminiClient   OpenAIClient    AnthropicClient    AzureOpenAIClient   BedrockAnthropicClient
  (public API)   (public API)    (public API)     (customer Azure tenant)  (customer AWS account)

                                          │
                              OpenAICompatibleClient
                    (openai_compatible, ollama, lmstudio, vllm, llamacpp --
                     one class, five names, each with its own default base_url)
```

`network_policy="strict"` short-circuits all of this: `create_llm_client` returns a
`NetworkForbiddenClient` instead of constructing any real client, so no credential is
ever read and no request can ever be made -- see
[`NetworkForbiddenClient`](../buffdata/engine/client.py) for the single class a
security reviewer needs to read to verify that guarantee, rather than auditing every
optimizer's internal logic.

Secrets (`GEMINI_API_KEY`, etc.) resolve through a swappable backend
(`buffdata/engine/secrets.py`) rather than `os.getenv` directly, so pointing
`BUFFDATA_SECRET_BACKEND` at Vault or a cloud secret manager changes every provider's
credential source without touching a client class. See [providers.md](providers.md).

## Data model

`DatasetItem` ([`buffdata/models/schemas.py`](../buffdata/models/schemas.py)) is the
one internal representation every input format is normalized into and every output
format is written back out from
([`buffdata/models/formats.py`](../buffdata/models/formats.py) handles
JSON/JSONL/CSV/TSV/Parquet/Arrow/YAML/plain-text/Hugging Face directories, gzip
variants, and `s3://`/`gs://`/`az://` URLs via `fsspec`). It carries the record's text
or chat/DPO structure, its label(s), its quality score once scored, and a `metadata`
dict every stage can attach its own findings to (a `rejection` reason, PII entity
counts, a dedup reason, ...) -- which is exactly what
`OptimizationPipeline._rejection_counts()` reads to build the report's rejection
breakdown.

`OptimizationRunResult` is what every run produces: `accepted`/`rejected` item lists,
the inferred `profile`, a `metrics` dict (per-stage accepted/rejected counts, token
usage, provider/model, the accuracy contract and network policy in force), and the
paths of the output/rejected/report files actually written.

## Code map

| Path | What lives there |
|---|---|
| `buffdata/cli/main.py` | Every CLI command -- the thinnest possible layer over the engine/optimizers below |
| `buffdata/engine/` | `pipeline.py` (the 7-stage orchestrator), `client.py` (provider contract), `secrets.py`, `limiter.py` (async rate limiting + retry), `validator.py`, `profiler.py`, `checkpoint.py` |
| `buffdata/optimizers/` | One module per stage's actual logic: `scorer.py`, `refiner.py`, `dedup.py`, `scrubber.py` (PII), `classifier.py`, `evolver.py`, `augmenter.py`, `preference.py` (DPO), `perplexity.py`, `gated_generator.py` (the `generate` command's accuracy-gated loop) |
| `buffdata/models/` | `schemas.py` (`DatasetItem`, `PipelineConfig`, ...), `formats.py` (every I/O format, local and cloud) |
| `buffdata/governance/` | `access.py` (RBAC), `oidc.py` (SSO), `audit_store.py`, `contract.py` (Data Contracts), `sbom.py` |
| `buffdata/integrations/` | `sharding.py`, `ray_executor.py`, `huggingface.py`, `train_export.py`, `graphify.py`/`context7.py` (developer-productivity tools) |
| `buffdata/evaluation/accuracy_gate.py` | The PyTorch proxy classifier `generate`/`optimize --require-positive-gain` use to prove a candidate dataset actually improves accuracy before publishing it |
| `buffdata/observability.py` | OpenTelemetry spans + Prometheus metrics, opt-in via `PipelineConfig.observability` |
| `buffdata/security/validator.py` | Structural/safety checks distinct from `engine/validator.py`'s schema checks |
| `buffdata/synthesizer/generator.py` | `buffdata synthesize`'s document-to-instruction-pairs logic |

See [cli-reference.md](cli-reference.md) for what each command in `cli/main.py`
actually exposes, and [configuration.md](configuration.md) for every field
`PipelineConfig` accepts.
