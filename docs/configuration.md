# Configuration reference

`PipelineConfig` ([`buffdata/models/schemas.py`](../buffdata/models/schemas.py)) is
the single source of truth for how `optimize`/`pipeline`/`generate` behave -- every
CLI flag on those commands sets one of these fields. `buffdata pipeline` reads it
straight from YAML ([`examples/pipeline_config.yaml`](../examples/pipeline_config.yaml)
is a full worked example); `buffdata optimize` merges an optional `--config` YAML file
with individual flag overrides, flags winning.

## Provider

| Field | Type | Default | What it controls |
|---|---|---|---|
| `provider` | `str` | `$BUFFDATA_PROVIDER` or `"gemini"` | Which client `create_llm_client` constructs -- see [providers.md](providers.md) for the full list |
| `model` | `str \| None` | `None` (provider's balanced default) | Model override; required (no default) for `azure_openai`, `openai_compatible`, and the four local-server providers |
| `base_url` | `str \| None` | `None` (provider's preset, if any) | Endpoint for `openai_compatible`/local-server providers |
| `network_policy` | `"unrestricted" \| "local" \| "strict"` | `"unrestricted"` | Whether/where calls are allowed -- see [architecture.md](architecture.md#the-provider-neutral-client-contract) |
| `concurrency` | `int` | `10` | Max concurrent async requests |
| `max_rpm` | `int` | `60` | Max requests per minute (token-bucket rate limit) |
| `fast_model` | `str` | `"gemini-3.5-flash-lite"` | Model used for high-throughput, lower-stakes calls (classification, sampled audits) regardless of the main `provider`/`model` |

## Quality scoring and refinement

| Field | Type | Default | What it controls |
|---|---|---|---|
| `quality_mode` | `"llm" \| "sampled" \| "off"` | `"llm"` | `llm` scores and filters every row; `sampled` audits a representative subset (batched, no filtering); `off` skips LLM quality scoring entirely |
| `quality_sample_size` | `int` (1-1000) | `50` | Rows audited when `quality_mode="sampled"` |
| `quality_audit_batch_size` | `int` (1-100) | `20` | Records grouped into one structured LLM request |
| `filter_min_score` | `float` | `7.0` | Minimum quality score to survive the `filter` stage (only meaningful with `quality_mode="llm"`) |
| `refine_below_score` | `float` (0-10) | `8.0` | Rows scoring below this get one refinement pass before re-scoring |
| `refine_mode` | `str` | `"all"` | `all`, `response_only`, or `prompt_only` |

## Deduplication

| Field | Type | Default | What it controls |
|---|---|---|---|
| `dedup_method` | `str` | `"auto"` | `auto` (exact for labeled classification, minhash otherwise), `exact`, `minhash`, `semantic-local` (local sentence-transformers), or `semantic` (provider embeddings -- the only dedup method that calls a provider) |
| `dedup_threshold` | `float` | `0.88` | Similarity threshold for minhash/semantic dedup |
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | Local sentence-transformers model for `semantic-local` |
| `embedding_provider` | `str` | `"local"` | Present for forward compatibility; local embeddings are provider-neutral today |

## Classification

| Field | Type | Default | What it controls |
|---|---|---|---|
| `classification` | `ClassificationMode` | `AUTO` | `auto`, `off`, `binary`, `multi-class`, `multi-label` |
| `classes` | `list[str]` | `[]` | Explicit class names (required for binary/multi-class/multi-label unless letting `auto` infer them) |
| `classification_confidence` | `float` (0-1) | `0.75` | Minimum profiler confidence before classification is attempted |
| `classification_sample_size` | `int` (1-1000) | `100` | Rows sampled to infer the task/classes when not already labeled |

## Privacy

| Field | Type | Default | What it controls |
|---|---|---|---|
| `scrub_pii` | `bool` | `True` | Local Presidio-based PII redaction before any content reaches a provider |
| `classification_pii_mode` | `"identifiers" \| "all" \| "off"` | `"identifiers"` | For labeled classification: redact only high-confidence direct identifiers, every Presidio entity type, or nothing |

`presidio-anonymizer` (the redaction half of Presidio; `presidio-analyzer`, detection, has no
such conflict and is always installed) is not a base dependency -- see
[dependency-release-blocker.md](dependency-release-blocker.md). Without it importable
in-process *and* without `SecurityPolicy.presidio_anonymizer_python`/
`BUFFDATA_PRESIDIO_ANONYMIZER_PYTHON` pointed at an isolated venv that has it, `scrub_pii`
falls back to the deterministic regex patterns for common identifiers (email, phone, IP,
credit card, API key) only -- broader entity types (names, locations, organizations) are not
redacted by the fallback. `pip install buffdata[presidio-anonymizer]` for a single-process
setup, or configure the isolated venv for the team deployment.

## Accuracy contract

| Field | Type | Default | What it controls |
|---|---|---|---|
| `accuracy_contract` | `"balanced" \| "strict"` | `"balanced"` | `strict` disables labeled-text redaction, refinement, quality filtering, relabeling, and all deduplication -- every valid labeled row survives with byte-identical text/labels. This is what the README's accuracy-recovery benchmark runs under. |

## Evolution (Evol-Instruct)

| Field | Type | Default | What it controls |
|---|---|---|---|
| `evolution_strategies` | `list[str]` | `["deepen_reasoning", "add_constraints", "concretize"]` | Which evolution transforms `buffdata evolve` applies |
| `evolution_depth` | `int` | `1` | Evolution iterations per item |

## Reporting and observability

| Field | Type | Default | What it controls |
|---|---|---|---|
| `report_html` | `bool` | `False` | Also write `<output>.report.html`, an interactive audit report |
| `observability` | `bool` | `False` | Emit an OTel span + Prometheus metrics per stage, and a `<output>.metrics.prom` textfile sidecar -- see [observability.md](observability.md) |

## Example

```yaml
# examples/pipeline_config.yaml (abridged -- see the file for the full worked example)
provider: gemini
model: gemini-3.7-flash
network_policy: unrestricted
quality_mode: sampled
quality_sample_size: 40
dedup_method: auto
classification: auto
scrub_pii: true
classification_pii_mode: identifiers
accuracy_contract: balanced
observability: true
report_html: true
```

```bash
buffdata pipeline examples/pipeline_config.yaml -i data.jsonl -o optimized.jsonl
```
