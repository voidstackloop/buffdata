# BuffData

## Verified benchmark results

The latest accuracy gate directly compares matching original AG News records with
the previously generated output that caused the reported regression and BuffData's
repaired generated output. Each condition trained the same PyTorch classifier on
3,000 records and evaluated the same 2,000-record test set across three fixed seeds.

| 2k+ accuracy gate | Mean accuracy | Mean macro-F1 |
|---|---:|---:|
| Matching original dataset | 83.15% | 83.07% |
| Regressed generated dataset | 80.70% | 80.61% |
| Strict generated dataset | **83.15%** | **83.07%** |

The strict accuracy contract recovers the full **+2.45 accuracy points**, changes
zero classification texts or labels, and produces exactly the same per-seed results
as the original. The gate now requires generated accuracy to be equal to or greater
than original accuracy; no negative tolerance is accepted.

The same strict gate was rerun on large clean datasets. It requires the full ordered
row sequence, every text, every label, accuracy, and macro-F1 to match exactly.

| Strict scale gate | Original accuracy | Strict generated accuracy | Row/text/label identity |
|---:|---:|---:|---:|
| 20,000 train / 7,000 test | 88.11% | **88.11%** | **exact** |
| 100,000 train / 7,000 test | 89.64% | **89.64%** | **exact** |

The same original-versus-generated value test was then scaled to 20,000 and 100,000
real source records. The original conditions contain disclosed conflicting-label
duplicates, class-skewing duplicates, and empty rows; BuffData generates the optimized
conditions from those exact inputs.

| AG News source size | Contaminated original rows | Original accuracy | Generated optimized accuracy | Gain |
|---:|---:|---:|---:|---:|
| 3,000 | 5,700 | 75.15% | **84.25%** | **+9.10 points** |
| 20,000 | 38,000 | 80.14% | **87.80%** | **+7.66 points** |
| 100,000 | 190,000 | 83.39% | **89.42%** | **+6.03 points** |

At 20k, the clean original and clean generated control both scored 87.80%. At 100k,
the generated clean control scored 89.42% versus 89.38% original after removing 19
naturally duplicated texts. The 100k generated dataset also cut mean training time
from 80.15 to 42.26 seconds versus the 190k contaminated original. Gemini audited
20 representative rows at each scale in one structured request with zero failures;
the deterministic validation and deduplication stages caused the measured recovery.

- [Original-versus-generated report](benchmarks/results-regression/REPORT.md)
- [Per-seed regression metrics](benchmarks/results-regression/results.json)
- [Accuracy-gate source](benchmarks/benchmark_accuracy_regression.py)
- [20k/100k strict parity report](benchmarks/results-strict-scale/REPORT.md)
- [Strict scale benchmark source](benchmarks/benchmark_strict_scale.py)
- [20k+ scale report](benchmarks/results-20k/REPORT.md)
- [100k+ scale report](benchmarks/results-100k/REPORT.md)

### Larger dataset benchmark

BuffData was tested with identical PyTorch classifiers across three fixed seeds on
12,000 real training records from each of DBpedia-14, Yelp Polarity, and DAIR.AI
Emotion. After BuffData processed the controlled dirty variants, mean test accuracy
recovered by **7.07–12.53 percentage points** and training time fell by approximately
**49%** because invalid and duplicated rows were removed.

| Dataset | Dirty baseline | After BuffData | Accuracy recovery | Macro-F1 recovery |
|---|---:|---:|---:|---:|
| DBpedia-14 | 86.97% | 96.68% | **+9.72 points** | **+9.77 points** |
| Yelp Polarity | 83.53% | 90.60% | **+7.07 points** | **+7.24 points** |
| Emotion | 71.23% | 83.77% | **+12.53 points** | **+14.55 points** |

Clean-data controls were unchanged for DBpedia-14 and Yelp Polarity. On Emotion,
BuffData removed 22 naturally duplicated texts and mean accuracy changed from 83.60%
to 83.77%. The run trained 36 models and used Gemini to audit 60 representative
records in only three structured batch requests, with zero failed audit scores.

These numbers are a reproducible stress test, not a claim that every dataset improves.
The dirty variants deliberately added conflicting-label duplicates, class-skewing
duplicates, and empty rows to real Hugging Face records. Validation and exact
deduplication produced the measured training recovery; Gemini's sampled semantic
scores were informational and did not decide which rows were retained.

- [Full large-dataset report](benchmarks/results-large/REPORT.md)
- [Per-seed metrics and Gemini usage](benchmarks/results-large/results.json)
- [Benchmark source and reproduction instructions](benchmarks/README.md)

### 5-dataset x 3-scale real-world recovery benchmark

The most comprehensive proof to date: five real Hugging Face classification datasets
spanning 2 to 14 classes and both formal and noisy user-generated text, each tested at
three realistic training-set sizes (10k / 25k / 50k rows), three fixed seeds, six epochs
per run -- 15 dataset x scale combinations in total. Every condition trains an identical
PyTorch classifier; the only variable is whether BuffData's local validate and
exact-deduplication stages processed the contaminated input first. No LLM is involved in
this recovery -- it comes entirely from BuffData's deterministic local stages.

Across all 15 combinations, BuffData recovered an average of **+8.14 accuracy points**
(range: +4.73 to +10.73) from data contaminated with disclosed conflicting-label
duplicates (40%), class-skewing duplicates (40%), and empty rows (10%):

| Dataset | Avg. recovery across 10k/25k/50k |
|---|---:|
| dbpedia_14 | **+10.29 points** |
| yahoo_answers_topics | **+9.01 points** |
| ag_news | **+8.54 points** |
| yelp_polarity | **+6.59 points** |
| amazon_polarity | **+6.26 points** |

The clean-data control -- BuffData run against data that was never contaminated --
stayed within **0.10 points** of zero across every one of the 15 combinations (mean
-0.007, max magnitude 0.10), confirming BuffData does not damage already-clean data; it
only recovers accuracy that contamination actually cost.

<details>
<summary>Full 15-combination results</summary>

| Dataset | Scale | Dirty raw acc | Dirty optimized acc | Recovery | Clean delta |
|---|---:|---:|---:|---:|---:|
| ag_news | 10,000 | 77.88% | 87.53% | +9.65 | +0.00 |
| ag_news | 25,000 | 79.65% | 88.83% | +9.18 | +0.00 |
| ag_news | 50,000 | 81.83% | 88.63% | +6.80 | -0.10 |
| dbpedia_14 | 10,000 | 86.22% | 96.00% | +9.78 | +0.00 |
| dbpedia_14 | 25,000 | 86.15% | 96.88% | +10.73 | +0.00 |
| dbpedia_14 | 50,000 | 86.67% | 97.03% | +10.37 | +0.00 |
| yelp_polarity | 10,000 | 81.32% | 88.85% | +7.53 | +0.00 |
| yelp_polarity | 25,000 | 82.33% | 89.83% | +7.50 | +0.00 |
| yelp_polarity | 50,000 | 85.42% | 90.15% | +4.73 | +0.00 |
| amazon_polarity | 10,000 | 79.05% | 85.93% | +6.88 | +0.00 |
| amazon_polarity | 25,000 | 79.18% | 85.88% | +6.70 | +0.00 |
| amazon_polarity | 50,000 | 82.52% | 87.70% | +5.18 | +0.00 |
| yahoo_answers_topics | 10,000 | 53.67% | 62.92% | +9.25 | +0.00 |
| yahoo_answers_topics | 25,000 | 52.53% | 62.53% | +10.00 | +0.00 |
| yahoo_answers_topics | 50,000 | 54.50% | 62.27% | +7.77 | +0.00 |

</details>

- [Full report](benchmarks/results-scale-matrix-gemini-3.7-flash/REPORT.md)
- [Per-seed metrics](benchmarks/results-scale-matrix-gemini-3.7-flash/results.json)
- [Benchmark source](benchmarks/benchmark_scale_matrix.py)

### What "recovery" actually removed, stage by stage

Every dirty condition above adds three disclosed defect types on top of the real,
untouched source rows -- BuffData is never told which rows are which; it processes the
mixed set the same way it would any raw input. These are two-field records (`text` +
`label`), so nothing here is a "column" operation: BuffData's stages decide whether to
keep or drop a whole *row*, they never drop a field from a row's schema. (The one place
BuffData modifies content *within* a row instead of deciding whether to keep it is PII
redaction, which this benchmark disables via `scrub_pii=False` specifically to isolate
the validate/dedup recovery number from a second, unrelated variable -- see
[Adaptive workflow](#adaptive-workflow) for how that's scoped when it's on.)

The pipeline's own per-stage counts -- not an estimate, pulled directly from each run's
`report.json` -- show exactly which stage caught each defect type:

- **`validate`** rejects rows that fail structural checks; in this benchmark that's the
  empty-text rows. It rejected **exactly** the injected empty-row count on all 15
  combinations, no more, no less.
- **`dedup`** (exact method) rejects any row whose text content hashes identically to a
  row already kept. Both remaining defect types -- conflicting-label duplicates and
  class-skew duplicates -- are literal or near-literal text copies, so both land here
  regardless of label, since exact dedup keys on text content only.

<details>
<summary>Full 15-combination stage breakdown</summary>

| Dataset | Scale | Input rows (dirty) | Rejected: validate | Rejected: dedup | Final rows kept |
|---|---:|---:|---:|---:|---:|
| ag_news | 10,000 | 19,000 | 1,000 | 8,000 | 10,000 |
| ag_news | 25,000 | 47,500 | 2,500 | 20,000 | 25,000 |
| ag_news | 50,000 | 95,000 | 5,000 | 40,002 | 49,998 |
| dbpedia_14 | 10,000 | 19,000 | 1,000 | 8,000 | 10,000 |
| dbpedia_14 | 25,000 | 47,500 | 2,500 | 20,000 | 25,000 |
| dbpedia_14 | 50,000 | 95,000 | 5,000 | 40,000 | 50,000 |
| yelp_polarity | 10,000 | 19,000 | 1,000 | 8,000 | 10,000 |
| yelp_polarity | 25,000 | 47,500 | 2,500 | 20,000 | 25,000 |
| yelp_polarity | 50,000 | 95,000 | 5,000 | 40,000 | 50,000 |
| amazon_polarity | 10,000 | 19,000 | 1,000 | 8,000 | 10,000 |
| amazon_polarity | 25,000 | 47,500 | 2,500 | 20,000 | 25,000 |
| amazon_polarity | 50,000 | 95,000 | 5,000 | 40,000 | 50,000 |
| yahoo_answers_topics | 10,000 | 19,000 | 1,000 | 8,000 | 10,000 |
| yahoo_answers_topics | 25,000 | 47,500 | 2,500 | 20,000 | 25,000 |
| yahoo_answers_topics | 50,000 | 95,000 | 5,000 | 40,000 | 50,000 |
| **Total** | | **807,500** | **42,500** | **340,002** | **424,998** |

</details>

Two numbers above don't match the injected totals exactly, and that's a real finding,
not a bug: at ag_news/50,000, dedup rejected 40,002 rows against 40,000 injected
duplicates (20,000 conflicting-label + 20,000 class-skew). Running BuffData against
that same scale's **clean, uncontaminated** control independently rejected 2 rows at
the dedup stage too -- meaning the real AG News data itself already contains 2
naturally-occurring exact-duplicate rows at that sample size, something BuffData
catches for free whether or not you've deliberately contaminated anything. (That's also
where this section's own -0.10-point "clean delta" for ag_news/50,000, in the full
results table further up, comes from -- and the same phenomenon, at different scales
and datasets, is what removed the "naturally duplicated texts" mentioned in the
20k/100k-scale and DBpedia/Yelp/Emotion benchmarks earlier in this README.)

Across all 15 combinations: 807,500 dirty input rows in, 42,500 rejected by validate
(100% of the 42,500 empty rows injected), 340,002 rejected by dedup (170,000
conflicting-label + 170,000 class-skew injected duplicates, plus the 2 natural
duplicates above), leaving **424,998 final training rows -- 52.6% of the dirty input,
kept because every one of them was validated and confirmed unique**, not because of
any quality judgment call.

### Gemini judge model comparison: gemini-3.5-flash-lite vs. gemini-3.7-flash

`--quality-mode sampled` optionally has a Gemini model sample-audit a representative
slice of the cleaned output -- informational only; it never decides which rows survive
validate/dedup (that's what produces every accuracy number above). Rerunning the exact
same 15-combination benchmark above with only the audit model swapped confirms this by
construction: **training accuracy was bit-for-bit identical across all 15 combinations**,
regardless of which model judged the output.

| | gemini-3.5-flash-lite | gemini-3.7-flash (current default) |
|---|---:|---:|
| Avg. quality score (of 10, 300 rows audited) | 9.78 | **9.93** |
| Total audit tokens (300 rows) | 115,454 | 115,302 |

The score gap is not a broad shift -- four of the five datasets were already rated
~10/10 by both models, leaving no room to move. It's concentrated entirely in
`yahoo_answers_topics`, the hardest/noisiest dataset in the set (10-class,
user-generated Q&A): gemini-3.5-flash-lite averaged 8.95-8.97/10 there across the three
scales, gemini-3.7-flash hit a perfect 10.00/10 at two of the three. Token usage was a
wash (-0.13% overall) -- the newer model isn't more verbose for this workload.

gemini-3.7-flash is BuffData's current default model (see provider config below) because it's
at least as good everywhere and meaningfully better on the hardest data, at comparable
token cost. Per-token API pricing for the standard flash tier is typically higher than
flash-lite; BuffData never hardcodes provider pricing anywhere (`buffdata audit
usage-report` requires you to supply your own rates) -- confirm current pricing for
your account before choosing between them at scale.

- [gemini-3.5-flash-lite report](benchmarks/results-scale-matrix-gemini/REPORT.md)
- [gemini-3.7-flash report](benchmarks/results-scale-matrix-gemini-3.7-flash/REPORT.md)

BuffData is a provider-neutral Python CLI and SDK for turning raw or inconsistent
datasets into safer, higher-quality AI training data. It validates structure,
redacts PII locally, removes duplicates, scores and refines records, quarantines
unusable data, and applies closed-set classification only when the dataset supports it.

Gemini, OpenAI, Anthropic, and a local LLM running on your own machine or your team's
network (Ollama, LM Studio, vLLM, llama.cpp) are all supported through one client
contract. BuffData never sends a failed request to a different provider automatically.

## Features

- Dataset-aware binary, multi-class, and multi-label detection.
- User overrides for classification task and class list.
- Gemini, OpenAI Responses API, and Anthropic Messages API adapters.
- Local LLM servers (Ollama, LM Studio, vLLM, llama.cpp) or any other OpenAI-compatible
  endpoint, on this machine or another one on your network -- with an optional
  structural guarantee (`--network-policy local`) that calls can never leave it.
- Local PII redaction before any dataset content reaches a provider.
- Schema-preserving JSON/JSONL/NDJSON, CSV/TSV, Parquet, Arrow/Feather/IPC,
  YAML, plain-text, and Hugging Face on-disk dataset processing.
- Gzip support for JSON, JSONL/NDJSON, CSV/TSV, and plain-text datasets.
- Exact, lexical, local-semantic, and provider-embedding deduplication.
- LLM quality scoring, targeted refinement, DPO generation, and Evol-Instruct.
- Scalable batched quality audits: score representative samples without one API call per row.
- Accepted output, rejected-record quarantine, audit report, and resumable checkpoints.

## Install

```bash
cd ~/projects/buffdata
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Developer productivity integrations

BuffData can turn its own repository into a local Graphify knowledge graph, and can
retrieve current library documentation through Context7. Graphify's AST-only build
keeps source code local. Context7 is opt-in and requires `CONTEXT7_API_KEY`.

```bash
pip install -e ".[productivity]"
buffdata productivity graph-build .
buffdata productivity graph-query "How does optimize reach the quality scorer?"
buffdata productivity docs-search pydantic
buffdata productivity docs-context /pydantic/pydantic "How do model validators work?"
```

Graph output is written to `graphify-out/` and is excluded from source control.

Configure only the providers you use:

```env
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
BUFFDATA_PROVIDER=gemini
```

Balanced defaults are `gemini-3.7-flash`, `gpt-5.4-mini`, and
`claude-sonnet-4-6`. Use `--model` or pipeline YAML to override them.

## Local LLM servers

Every command that talks to an LLM (`optimize`, `score`, `refine`, `evolve`, `dpo`,
`augment`, `classify`, `filter-ppl`, ...) can point at a model running on your own
machine or another machine on your network instead of a public cloud API -- no data
leaves your infrastructure, and there's no per-token bill. Four common local servers
have a named `--provider` that fills in their well-known default port automatically:

| `--provider` | Default `--base-url` | Override env var |
|---|---|---|
| `ollama` | `http://localhost:11434/v1` | `OLLAMA_BASE_URL` |
| `lmstudio` | `http://localhost:1234/v1` | `LMSTUDIO_BASE_URL` |
| `vllm` | `http://localhost:8000/v1` | `VLLM_BASE_URL` |
| `llamacpp` | `http://localhost:8080/v1` | `LLAMACPP_BASE_URL` |

All four (and the fully-generic `openai_compatible`, for anything else that speaks the
OpenAI chat-completions protocol -- an internal gateway, LiteLLM, TGI, ...) construct
the same client under the hood; they exist as separate names only so you don't have to
spell the base URL out by hand for the common case:

```bash
# A model already running via `ollama serve` on this machine
buffdata score data.jsonl -o scored.jsonl --provider ollama --model llama3.1

# The same model, but served from a shared GPU box elsewhere on the network --
# --base-url (or that provider's own *_BASE_URL env var) always overrides the default
buffdata score data.jsonl -o scored.jsonl --provider ollama \
  --base-url http://192.168.1.50:11434/v1 --model llama3.1

# vLLM, LM Studio, llama.cpp's server, or anything else OpenAI-compatible work the same way
buffdata optimize data.jsonl -o optimized.jsonl --provider vllm --model mistral-7b-instruct
buffdata refine data.jsonl -o refined.jsonl --provider openai_compatible \
  --base-url http://internal-llm-gateway.corp:8000/v1 --model house-model
```

None of these servers need an API key by default (`OLLAMA_API_KEY` etc. are read if you
*have* put one in front of a shared server -- most local setups don't need to).

### `--network-policy local`: a structural guarantee, not just a default

`--network-policy strict` (see below) blocks every LLM call outright. `local` is the
middle ground for exactly this use case: calls are allowed, but only to a provider
whose endpoint you control (`openai_compatible` or one of the four above), and only
when that endpoint's host is unambiguously loopback or private --

```bash
buffdata optimize data.jsonl -o optimized.jsonl --provider ollama --network-policy local
```

-- `gemini`/`openai`/`anthropic`/`azure_openai`/`bedrock_anthropic` are rejected
outright (checked at config-construction time, before any data is touched), and a
`--base-url` pointed at a public hostname is rejected too (checked before any client is
constructed). The host check is deliberately DNS-free: it only recognizes a literal
loopback/private IP (`127.x`, `10.x`, `172.16-31.x`, `192.168.x`), the literal
`localhost`, or a `.local` mDNS hostname -- a real internet hostname is never treated as
local even if it happens to resolve to a private address today, since that resolution
could change later without this guarantee changing with it. Same idea as
`buffdata/engine/client.py`'s `NetworkForbiddenClient` for `strict`: a reviewer can
verify "this run can only ever reach our own network" from this one check, not by
auditing every stage.

Verified for real in `tests/test_local_llm.py`: a minimal local HTTP server standing in
for Ollama/vLLM/LM Studio's real `/v1/chat/completions` endpoint, with the actual
`openai` SDK client making genuine HTTP requests to it over loopback (both sync and
async), plus every host-classification edge case (`localhost`, `127.x`, all three
private ranges, `169.254.x` link-local, `.local` hostnames vs. real public IPs and
real provider hostnames like `api.openai.com`).

## Adaptive workflow

```bash
buffdata optimize data.jsonl -o optimized.jsonl --provider openai
```

Input and output containers are selected from their paths. Formats can be converted
while optimizing:

```bash
# CSV to Parquet
buffdata optimize train.csv -o train.parquet --accuracy-contract strict

# Compressed NDJSON to Arrow IPC
buffdata optimize train.ndjson.gz -o train.arrow --accuracy-contract strict

# Hugging Face save_to_disk directory to compressed TSV
buffdata optimize train.hf -o train.tsv.gz --accuracy-contract strict

# Parquet to an atomic Hugging Face dataset directory
buffdata optimize train.parquet -o train.hf --accuracy-contract strict
```

Supported suffixes are `.json`, `.jsonl`, `.ndjson`, `.csv`, `.tsv`, `.parquet`,
`.pq`, `.arrow`, `.feather`, `.ipc`, `.yaml`, `.yml`, `.txt`, their documented gzip
variants, and `.hf` directories. Existing Hugging Face `save_to_disk` directories
are detected directly, including `DatasetDict` splits. Unknown extensions fail
explicitly instead of being silently interpreted as JSONL.

The run produces:

- `optimized.jsonl`: accepted, optimized records.
- `optimized.rejected.jsonl`: invalid, duplicate, unsafe, failed, or low-quality records with reasons.
- `optimized.report.json`: profile, stage metrics, classification decision, provider/model, and token usage.

Automatic classification is conservative. Existing labels are inspected locally;
unlabeled raw/custom datasets are profiled using up to 100 evenly distributed,
PII-redacted samples. Classification is skipped for chat, Alpaca, DPO, pretraining,
open-ended, or low-confidence data.

For labeled classification, the default PII policy redacts only high-confidence
direct identifiers while preserving names, locations, dates, and topic-bearing
entities. When accuracy must never fall below a clean original, explicitly select
`--accuracy-contract strict`. Strict mode disables labeled text redaction, refinement,
quality filtering, relabeling, and all deduplication. Every valid labeled row remains
in the output with identical classification text and labels. This opt-in preserves
the privacy-safe default for other runs; use balanced mode for measured cleaning gains.

Overrides:

```bash
# Disable classification
buffdata optimize data.jsonl -o clean.jsonl --classification off

# Choose the labeled-classification privacy/accuracy tradeoff explicitly
buffdata optimize data.jsonl -o clean.jsonl --classification-pii-mode identifiers

# Enforce the lossless labeled-data path
buffdata optimize data.jsonl -o clean.jsonl --accuracy-contract strict

# Supply task and classes explicitly
buffdata optimize data.jsonl -o labeled.jsonl \
  --classification binary --classes negative,positive

# Infer classes inside a required task
buffdata classify data.jsonl -o labeled.jsonl \
  --type multi-label --provider anthropic

# Large labeled dataset: validate/deduplicate every row and audit 40 rows in two batches
buffdata optimize data.jsonl -o optimized.jsonl --provider gemini \
  --model gemini-3.5-flash-lite --quality-mode sampled --quality-sample-size 40
```

Quality modes are explicit. `llm` scores and filters every row, `sampled` audits a
representative subset in structured batches without filtering unscored rows, and
`off` runs only local stages. Reports distinguish these modes and record provider
token usage. Sampled mode is recommended for initial audits of large labeled datasets.

Run from YAML:

```bash
buffdata pipeline examples/pipeline_config.yaml -i data.jsonl -o optimized.jsonl
```

## Existing focused commands

API-backed commands accept `--provider` and `--model`:

```bash
buffdata score data.jsonl -o scored.jsonl --provider openai
buffdata refine data.jsonl -o refined.jsonl --provider anthropic
buffdata evolve data.jsonl -o evolved.jsonl --provider gemini
buffdata dpo data.jsonl -o dpo.jsonl --provider openai
buffdata synthesize source.pdf -o synthetic.jsonl --provider anthropic
buffdata augment labeled.jsonl -o augmented.jsonl --provider gemini --multiplier 1
```

Local and integration commands remain available:

```bash
buffdata dedup data.jsonl -o deduped.jsonl --method minhash
buffdata scrub data.jsonl -o scrubbed.jsonl
buffdata validate data.jsonl
buffdata stats data.jsonl
buffdata report data.jsonl -o report.html
buffdata export data.jsonl training.json --format unsloth
```

## Python SDK

```python
import asyncio
from buffdata import OptimizationPipeline, PipelineConfig

config = PipelineConfig(provider="openai", classification="auto")
result = asyncio.run(
    OptimizationPipeline(config).run_file("data.jsonl", "optimized.jsonl")
)
print(result.metrics)
```

Legacy `GeminiClient()` construction remains supported. New integrations should use
`create_llm_client()` or `OptimizationPipeline`.

## Tests

```bash
pytest tests/ -v
```

Normal tests are offline and mock provider calls. Live API smoke tests should be run
separately with explicit credentials.

## License

MIT
