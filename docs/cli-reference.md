# CLI reference

Every command supports `--help` for its complete, always-current flag list -- this
page covers what each one is *for* and the flags that matter most for choosing between
them. Provider-calling commands share a common flag set covered once in
[Shared provider/governance flags](#shared-providergovernance-flags) rather than
repeated per command.

## The pipeline commands

### `buffdata optimize`

The main entry point: runs the full 7-stage pipeline (see
[architecture.md](architecture.md)) with CLI flags overriding a `--config` YAML file
where both are given.

```bash
buffdata optimize data.jsonl -o optimized.jsonl --provider openai
buffdata optimize train.csv -o train.parquet --accuracy-contract strict
buffdata optimize data.jsonl -o out.jsonl --classification off
buffdata optimize data.jsonl -o out.jsonl --classification-pii-mode identifiers
buffdata optimize data.jsonl -o out.jsonl \
  --provider gemini --model gemini-3.5-flash-lite \
  --quality-mode sampled --quality-sample-size 40
```

Input/output format is inferred from the file extension -- `.json`, `.jsonl`,
`.ndjson`, `.csv`, `.tsv`, `.parquet`/`.pq`, `.arrow`/`.feather`/`.ipc`, `.yaml`/`.yml`,
`.txt`, their `.gz` variants, and `.hf` (Hugging Face `save_to_disk`) directories --
converting between formats is free (`optimize train.csv -o train.parquet`).
`optimize`/`pipeline` don't accept cloud (`s3://`/`gs://`/`az://`) URLs, since their
checkpoint/report/rejected sidecar files assume a local filesystem; every other
provider-calling command below does.

Produces three files beside the output: `<name>.rejected.jsonl` (every dropped row
with a reason), `<name>.report.json` (profile, per-stage metrics, provider/model,
token usage -- what `buffdata audit record` and `buffdata contract check` both read),
and, with `--observability`, `<name>.metrics.prom` (see
[observability.md](observability.md)).

Accuracy-gate flags (`--validation-file`, `--require-positive-gain`,
`--accuracy-min-gain`, `--accuracy-seeds`, `--accuracy-epochs`,
`--accuracy-max-train-rows`) publish the output only when it's proven, against a
held-out labeled set, to not regress accuracy -- the same machinery `buffdata
generate` uses (below).

### `buffdata pipeline <config.yaml> -i input -o output`

Same engine as `optimize`, but every setting comes from a
[`PipelineConfig`](configuration.md) YAML file (see
[`examples/pipeline_config.yaml`](../examples/pipeline_config.yaml)) with no
per-flag overrides. Use this when a run needs to be checked into source control as
config, or wired into an orchestrator that shells out with a config path rather than a
long flag list.

### `buffdata generate <labeled.jsonl> -o generated.jsonl --validation-file held_out.jsonl`

Iteratively augments a labeled classification dataset -- generating variations,
retraining a proxy classifier, measuring the accuracy gain against
`--validation-file` -- until it clears `--min-relative-gain` (default 10%) or exhausts
`--max-iterations` (default 5), reporting the shortfall honestly rather than
publishing an unproven result either way. `--weak-class-count` targets retry rounds at
whichever classes are dragging accuracy down. Backed by
`buffdata/optimizers/gated_generator.py` and the same
`buffdata/evaluation/accuracy_gate.py` proxy-classifier machinery `optimize
--require-positive-gain` uses.

## Focused single-stage commands

Each does one pipeline stage's job standalone, and (unlike `optimize`/`pipeline`)
every one of these accepts cloud URLs directly:

```bash
buffdata score data.jsonl -o scored.jsonl --provider openai        # LLM-as-a-judge quality scoring
buffdata refine data.jsonl -o refined.jsonl --provider anthropic   # clean/expand/de-boilerplate
buffdata evolve data.jsonl -o evolved.jsonl --provider gemini      # Evol-Instruct complexity growth
buffdata dpo data.jsonl -o dpo.jsonl --provider openai             # chosen/rejected preference pairs
buffdata augment labeled.jsonl -o augmented.jsonl --provider gemini --multiplier 1  # label-guarded synthetic variations
buffdata classify data.jsonl -o labeled.jsonl --type multi-label --provider anthropic
buffdata dedup data.jsonl -o deduped.jsonl --method minhash        # exact | minhash | semantic-local | semantic
buffdata scrub data.jsonl -o scrubbed.jsonl                        # local PII redaction, no provider needed
buffdata validate data.jsonl                                       # schema/structural check only
buffdata filter-ppl data.jsonl -o filtered.jsonl                   # local PyTorch perplexity filter, no provider needed
buffdata synthesize source.pdf -o synthetic.jsonl --provider anthropic  # documents -> instruction pairs
buffdata stats data.jsonl                                          # quality/statistics summary
buffdata report data.jsonl -o report.html                          # interactive HTML audit report
```

## Data movement

```bash
buffdata push optimized.jsonl user/my-dataset --token $HF_TOKEN   # to the Hugging Face Hub
buffdata export data.jsonl training.json --format unsloth          # to Axolotl/Unsloth training format
```

## Scale and distribution

See [scaling.md](scaling.md) for the full picture; in short:

```bash
buffdata shard data.jsonl -o parts/ -n 8                    # split for orchestrator-level parallelism
# ... N parallel `buffdata optimize` runs, one per partition-NNNN.jsonl (Airflow/Dagster/k8s/xargs -P) ...
buffdata run-ray pipeline.yaml parts/partition-*.jsonl -o out/  # or: distribute those same partitions as Ray tasks instead
buffdata merge out/*.jsonl -o final.jsonl                    # recombine either way
```

## Governance and compliance

See [governance.md](governance.md) for the full picture.

```bash
buffdata sbom -o sbom.json                                          # CycloneDX 1.5 SBOM of installed packages
buffdata contract check contract.yaml --report run.report.json      # CI gate: exit 0 pass / 1 fail
buffdata audit record run.report.json --command optimize            # ingest a run into the durable audit log
buffdata audit query --command optimize --since 2026-01-01           # list recorded runs
buffdata audit usage-report --pricing my_rates.yaml                  # aggregate token usage (never guesses $ pricing)
```

## Developer productivity

Opt-in, requires `pip install -e ".[productivity]"`:

```bash
buffdata productivity graph-build .                                          # local AST-only Graphify knowledge graph
buffdata productivity graph-query "How does optimize reach the quality scorer?"
buffdata productivity docs-search pydantic                                    # find a Context7 library id
buffdata productivity docs-context /pydantic/pydantic "How do model validators work?"
```

## Shared provider/governance flags

Present on every command that can call an LLM (`optimize`, `score`, `refine`,
`evolve`, `dpo`, `augment`, `classify`, `filter-ppl` where relevant, `synthesize`,
`generate`, `pipeline` via YAML):

| Flag | Purpose |
|---|---|
| `--provider` | `gemini`, `openai`, `anthropic`, `azure_openai`, `bedrock_anthropic`, `openai_compatible`, or a local server (`ollama`, `lmstudio`, `vllm`, `llamacpp`) -- see [providers.md](providers.md) |
| `--model` | Provider model override; each provider has a balanced default except the private/local ones, which have none (there's no universal default a customer gateway or a locally-pulled model could safely assume) |
| `--base-url` | Endpoint for `openai_compatible`/local-server providers -- another machine on your network, e.g. `http://192.168.1.50:11434/v1` |
| `--network-policy` | `unrestricted` (default), `local` (only a provider whose endpoint you control, on a loopback/private address), or `strict` (zero network calls, guaranteed structurally) |
| `--actor` / `--policy` | Check this actor's permissions against an access-policy YAML before running -- [governance.md](governance.md#access-policy-rbac) |
| `--bearer-token` / `--oidc-config` | Verify a real OIDC bearer token instead of trusting a plain `--actor` string -- [governance.md](governance.md#oidc-bearer-token-verification) |
| `--concurrency` / `--rpm` | Async request concurrency and rate limit |
