# Observability

Opt-in OpenTelemetry tracing and Prometheus metrics for a pipeline run --
[`buffdata/observability.py`](../buffdata/observability.py). A no-op, not an error,
whenever the libraries aren't installed or `observability` isn't explicitly enabled:
nothing about a normal run's behavior, output, or performance changes unless you opt in.

```bash
pip install -e ".[observability]"    # opentelemetry-sdk, opentelemetry-exporter-otlp, prometheus-client
```

```yaml
# pipeline.yaml
observability: true
```

```bash
buffdata pipeline pipeline.yaml -i data.jsonl -o optimized.jsonl
```

## What you get

`ObservabilityRegistry` ([`buffdata/observability.py`](../buffdata/observability.py))
is constructed once per `OptimizationPipeline` run and wraps every one of the 7 stages
(see [architecture.md](architecture.md)):

- **A span per stage**: `buffdata.stage.<name>` (`buffdata.stage.validate`,
  `buffdata.stage.dedup`, ...), timed into a `buffdata_stage_duration_seconds`
  histogram either way. When an OpenTelemetry `TracerProvider` is configured, spans
  nest under whatever span is already current -- standard OTel behavior, no BuffData-
  specific wiring needed on the consuming end.
- **`buffdata_records_total{stage, outcome}`**: accepted/rejected counts per stage,
  attributed to whichever stage actually caused the rejection (verified in
  `tests/test_pipeline_observability.py` -- e.g. a rejection lands on `dedup`, not
  `validate`, when that's genuinely where it happened).
- **`buffdata_tokens_total{provider, model, direction}`**: input/output token usage
  from the run's real client usage counters, recorded once at the end of the run.
- **`buffdata_llm_errors_total{provider}`**: exists in the registry API, not currently
  wired into pipeline stages (see [Known gaps](#known-gaps)).

## Where the metrics go

This module only does the instrumentation -- creating spans, recording values. Where
they're actually exported (Datadog, Jaeger, a Prometheus scrape endpoint, ...) is
standard OpenTelemetry SDK / Prometheus configuration your deployment does on its own,
not something this flag controls.

For a batch CLI run specifically (not a long-lived process with a `/metrics` HTTP
endpoint to scrape), enabling `observability` also writes a
`<output>.metrics.prom` sidecar file next to the output dataset -- the standard
Prometheus `node_exporter` **textfile collector** format. Point
`--collector.textfile.directory` at wherever that lands (a local directory, or a
PVC in the [Helm chart](deployment.md#helm-chart)) and a normal Prometheus setup
picks it up on its own schedule; no push gateway or long-running process required.

## Known gaps

- `record_llm_error()` is implemented and tested in isolation
  (`tests/test_observability.py`) but not yet called from the 6 optimizer stages that
  can raise a provider error (`scorer.py`, `refiner.py`, `evolver.py`, `preference.py`,
  `classifier.py`, `augmenter.py`) -- wiring it in would require touching each of those
  again at the same site where `except ProviderError: raise` already lives. Not
  critical today since failure/success is already visible via whether the run
  completes or raises, and per-item errors already roll into the report's
  `rejection_reasons`.
- No dedicated `buffdata metrics` command to print or serve `export_prometheus_text()`
  standalone -- the `.metrics.prom` sidecar file is the current hand-off point.

Fully tested: 11 unit tests (`tests/test_observability.py`, including a real OTel span
captured via `InMemorySpanExporter`, no mocking) plus 6 pipeline-integration tests
(`tests/test_pipeline_observability.py`) covering the off-by-default path, per-stage
span+outcome attribution, token recording, and the `.metrics.prom` sidecar itself.
