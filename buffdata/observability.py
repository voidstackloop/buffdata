"""Optional OpenTelemetry tracing + Prometheus metrics for OptimizationPipeline runs.

Both libraries are optional (buffdata[observability]) and this module degrades to a real
no-op -- not an error -- whenever they aren't installed or observability isn't explicitly
enabled (PipelineConfig.observability defaults to False), so nothing about a normal run's
behavior, output, or performance changes unless a caller opts in.

This module only does the instrumentation: creating spans, recording metric values. Where
those spans/metrics actually go (Datadog, Grafana via a Prometheus scrape, Jaeger, ...) is
standard OpenTelemetry SDK / Prometheus configuration the deployment does on its own --
setting a TracerProvider with the exporter of its choice, or serving
export_prometheus_text() from an HTTP endpoint a Prometheus server scrapes. This module
never picks a destination for you.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:
    from opentelemetry import trace as _otel_trace
    _OTEL_AVAILABLE = True
except ImportError:
    _otel_trace = None
    _OTEL_AVAILABLE = False

try:
    from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


@contextmanager
def _null_context() -> Iterator[None]:
    yield None


class ObservabilityRegistry:
    """One instance per pipeline run. Holds its own Prometheus CollectorRegistry (never the
    global default registry) so concurrent runs -- or concurrent tests -- never collide on
    shared metric state.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled) and (_OTEL_AVAILABLE or _PROMETHEUS_AVAILABLE)
        self.otel_enabled = self.enabled and _OTEL_AVAILABLE
        self.prometheus_enabled = self.enabled and _PROMETHEUS_AVAILABLE

        self.registry: Optional["CollectorRegistry"] = None
        self._stage_duration = None
        self._records_total = None
        self._llm_errors_total = None
        self._tokens_total = None
        if self.prometheus_enabled:
            self.registry = CollectorRegistry()
            self._stage_duration = Histogram(
                "buffdata_stage_duration_seconds", "Time spent in each pipeline stage",
                ["stage"], registry=self.registry,
            )
            self._records_total = Counter(
                "buffdata_records_total", "Records processed, by stage and outcome",
                ["stage", "outcome"], registry=self.registry,
            )
            self._llm_errors_total = Counter(
                "buffdata_llm_errors_total", "LLM call errors, by provider",
                ["provider"], registry=self.registry,
            )
            self._tokens_total = Counter(
                "buffdata_tokens_total", "LLM tokens used, by provider/model and direction",
                ["provider", "model", "direction"], registry=self.registry,
            )

        self._tracer = _otel_trace.get_tracer("buffdata") if self.otel_enabled else None

    @contextmanager
    def stage_span(self, stage: str) -> Iterator[Any]:
        """Times `stage` into the duration histogram, and -- when OpenTelemetry is enabled
        -- wraps it in a span named buffdata.stage.<stage> so it shows up as a child of
        whatever span is current (e.g. one overall buffdata.pipeline.run span), the same way
        any other instrumented library's spans nest into a caller's trace.
        """
        started = time.perf_counter()
        span_cm = self._tracer.start_as_current_span(f"buffdata.stage.{stage}") if self._tracer else _null_context()
        with span_cm as span:
            try:
                yield span
            finally:
                if self._stage_duration is not None:
                    self._stage_duration.labels(stage=stage).observe(time.perf_counter() - started)

    def record_stage_outcome(self, stage: str, *, accepted: int = 0, rejected: int = 0) -> None:
        if self._records_total is None:
            return
        if accepted:
            self._records_total.labels(stage=stage, outcome="accepted").inc(accepted)
        if rejected:
            self._records_total.labels(stage=stage, outcome="rejected").inc(rejected)

    def record_llm_error(self, provider: str) -> None:
        if self._llm_errors_total is not None:
            self._llm_errors_total.labels(provider=provider).inc()

    def record_tokens(self, provider: str, model: str, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        if self._tokens_total is None:
            return
        if input_tokens:
            self._tokens_total.labels(provider=provider, model=model, direction="input").inc(input_tokens)
        if output_tokens:
            self._tokens_total.labels(provider=provider, model=model, direction="output").inc(output_tokens)

    def export_prometheus_text(self) -> str:
        """Current metric values in Prometheus text exposition format -- what a /metrics
        HTTP endpoint would serve to a real Prometheus scraper. Empty string when
        Prometheus support isn't enabled/installed."""
        if self.registry is None:
            return ""
        return generate_latest(self.registry).decode("utf-8")


_disabled_registry = ObservabilityRegistry(enabled=False)


def disabled_registry() -> ObservabilityRegistry:
    """A shared, always-disabled registry -- every method is a real no-op. Callers that
    don't want to construct their own ObservabilityRegistry (most call sites, since
    observability is opt-in) can default to this instead of writing `if self._obs:` checks
    everywhere.
    """
    return _disabled_registry
