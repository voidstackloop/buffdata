import time

import pytest

from buffdata.observability import ObservabilityRegistry, disabled_registry


# --- disabled by default / opt-out ---------------------------------------------------------

def test_disabled_registry_every_method_is_a_real_noop():
    obs = ObservabilityRegistry(enabled=False)
    assert obs.enabled is False
    assert obs.registry is None

    with obs.stage_span("validate"):
        pass  # must not raise
    obs.record_stage_outcome("validate", accepted=10, rejected=2)
    obs.record_llm_error("gemini")
    obs.record_tokens("gemini", "gemini-3.7-flash", input_tokens=100, output_tokens=50)
    assert obs.export_prometheus_text() == ""


def test_disabled_registry_helper_is_shared_and_disabled():
    a = disabled_registry()
    b = disabled_registry()
    assert a is b
    assert a.enabled is False


# --- enabled: Prometheus metrics ------------------------------------------------------------

def test_enabled_registry_reports_both_backends_available():
    obs = ObservabilityRegistry(enabled=True)
    # Both opentelemetry-sdk and prometheus_client are installed in this environment.
    assert obs.prometheus_enabled is True
    assert obs.otel_enabled is True


def test_stage_span_records_duration_histogram():
    obs = ObservabilityRegistry(enabled=True)
    with obs.stage_span("validate"):
        time.sleep(0.01)

    text = obs.export_prometheus_text()
    assert 'buffdata_stage_duration_seconds_count{stage="validate"} 1.0' in text
    assert "buffdata_stage_duration_seconds_sum" in text


def test_stage_span_records_even_when_body_raises():
    obs = ObservabilityRegistry(enabled=True)
    with pytest.raises(ValueError):
        with obs.stage_span("dedup"):
            raise ValueError("boom")

    text = obs.export_prometheus_text()
    assert 'buffdata_stage_duration_seconds_count{stage="dedup"} 1.0' in text


def test_record_stage_outcome_increments_correct_labels():
    obs = ObservabilityRegistry(enabled=True)
    obs.record_stage_outcome("validate", accepted=8, rejected=2)
    obs.record_stage_outcome("validate", accepted=3, rejected=0)

    text = obs.export_prometheus_text()
    assert 'buffdata_records_total{outcome="accepted",stage="validate"} 11.0' in text
    assert 'buffdata_records_total{outcome="rejected",stage="validate"} 2.0' in text


def test_record_stage_outcome_skips_zero_values_without_erroring():
    obs = ObservabilityRegistry(enabled=True)
    obs.record_stage_outcome("filter", accepted=0, rejected=0)
    # No exception, and no bogus zero-valued series forced into existence for labels that
    # never actually occurred.
    text = obs.export_prometheus_text()
    assert 'stage="filter"' not in text


def test_record_llm_error_increments_by_provider():
    obs = ObservabilityRegistry(enabled=True)
    obs.record_llm_error("gemini")
    obs.record_llm_error("gemini")
    obs.record_llm_error("anthropic")

    text = obs.export_prometheus_text()
    assert 'buffdata_llm_errors_total{provider="gemini"} 2.0' in text
    assert 'buffdata_llm_errors_total{provider="anthropic"} 1.0' in text


def test_record_tokens_splits_by_direction():
    obs = ObservabilityRegistry(enabled=True)
    obs.record_tokens("gemini", "gemini-3.7-flash", input_tokens=100, output_tokens=40)

    text = obs.export_prometheus_text()
    assert 'buffdata_tokens_total{direction="input",model="gemini-3.7-flash",provider="gemini"} 100.0' in text
    assert 'buffdata_tokens_total{direction="output",model="gemini-3.7-flash",provider="gemini"} 40.0' in text


def test_two_registries_do_not_collide():
    # Each ObservabilityRegistry must own its own CollectorRegistry -- using Prometheus's
    # shared global default registry would make a second pipeline run (or a second test)
    # raise "Duplicated timeseries" on construction.
    first = ObservabilityRegistry(enabled=True)
    second = ObservabilityRegistry(enabled=True)  # must not raise
    first.record_llm_error("gemini")
    second.record_llm_error("openai")

    assert "gemini" in first.export_prometheus_text()
    assert "gemini" not in second.export_prometheus_text()
    assert "openai" in second.export_prometheus_text()


# --- enabled: OpenTelemetry spans ------------------------------------------------------------

def test_stage_span_creates_a_real_otel_span_when_a_provider_is_configured():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    # OpenTelemetry's TracerProvider can only be set once per process (later calls are a
    # documented no-op), so this test relies on being the only one in the suite that sets
    # one -- get_tracer("buffdata") is called *after* this, and correctly resolves against
    # whatever the current global provider is via OTel's proxy-tracer mechanism, which is
    # exactly what makes this work regardless of when a real SDK provider gets configured.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    obs = ObservabilityRegistry(enabled=True)
    with obs.stage_span("classify"):
        pass
    spans = exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].name == "buffdata.stage.classify"
