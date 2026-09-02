import pytest

from buffdata.engine.pipeline import OptimizationPipeline, STAGES
from buffdata.models.schemas import DatasetItem, PipelineConfig
from buffdata.observability import disabled_registry
from tests.test_adaptive_pipeline import PipelineFakeClient


@pytest.mark.asyncio
async def test_observability_off_by_default_uses_shared_disabled_registry():
    client = PipelineFakeClient()
    items = [DatasetItem.from_dict({"text": f"Record {index}", "label": index % 2}) for index in range(4)]
    config = PipelineConfig(
        provider="anthropic",
        model="fake-balanced",
        dedup_method="exact",
        scrub_pii=False,
        quality_mode="off",
    )

    pipeline = OptimizationPipeline(config, client=client)
    assert pipeline._observability is disabled_registry()

    await pipeline.run(items)

    assert pipeline._observability.export_prometheus_text() == ""


@pytest.mark.asyncio
async def test_observability_enabled_records_a_span_and_outcome_per_stage():
    client = PipelineFakeClient()
    items = [
        DatasetItem.from_dict({"text": "Michael Phelps won in Athens; email person@example.com", "label": 1}),
        DatasetItem.from_dict({"text": "Stocks rose after the earnings report", "label": 0}),
    ]
    config = PipelineConfig(
        provider="anthropic",
        model="fake-balanced",
        quality_mode="off",
        classification="off",
        dedup_method="exact",
        scrub_pii=True,
        observability=True,
    )

    pipeline = OptimizationPipeline(config, client=client)
    result = await pipeline.run(items)

    text = pipeline._observability.export_prometheus_text()
    for stage in STAGES:
        assert f'buffdata_stage_duration_seconds_count{{stage="{stage}"}} 1.0' in text

    assert 'buffdata_records_total{outcome="accepted",stage="validate"} 2.0' in text
    assert len(result.accepted) == 2


@pytest.mark.asyncio
async def test_observability_enabled_records_token_usage_from_the_client():
    client = PipelineFakeClient()
    assert client.usage == {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
    items = [DatasetItem.from_dict({"text": "Stocks rose after the earnings report", "label": 0})]
    config = PipelineConfig(
        provider="anthropic",
        model="fake-balanced",
        quality_mode="off",
        classification="off",
        dedup_method="exact",
        scrub_pii=False,
        observability=True,
    )

    pipeline = OptimizationPipeline(config, client=client)
    await pipeline.run(items)

    text = pipeline._observability.export_prometheus_text()
    assert (
        'buffdata_tokens_total{direction="input",model="fake-balanced",provider="anthropic"} 12.0' in text
    )
    assert (
        'buffdata_tokens_total{direction="output",model="fake-balanced",provider="anthropic"} 4.0' in text
    )


@pytest.mark.asyncio
async def test_observability_enabled_writes_a_prometheus_textfile_sidecar(tmp_path):
    client = PipelineFakeClient()
    items = [DatasetItem.from_dict({"text": "Stocks rose after the earnings report", "label": 0})]
    config = PipelineConfig(
        provider="anthropic",
        model="fake-balanced",
        quality_mode="off",
        classification="off",
        dedup_method="exact",
        scrub_pii=False,
        observability=True,
    )
    output = tmp_path / "optimized.jsonl"

    await OptimizationPipeline(config, client=client).run(items, output_path=output)

    sidecar = tmp_path / "optimized.metrics.prom"
    assert sidecar.exists()
    text = sidecar.read_text(encoding="utf-8")
    assert 'buffdata_stage_duration_seconds_count{stage="validate"} 1.0' in text


@pytest.mark.asyncio
async def test_observability_disabled_writes_no_prometheus_sidecar(tmp_path):
    client = PipelineFakeClient()
    items = [DatasetItem.from_dict({"text": "Stocks rose after the earnings report", "label": 0})]
    config = PipelineConfig(
        provider="anthropic",
        model="fake-balanced",
        quality_mode="off",
        classification="off",
        dedup_method="exact",
        scrub_pii=False,
    )
    output = tmp_path / "optimized.jsonl"

    await OptimizationPipeline(config, client=client).run(items, output_path=output)

    assert not (tmp_path / "optimized.metrics.prom").exists()


@pytest.mark.asyncio
async def test_observability_records_rejections_at_the_stage_that_caused_them():
    client = PipelineFakeClient()
    items = [
        DatasetItem.from_dict({"text": "Stocks rose after the earnings report", "label": 0}),
        DatasetItem.from_dict({"text": "Stocks rose after the earnings report", "label": 0}),  # exact duplicate
    ]
    config = PipelineConfig(
        provider="anthropic",
        model="fake-balanced",
        quality_mode="off",
        classification="off",
        dedup_method="exact",
        scrub_pii=False,
        observability=True,
    )

    pipeline = OptimizationPipeline(config, client=client)
    result = await pipeline.run(items)

    assert len(result.rejected) == 1
    text = pipeline._observability.export_prometheus_text()
    assert 'buffdata_records_total{outcome="rejected",stage="dedup"} 1.0' in text
    # No other stage should have claimed this rejection.
    assert 'buffdata_records_total{outcome="rejected",stage="validate"}' not in text
