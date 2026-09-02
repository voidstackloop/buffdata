import json
import os
import stat
import sys

import pytest

from buffdata.engine.client import LLMProvider
from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.models.schemas import (
    ClassificationTask,
    DatasetItem,
    PipelineConfig,
    QualityAuditBatch,
    QualityAuditEntry,
    QualityScore,
)
from buffdata.optimizers.classifier import ClassificationResult, ClassificationSchema
from buffdata.optimizers.scrubber import PIIScrubber, _scrub_worker


class PipelineFakeClient:
    provider = LLMProvider.ANTHROPIC
    default_model = "fake-balanced"

    def __init__(self):
        self.prompts = []
        self.usage = {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}

    async def generate_structured_async(self, prompt, response_schema, **kwargs):
        self.prompts.append(prompt)
        if response_schema is ClassificationSchema:
            return ClassificationSchema(
                applicable=True,
                task_type=ClassificationTask.BINARY,
                classes=["negative", "positive"],
                confidence=0.92,
                reasoning="Closed-set sentiment dataset.",
            )
        if response_schema.__name__ == "ClassificationResult":
            return response_schema(id="dummy", labels=["positive"], confidence=0.9)
        if response_schema.__name__ == "BatchClassificationResponse":
            import re
            ids = re.findall(r"ID: (.*?)\n", prompt)
            from buffdata.optimizers.classifier import BatchClassificationResponse, ClassificationResult
            results = [ClassificationResult(id=id, labels=["positive"], confidence=0.9) for id in ids]
            return BatchClassificationResponse(results=results)
        if response_schema is QualityScore:
            return QualityScore(
                overall_score=9.0,
                clarity=9.0,
                factual_accuracy=9.0,
                reasoning_depth=8.0,
                instruction_following=9.0,
                is_safe=True,
            )
        if response_schema is QualityAuditBatch:
            import re

            item_ids = re.findall(r'"item_id":\s*"([^"]+)"', prompt)
            return QualityAuditBatch(entries=[
                QualityAuditEntry(
                    item_id=item_id,
                    score=QualityScore(
                        overall_score=9.0,
                        clarity=9.0,
                        factual_accuracy=9.0,
                        reasoning_depth=8.0,
                        instruction_following=9.0,
                        is_safe=True,
                    ),
                )
                for item_id in item_ids
            ])
        raise AssertionError(response_schema)


@pytest.mark.asyncio
async def test_pipeline_redacts_before_remote_calls_and_writes_sidecars(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "optimized.jsonl"
    source.write_text(
        json.dumps({"text": "A great result from person@example.com", "source": "fixture"}) + "\n",
        encoding="utf-8",
    )
    client = PipelineFakeClient()
    config = PipelineConfig(
        provider="anthropic",
        model="fake-balanced",
        dedup_method="exact",
        filter_min_score=7.0,
        scrub_pii=True,
    )
    result = await OptimizationPipeline(config, client=client).run_file(source, output)

    assert len(result.accepted) == 1
    assert result.accepted[0].labels == "positive"
    assert result.accepted[0].to_dict()["source"] == "fixture"
    assert all("person@example.com" not in prompt for prompt in client.prompts)
    assert "<EMAIL_ADDRESS>" in result.accepted[0].text
    assert output.exists()
    assert (tmp_path / "optimized.rejected.jsonl").exists()
    report = json.loads((tmp_path / "optimized.report.json").read_text(encoding="utf-8"))
    assert report["metrics"]["provider"] == "anthropic"
    assert not (tmp_path / ".optimized.checkpoint.json").exists()


@pytest.mark.asyncio
async def test_sampled_quality_mode_audits_without_filtering_all_rows():
    client = PipelineFakeClient()
    items = [DatasetItem.from_dict({"text": f"Record {index}", "label": index % 2}) for index in range(20)]
    config = PipelineConfig(
        provider="anthropic",
        model="fake-balanced",
        dedup_method="exact",
        scrub_pii=False,
        quality_mode="sampled",
        quality_sample_size=5,
    )

    result = await OptimizationPipeline(config, client=client).run(items)

    assert len(result.accepted) == 20
    assert len(result.rejected) == 0
    assert result.metrics["stages"]["score_refine"]["scored"] == 5
    assert result.metrics["stages"]["score_refine"]["remote_batches"] == 1
    assert any("labeled binary classification records" in prompt for prompt in client.prompts)
    assert result.metrics["stages"]["filter"]["action"] == "skipped"
    assert all(item.quality_score is None for item in result.accepted)


@pytest.mark.asyncio
async def test_labeled_classification_preserves_entities_and_uses_exact_dedup():
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
        classification_pii_mode="identifiers",
    )

    result = await OptimizationPipeline(config, client=client).run(items)

    assert "Michael Phelps" in result.accepted[0].text
    assert "Athens" in result.accepted[0].text
    assert "person@example.com" not in result.accepted[0].text
    assert result.metrics["stages"]["pii"]["policy"] == "identifiers"
    assert result.metrics["stages"]["dedup"]["method"] == "exact"


@pytest.mark.asyncio
async def test_strict_accuracy_contract_preserves_labeled_text_exactly():
    client = PipelineFakeClient()
    original = "Michael Phelps won in Athens; email person@example.com"
    items = [
        DatasetItem.from_dict({"text": original, "label": 1}),
        DatasetItem.from_dict({"text": "Stocks rose after the earnings report", "label": 0}),
    ]

    result = await OptimizationPipeline(
        PipelineConfig(
            provider="anthropic",
            model="fake-balanced",
            quality_mode="llm",
            classification="multi-class",
            dedup_method="minhash",
            classification_pii_mode="all",
            accuracy_contract="strict",
        ),
        client=client,
    ).run(items)

    assert result.accepted[0].text == original
    assert result.metrics["stages"]["pii"] == {
        "enabled": False,
        "policy": "off",
        "redacted_records": 0,
    }
    assert result.metrics["stages"]["dedup"]["method"] == "off"
    assert result.metrics["stages"]["score_refine"]["mode"] == "off"
    assert result.metrics["stages"]["filter"]["action"] == "skipped"
    assert result.metrics["stages"]["classify"]["action"] == "strict_existing_labels"
    assert result.metrics["accuracy_contract"] == "strict"


def test_parallel_scrub_worker_preserves_identifier_only_policy():
    item = DatasetItem.from_dict({
        "text": "Michael Phelps won in Athens; email person@example.com",
        "label": 1,
    })

    scrubbed = _scrub_worker(
        [item],
        "en",
        sorted(PIIScrubber.IDENTIFIER_ENTITIES),
    )[0]

    assert "Michael Phelps" in scrubbed.text
    assert "Athens" in scrubbed.text
    assert "person@example.com" not in scrubbed.text


def test_identifier_policy_uses_high_precision_phone_fallback():
    scrubber = PIIScrubber(entities=sorted(PIIScrubber.IDENTIFIER_ENTITIES))
    text = "Radio bands are 900/1800/1900 MHz; call +1 (555) 123-4567."

    redacted = scrubber.scrub_text(text)

    assert "900/1800/1900 MHz" in redacted
    assert "+1 (555) 123-4567" not in redacted
    assert "<PHONE_NUMBER>" in redacted


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="chmod doesn't express owner-only ACLs on Windows")
async def test_report_json_is_written_with_owner_only_permissions(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "optimized.jsonl"
    source.write_text(json.dumps({"text": "a record", "source": "fixture"}) + "\n", encoding="utf-8")
    client = PipelineFakeClient()
    config = PipelineConfig(provider="anthropic", model="fake-balanced", dedup_method="exact", scrub_pii=False)

    await OptimizationPipeline(config, client=client).run_file(source, output)

    report_path = tmp_path / "optimized.report.json"
    assert report_path.exists()
    mode = stat.S_IMODE(os.stat(report_path).st_mode)
    assert mode == 0o600
