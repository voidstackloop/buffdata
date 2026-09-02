import pytest

from buffdata.engine.client import LLMProvider
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.engine.profiler import DatasetProfiler, representative_sample
from buffdata.models.schemas import (
    ClassificationMode,
    ClassificationTask,
    DatasetFormat,
    DatasetItem,
)
from buffdata.optimizers.classifier import (
    ClassificationResult,
    ClassificationSchema,
    DatasetClassifier,
)


class FakeClient:
    provider = LLMProvider.OPENAI
    default_model = "fake-balanced"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def generate_structured_async(self, prompt, response_schema, **kwargs):
        if response_schema is ClassificationSchema:
            return ClassificationSchema(
                applicable=True,
                task_type=ClassificationTask.BINARY,
                classes=["negative", "positive"],
                confidence=0.9,
                reasoning="Sentiment records with two exclusive outcomes.",
            )
        if response_schema.__name__ == "BatchClassificationResponse":
            import re
            ids = re.findall(r"ID: (.*?)\n", prompt)
            from buffdata.optimizers.classifier import BatchClassificationResponse, ClassificationResult
            results = []
            for id in ids:
                label = "positive" if "great" in prompt.lower() else "negative"
                results.append(ClassificationResult(id=id, labels=[label], confidence=0.88))
            return BatchClassificationResponse(results=results)
        raise AssertionError(response_schema)


def test_representative_sample_spans_dataset():
    items = [DatasetItem(id=str(i), format=DatasetFormat.RAW, text=str(i)) for i in range(1000)]
    sample = representative_sample(items, 5)
    assert [item.id for item in sample] == ["0", "250", "500", "749", "999"]


@pytest.mark.asyncio
async def test_profile_derives_existing_multilabel_schema_without_llm():
    items = [
        DatasetItem(format=DatasetFormat.RAW, text="one", labels=["a", "b"]),
        DatasetItem(format=DatasetFormat.RAW, text="two", labels=["b"]),
    ]
    profile = await DatasetProfiler(FakeClient()).profile(items)
    assert profile.task_type == ClassificationTask.MULTI_LABEL
    assert profile.classes == ["a", "b"]
    assert profile.confidence == 1.0


@pytest.mark.asyncio
async def test_classifier_applies_schema_and_preserves_unknown_columns():
    item = DatasetItem.from_dict({"text": "A great result", "source": "fixture"})
    classifier = DatasetClassifier(FakeClient(), AsyncRateLimiter(max_rpm=1000, concurrency=2))
    schema = await classifier.resolve_schema([item], ClassificationMode.AUTO)
    result = await classifier.classify_batch([item], schema)
    output = result[0].to_dict()
    assert output["labels"] == "positive"
    assert output["source"] == "fixture"
    assert output["_buffdata_metadata"]["classification"]["provider"] == "openai"


def test_binary_override_rejects_invalid_cardinality():
    with pytest.raises(ValueError, match="exactly two"):
        ClassificationSchema(
            applicable=True,
            task_type=ClassificationTask.BINARY,
            classes=["only-one"],
            confidence=1.0,
        )

