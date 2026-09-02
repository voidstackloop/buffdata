import pytest

from buffdata.engine.client import LLMProvider
from buffdata.models.schemas import AugmentationResult, BatchAugmentationResponse, DatasetItem
from buffdata.optimizers.augmenter import DataAugmenter


class AugmentFakeClient:
    provider = LLMProvider.GEMINI
    default_model = "fake"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def generate_structured_async(self, prompt, response_schema, **kwargs):
        ids = [line.removeprefix("ID: ") for line in prompt.splitlines() if line.startswith("ID: ")]
        return BatchAugmentationResponse(results=[
            AugmentationResult(
                id=item_id,
                variations=[
                    "The team won the football championship after a strong final match.",
                    "technology shares dropped after the company earnings announcement",
                ],
            )
            for item_id in ids
        ])


@pytest.mark.asyncio
async def test_augmentation_keeps_labels_rejects_drift_and_has_stable_ids():
    items = [
        DatasetItem.from_dict({"id": "s1", "text": "The football team won the final match", "label": 0}),
        DatasetItem.from_dict({"id": "s2", "text": "The basketball team won the league game", "label": 0}),
        DatasetItem.from_dict({"id": "b1", "text": "Company shares rose after earnings", "label": 1}),
        DatasetItem.from_dict({"id": "b2", "text": "Markets fell as technology stocks declined", "label": 1}),
    ]
    augmenter = DataAugmenter(client=AugmentFakeClient())

    first = await augmenter.augment_batch_async(items, multiplier=2, min_label_confidence=0.5)
    second = await augmenter.augment_batch_async(items, multiplier=2, min_label_confidence=0.5)

    synthetic = [item for item in first if item.metadata.get("is_synthetic")]
    source_labels = {item.id: item.labels for item in items}
    assert synthetic
    assert all(item.labels == source_labels[item.metadata["source_id"]] for item in synthetic)
    assert {item.labels for item in synthetic} == {0, 1}
    assert [item.id for item in first] == [item.id for item in second]
    assert len({item.get_classification_text().lower() for item in first}) == len(first)
