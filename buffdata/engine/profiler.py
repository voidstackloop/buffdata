"""Dataset profiling and classification applicability detection."""

from __future__ import annotations

from typing import Iterable, List, Optional

from buffdata.engine.client import LLMClient
from buffdata.models.schemas import (
    ClassificationTask,
    DatasetFormat,
    DatasetItem,
    DatasetProfile,
)


def representative_sample(items: List[DatasetItem], limit: int = 100) -> List[DatasetItem]:
    """Select deterministic, evenly distributed records instead of only the file head."""
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indices = {round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)}
    return [items[index] for index in sorted(indices)]


def _flatten_labels(items: Iterable[DatasetItem]) -> tuple[List[str], bool]:
    values: set[str] = set()
    has_multiple = False
    for item in items:
        labels = item.labels
        if labels is None:
            continue
        if isinstance(labels, list):
            clean = [str(value).strip() for value in labels if str(value).strip()]
            has_multiple = has_multiple or len(clean) > 1
            values.update(clean)
        else:
            value = str(labels).strip()
            if value:
                values.add(value)
    return sorted(values), has_multiple


class DatasetProfiler:
    def __init__(
        self,
        client: LLMClient,
        model: Optional[str] = None,
        sample_size: int = 100,
    ):
        self.client = client
        self.model = model or client.default_model
        self.sample_size = sample_size

    async def profile(self, items: List[DatasetItem]) -> DatasetProfile:
        if not items:
            return DatasetProfile(
                format=DatasetFormat.CUSTOM,
                row_count=0,
                confidence=1.0,
                reasoning="Dataset is empty.",
            )

        formats = {item.format for item in items}
        primary_format = next(iter(formats)) if len(formats) == 1 else DatasetFormat.CUSTOM
        sample = representative_sample(items, self.sample_size)
        text_fields = self._text_fields(items)
        classes, has_multiple = _flatten_labels(items)
        label_field = self._label_field(items)

        if classes:
            if has_multiple:
                task = ClassificationTask.MULTI_LABEL
            elif len(classes) == 2:
                task = ClassificationTask.BINARY
            elif len(classes) > 2:
                task = ClassificationTask.MULTI_CLASS
            else:
                task = None
            return DatasetProfile(
                format=primary_format,
                row_count=len(items),
                text_fields=text_fields,
                existing_label_field=label_field,
                classification_applicable=task is not None,
                task_type=task,
                classes=classes,
                confidence=1.0 if task else 0.4,
                reasoning=(
                    "Derived the classification task from existing labels."
                    if task
                    else "Only one distinct existing label was found."
                ),
                sampled_ids=[item.id for item in sample],
            )

        if primary_format in {DatasetFormat.ALPACA, DatasetFormat.CHAT, DatasetFormat.DPO}:
            return DatasetProfile(
                format=primary_format,
                row_count=len(items),
                text_fields=text_fields,
                classification_applicable=False,
                confidence=0.99,
                reasoning=f"{primary_format.value} records are generative training data, not a closed-set classification dataset.",
                sampled_ids=[item.id for item in sample],
            )

        sample_texts = [item.get_classification_text() for item in sample]
        sample_texts = [text[:4000] for text in sample_texts if text.strip()]
        if not sample_texts:
            return DatasetProfile(
                format=primary_format,
                row_count=len(items),
                text_fields=text_fields,
                confidence=1.0,
                reasoning="No classifiable text field was found.",
                sampled_ids=[item.id for item in sample],
            )

        from buffdata.optimizers.classifier import ClassificationSchema

        prompt = """Analyze these representative, PII-redacted dataset records.
Decide whether they form a closed-set text classification dataset. Do not force
classification on open-ended pretraining, instruction-response, chat, preference,
or generative data. If applicable, choose binary, multi-class, or multi-label and
infer a concise, stable class list. Confidence must reflect ambiguity.

Records:
""" + "\n---\n".join(sample_texts)
        schema = await self.client.generate_structured_async(
            prompt=prompt,
            response_schema=ClassificationSchema,
            model=self.model,
            temperature=0.0,
        )
        return DatasetProfile(
            format=primary_format,
            row_count=len(items),
            text_fields=text_fields,
            classification_applicable=schema.applicable,
            task_type=schema.task_type,
            classes=schema.classes,
            confidence=schema.confidence,
            reasoning=schema.reasoning,
            sampled_ids=[item.id for item in sample],
        )

    @staticmethod
    def _label_field(items: List[DatasetItem]) -> Optional[str]:
        for field in ("labels", "label", "category", "target"):
            if any(field in item.raw_data for item in items):
                return field
        return None

    @staticmethod
    def _text_fields(items: List[DatasetItem]) -> List[str]:
        candidates = ("text", "content", "query", "prompt", "instruction", "input", "messages")
        return [field for field in candidates if any(field in item.raw_data for item in items)]
