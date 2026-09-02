"""Dataset-aware closed-set classification."""

from __future__ import annotations

import asyncio
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from buffdata.engine.client import LLMClient, ProviderError
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.schemas import ClassificationMode, ClassificationTask, DatasetItem


class ClassificationSchema(BaseModel):
    applicable: bool = Field(True, description="Whether closed-set classification is appropriate")
    task_type: Optional[ClassificationTask] = None
    classes: List[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("classes")
    @classmethod
    def normalize_classes(cls, values: List[str]) -> List[str]:
        clean: List[str] = []
        for value in values:
            label = str(value).strip()
            if label and label not in clean:
                clean.append(label)
        return clean

    @model_validator(mode="after")
    def validate_cardinality(self):
        if not self.applicable:
            return self
        if self.task_type is None:
            raise ValueError("task_type is required when classification is applicable")
        if self.task_type == ClassificationTask.BINARY and len(self.classes) != 2:
            raise ValueError("binary classification requires exactly two classes")
        if self.task_type == ClassificationTask.MULTI_CLASS and len(self.classes) < 3:
            raise ValueError("multi-class classification requires at least three classes")
        if self.task_type == ClassificationTask.MULTI_LABEL and len(self.classes) < 2:
            raise ValueError("multi-label classification requires at least two classes")
        return self


class ClassificationResult(BaseModel):
    id: str = Field(description="The exact ID of the dataset item")
    labels: List[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResult]


class DatasetClassifier:
    def __init__(
        self,
        client: LLMClient,
        limiter: AsyncRateLimiter,
        model: Optional[str] = None,
    ):
        self.client = client
        self.limiter = limiter
        self.model = model or client.default_model

    async def detect_schema(
        self,
        sample_texts: List[str],
        task_hint: Optional[ClassificationTask] = None,
        allowed_classes: Optional[List[str]] = None,
    ) -> ClassificationSchema:
        constraints = []
        if task_hint:
            constraints.append(f"The user requires task type: {task_hint.value}.")
        if allowed_classes:
            constraints.append(f"The user requires this class set: {allowed_classes}.")
        prompt = """Determine whether the representative records form a closed-set
classification dataset. When appropriate, select binary, multi-class, or multi-label.
Never force classification on open-ended generative data. Return a stable class list
and calibrated confidence.

""" + "\n".join(constraints) + "\n\nRecords:\n" + "\n---\n".join(sample_texts)
        schema = await self.limiter.execute_with_retry(
            lambda: self.client.generate_structured_async(
                prompt=prompt,
                response_schema=ClassificationSchema,
                model=self.model,
                temperature=0.0,
            )
        )
        if task_hint:
            schema.task_type = task_hint
            schema.applicable = True
        if allowed_classes:
            schema.classes = allowed_classes
            schema.applicable = True
        return ClassificationSchema.model_validate(schema.model_dump())

    async def resolve_schema(
        self,
        items: List[DatasetItem],
        mode: ClassificationMode | str = ClassificationMode.AUTO,
        classes: Optional[List[str]] = None,
        sample_size: int = 100,
    ) -> ClassificationSchema:
        selected = mode if isinstance(mode, ClassificationMode) else ClassificationMode(mode)
        requested_classes = [value.strip() for value in (classes or []) if value.strip()]
        if selected == ClassificationMode.OFF:
            return ClassificationSchema(
                applicable=False,
                confidence=1.0,
                reasoning="Classification disabled by user override.",
            )

        from buffdata.engine.profiler import DatasetProfiler, representative_sample

        task_hint = None if selected == ClassificationMode.AUTO else ClassificationTask(selected.value)
        if task_hint and requested_classes:
            return ClassificationSchema(
                applicable=True,
                task_type=task_hint,
                classes=requested_classes,
                confidence=1.0,
                reasoning="User-provided task type and classes.",
            )

        if task_hint or requested_classes:
            sample = representative_sample(items, sample_size)
            texts = [item.get_classification_text()[:4000] for item in sample]
            return await self.detect_schema(
                [text for text in texts if text.strip()],
                task_hint=task_hint,
                allowed_classes=requested_classes or None,
            )

        profile = await DatasetProfiler(
            client=self.client,
            model=self.model,
            sample_size=sample_size,
        ).profile(items)
        return ClassificationSchema(
            applicable=profile.classification_applicable,
            task_type=profile.task_type,
            classes=profile.classes,
            confidence=profile.confidence,
            reasoning=profile.reasoning,
        )

    @staticmethod
    def _validate_result(result: ClassificationResult, schema: ClassificationSchema) -> List[str]:
        labels = list(dict.fromkeys(label.strip() for label in result.labels if label.strip()))
        unknown = [label for label in labels if label not in schema.classes]
        if unknown:
            raise ValueError(f"Provider returned labels outside the allowed set: {unknown}")
        if schema.task_type in {ClassificationTask.BINARY, ClassificationTask.MULTI_CLASS}:
            if len(labels) != 1:
                raise ValueError("Binary and multi-class results require exactly one label")
        elif schema.task_type == ClassificationTask.MULTI_LABEL and not labels:
            raise ValueError("Multi-label results require at least one label")
        return labels

    async def _classify_chunk(self, chunk: List[DatasetItem], schema: ClassificationSchema) -> None:
        valid_items = [it for it in chunk if it.get_classification_text().strip()]
        for it in chunk:
            if not it.get_classification_text().strip():
                it.metadata["classification_error"] = "No classifiable text found."
                
        if not valid_items:
            return

        prompt = f"""Task type: {schema.task_type.value}
Allowed classes: {schema.classes}

Assign only labels from the allowed class list. Binary and multi-class require exactly
one label. Multi-label requires one or more applicable labels.
Return a list of results, matching exactly the IDs provided.

"""
        for item in valid_items:
            text = item.get_classification_text()[:4000] # Defensive token truncation
            prompt += f"\nID: {item.id}\nText:\n{text}\n---\n"

        last_error: Optional[Exception] = None
        for _ in range(2):
            try:
                async with self.limiter:
                    response = await self.client.generate_structured_async(
                        prompt=prompt,
                        response_schema=BatchClassificationResponse,
                        model=self.model,
                        temperature=0.0,
                    )
                
                # Map results by ID
                result_map = {res.id: res for res in response.results}
                
                for item in valid_items:
                    res = result_map.get(item.id)
                    if not res:
                        raise ValueError(f"Provider missed ID {item.id} in batch response")
                        
                    labels = self._validate_result(res, schema)
                    item.labels = labels if schema.task_type == ClassificationTask.MULTI_LABEL else labels[0]
                    item.metadata["classification"] = {
                        "provider": self.client.provider.value,
                        "model": self.model,
                        "task_type": schema.task_type.value,
                        "classes": schema.classes,
                        "schema_confidence": schema.confidence,
                        "item_confidence": res.confidence,
                    }
                    item.metadata.pop("classification_error", None)
                return
            except ProviderError:
                raise
            except Exception as exc:
                last_error = exc
                
        for item in valid_items:
            item.metadata["classification_error"] = str(last_error)

    async def classify_batch(
        self,
        items: List[DatasetItem],
        schema: ClassificationSchema,
    ) -> List[DatasetItem]:
        if not schema.applicable:
            return items
            
        import asyncio
        chunk_size = 10
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        await asyncio.gather(*(self._classify_chunk(chunk, schema) for chunk in chunks))
        return items

    @staticmethod
    def partition_failures(items: List[DatasetItem]) -> tuple[List[DatasetItem], List[DatasetItem]]:
        accepted, rejected = [], []
        for item in items:
            (rejected if item.metadata.get("classification_error") else accepted).append(item)
        return accepted, rejected
