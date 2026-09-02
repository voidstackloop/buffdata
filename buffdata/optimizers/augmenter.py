from collections import Counter, defaultdict
import hashlib
import math
import re
from typing import List, Optional
from buffdata.engine.client import GeminiClient, LLMClient, ProviderError
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.schemas import DatasetItem, AugmentationResult, BatchAugmentationResponse

AUGMENTER_SYSTEM_PROMPT = """You are an expert AI dataset synthesizer specializing in data augmentation for supervised classification.
Your objective is to take a set of labeled source records and generate high-quality synthetic variations of each record.

Rules:
1. Preserve the original semantic label and intent perfectly.
2. Preserve every named entity, number, date, product, place, and factual claim unless a purely grammatical rewrite requires no semantic change.
3. Introduce useful lexical and structural diversity without adding new facts.
4. Ensure the text length and complexity remain similar to the source.
5. Output exactly N variations for each requested ID. Never copy the source verbatim.
"""

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def _normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


class _LabelGuard:
    """Small deterministic multinomial Naive Bayes teacher for label-drift rejection."""

    def __init__(self, items: List[DatasetItem]):
        self.class_docs: Counter[str] = Counter()
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.token_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        for item in items:
            if item.labels is None or isinstance(item.labels, list):
                continue
            label = str(item.labels)
            tokens = TOKEN_RE.findall(item.get_classification_text().lower())
            if not tokens:
                continue
            self.class_docs[label] += 1
            self.token_counts[label].update(tokens)
            self.token_totals[label] += len(tokens)
            self.vocabulary.update(tokens)

    @property
    def usable(self) -> bool:
        return len(self.class_docs) >= 2 and bool(self.vocabulary)

    def predict(self, text: str) -> tuple[str | None, float]:
        if not self.usable:
            return None, 0.0
        tokens = TOKEN_RE.findall(text.lower())
        total_docs = sum(self.class_docs.values())
        vocab_size = len(self.vocabulary)
        scores: dict[str, float] = {}
        for label, documents in self.class_docs.items():
            score = math.log(documents / total_docs)
            denominator = self.token_totals[label] + vocab_size
            counts = self.token_counts[label]
            score += sum(math.log((counts[token] + 1) / denominator) for token in tokens)
            scores[label] = score
        best = max(scores, key=scores.get)
        peak = scores[best]
        probability = 1.0 / sum(math.exp(score - peak) for score in scores.values())
        return best, probability

class DataAugmenter:
    def __init__(
        self,
        client: Optional[LLMClient] = None,
        limiter: Optional[AsyncRateLimiter] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.client = client or GeminiClient()
        self.limiter = limiter or AsyncRateLimiter()
        self.model = model or self.client.default_model
        self.system_prompt = system_prompt or AUGMENTER_SYSTEM_PROMPT

    async def _augment_chunk(
        self,
        chunk: List[DatasetItem],
        multiplier: int,
        label_guard: _LabelGuard,
        min_label_confidence: float,
        extra_prompt: str = "",
    ) -> List[DatasetItem]:
        valid_items = [it for it in chunk if it.get_classification_text().strip()]
        if not valid_items:
            return chunk

        prompt = extra_prompt + f"Generate {multiplier} synthetic variations for each of the following labeled records.\n\n"
        for item in valid_items:
            text = item.get_classification_text()[:4000]
            label = getattr(item, "labels", "Unknown")
            if not label and "labels" in item.raw_data:
                label = item.raw_data["labels"]
            prompt += f"ID: {item.id}\nLabel: {label}\nText:\n{text}\n---\n"

        augmented = list(chunk)  # Keep originals
        try:
            response = await self.limiter.execute_with_retry(
                lambda: self.client.generate_structured_async(
                        prompt=prompt,
                        response_schema=BatchAugmentationResponse,
                        model=self.model,
                        system_instruction=self.system_prompt,
                        temperature=0.45,
                    )
            )
        except ProviderError:
            raise
        except Exception as exc:
            for item in chunk:
                item.metadata["augmentation_error"] = str(exc)
            return chunk

        result_map = {res.id: res for res in response.results}
        seen = {_normalize(item.get_classification_text()) for item in chunk}
        for item in valid_items:
            res = result_map.get(item.id)
            if not res:
                continue
            source_text = item.get_classification_text()
            source_length = max(1, len(TOKEN_RE.findall(source_text)))
            for var_text in res.variations[:multiplier]:
                if not isinstance(var_text, str):
                    continue
                normalized = _normalize(var_text)
                length_ratio = len(TOKEN_RE.findall(var_text)) / source_length
                if not normalized or normalized in seen or not 0.5 <= length_ratio <= 2.0:
                    continue
                predicted_label, confidence = label_guard.predict(var_text)
                if label_guard.usable and (
                    predicted_label != str(item.labels) or confidence < min_label_confidence
                ):
                    continue
                seen.add(normalized)
                new_item = DatasetItem.from_dict(item.to_dict())
                digest = hashlib.sha256(f"{item.id}\0{normalized}".encode()).hexdigest()[:10]
                new_item.id = f"{item.id}_aug_{digest}"
                if new_item.format.value == "raw":
                    new_item.text = var_text.strip()
                    if "text" in new_item.raw_data:
                        new_item.raw_data["text"] = new_item.text
                new_item.labels = item.labels
                new_item.metadata["is_synthetic"] = True
                new_item.metadata["source_id"] = str(item.id)
                new_item.metadata["label_guard_confidence"] = confidence
                augmented.append(new_item)
        return augmented

    async def augment_batch_async(
        self,
        items: List[DatasetItem],
        multiplier: int = 1,
        chunk_size: int = 20,
        min_label_confidence: float = 0.55,
        extra_prompt: str = "",
    ) -> List[DatasetItem]:
        if multiplier < 1 or multiplier > 3:
            raise ValueError("multiplier must be between 1 and 3 to avoid synthetic data dominating originals")
        if any(item.labels is None or isinstance(item.labels, list) for item in items):
            raise ValueError("augmentation requires a scalar label on every record")
        label_guard = _LabelGuard(items)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        import asyncio
        results = await asyncio.gather(*(
            self._augment_chunk(chunk, multiplier, label_guard, min_label_confidence, extra_prompt)
            for chunk in chunks
        ))

        final_items: List[DatasetItem] = []
        seen: set[str] = set()
        for chunk_res in results:
            for item in chunk_res:
                normalized = _normalize(item.get_classification_text())
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    final_items.append(item)
        return final_items
