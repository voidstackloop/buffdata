import asyncio
import json
from typing import List, Optional
from buffdata.engine.client import GeminiClient, LLMClient, ProviderError
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.schemas import BatchPreferenceResponse, DatasetFormat, DatasetItem, PreferenceResult

DPO_SYSTEM_PROMPT = """You are an expert Alignment & Preference Dataset Architect.
Your task is to take an instruction/prompt and generate a state-of-the-art DPO / RLHF preference pair:
1. 'chosen': The absolute gold-standard response. Comprehensive, insightful, impeccably formatted, accurate, and deeply reasoned.
2. 'rejected': A subtly flawed or suboptimal response that reflects common LLM pitfalls (e.g. subtle calculation slip, superficial explanation, unhandled edge cases, verbose filler, or minor instruction deviation).
3. 'rejection_reason': A clear diagnostic of why 'chosen' is superior to 'rejected'.
"""

class PreferenceBuilder:
    """Generates high-contrast DPO/RLHF preference pairs for alignment training."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        limiter: Optional[AsyncRateLimiter] = None,
        model: Optional[str] = None,
    ):
        self.client = client or GeminiClient()
        self.limiter = limiter or AsyncRateLimiter()
        self.model = model or self.client.default_model

    async def build_dpo_pair_async(self, item: DatasetItem) -> DatasetItem:
        """Convert an SFT item into a DPO pair."""
        prompt, response = item.get_prompt_and_response()

        dpo_prompt = f"""Generate a high-contrast DPO preference pair for this prompt:

[PROMPT]
{prompt}

[EXISTING REFERENCE RESPONSE (use as reference or enhance)]
{response}
"""
        try:
            result: PreferenceResult = await self.limiter.execute_with_retry(
                lambda: self.client.generate_structured_async(
                    prompt=dpo_prompt,
                    response_schema=PreferenceResult,
                    model=self.model,
                    system_instruction=DPO_SYSTEM_PROMPT,
                    temperature=0.3,
                )
            )

            return DatasetItem(
                format=DatasetFormat.DPO,
                prompt=result.prompt,
                chosen=result.chosen,
                rejected=result.rejected,
                metadata={
                    "original_id": str(item.id),
                    "rejection_reason": result.rejection_reason,
                }
            )
        except ProviderError:
            raise
        except Exception as e:
            item.metadata["dpo_error"] = str(e)
            return item

    async def _build_dpo_chunk(
        self,
        chunk: List[DatasetItem],
        max_field_chars: int,
        on_progress: Optional[callable],
    ) -> List[DatasetItem]:
        records = []
        for item in chunk:
            prompt, response = item.get_prompt_and_response()
            records.append({
                "item_id": item.id,
                "prompt": prompt[:max_field_chars],
                "reference_response": response[:max_field_chars],
            })
        batch_prompt = (
            "Generate a high-contrast DPO preference pair for every prompt below. Return "
            "exactly one entry per item_id, preserve each item_id verbatim, and apply the "
            "preference-pair rules independently to each record.\n\n"
            + json.dumps(records, ensure_ascii=False)
        )
        try:
            response: BatchPreferenceResponse = await self.limiter.execute_with_retry(
                lambda: self.client.generate_structured_async(
                    prompt=batch_prompt,
                    response_schema=BatchPreferenceResponse,
                    model=self.model,
                    system_instruction=DPO_SYSTEM_PROMPT,
                    temperature=0.3,
                )
            )
            by_id = {entry.item_id: entry.result for entry in response.entries}
        except ProviderError:
            raise
        except Exception as exc:
            results = []
            for item in chunk:
                item.metadata["dpo_error"] = str(exc)
                if on_progress:
                    on_progress()
                results.append(item)
            return results

        results = []
        for item in chunk:
            result = by_id.get(item.id)
            if result is None:
                item.metadata["dpo_error"] = "Batch DPO generation returned no entry for this item_id."
                if on_progress:
                    on_progress()
                results.append(item)
                continue
            results.append(DatasetItem(
                format=DatasetFormat.DPO,
                prompt=result.prompt,
                chosen=result.chosen,
                rejected=result.rejected,
                metadata={
                    "original_id": str(item.id),
                    "rejection_reason": result.rejection_reason,
                },
            ))
            if on_progress:
                on_progress()
        return results

    async def build_dpo_batch_async(
        self,
        items: List[DatasetItem],
        batch_size: int = 20,
        max_field_chars: int = 6000,
        on_progress: Optional[callable] = None,
    ) -> List[DatasetItem]:
        """Generate DPO pairs for a batch of items, grouping records into structured
        requests of up to `batch_size` so the system prompt is sent once per group instead
        of once per record.
        """
        chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        chunk_results = await asyncio.gather(*(
            self._build_dpo_chunk(chunk, max_field_chars, on_progress)
            for chunk in chunks
        ))
        return [item for chunk_res in chunk_results for item in chunk_res]
