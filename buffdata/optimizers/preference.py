import asyncio
from typing import List, Optional
from buffdata.engine.client import GeminiClient
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.schemas import DatasetFormat, DatasetItem, PreferenceResult

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
        client: Optional[GeminiClient] = None,
        limiter: Optional[AsyncRateLimiter] = None,
        model: str = "gemini-3.7-flash",
    ):
        self.client = client or GeminiClient()
        self.limiter = limiter or AsyncRateLimiter()
        self.model = model

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
        except Exception as e:
            item.metadata["dpo_error"] = str(e)
            return item

    async def build_dpo_batch_async(
        self,
        items: List[DatasetItem],
        on_progress: Optional[callable] = None,
    ) -> List[DatasetItem]:
        """Generate DPO pairs for a batch of items concurrently."""
        tasks = []
        for it in items:
            async def wrapped(item=it):
                res = await self.build_dpo_pair_async(item)
                if on_progress:
                    on_progress()
                return res
            tasks.append(wrapped())
        return await asyncio.gather(*tasks)
