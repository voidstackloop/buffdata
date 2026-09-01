import asyncio
from typing import List, Optional
from buffdata.engine.client import GeminiClient
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.schemas import DatasetItem, RefinementResult

REFINER_SYSTEM_PROMPT = """You are an expert AI Training Data Refiner.
Your mission is to upgrade the quality of training data for state-of-the-art LLMs.
Apply the following strict guidelines:
1. Strip AI Preamble & Boilerplate: Remove conversational filler like "Certainly! Here is...", "As an AI language model...", "I hope this helps!", etc.
2. Elevate Reasoning Depth: Where appropriate, provide clear, step-by-step chain-of-thought derivations, logical deductions, or explanations before giving final conclusions.
3. Perfect Formatting & Syntax: Fix broken markdown, ensure code blocks have appropriate language identifiers (e.g. ```python), proper indentation, and clean tables/lists.
4. Improve Prompt Precision: Disambiguate unclear instructions and eliminate meta-prompt artifacts without altering the original user intent.
5. Preserve Core Meaning: Do not change the fundamental task or introduce fabricated assertions.
"""

class DataRefiner:
    """Polishes prompts, expands reasoning steps, fixes formatting, and removes boilerplate."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        limiter: Optional[AsyncRateLimiter] = None,
        model: str = "gemini-3.7-flash",
    ):
        self.client = client or GeminiClient()
        self.limiter = limiter or AsyncRateLimiter()
        self.model = model

    async def refine_item_async(self, item: DatasetItem, mode: str = "all") -> DatasetItem:
        """Refine prompt and/or response of a single dataset item."""
        prompt, response = item.get_prompt_and_response()

        refine_instruction = f"""Refine this training sample (mode: {mode}):

[ORIGINAL PROMPT]
{prompt}

[ORIGINAL RESPONSE]
{response}
"""
        try:
            result: RefinementResult = await self.limiter.execute_with_retry(
                lambda: self.client.generate_structured_async(
                    prompt=refine_instruction,
                    response_schema=RefinementResult,
                    model=self.model,
                    system_instruction=REFINER_SYSTEM_PROMPT,
                    temperature=0.2,
                )
            )

            new_prompt = result.refined_prompt if (mode in ("all", "prompt_only") and result.refined_prompt) else prompt
            new_response = result.refined_response if mode in ("all", "response_only") else response

            item.update_content(new_prompt=new_prompt, new_response=new_response)
            item.metadata["refined"] = True
            item.metadata["refinement_info"] = {
                "reasoning_added": result.reasoning_added,
                "formatting_fixed": result.formatting_fixed,
                "artifacts_removed": result.artifacts_removed,
                "explanation": result.explanation_of_changes,
            }
        except Exception as e:
            item.metadata["refinement_error"] = str(e)

        return item

    async def refine_batch_async(
        self,
        items: List[DatasetItem],
        mode: str = "all",
        on_progress: Optional[callable] = None,
    ) -> List[DatasetItem]:
        """Refine a batch of items concurrently."""
        tasks = []
        for it in items:
            async def wrapped(item=it):
                res = await self.refine_item_async(item, mode=mode)
                if on_progress:
                    on_progress()
                return res
            tasks.append(wrapped())
        return await asyncio.gather(*tasks)
