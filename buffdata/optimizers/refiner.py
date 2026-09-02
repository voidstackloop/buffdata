import asyncio
import json
from typing import List, Optional
from buffdata.engine.client import GeminiClient, LLMClient, ProviderError
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.schemas import BatchRefinementResponse, DatasetItem, RefinementResult

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
        client: Optional[LLMClient] = None,
        limiter: Optional[AsyncRateLimiter] = None,
        model: Optional[str] = None,
    ):
        self.client = client or GeminiClient()
        self.limiter = limiter or AsyncRateLimiter()
        self.model = model or self.client.default_model

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
        except ProviderError:
            raise
        except Exception as e:
            item.metadata["refinement_error"] = str(e)

        return item

    async def _refine_chunk(
        self,
        chunk: List[DatasetItem],
        mode: str,
        max_field_chars: int,
        on_progress: Optional[callable],
    ) -> None:
        records = []
        for item in chunk:
            prompt, response = item.get_prompt_and_response()
            records.append({
                "item_id": item.id,
                "prompt": prompt[:max_field_chars],
                "response": response[:max_field_chars],
            })
        batch_prompt = (
            f"Refine every training sample below (mode: {mode}). Return exactly one entry per "
            "item_id, preserve each item_id verbatim, and apply the refinement guidelines "
            "independently to each record.\n\n" + json.dumps(records, ensure_ascii=False)
        )
        try:
            response: BatchRefinementResponse = await self.limiter.execute_with_retry(
                lambda: self.client.generate_structured_async(
                    prompt=batch_prompt,
                    response_schema=BatchRefinementResponse,
                    model=self.model,
                    system_instruction=REFINER_SYSTEM_PROMPT,
                    temperature=0.2,
                )
            )
            by_id = {entry.item_id: entry.result for entry in response.entries}
        except ProviderError:
            raise
        except Exception as exc:
            for item in chunk:
                item.metadata["refinement_error"] = str(exc)
                if on_progress:
                    on_progress()
            return

        for item in chunk:
            result = by_id.get(item.id)
            if result is None:
                item.metadata["refinement_error"] = "Batch refinement returned no entry for this item_id."
                if on_progress:
                    on_progress()
                continue
            prompt, response_text = item.get_prompt_and_response()
            new_prompt = result.refined_prompt if (mode in ("all", "prompt_only") and result.refined_prompt) else prompt
            new_response = result.refined_response if mode in ("all", "response_only") else response_text
            item.update_content(new_prompt=new_prompt, new_response=new_response)
            item.metadata["refined"] = True
            item.metadata["refinement_info"] = {
                "reasoning_added": result.reasoning_added,
                "formatting_fixed": result.formatting_fixed,
                "artifacts_removed": result.artifacts_removed,
                "explanation": result.explanation_of_changes,
            }
            if on_progress:
                on_progress()

    async def refine_batch_async(
        self,
        items: List[DatasetItem],
        mode: str = "all",
        batch_size: int = 20,
        max_field_chars: int = 6000,
        on_progress: Optional[callable] = None,
    ) -> List[DatasetItem]:
        """Refine a batch of items, grouping records into structured requests of up to
        `batch_size` so the (fairly long) system prompt is sent once per group instead of
        once per record -- cuts both token cost and network round-trips versus one call
        per item, with no change in refinement quality per record.
        """
        chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        await asyncio.gather(*(
            self._refine_chunk(chunk, mode, max_field_chars, on_progress)
            for chunk in chunks
        ))
        return items
