import asyncio
import json
from typing import List, Optional, Tuple
from buffdata.engine.client import GeminiClient, LLMClient, ProviderError
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.schemas import BatchEvolutionResponse, DatasetFormat, DatasetItem, EvolutionResult

EVOLVER_SYSTEM_PROMPT = """You are an expert AI prompt engineer specializing in Evol-Instruct data synthesis.
Your objective is to take an existing instruction-response pair and evolve it into a more complex, challenging, and high-value training example.

Strategies:
1. 'deepen_reasoning': Require deeper multi-step deductive reasoning, mathematical analysis, edge case exploration, or architectural justification.
2. 'add_constraints': Introduce realistic, strict constraints (e.g. time/space complexity O(N), memory boundaries, specific style constraints, negative requirements).
3. 'concretize': Transform abstract or generic questions into concrete, detailed real-world scenarios with complex inputs.
4. 'in_breadth': Mutate the domain to an adjacent, diverse, high-value field while preserving the problem structure.

Rules:
- The evolved prompt must remain clear, unambiguous, and solvable.
- The evolved response must be an absolute gold-standard, thorough, step-by-step masterclass answer.
"""

class DataEvolver:
    """Applies Evol-Instruct techniques to scale dataset reasoning complexity and diversity."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        limiter: Optional[AsyncRateLimiter] = None,
        model: Optional[str] = None,
    ):
        self.client = client or GeminiClient()
        self.limiter = limiter or AsyncRateLimiter()
        self.model = model or self.client.default_model

    async def evolve_item_async(
        self,
        item: DatasetItem,
        strategy: str = "deepen_reasoning",
    ) -> DatasetItem:
        """Evolve a single dataset item into a higher-complexity item."""
        prompt, response = item.get_prompt_and_response()

        evolve_prompt = f"""Evolve the following instruction-response pair using strategy '{strategy}':

[ORIGINAL INSTRUCTION]
{prompt}

[ORIGINAL RESPONSE]
{response}
"""
        try:
            result: EvolutionResult = await self.limiter.execute_with_retry(
                lambda: self.client.generate_structured_async(
                    prompt=evolve_prompt,
                    response_schema=EvolutionResult,
                    model=self.model,
                    system_instruction=EVOLVER_SYSTEM_PROMPT,
                    temperature=0.4,
                )
            )

            evolved_item = DatasetItem(
                format=item.format,
                metadata={
                    "original_id": str(item.id),
                    "evolution_type": result.evolution_type,
                    "complexity_score": result.complexity_score,
                }
            )
            evolved_item.update_content(new_prompt=result.evolved_prompt, new_response=result.evolved_response)
            return evolved_item
        except ProviderError:
            raise
        except Exception as e:
            # Return original with error metadata on failure
            item.metadata["evolution_error"] = str(e)
            return item

    async def _evolve_chunk(
        self,
        chunk: List[Tuple[DatasetItem, str]],
        max_field_chars: int,
        on_progress: Optional[callable],
    ) -> List[DatasetItem]:
        records = []
        for item, strategy in chunk:
            prompt, response = item.get_prompt_and_response()
            records.append({
                "item_id": item.id,
                "strategy": strategy,
                "prompt": prompt[:max_field_chars],
                "response": response[:max_field_chars],
            })
        batch_prompt = (
            "Evolve every instruction-response pair below using the strategy given for its "
            "item_id. Return exactly one entry per item_id, preserve each item_id verbatim, "
            "and apply the evolution rules independently to each record.\n\n"
            + json.dumps(records, ensure_ascii=False)
        )
        try:
            response: BatchEvolutionResponse = await self.limiter.execute_with_retry(
                lambda: self.client.generate_structured_async(
                    prompt=batch_prompt,
                    response_schema=BatchEvolutionResponse,
                    model=self.model,
                    system_instruction=EVOLVER_SYSTEM_PROMPT,
                    temperature=0.4,
                )
            )
            by_id = {entry.item_id: entry.result for entry in response.entries}
        except ProviderError:
            raise
        except Exception as exc:
            results = []
            for item, _ in chunk:
                item.metadata["evolution_error"] = str(exc)
                if on_progress:
                    on_progress()
                results.append(item)
            return results

        results = []
        for item, _ in chunk:
            result = by_id.get(item.id)
            if result is None:
                item.metadata["evolution_error"] = "Batch evolution returned no entry for this item_id."
                if on_progress:
                    on_progress()
                results.append(item)
                continue
            evolved_item = DatasetItem(
                format=item.format,
                metadata={
                    "original_id": str(item.id),
                    "evolution_type": result.evolution_type,
                    "complexity_score": result.complexity_score,
                },
            )
            evolved_item.update_content(new_prompt=result.evolved_prompt, new_response=result.evolved_response)
            if on_progress:
                on_progress()
            results.append(evolved_item)
        return results

    async def evolve_batch_async(
        self,
        items: List[DatasetItem],
        strategies: Optional[List[str]] = None,
        batch_size: int = 20,
        max_field_chars: int = 6000,
        on_progress: Optional[callable] = None,
    ) -> List[DatasetItem]:
        """Evolve a batch of items, cycling through multiple strategies. Records are grouped
        into structured requests of up to `batch_size` so the system prompt is sent once per
        group instead of once per record.
        """
        avail_strategies = strategies or ["deepen_reasoning", "add_constraints", "concretize"]
        assigned = [(item, avail_strategies[i % len(avail_strategies)]) for i, item in enumerate(items)]
        chunks = [assigned[i:i + batch_size] for i in range(0, len(assigned), batch_size)]
        chunk_results = await asyncio.gather(*(
            self._evolve_chunk(chunk, max_field_chars, on_progress)
            for chunk in chunks
        ))
        return [item for chunk_res in chunk_results for item in chunk_res]
