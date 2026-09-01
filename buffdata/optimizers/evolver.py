import asyncio
from typing import List, Optional
from buffdata.engine.client import GeminiClient
from buffdata.engine.limiter import AsyncRateLimiter
from buffdata.models.schemas import DatasetFormat, DatasetItem, EvolutionResult

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
        client: Optional[GeminiClient] = None,
        limiter: Optional[AsyncRateLimiter] = None,
        model: str = "gemini-3.7-flash",
    ):
        self.client = client or GeminiClient()
        self.limiter = limiter or AsyncRateLimiter()
        self.model = model

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
        except Exception as e:
            # Return original with error metadata on failure
            item.metadata["evolution_error"] = str(e)
            return item

    async def evolve_batch_async(
        self,
        items: List[DatasetItem],
        strategies: Optional[List[str]] = None,
        on_progress: Optional[callable] = None,
    ) -> List[DatasetItem]:
        """Evolve a batch of items, optionally cycling through multiple strategies."""
        avail_strategies = strategies or ["deepen_reasoning", "add_constraints", "concretize"]
        tasks = []
        for i, it in enumerate(items):
            strat = avail_strategies[i % len(avail_strategies)]
            async def wrapped(item=it, s=strat):
                res = await self.evolve_item_async(item, strategy=s)
                if on_progress:
                    on_progress()
                return res
            tasks.append(wrapped())
        return await asyncio.gather(*tasks)
