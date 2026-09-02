import json

import pytest

from buffdata.engine.client import LLMProvider
from buffdata.models.schemas import (
    BatchEvolutionResponse,
    BatchPreferenceResponse,
    BatchRefinementResponse,
    DatasetItem,
    EvolutionEntry,
    EvolutionResult,
    PreferenceEntry,
    PreferenceResult,
    RefinementEntry,
    RefinementResult,
)
from buffdata.optimizers.evolver import DataEvolver
from buffdata.optimizers.preference import PreferenceBuilder
from buffdata.optimizers.refiner import DataRefiner


def _records_from_prompt(prompt: str) -> list[dict]:
    return json.loads(prompt[prompt.index("["):])


class _CountingClient:
    provider = LLMProvider.GEMINI
    default_model = "fake"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def __init__(self):
        self.calls = 0
        self.captured_records: list[list[dict]] = []


class RefineBatchFakeClient(_CountingClient):
    async def generate_structured_async(self, prompt, response_schema, **kwargs):
        self.calls += 1
        records = _records_from_prompt(prompt)
        self.captured_records.append(records)
        return BatchRefinementResponse(entries=[
            RefinementEntry(
                item_id=r["item_id"],
                result=RefinementResult(
                    refined_prompt=r["prompt"] + " (clarified)",
                    refined_response=r["response"].strip() + " Clean, well-formatted answer.",
                    reasoning_added=True,
                    formatting_fixed=True,
                    artifacts_removed=["Certainly!"],
                    explanation_of_changes="Removed boilerplate.",
                ),
            )
            for r in records
        ])


class DroppingRefineFakeClient(_CountingClient):
    """Omits the first record's entry, to exercise the missing-item_id path."""

    async def generate_structured_async(self, prompt, response_schema, **kwargs):
        self.calls += 1
        records = _records_from_prompt(prompt)
        return BatchRefinementResponse(entries=[
            RefinementEntry(
                item_id=r["item_id"],
                result=RefinementResult(refined_prompt=r["prompt"], refined_response=r["response"]),
            )
            for r in records[1:]
        ])


class FailingClient(_CountingClient):
    async def generate_structured_async(self, prompt, response_schema, **kwargs):
        self.calls += 1
        raise RuntimeError("provider unavailable")


class EvolveBatchFakeClient(_CountingClient):
    async def generate_structured_async(self, prompt, response_schema, **kwargs):
        self.calls += 1
        records = _records_from_prompt(prompt)
        self.captured_records.append(records)
        return BatchEvolutionResponse(entries=[
            EvolutionEntry(
                item_id=r["item_id"],
                result=EvolutionResult(
                    evolved_prompt=r["prompt"] + " with edge cases",
                    evolved_response=r["response"] + " (deeper reasoning)",
                    evolution_type=r["strategy"],
                    complexity_score=7.5,
                ),
            )
            for r in records
        ])


class DpoBatchFakeClient(_CountingClient):
    async def generate_structured_async(self, prompt, response_schema, **kwargs):
        self.calls += 1
        records = _records_from_prompt(prompt)
        self.captured_records.append(records)
        return BatchPreferenceResponse(entries=[
            PreferenceEntry(
                item_id=r["item_id"],
                result=PreferenceResult(
                    prompt=r["prompt"],
                    chosen="A thorough, correct answer.",
                    rejected="A vague, incomplete answer.",
                    rejection_reason="Missing detail and rigor.",
                ),
            )
            for r in records
        ])


def _alpaca_items(n: int) -> list[DatasetItem]:
    return [
        DatasetItem.from_dict({
            "instruction": f"Explain concept {i}",
            "output": f"Certainly! Concept {i} is...",
        })
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_refine_batch_groups_records_into_few_requests():
    items = _alpaca_items(5)
    client = RefineBatchFakeClient()
    refiner = DataRefiner(client=client)

    result = await refiner.refine_batch_async(items, batch_size=2)

    assert client.calls == 3  # ceil(5/2)
    assert all(item.metadata.get("refined") for item in result)
    assert all("(clarified)" in item.get_prompt_and_response()[0] for item in result)
    assert sum(len(chunk) for chunk in client.captured_records) == 5


@pytest.mark.asyncio
async def test_refine_batch_reports_per_item_error_on_missing_entry():
    items = _alpaca_items(3)
    client = DroppingRefineFakeClient()
    refiner = DataRefiner(client=client)

    result = await refiner.refine_batch_async(items, batch_size=10)

    assert client.calls == 1
    missing = [item for item in result if item.metadata.get("refinement_error")]
    refined = [item for item in result if item.metadata.get("refined")]
    assert len(missing) == 1
    assert len(refined) == 2


@pytest.mark.asyncio
async def test_refine_batch_chunk_failure_does_not_crash_other_chunks():
    items = _alpaca_items(4)
    client = FailingClient()
    refiner = DataRefiner(client=client)

    result = await refiner.refine_batch_async(items, batch_size=2)

    # execute_with_retry retries each failing chunk the same number of times, so calls is
    # a multiple of the 2 chunks -- what matters is both chunks failed independently and
    # every item still gets a clear error instead of the whole run crashing.
    assert client.calls >= 2 and client.calls % 2 == 0
    assert all("provider unavailable" in item.metadata.get("refinement_error", "") for item in result)


@pytest.mark.asyncio
async def test_refine_batch_truncates_oversized_fields():
    items = [DatasetItem.from_dict({"instruction": "x" * 20000, "output": "y" * 20000})]
    client = RefineBatchFakeClient()
    refiner = DataRefiner(client=client)

    await refiner.refine_batch_async(items, batch_size=10, max_field_chars=100)

    sent = client.captured_records[0][0]
    assert len(sent["prompt"]) <= 100
    assert len(sent["response"]) <= 100


@pytest.mark.asyncio
async def test_refine_item_async_single_item_path_still_works(monkeypatch):
    from buffdata.engine.client import GeminiClient

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    refiner = DataRefiner(client=GeminiClient(allow_mock=True))
    item = DatasetItem.from_dict({"instruction": "Explain recursion", "output": "Certainly! Recursion is..."})
    result = await refiner.refine_item_async(item)
    assert result.metadata.get("refined") is True


@pytest.mark.asyncio
async def test_evolve_batch_cycles_strategies_and_groups_requests():
    items = _alpaca_items(4)
    client = EvolveBatchFakeClient()
    evolver = DataEvolver(client=client)

    result = await evolver.evolve_batch_async(items, strategies=["deepen_reasoning", "concretize"], batch_size=2)

    assert client.calls == 2
    assert len(result) == 4
    assert [item.metadata["evolution_type"] for item in result] == [
        "deepen_reasoning", "concretize", "deepen_reasoning", "concretize",
    ]
    assert all(item.metadata["original_id"] for item in result)


@pytest.mark.asyncio
async def test_dpo_batch_groups_requests_and_builds_pairs():
    items = _alpaca_items(5)
    client = DpoBatchFakeClient()
    dpo_builder = PreferenceBuilder(client=client)

    result = await dpo_builder.build_dpo_batch_async(items, batch_size=2)

    assert client.calls == 3
    assert len(result) == 5
    assert all(item.chosen and item.rejected for item in result)
