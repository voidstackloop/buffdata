import pytest
from buffdata.models.schemas import DatasetItem, DatasetFormat
from buffdata.optimizers.scorer import FastRuleFilter, QualityScorer
from buffdata.optimizers.refiner import DataRefiner
from buffdata.optimizers.evolver import DataEvolver
from buffdata.optimizers.preference import PreferenceBuilder
from buffdata.optimizers.dedup import Deduplicator
from buffdata.engine.client import GeminiClient

def test_fast_rule_filter():
    rf = FastRuleFilter(min_prompt_len=5, min_response_len=5)
    passed, issues = rf.evaluate("hi", "bye")
    assert not passed
    assert len(issues) > 0

    passed_ok, issues_ok = rf.evaluate("Explain quantum physics in detail.", "Quantum physics is the study of matter and energy at the fundamental level.")
    assert passed_ok
    assert len(issues_ok) == 0

@pytest.mark.asyncio
async def test_mock_scorer():
    client = GeminiClient() # in mock mode when no API key
    scorer = QualityScorer(client=client)
    item = DatasetItem(format=DatasetFormat.ALPACA, instruction="What is 2+2?", output="2+2 is 4.")
    res = await scorer.score_item_async(item)
    assert res.quality_score is not None
    assert res.quality_score.overall_score >= 0.0

@pytest.mark.asyncio
async def test_mock_refiner():
    client = GeminiClient()
    refiner = DataRefiner(client=client)
    item = DatasetItem(format=DatasetFormat.ALPACA, instruction="Explain recursion", output="Certainly! Recursion is...")
    res = await refiner.refine_item_async(item)
    assert res.metadata.get("refined") is True

@pytest.mark.asyncio
async def test_mock_evolver():
    client = GeminiClient()
    evolver = DataEvolver(client=client)
    item = DatasetItem(format=DatasetFormat.ALPACA, instruction="Sort a list", output="Use sort()")
    res = await evolver.evolve_item_async(item)
    assert res is not None

@pytest.mark.asyncio
async def test_mock_preference():
    client = GeminiClient()
    dpo_builder = PreferenceBuilder(client=client)
    item = DatasetItem(format=DatasetFormat.ALPACA, instruction="What is gravity?", output="A force.")
    dpo_item = await dpo_builder.build_dpo_pair_async(item)
    assert dpo_item.format == DatasetFormat.DPO
    assert dpo_item.chosen is not None
    assert dpo_item.rejected is not None

def test_dedup_exact_and_minhash():
    dedup = Deduplicator()
    items = [
        DatasetItem(id="1", format=DatasetFormat.ALPACA, instruction="prompt A", output="response A"),
        DatasetItem(id="2", format=DatasetFormat.ALPACA, instruction="prompt A", output="response A"),
        DatasetItem(id="3", format=DatasetFormat.ALPACA, instruction="prompt B", output="response B"),
    ]
    kept, dropped = dedup.deduplicate_exact(items)
    assert len(kept) == 2
    assert len(dropped) == 1
