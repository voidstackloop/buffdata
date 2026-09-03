import pytest
from buffdata.models.schemas import DatasetItem, DatasetFormat
from buffdata.optimizers.scorer import FastRuleFilter, QualityScorer
from buffdata.optimizers.refiner import DataRefiner
from buffdata.optimizers.evolver import DataEvolver
from buffdata.optimizers.preference import PreferenceBuilder
from buffdata.optimizers.dedup import Deduplicator
from buffdata.engine.client import GeminiClient


@pytest.fixture
def mock_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return GeminiClient(allow_mock=True)

def test_fast_rule_filter():
    rf = FastRuleFilter(min_prompt_len=5, min_response_len=5)
    passed, issues = rf.evaluate("hi", "bye")
    assert not passed
    assert len(issues) > 0

    passed_ok, issues_ok = rf.evaluate("Explain quantum physics in detail.", "Quantum physics is the study of matter and energy at the fundamental level.")
    assert passed_ok
    assert len(issues_ok) == 0

@pytest.mark.asyncio
async def test_mock_scorer(mock_gemini):
    client = mock_gemini
    scorer = QualityScorer(client=client)
    item = DatasetItem(format=DatasetFormat.ALPACA, instruction="What is 2+2?", output="2+2 is 4.")
    res = await scorer.score_item_async(item)
    assert res.quality_score is not None
    assert res.quality_score.overall_score >= 0.0

@pytest.mark.asyncio
async def test_mock_refiner(mock_gemini):
    client = mock_gemini
    refiner = DataRefiner(client=client)
    item = DatasetItem(format=DatasetFormat.ALPACA, instruction="Explain recursion", output="Certainly! Recursion is...")
    res = await refiner.refine_item_async(item)
    assert res.metadata.get("refined") is True

@pytest.mark.asyncio
async def test_mock_evolver(mock_gemini):
    client = mock_gemini
    evolver = DataEvolver(client=client)
    item = DatasetItem(format=DatasetFormat.ALPACA, instruction="Sort a list", output="Use sort()")
    res = await evolver.evolve_item_async(item)
    assert res is not None

@pytest.mark.asyncio
async def test_mock_preference(mock_gemini):
    client = mock_gemini
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


def test_dedup_minhash_drops_near_duplicates_above_threshold():
    dedup = Deduplicator()
    items = [
        DatasetItem(id="1", format=DatasetFormat.ALPACA, instruction="the quick brown fox jumps over the lazy dog",
                    output="a response"),
        DatasetItem(id="2", format=DatasetFormat.ALPACA, instruction="the quick brown fox jumps over the lazy cat",
                    output="a response"),
        DatasetItem(id="3", format=DatasetFormat.ALPACA, instruction="completely unrelated text about astronomy",
                    output="a response"),
    ]
    kept, dropped = dedup.deduplicate_minhash(items, threshold=0.5)
    assert [item.id for item in kept] == ["1", "3"]
    assert [item.id for item in dropped] == ["2"]
    assert "minhash_near_dup" in dropped[0].metadata["dedup_reason"]


def _reference_deduplicate_minhash(items, threshold=0.85, shingle_size=3):
    """The pre-hashing implementation (raw shingle strings), kept only in this test as a
    reference oracle for the hashed-fingerprint version now in dedup.py."""
    def get_shingles(text):
        words = text.lower().split()
        if len(words) < shingle_size:
            return set(words)
        return set(" ".join(words[i:i + shingle_size]) for i in range(len(words) - shingle_size + 1))

    shingle_sets, kept, dropped = [], [], []
    for it in items:
        prompt, response = it.get_prompt_and_response()
        content = (f"{prompt} {response}".strip() or it.get_classification_text())
        curr = get_shingles(content)
        is_dup = False
        for prev in shingle_sets:
            union = len(curr | prev)
            if union > 0 and len(curr & prev) / union >= threshold:
                is_dup = True
                break
        if is_dup:
            dropped.append(it.id)
        else:
            shingle_sets.append(curr)
            kept.append(it.id)
    return kept, dropped


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.85, 0.95])
def test_dedup_minhash_hashed_shingles_match_string_reference(threshold):
    import random
    random.seed(42)
    vocabulary = ["alpha", "beta", "gamma", "delta", "the", "quick", "fox", "dog", "cat",
                  "jumps", "runs", "sits", "über", "café", "naïve", "a", "b"]
    items = []
    for i in range(60):
        length = random.randint(1, 12)
        text = " ".join(random.choice(vocabulary) for _ in range(length))
        items.append(DatasetItem(id=str(i), format=DatasetFormat.RAW, text=text))

    reference_kept, reference_dropped = _reference_deduplicate_minhash(
        [it.model_copy(deep=True) for it in items], threshold=threshold)
    actual_kept, actual_dropped = Deduplicator().deduplicate_minhash(items, threshold=threshold)

    assert [it.id for it in actual_kept] == reference_kept
    assert [it.id for it in actual_dropped] == reference_dropped
