from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from benchmark_generative_utility import (  # noqa: E402
    language_model_metrics,
    preference_metrics,
    retrieval_metrics,
    rouge_l_f1,
    token_f1,
)


def test_text_metrics_reward_exact_and_partial_overlap() -> None:
    assert token_f1("the exact answer", "the exact answer") == 1.0
    assert 0 < token_f1("exact answer", "the exact long answer") < 1
    assert rouge_l_f1("a b c", "a b c") == 1.0
    assert rouge_l_f1("a c", "a b c") > 0


def test_retrieval_reports_held_out_response_accuracy() -> None:
    train = [
        {"instruction": "capital of france", "input": "", "output": "Paris"},
        {"instruction": "capital of italy", "input": "", "output": "Rome"},
    ]
    result = retrieval_metrics(
        train,
        [{"instruction": "what is the capital of france", "input": "", "output": "Paris"}],
        "instruction/SFT",
    )
    assert result["response_exact_match"] == 1.0
    assert result["response_token_f1"] == 1.0


def test_preference_and_language_model_metrics_are_bounded() -> None:
    dpo = [
        {"prompt": "math", "chosen": "correct useful answer", "rejected": "wrong"},
        {"prompt": "science", "chosen": "correct clear explanation", "rejected": "wrong"},
    ] * 10
    preference = preference_metrics(dpo, dpo[:2], seed=1)
    assert 0 <= preference["preference_accuracy"] <= 1

    raw = [{"text": "a b a b a b"}, {"text": "a b c"}]
    language = language_model_metrics(raw, [{"text": "a b a b"}])
    assert 0 <= language["next_token_accuracy"] <= 1
    assert language["perplexity"] > 0
