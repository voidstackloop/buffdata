from __future__ import annotations

import asyncio
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from benchmark_generative_matrix import (  # noqa: E402
    CATEGORY_CLEAN,
    CATEGORY_DUPLICATE,
    CATEGORY_INVALID,
    DATASETS,
    make_dirty,
    normalize_row,
    optimize,
    score_recovery_accuracy,
    summarize_run,
)


def _clean_alpaca() -> list[dict]:
    return [
        {
            "id": f"alpaca_sft-clean-{index:06d}",
            "instruction": f"Question {index}",
            "input": f"Context {index}",
            "output": f"Answer {index}",
            "_buffdata_metadata": {"defect_category": CATEGORY_CLEAN},
        }
        for index in range(10)
    ]


def test_catalog_covers_six_nonclassification_families_and_four_formats() -> None:
    assert len(DATASETS) == 6
    assert {spec["family"] for spec in DATASETS.values()} == {
        "instruction/SFT",
        "multi-turn chat",
        "preference/DPO",
        "extractive QA",
        "summarization",
        "raw language modeling",
    }
    assert {spec["format"] for spec in DATASETS.values()} == {"alpaca", "chat", "dpo", "raw"}


def test_live_schema_adapters_normalize_each_task_shape() -> None:
    assert normalize_row("alpaca_sft", {"instruction": "Do it", "input": "x", "output": "done"})
    assert normalize_row(
        "ultrachat",
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
    ) == {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    assert normalize_row(
        "ultrafeedback_dpo",
        {
            "prompt": "p",
            "chosen": [{"role": "assistant", "content": "good"}],
            "rejected": [{"role": "assistant", "content": "bad"}],
        },
    ) == {"prompt": "p", "chosen": "good", "rejected": "bad"}
    assert normalize_row(
        "squad_qa",
        {"question": "q", "context": "c", "answers": {"text": ["a"]}},
    ) == {"instruction": "q", "input": "c", "output": "a"}
    assert normalize_row("xsum", {"document": "doc", "summary": "sum"})["output"] == "sum"
    assert normalize_row("wikitext_103", {"text": "  prose  "}) == {"text": "prose"}


def test_dirty_injection_does_not_modify_source_and_tags_defects() -> None:
    clean = _clean_alpaca()
    snapshot = [dict(row) for row in clean]
    dirty, counts = make_dirty("alpaca_sft", clean, 0.4, 0.1, seed=7)

    assert clean == snapshot
    assert counts == {"source": 10, "duplicates": 4, "invalid": 1}
    assert len(dirty) == 15
    categories = [row["_buffdata_metadata"]["defect_category"] for row in dirty]
    assert categories.count(CATEGORY_CLEAN) == 10
    assert categories.count(CATEGORY_DUPLICATE) == 4
    assert categories.count(CATEGORY_INVALID) == 1


def test_real_pipeline_removes_injected_defects_and_preserves_source() -> None:
    clean = _clean_alpaca()
    dirty, _ = make_dirty("alpaca_sft", clean, 0.4, 0.1, seed=9)
    accepted, rejected, run = asyncio.run(optimize(dirty))
    summary = summarize_run(dirty, accepted, rejected, run["elapsed"])

    assert summary["source_lost"] == 0
    assert summary["source_changed"] == 0
    assert summary["categories"][CATEGORY_DUPLICATE]["removed"] == 4
    assert summary["categories"][CATEGORY_INVALID]["removed"] == 1
    assert summary["duplicate_removal_recall"] == 1.0
    assert summary["invalid_removal_recall"] == 1.0
    assert summary["injected_defect_leakage"] == 0
    assert summary["rejected_by_stage"] == {"dedup": 4, "validate": 1}
    assert run["pipeline"]["usage"]["total_tokens"] == 0


def test_recovery_accuracy_scores_keep_delete_and_content_errors() -> None:
    dirty, _ = make_dirty("alpaca_sft", _clean_alpaca(), 0.4, 0.1, seed=12)
    accepted, _, _ = asyncio.run(optimize(dirty))
    clean_reference = {
        item.id: __import__("json").dumps(
            [item.instruction or "", item.input or "", item.output or ""],
            ensure_ascii=False,
            sort_keys=True,
        )
        for item in accepted
        if item.metadata.get("defect_category") == CATEGORY_CLEAN
    }
    accuracy = score_recovery_accuracy(dirty, accepted, clean_reference)

    assert accuracy["cleaning_decision_accuracy"] == 1.0
    assert accuracy["deletion_precision"] == 1.0
    assert accuracy["deletion_recall"] == 1.0
    assert accuracy["deletion_f1"] == 1.0
    assert accuracy["output_content_accuracy"] == 1.0
    assert accuracy["true_positive_deleted"] == 5
    assert accuracy["true_negative_retained"] == 10
    assert accuracy["false_positive_deleted"] == 0
    assert accuracy["false_negative_deleted"] == 0
