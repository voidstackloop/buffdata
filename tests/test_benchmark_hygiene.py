from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from benchmark_buffdata import (  # noqa: E402
    DATASETS,
    DEFECT_CLEAN,
    DEFECT_CONFLICT,
    DEFECT_EMPTY,
    DEFECT_SKEW,
    available_row_count,
    summarize_data_hygiene,
)
from buffdata.models.schemas import DatasetItem  # noqa: E402


def _item(text: str, label: int, category: str, stage: str | None = None) -> DatasetItem:
    metadata = {"defect_category": category}
    if stage is not None:
        metadata["rejection"] = {"stage": stage, "reason": f"{stage}-reason"}
    return DatasetItem.from_dict(
        {"text": text, "label": label, "_buffdata_metadata": metadata}
    )


def test_hygiene_summary_distinguishes_present_and_deleted_duplicates() -> None:
    rows = [
        {"text": "alpha", "label": 0, "_buffdata_metadata": {"defect_category": DEFECT_CLEAN}},
        {"text": "beta", "label": 1, "_buffdata_metadata": {"defect_category": DEFECT_CLEAN}},
        {"text": "alpha", "label": 0, "_buffdata_metadata": {"defect_category": DEFECT_SKEW}},
        {"text": "alpha", "label": 1, "_buffdata_metadata": {"defect_category": DEFECT_CONFLICT}},
        {"text": "   ", "label": 0, "_buffdata_metadata": {"defect_category": DEFECT_EMPTY}},
    ]
    accepted = [
        _item("alpha", 0, DEFECT_CLEAN),
        _item("beta", 1, DEFECT_CLEAN),
    ]
    rejected = [
        _item("alpha", 0, DEFECT_SKEW, "dedup"),
        _item("alpha", 1, DEFECT_CONFLICT, "dedup"),
        _item("   ", 0, DEFECT_EMPTY, "validate"),
    ]

    hygiene = summarize_data_hygiene(rows, accepted, rejected)

    assert hygiene["input_rows"] == 5
    assert hygiene["retained_rows"] == 2
    assert hygiene["deleted_rows"] == 3
    assert hygiene["deletion_rate"] == 0.6
    assert hygiene["unique_content_rows"] == 3
    assert hygiene["duplicate_groups_in_input"] == 1
    assert hygiene["duplicate_rows_in_input"] == 2
    assert hygiene["conflicting_label_groups_in_input"] == 1
    assert hygiene["conflicting_label_rows_in_input"] == 1
    assert hygiene["empty_text_rows_in_input"] == 1
    assert hygiene["duplicate_rows_deleted"] == 2
    assert hygiene["invalid_rows_deleted"] == 1
    assert hygiene["injected_defect_rows"] == 3
    assert hygiene["injected_defects_removed"] == 3
    assert hygiene["injected_defects_retained"] == 0
    assert hygiene["clean_rows_deleted"] == 0


def test_available_row_count_caps_balanced_and_random_samples() -> None:
    split = [
        {"target": 0},
        {"target": 0},
        {"target": 0},
        {"target": 1},
    ]

    assert available_row_count(split, 10, 2, "balanced", "target") == 2
    assert available_row_count(split, 10, 2, "random", "target") == 4


def test_extended_registry_covers_binary_and_multiclass_sources() -> None:
    assert len(DATASETS) >= 19
    assert DATASETS["yahoo_answers_topics"]["label_field"] == "topic"
    assert DATASETS["trec_coarse"]["label_field"] == "label_coarse"
    assert {spec["classes"] for spec in DATASETS.values()} >= {2, 3, 4, 6, 10, 14, 20}
