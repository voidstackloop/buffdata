import json

import pytest
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.integrations.sharding import merge_partition_results, partition_dataset
from buffdata.models.formats import read_dataset, write_dataset
from buffdata.models.schemas import DatasetItem


def _labeled_rows(n: int) -> list[DatasetItem]:
    return [
        DatasetItem.from_dict({"text": f"row {i}", "label": i % 3})
        for i in range(n)
    ]


def test_partition_dataset_splits_into_requested_count(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(30), source)

    paths = partition_dataset(str(source), str(tmp_path / "parts"), num_partitions=4)

    assert len(paths) == 4
    total_rows = sum(len(read_dataset(p)) for p in paths)
    assert total_rows == 30


def test_partition_dataset_stratifies_by_label(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(30), source)  # 10 rows each of labels 0, 1, 2

    paths = partition_dataset(str(source), str(tmp_path / "parts"), num_partitions=3, stratify_by_label=True)

    for path in paths:
        items = read_dataset(path)
        labels = {str(item.labels) for item in items}
        assert labels == {"0", "1", "2"}  # every partition sees every class


def test_partition_dataset_no_stratify_still_covers_all_rows(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(30), source)

    paths = partition_dataset(str(source), str(tmp_path / "parts"), num_partitions=4, stratify_by_label=False)
    total_rows = sum(len(read_dataset(p)) for p in paths)
    assert total_rows == 30


def test_partition_dataset_skips_empty_partitions(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(3), source)  # fewer rows than requested partitions

    paths = partition_dataset(str(source), str(tmp_path / "parts"), num_partitions=10)

    assert len(paths) <= 3  # no empty partition files created
    total_rows = sum(len(read_dataset(p)) for p in paths)
    assert total_rows == 3


def test_partition_dataset_handles_mixed_labeled_and_unlabeled_rows(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = _labeled_rows(12) + [DatasetItem.from_dict({"text": f"unlabeled {i}"}) for i in range(6)]
    write_dataset(rows, source)

    paths = partition_dataset(str(source), str(tmp_path / "parts"), num_partitions=3)

    total_rows = sum(len(read_dataset(p)) for p in paths)
    assert total_rows == 18
    # every labeled class still appears in every partition; unlabeled rows just ride along
    # on the separate plain round-robin counter without crashing anything.
    for path in paths:
        items = read_dataset(path)
        labels = {str(item.labels) for item in items if item.labels is not None}
        assert labels == {"0", "1", "2"}


def test_partition_dataset_at_moderate_scale_preserves_every_row(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(600), source)

    paths = partition_dataset(str(source), str(tmp_path / "parts"), num_partitions=6)

    total_rows = sum(len(read_dataset(p)) for p in paths)
    assert total_rows == 600
    for path in paths:
        assert 90 <= len(read_dataset(path)) <= 110  # roughly balanced


def test_partition_dataset_rejects_invalid_partition_count(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(5), source)
    with pytest.raises(ValueError, match="num_partitions"):
        partition_dataset(str(source), str(tmp_path / "parts"), num_partitions=0)


def test_partition_dataset_rejects_empty_input(tmp_path):
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No rows found"):
        partition_dataset(str(source), str(tmp_path / "parts"), num_partitions=2)


def test_merge_partition_results_combines_rows_and_reports(tmp_path):
    part1 = tmp_path / "partition-0000.jsonl"
    part2 = tmp_path / "partition-0001.jsonl"
    write_dataset(_labeled_rows(5), part1)
    write_dataset(_labeled_rows(7), part2)

    # Simulate companion artifacts a real buffdata run would leave next to each partition.
    (tmp_path / "partition-0000.rejected.jsonl").write_text(
        json.dumps({"text": "bad row", "label": 0}) + "\n", encoding="utf-8"
    )
    (tmp_path / "partition-0000.report.json").write_text(json.dumps({
        "metrics": {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
        "rejection_reasons": {"validate: too short": 1},
    }), encoding="utf-8")
    (tmp_path / "partition-0001.report.json").write_text(json.dumps({
        "metrics": {"usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}},
        "rejection_reasons": {"validate: too short": 2},
    }), encoding="utf-8")

    output = tmp_path / "final.jsonl"
    summary = merge_partition_results([str(part1), str(part2)], str(output))

    assert summary["partitions_merged"] == 2
    assert summary["accepted_records"] == 12
    assert summary["rejected_records"] == 1
    assert summary["usage"] == {"input_tokens": 30, "output_tokens": 13, "total_tokens": 43}
    assert summary["rejection_reasons"] == {"validate: too short": 3}
    assert len(read_dataset(output)) == 12
    assert (tmp_path / "final.rejected.jsonl").exists()
    assert (tmp_path / "final.report.json").exists()


def test_merge_partition_results_without_companion_artifacts(tmp_path):
    part1 = tmp_path / "a.jsonl"
    part2 = tmp_path / "b.jsonl"
    write_dataset(_labeled_rows(3), part1)
    write_dataset(_labeled_rows(4), part2)

    output = tmp_path / "final.jsonl"
    summary = merge_partition_results([str(part1), str(part2)], str(output))

    assert summary["accepted_records"] == 7
    assert summary["rejected_records"] == 0
    assert not (tmp_path / "final.rejected.jsonl").exists()


def test_merge_partition_results_streams_the_default_jsonl_output(tmp_path, monkeypatch):
    part1 = tmp_path / "a.jsonl"
    write_dataset(_labeled_rows(3), part1)
    output = tmp_path / "final.jsonl"

    # No format_override and a .jsonl destination -> the streaming path, which writes
    # accepted rows straight to a file handle and never calls write_dataset(all_items, ...)
    # for them at all (see the source for why that's the point).
    accepted_item_call_sizes = []

    def spy_write_dataset(items, *args, **kwargs):
        accepted_item_call_sizes.append(len(list(items)))
        return write_dataset(items, *args, **kwargs)

    monkeypatch.setattr("buffdata.integrations.sharding.write_dataset", spy_write_dataset)
    merge_partition_results([str(part1)], str(output))

    # write_dataset is never invoked with the 3 accepted rows -- only possibly for an
    # (empty, here) rejected set, since there are no .rejected.jsonl companions in this test.
    assert accepted_item_call_sizes == []
    assert len(read_dataset(output)) == 3


def test_merge_partition_results_with_format_override_uses_non_streaming_path(tmp_path):
    part1 = tmp_path / "a.jsonl"
    write_dataset(_labeled_rows(5), part1)
    output = tmp_path / "final.parquet"

    summary = merge_partition_results([str(part1)], str(output), format_override="parquet")

    assert summary["accepted_records"] == 5
    assert output.exists()
    assert len(read_dataset(output)) == 5


# --- CLI wiring -------------------------------------------------------------------------

def test_cli_shard_and_merge_round_trip(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_rows(30), source)
    parts_dir = tmp_path / "parts"

    shard_result = CliRunner().invoke(app, [
        "shard", str(source), "-o", str(parts_dir), "-n", "3",
    ])
    assert shard_result.exit_code == 0, shard_result.output

    partitions = sorted(parts_dir.glob("partition-*.jsonl"))
    assert len(partitions) == 3

    merged = tmp_path / "merged.jsonl"
    merge_result = CliRunner().invoke(app, [
        "merge", *[str(p) for p in partitions], "-o", str(merged),
    ])
    assert merge_result.exit_code == 0, merge_result.output
    assert len(read_dataset(merged)) == 30
