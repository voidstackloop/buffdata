import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from buffdata.cli.main import app
from buffdata.integrations.ray_executor import RayExecutionError, run_partitions_with_ray
from buffdata.integrations.sharding import merge_partition_results, partition_dataset
from buffdata.models.formats import read_dataset, write_dataset
from buffdata.models.schemas import DatasetItem, PipelineConfig

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module", autouse=True)
def local_ray_runtime():
    # A real, if single-machine, Ray runtime -- not a mock. Module-scoped so the whole file
    # pays Ray's startup cost once rather than once per test.
    ray.init(num_cpus=2, ignore_reinit_error=True, logging_level="warning")
    yield
    ray.shutdown()


def _offline_config(**overrides) -> PipelineConfig:
    # Mirrors tests/test_network_policy.py's fully-offline strict config: labeled data,
    # quality_mode off, exact dedup, no LLM call ever attempted -- so these tests don't
    # depend on any real provider credentials, inside or outside the Ray worker process.
    defaults = dict(
        provider="gemini",
        network_policy="strict",
        quality_mode="off",
        dedup_method="exact",
        classification="auto",
        scrub_pii=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _labeled_items(n: int, offset: int = 0) -> list[DatasetItem]:
    return [
        DatasetItem.from_dict({"text": f"apple orchard fruit stand {i}", "label": 0})
        if i % 2 == 0
        else DatasetItem.from_dict({"text": f"banana market fruit stand {i}", "label": 1})
        for i in range(offset, offset + n)
    ]


def test_run_partitions_with_ray_processes_each_shard_and_returns_ordered_outputs(tmp_path):
    source = tmp_path / "input.jsonl"
    write_dataset(_labeled_items(12), source)
    partitions = partition_dataset(str(source), str(tmp_path / "shards"), num_partitions=3)
    assert len(partitions) == 3

    output_dir = tmp_path / "outputs"
    outputs = run_partitions_with_ray(partitions, output_dir, _offline_config())

    assert len(outputs) == 3
    assert [Path(p).stem for p in outputs] == [f"{p.stem}.optimized" for p in partitions]
    total_rows = 0
    for output_path, partition_path in zip(outputs, partitions):
        assert output_path is not None
        rows = [json.loads(line) for line in open(output_path, encoding="utf-8")]
        expected = sum(1 for _ in open(partition_path, encoding="utf-8"))
        assert len(rows) == expected
        total_rows += len(rows)
    assert total_rows == 12


def test_run_partitions_with_ray_composes_with_merge_partition_results(tmp_path):
    source = tmp_path / "input.jsonl"
    write_dataset(_labeled_items(20), source)
    partitions = partition_dataset(str(source), str(tmp_path / "shards"), num_partitions=4)

    outputs = run_partitions_with_ray(partitions, tmp_path / "outputs", _offline_config())

    final_output = tmp_path / "merged.jsonl"
    summary = merge_partition_results(outputs, str(final_output))

    assert summary["accepted_records"] == 20
    assert final_output.exists()
    merged_rows = [json.loads(line) for line in open(final_output, encoding="utf-8")]
    assert len(merged_rows) == 20


def test_run_partitions_with_ray_raises_on_shard_failure(tmp_path):
    # Unlabeled data under network_policy=strict + classification=auto needs a remote
    # classification call NetworkForbiddenClient refuses -- same failure
    # tests/test_network_policy.py exercises for a single (non-Ray) run, here surfacing
    # through a Ray task instead.
    unlabeled = [DatasetItem.from_dict({"text": f"unlabeled sentence {i}"}) for i in range(4)]
    source = tmp_path / "input.jsonl"
    write_dataset(unlabeled, source)
    partitions = partition_dataset(str(source), str(tmp_path / "shards"), num_partitions=1)

    with pytest.raises(RayExecutionError, match="shard task"):
        run_partitions_with_ray(partitions, tmp_path / "outputs", _offline_config())


def test_run_partitions_with_ray_rejects_empty_partition_list(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        run_partitions_with_ray([], tmp_path / "outputs", _offline_config())


# --- CLI wiring -------------------------------------------------------------------------

def test_cli_shard_run_ray_merge_round_trip(tmp_path):
    source = tmp_path / "train.jsonl"
    write_dataset(_labeled_items(30), source)
    parts_dir = tmp_path / "parts"

    shard_result = CliRunner().invoke(app, ["shard", str(source), "-o", str(parts_dir), "-n", "3"])
    assert shard_result.exit_code == 0, shard_result.output
    partitions = sorted(parts_dir.glob("partition-*.jsonl"))
    assert len(partitions) == 3

    config_file = tmp_path / "pipeline.yaml"
    config_file.write_text(
        yaml.safe_dump({
            "provider": "gemini", "network_policy": "strict", "quality_mode": "off",
            "dedup_method": "exact", "classification": "auto", "scrub_pii": False,
        }),
        encoding="utf-8",
    )
    outputs_dir = tmp_path / "outputs"

    ray_result = CliRunner().invoke(app, [
        "run-ray", str(config_file), *[str(p) for p in partitions],
        "-o", str(outputs_dir), "--num-cpus", "2",
    ])
    assert ray_result.exit_code == 0, ray_result.output

    outputs = sorted(outputs_dir.glob("*.optimized.jsonl"))
    assert len(outputs) == 3

    merged = tmp_path / "merged.jsonl"
    merge_result = CliRunner().invoke(app, [
        "merge", *[str(p) for p in outputs], "-o", str(merged),
    ])
    assert merge_result.exit_code == 0, merge_result.output
    assert len(read_dataset(merged)) == 30


def test_ray_executor_gives_a_clear_error_when_ray_is_not_installed(monkeypatch, tmp_path):
    import builtins

    from buffdata.integrations import ray_executor as ray_executor_module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ray":
            raise ImportError("simulated: ray not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RayExecutionError, match=r"pip install buffdata\[enterprise\]"):
        ray_executor_module.run_partitions_with_ray(["a.jsonl"], tmp_path / "out", _offline_config())
