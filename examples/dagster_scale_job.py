"""Example Dagster job: shard a large dataset, clean each partition in parallel, merge.

NOT installed as part of the buffdata package, and not run against a live Dagster instance
in the session that wrote it -- copy it into your own Dagster project and adjust before
relying on it. Like the Airflow example next to this file, every op shells out to the
`buffdata` CLI via subprocess rather than importing buffdata's Python API directly into the
Dagster process: it isolates buffdata's dependencies (torch, transformers,
sentence-transformers, presidio, ...) from Dagster's own environment, and it only depends on
Dagster's stable op/job/DynamicOut API rather than anything buffdata-specific this project
has no way to verify against.

Pipeline shape:

    shard_op  ->  optimize_partition_op (one per partition, fanned out, parallel)  ->  merge_op

Requires GEMINI_API_KEY (or whichever provider you configure) available to the process each
op's subprocess runs in -- see buffdata's secrets backends (buffdata/engine/secrets.py) if
you'd rather source it from Vault/AWS/GCP/Azure than a plain environment variable.
"""

from __future__ import annotations

import subprocess

from dagster import DynamicOut, DynamicOutput, In, Out, job, op

NUM_PARTITIONS = 8
INPUT_DATASET = "/data/raw/train.jsonl"
PARTITIONS_DIR = "/data/partitions"
OPTIMIZED_DIR = "/data/optimized"
FINAL_OUTPUT = "/data/final/train.optimized.jsonl"
BUFFDATA_PROVIDER = "gemini"


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


@op(out=Out(str))
def shard_op(context) -> str:
    _run(["buffdata", "shard", INPUT_DATASET, "-o", PARTITIONS_DIR, "-n", str(NUM_PARTITIONS)])
    context.log.info(f"Wrote {NUM_PARTITIONS} partitions to {PARTITIONS_DIR}")
    return PARTITIONS_DIR


@op(ins={"partitions_dir": In(str)}, out=DynamicOut(int))
def fan_out_partitions(partitions_dir: str):
    # Partition files are named partition-0000.jsonl .. partition-NNNN.jsonl by
    # buffdata shard; NUM_PARTITIONS is the same constant passed to `-n` above, so the
    # indices are already known rather than requiring a directory listing here.
    for index in range(NUM_PARTITIONS):
        yield DynamicOutput(index, mapping_key=f"partition_{index:04d}")


@op(ins={"partition_index": In(int)}, out=Out(str))
def optimize_partition_op(context, partition_index: int) -> str:
    input_path = f"{PARTITIONS_DIR}/partition-{partition_index:04d}.jsonl"
    output_path = f"{OPTIMIZED_DIR}/partition-{partition_index:04d}.jsonl"
    # --network-policy strict here would fail outright since this example dataset isn't
    # already labeled -- left unrestricted deliberately. Set it to "strict" once your data
    # is pre-labeled and you want the zero-network guarantee from Phase 1(d) per partition.
    _run([
        "buffdata", "optimize", input_path, "-o", output_path,
        "--provider", BUFFDATA_PROVIDER, "--quality-mode", "sampled",
    ])
    context.log.info(f"Optimized partition {partition_index} -> {output_path}")
    return output_path


@op(ins={"partition_outputs": In(list)})
def merge_op(context, partition_outputs: list[str]) -> None:
    _run(["buffdata", "merge", *partition_outputs, "-o", FINAL_OUTPUT])
    context.log.info(f"Merged {len(partition_outputs)} partitions -> {FINAL_OUTPUT}")


@job
def buffdata_scale_optimize_job():
    partitions_dir = shard_op()
    partition_indices = fan_out_partitions(partitions_dir)
    optimized_paths = partition_indices.map(optimize_partition_op)
    merge_op(optimized_paths.collect())
