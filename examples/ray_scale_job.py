"""Example: shard a large dataset, process each partition as a Ray task, merge the results.

Unlike the Airflow/Dagster examples next to this file, this one *has* been run for real in
this session -- tests/test_ray_executor.py exercises the exact same partition_dataset ->
run_partitions_with_ray -> merge_partition_results flow below against a real (if
single-machine) local Ray runtime. What hasn't been verified here is a real multi-node
remote cluster -- ray.init(address="ray://...") talks to one exactly the same way
ray.init() talks to a local runtime, but there was no cluster available in this environment
to connect to and confirm. Point RAY_ADDRESS at a real one and confirm before relying on it
in production; run this file as-is first to see the local-runtime path work.

Pipeline shape:

    partition_dataset()  ->  N Ray tasks, each one OptimizationPipeline.run_file()  ->  merge_partition_results()

Requires GEMINI_API_KEY (or whichever provider you configure) available wherever the Ray
workers run -- on a remote cluster, that means every worker node, not just the driver
machine this script runs on.
"""

from __future__ import annotations

import os

from buffdata.integrations.ray_executor import run_partitions_with_ray
from buffdata.integrations.sharding import merge_partition_results, partition_dataset
from buffdata.models.schemas import PipelineConfig

NUM_PARTITIONS = 8
INPUT_DATASET = "/data/raw/train.jsonl"
PARTITIONS_DIR = "/data/partitions"
OPTIMIZED_DIR = "/data/optimized"
FINAL_OUTPUT = "/data/final/train.optimized.jsonl"

# None (the default) starts/reuses a local Ray runtime -- exactly what this session's tests
# run against. Set RAY_ADDRESS (e.g. "ray://head-node:10001") to distribute across a real
# cluster instead; no other line in this file changes.
RAY_ADDRESS = os.environ.get("RAY_ADDRESS")


def main() -> None:
    partitions = partition_dataset(INPUT_DATASET, PARTITIONS_DIR, NUM_PARTITIONS)
    print(f"Wrote {len(partitions)} partitions to {PARTITIONS_DIR}")

    config = PipelineConfig(
        provider="gemini",
        quality_mode="sampled",
        # network_policy="strict" here would fail outright since this example dataset isn't
        # pre-labeled -- left unrestricted deliberately, same call the Dagster/Airflow
        # examples make. Switch to "strict" once your data is pre-labeled and you want the
        # zero-network guarantee (buffdata/engine/client.py's NetworkForbiddenClient) on
        # every worker.
    )
    outputs = run_partitions_with_ray(partitions, OPTIMIZED_DIR, config, ray_address=RAY_ADDRESS)
    print(f"Processed {len(outputs)} partitions into {OPTIMIZED_DIR}")

    summary = merge_partition_results(outputs, FINAL_OUTPUT)
    print(f"Merged into {FINAL_OUTPUT}: {summary['accepted_records']} accepted, {summary['rejected_records']} rejected")


if __name__ == "__main__":
    main()
