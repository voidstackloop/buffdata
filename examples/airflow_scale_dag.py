"""Example Airflow DAG: shard a large dataset, clean each partition in parallel, merge.

NOT installed as part of the buffdata package, and not run against a live Airflow instance
in the session that wrote it -- copy it into your own DAGs folder and adjust for your Airflow
version before relying on it. It's deliberately built on BashOperator calling the `buffdata`
CLI as a subprocess, not a custom buffdata-specific Operator class, for two reasons:

1. It only depends on things that are genuinely stable across Airflow versions (BashOperator
   and dynamic task mapping have existed since Airflow 2.3+), instead of an internal
   BaseOperator API this project has no way to verify compatibility against.
2. It's the pattern most Airflow deployments already prefer for third-party CLIs: buffdata's
   dependencies (torch, transformers, sentence-transformers, presidio, ...) never have to be
   installed into the Airflow scheduler/worker's own Python environment, which is exactly the
   kind of dependency collision that makes custom PythonOperator integrations fragile in
   practice. The bare `buffdata` command below assumes it's on PATH wherever the task
   actually executes -- swap in an absolute path to a dedicated venv's binary, or run these
   BashOperators through a KubernetesPodOperator / DockerOperator image that has `buffdata`
   installed, if your workers don't have it directly.

Pipeline shape:

    shard  ->  optimize (one mapped task per partition, parallel)  ->  merge

Requires GEMINI_API_KEY (or whichever provider you configure) available to the worker
environment the BashOperator tasks execute in -- see buffdata's secrets backends
(buffdata/engine/secrets.py) if you'd rather source it from Vault/AWS/GCP/Azure than a
plain environment variable.
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag
from airflow.operators.bash import BashOperator

NUM_PARTITIONS = 8
INPUT_DATASET = "/data/raw/train.jsonl"
PARTITIONS_DIR = "/data/partitions"
OPTIMIZED_DIR = "/data/optimized"
FINAL_OUTPUT = "/data/final/train.optimized.jsonl"
BUFFDATA_PROVIDER = "gemini"


@dag(
    dag_id="buffdata_scale_optimize",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["buffdata", "data-quality"],
)
def buffdata_scale_optimize():
    shard = BashOperator(
        task_id="shard",
        bash_command=(
            f"buffdata shard {INPUT_DATASET} "
            f"-o {PARTITIONS_DIR} -n {NUM_PARTITIONS}"
        ),
    )

    # NUM_PARTITIONS is a static constant known at DAG-parse time (it's the same value
    # passed to `buffdata shard -n` above), so the partition indices below are just that
    # constant expanded into a list -- no need for a runtime task to discover them.
    optimize_partition = BashOperator.partial(
        task_id="optimize_partition",
        # --network-policy strict here would fail this DAG's classification step outright --
        # left unrestricted deliberately since this example dataset isn't already labeled.
        # Set it to "strict" once your data is pre-labeled and you want the zero-network
        # guarantee from Phase 1(d) enforced per-partition.
    ).expand(
        bash_command=[
            (
                f"buffdata optimize {PARTITIONS_DIR}/partition-{index:04d}.jsonl "
                f"-o {OPTIMIZED_DIR}/partition-{index:04d}.jsonl "
                f"--provider {BUFFDATA_PROVIDER} --quality-mode sampled"
            )
            for index in range(NUM_PARTITIONS)
        ]
    )

    merge = BashOperator(
        task_id="merge",
        bash_command=(
            f"buffdata merge {OPTIMIZED_DIR}/partition-*.jsonl -o {FINAL_OUTPUT}"
        ),
    )

    shard >> optimize_partition >> merge


buffdata_scale_optimize()
