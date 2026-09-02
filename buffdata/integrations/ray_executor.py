"""Distributes OptimizationPipeline runs across a Ray cluster, one task per dataset shard.

sharding.py's own docstring describes the default pattern: partition_dataset() splits a
dataset, an external orchestrator (Airflow, Dagster, a k8s Job array, `xargs -P`) runs N
independent `buffdata` CLI invocations against those partitions, and merge_partition_results()
recombines the outputs. This module is an alternative to the "external orchestrator" part
for teams that already run a Ray cluster: it submits the same N-way work as Ray remote
tasks instead, each running OptimizationPipeline.run_file() in a Ray worker process rather
than a separately-launched `buffdata` subprocess. Everything upstream (partition_dataset)
and downstream (merge_partition_results) is unchanged and still composes with this directly.

Requires the optional `ray` dependency (buffdata[enterprise]). ray.init() with no address
starts (or reuses) a real, if single-machine, local Ray runtime -- not a mock -- which is
what this module's own tests run against; ray.init(address="ray://host:10001") connects to
a real remote cluster with no other code difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

from buffdata.models.schemas import PipelineConfig


class RayExecutionError(RuntimeError):
    """Raised when Ray itself isn't available, or one or more shard tasks fail."""


def _run_one_shard(input_path: str, output_path: str, config_dict: dict[str, Any]) -> dict[str, Any]:
    """Executed inside a Ray worker process (a separate process from the driver, possibly
    on a different machine) -- imports happen here, not at module import time on the
    driver, so the driver process never needs OptimizationPipeline's own heavy dependencies
    just to submit work.
    """
    import asyncio

    from buffdata.engine.pipeline import OptimizationPipeline
    from buffdata.models.schemas import PipelineConfig as _PipelineConfig

    config = _PipelineConfig(**config_dict)
    pipeline = OptimizationPipeline(config)
    result = asyncio.run(pipeline.run_file(Path(input_path), Path(output_path)))
    return {
        "input_path": input_path,
        "output_path": output_path,
        "accepted": len(result.accepted),
        "rejected": len(result.rejected),
    }


def run_partitions_with_ray(
    partition_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    config: PipelineConfig,
    *,
    ray_address: Optional[str] = None,
    num_cpus: Optional[int] = None,
) -> List[str]:
    """Runs OptimizationPipeline.run_file() once per entry in `partition_paths`, each as an
    independent Ray task, and returns the output path for each -- in the same order as
    `partition_paths` -- ready to pass straight into merge_partition_results().

    Connects to `ray_address` if given (a real cluster, e.g. "ray://host:10001"); otherwise
    starts or reuses whatever local Ray runtime is available. Raises RayExecutionError
    (never a partial/silent result) if `ray` isn't installed, or if any shard's task raises
    -- ray.get on the full list of futures surfaces the first task's exception, matching
    OptimizationPipeline.run's own fail-fast behavior for a single run.
    """
    try:
        import ray
    except ImportError as exc:
        raise RayExecutionError(
            "Install ray (pip install buffdata[enterprise]) to distribute pipeline runs across a Ray cluster."
        ) from exc

    if not partition_paths:
        raise ValueError("partition_paths must be non-empty.")

    if not ray.is_initialized():
        ray.init(address=ray_address, num_cpus=num_cpus, ignore_reinit_error=True, logging_level="warning")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.model_dump(mode="json")

    remote_run_one_shard = ray.remote(_run_one_shard)
    output_paths = [str(out_dir / f"{Path(path).stem}.optimized.jsonl") for path in partition_paths]
    futures = [
        remote_run_one_shard.remote(str(input_path), output_path, config_dict)
        for input_path, output_path in zip(partition_paths, output_paths)
    ]

    try:
        ray.get(futures)  # order matches futures/partition_paths; raises on the first task failure
    except Exception as exc:
        raise RayExecutionError(f"One or more shard tasks failed: {exc}") from exc

    return output_paths
