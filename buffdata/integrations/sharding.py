"""Dataset sharding for orchestrator-level parallelism.

BuffData's own execution model is deliberately unchanged by this module: one process, one
machine, the same proven pipeline. Scale to TB-sized datasets by handing partitions of the
data to whatever orchestrator you already run -- Airflow, Dagster, a Kubernetes Job array,
even a plain shell loop with `xargs -P` -- and letting it run N independent
`buffdata optimize`/`score`/`refine`/... invocations in parallel, one per partition. This
module provides the partition/merge logic every one of those needs, so BuffData doesn't have
to build or operate a distributed execution engine of its own to reach that scale.

For teams that already run a Ray cluster, buffdata.integrations.ray_executor is a direct
in-process alternative to the external-orchestrator step: same partition_dataset() /
merge_partition_results() on either end, Ray remote tasks distributing the middle instead
of N external CLI invocations.

partition_dataset streams the input via iter_dataset and writes each row straight to its
target partition file as it arrives, rather than collecting the whole dataset into N
in-memory lists first -- for jsonl/ndjson/txt inputs (where iter_dataset already reads line
by line) this holds only a small per-label counter and num_partitions open file handles in
memory, never the dataset itself. Other input formats (parquet/csv/tsv/hf) still read fully
into memory once inside iter_dataset's own fallback -- that's an existing limitation of
iter_dataset, not something introduced here; this at least avoids compounding it with an
extra full-dataset copy split across N lists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import orjson

from buffdata.models.formats import iter_dataset, write_dataset
from buffdata.models.schemas import DatasetItem


def partition_dataset(
    input_path: str,
    output_dir: str,
    num_partitions: int,
    *,
    stratify_by_label: bool = True,
) -> list[Path]:
    """Split a dataset into up to `num_partitions` files under `output_dir`, named
    partition-0000.jsonl .. partition-NNNN.jsonl (partitions are always written as jsonl,
    regardless of the input format, since it's the one container every downstream
    orchestrator task and every buffdata command reads without ambiguity). A partition that
    would end up empty (more partitions requested than rows available for it) is never
    created, so the returned list can be shorter than `num_partitions`.

    Each row with a scalar label is distributed round-robin *within its own label* (tracked
    via a small per-label counter, not by holding every prior row of that label), so every
    partition ends up with a proportional class mix -- important because classification and
    the accuracy gate both need every label represented in whatever subset they see. A row
    with no label, a list label, or `stratify_by_label=False` instead goes through a single
    shared round-robin counter.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digits = max(4, len(str(num_partitions - 1)))
    partition_paths = [out_dir / f"partition-{index:0{digits}d}.jsonl" for index in range(num_partitions)]
    handles: list[Any] = [None] * num_partitions
    label_counters: dict[str, int] = {}
    plain_counter = 0
    row_count = 0

    try:
        for item in iter_dataset(input_path):
            if stratify_by_label and item.labels is not None and not isinstance(item.labels, list):
                label = str(item.labels)
                seen = label_counters.get(label, 0)
                label_counters[label] = seen + 1
                index = seen % num_partitions
            else:
                index = plain_counter % num_partitions
                plain_counter += 1

            if handles[index] is None:
                handles[index] = open(partition_paths[index], "wb")
            handles[index].write(orjson.dumps(item.to_dict()) + b"\n")
            row_count += 1
    finally:
        for handle in handles:
            if handle is not None:
                handle.close()

    if row_count == 0:
        raise ValueError(f"No rows found in {input_path}")

    return [path for path, handle in zip(partition_paths, handles) if handle is not None]


def merge_partition_results(
    partition_output_paths: list[str],
    final_output_path: str,
    *,
    format_override: Optional[str] = None,
) -> dict[str, Any]:
    """Concatenate N partition outputs -- each already run independently through
    buffdata optimize/score/refine/evolve/dpo/augment/... -- into one final dataset. Also
    merges each partition's companion .rejected.jsonl / .report.json artifacts, when
    present alongside it, into a single rejected file and a summed metrics report. Returns
    that merged summary.

    Streams the accepted rows straight into the final file when the output format is
    jsonl/ndjson (the default, and the format every partition is already in) -- memory use
    then stays bounded by one partition's rejected-row list at a time, not the full merged
    dataset. Any other final format (an explicit format_override) still builds the full
    result in memory first, since every non-jsonl writer in models/formats.py needs a
    complete row list to build its DataFrame/container.
    """
    final_path = Path(final_output_path)
    stream_output = format_override is None and final_path.suffix.lower() in {".jsonl", ".ndjson"}

    merged_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    merged_rejection_reasons: dict[str, int] = {}
    accepted_count = 0
    all_items: list[DatasetItem] = []
    all_rejected: list[DatasetItem] = []

    accepted_handle = open(final_path, "wb") if stream_output else None
    try:
        for partition_path_str in partition_output_paths:
            partition_path = Path(partition_path_str)
            for item in iter_dataset(partition_path):
                if stream_output:
                    accepted_handle.write(orjson.dumps(item.to_dict()) + b"\n")
                else:
                    all_items.append(item)
                accepted_count += 1

            rejected_path = partition_path.with_name(f"{partition_path.stem}.rejected.jsonl")
            if rejected_path.exists():
                all_rejected.extend(iter_dataset(rejected_path))

            report_path = partition_path.with_name(f"{partition_path.stem}.report.json")
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                usage = report.get("metrics", {}).get("usage", {})
                for key in merged_usage:
                    merged_usage[key] += usage.get(key, 0)
                for reason, count in report.get("rejection_reasons", {}).items():
                    merged_rejection_reasons[reason] = merged_rejection_reasons.get(reason, 0) + count
    finally:
        if accepted_handle is not None:
            accepted_handle.close()

    if not stream_output:
        write_dataset(all_items, final_output_path, format_override=format_override)
    if all_rejected:
        write_dataset(all_rejected, final_path.with_name(f"{final_path.stem}.rejected.jsonl"))

    summary = {
        "partitions_merged": len(partition_output_paths),
        "accepted_records": accepted_count,
        "rejected_records": len(all_rejected),
        "usage": merged_usage,
        "rejection_reasons": merged_rejection_reasons,
    }
    final_path.with_name(f"{final_path.stem}.report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
