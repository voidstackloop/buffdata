# Scaling beyond one process

BuffData's own execution model stays deliberately unchanged at any scale: one
process, one machine, the same proven pipeline (see [architecture.md](architecture.md)).
Scale to TB-sized datasets by handing partitions of the data to *something else* that
runs N independent BuffData invocations in parallel, one per partition. Two
interchangeable ways to be that "something else":

```
                    partition_dataset()
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    external orchestrator          Ray remote tasks
  (Airflow / Dagster / a k8s        (run-ray, same
   Job array / xargs -P)             process, no extra
   -- N separate `buffdata`          infra beyond a Ray
   subprocess invocations)           cluster you already run)
              │                           │
              └─────────────┬─────────────┘
                             ▼
                  merge_partition_results()
```

Both ends -- `partition_dataset()` and `merge_partition_results()`
([`buffdata/integrations/sharding.py`](../buffdata/integrations/sharding.py)) -- are
identical either way; only the middle changes.

## Sharding

```bash
buffdata shard data.jsonl -o parts/ -n 8
```

Streams the input and writes `parts/partition-0000.jsonl` .. `partition-0007.jsonl`
without ever materializing the full dataset in memory. Rows with a scalar label are
distributed round-robin *within their own label* (`--no-stratify` disables this), so
every partition ends up with a proportional class mix -- important because
classification and the accuracy gate both need every label represented in whatever
subset they see.

```bash
buffdata merge out/partition-0000.jsonl out/partition-0001.jsonl ... -o final.jsonl
```

Concatenates N independently-processed outputs, merging each partition's own
`.rejected.jsonl`/`.report.json` artifacts (summed token usage, combined rejection
reasons) into one final summary. Streams straight to the output file when the target
format is jsonl/ndjson (the default) rather than building the full merged dataset in
memory first.

### External orchestrator

Run N `buffdata optimize`/`score`/... invocations against the partition files with
whatever you already operate -- worked, annotated examples for both are in
[`examples/airflow_scale_dag.py`](../examples/airflow_scale_dag.py) and
[`examples/dagster_scale_job.py`](../examples/dagster_scale_job.py). Both shell out to
the `buffdata` CLI via subprocess rather than importing BuffData's Python API into the
orchestrator's own process, deliberately isolating BuffData's dependencies (torch,
transformers, presidio, ...) from the orchestrator's environment.

**Status**: written against the real CLI, not run against a live Airflow/Dagster
instance in the environment they were built in -- copy them into your own project and
adjust before relying on them, same caveat as [deployment.md](deployment.md).

### Ray

[`buffdata/integrations/ray_executor.py`](../buffdata/integrations/ray_executor.py)
is the in-process alternative for teams that already run a Ray cluster: each partition
becomes an independent Ray remote task running a full `OptimizationPipeline.run_file()`
in a Ray worker process, instead of a separately-launched `buffdata` subprocess.

```bash
buffdata run-ray pipeline.yaml parts/partition-0000.jsonl parts/partition-0001.jsonl ... \
  -o outputs/ --ray-address ray://head-node:10001
```

Omit `--ray-address` to start/reuse a local Ray runtime instead -- a real, if
single-machine, execution mode, not a mock; that's exactly what
[`tests/test_ray_executor.py`](../tests/test_ray_executor.py) runs against for real (a
live `ray.init()` runtime, real remote tasks, real pipeline runs, plus the full
`shard -> run-ray -> merge` CLI round trip). A real remote cluster works identically --
`ray.init(address=...)` connects the same way regardless of what's on the other end;
that specific path (a genuine multi-node cluster) hasn't been exercised, since none
was available in the environment this was built in.

Requires `pip install -e ".[enterprise]"` for `ray[default]`.

## Which one to use

| | External orchestrator | Ray |
|---|---|---|
| Needs | Airflow/Dagster/k8s, or just a shell loop | A Ray cluster (local or remote) |
| Isolation | Full process isolation per partition | Ray worker processes, same cluster |
| Best fit | Already have a workflow orchestrator | Already have a Ray cluster, want one command instead of DAG boilerplate |
