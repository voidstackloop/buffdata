"""Private fixed worker entry point. No caller-supplied Python or shell commands."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import sys

from buffdata.runs.models import RunManifest, RunSpec
from buffdata.runs.service import hash_path, runtime_identity
from buffdata.runs.store import now_iso
from buffdata.security.policy import ExecutionContext, SecurityPolicy, execution_context, private_json


def execute(directory: Path):
    if os.name == "posix":
        os.umask(0o077)
    data = json.loads((directory / "execution.json").read_text())
    run, paths = data["run"], data["paths"]
    spec = RunSpec(**run["spec"])
    policy = SecurityPolicy(**data["policy"])
    if runtime_identity(policy) != run["implementation"]:
        raise ValueError("Worker implementation differs from submission")
    for kind, path in paths.items():
        if hash_path(Path(path)) != run["inputs"][kind]:
            raise ValueError("Immutable input integrity check failed")
    context = ExecutionContext(actor=run["actor"], project_id=spec.project_id,
                               network=spec.configuration["network_policy"], policy=policy)
    with execution_context(context):
        from buffdata.security.network import install_worker_network_guard
        install_worker_network_guard(context)
        from buffdata.models.schemas import PipelineConfig
        from buffdata.models.formats import read_dataset, write_dataset_atomic
        from buffdata.engine.pipeline import OptimizationPipeline
        pipeline = OptimizationPipeline(PipelineConfig(**spec.configuration))
        from buffdata.security.provider import SafeProvider
        pipeline.client = SafeProvider(pipeline.client)
        output = directory / ("optimized." + spec.output_format)
        if spec.require_positive_gain:
            from buffdata.evaluation.accuracy_gate import evaluate_accuracy_gain
            # Two independent parses: the baseline cannot alias mutable optimizer items.
            result = asyncio.run(pipeline.run(read_dataset(paths["source"])))
            gate = evaluate_accuracy_gain(read_dataset(paths["source"]), result.accepted,
                read_dataset(paths["validation"]), seeds=spec.accuracy_seeds, epochs=spec.accuracy_epochs,
                minimum_gain=spec.accuracy_min_gain, max_train_rows=spec.accuracy_max_train_rows)
            if not gate["accepted"]:
                raise ValueError("Accuracy gate failed; no output published")
            result.metrics["accuracy_gate"] = gate
            rejected, report, _ = pipeline.artifact_paths(output)
            write_dataset_atomic(result.accepted, output)
            write_dataset_atomic(result.rejected, rejected)
            private_json(report, {"profile": result.profile.model_dump(mode="json"), "metrics": result.metrics,
                                   "rejection_reasons": pipeline._rejection_counts(result.rejected)})
        else:
            result = asyncio.run(pipeline.run_file(paths["source"], output))
            rejected, report, _ = pipeline.artifact_paths(output)
        html_report = output.with_name(f"{output.stem}.report.html")
        # Encryption at rest (buffdata/security/keys.py, opt-in -- see
        # docs/system-design-roadmap.md #3.7): the data key, if any, reaches this subprocess
        # only via its own environment (set by runner.py's supervise(), never written to a
        # file), and is encrypted in place before manifest hashes are computed, so those
        # hashes -- like everything else about this file from here on -- are over ciphertext.
        run_data_key = os.environ.get("BUFFDATA_RUN_DATA_KEY")
        if run_data_key:
            import base64
            from buffdata.security.keys import encrypt_file_in_place
            data_key = base64.b64decode(run_data_key)
            for artifact_path in (output, rejected, report, html_report):
                if artifact_path.exists():
                    encrypt_file_in_place(artifact_path, data_key)
        artifacts = {name: {"path": path.relative_to(directory).as_posix(), "sha256": hash_path(path)}
                     for name, path in (("output", output), ("rejected", rejected), ("report", report))}
        if html_report.exists():
            artifacts["report_html"] = {"path": html_report.name, "sha256": hash_path(html_report)}
        manifest = RunManifest(run_id=run["id"], project_id=spec.project_id, parent_run_id=run.get("parent_run_id"),
            spec=spec, implementation=run["implementation"], inputs=run["inputs"], artifacts=artifacts,
            metrics=result.metrics, completed_at=now_iso())
        private_json(directory / "candidate-manifest.json", manifest.model_dump(mode="json"))


if __name__ == "__main__":
    try:
        execute(Path(sys.argv[1]))
    except Exception as exc:
        # Provider exceptions and dataset text are never copied into operational logs.
        private_json(Path(sys.argv[1]) / "failure.json", {"type": type(exc).__name__,
            "message": "Execution failed. Check configuration, policy, resources, and input integrity."})
        sys.exit(1)
