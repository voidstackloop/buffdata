"""Cached-data parity: original vs legacy vs managed. No provider/network calls."""
import asyncio
import json
from pathlib import Path
import time
import torch
from benchmark_buffdata import build_vocab, train_once, summarize
from buffdata.models.formats import read_dataset
from buffdata.models.schemas import PipelineConfig
from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.runs.service import RunService
from buffdata.runs.models import RunSpec
from buffdata.runs.runner import run_local


def main():
    torch.set_num_threads(8)
    root = Path("benchmarks/results-managed-security")
    root.mkdir(parents=True, exist_ok=True)
    service = RunService(root / "runs")
    service.store.ensure_project("local", {"policy": {}, "members": {"local": "administrator"}})
    config = dict(accuracy_contract="strict", quality_mode="off", network_policy="strict", scrub_pii=False)
    results = {"method": {"dataset": "cached AG News", "seeds": [17, 29, 43], "epochs": 6,
                          "model": "EmbeddingBag(mean, 64) + Linear", "remote_calls": 0}, "sizes": {}}
    for size, folder in [(3000, "results"), (20000, "results-20k"), (100000, "results-100k")]:
        source = Path("benchmarks") / folder / "ag_news/clean_train.jsonl"
        test = Path("benchmarks") / folder / "ag_news/clean_test.jsonl"
        original = read_dataset(source)
        assert len(original) == size, (size, len(original))
        started = time.monotonic()
        legacy = asyncio.run(OptimizationPipeline(PipelineConfig(**config)).run(original))
        legacy_seconds = time.monotonic() - started
        started = time.monotonic()
        dataset = service.register("local", source)
        spec = RunSpec(dataset_id=dataset["id"], configuration=config)
        run = service.submit(spec)
        finished = run_local(service, "local", run["id"])
        assert finished["status"] == "succeeded", finished
        managed = read_dataset(service.artifact("local", run["id"], "output"))
        managed_seconds = time.monotonic() - started
        def rows(items):
            return [{"text": x.get_classification_text(), "label": int(x.labels)} for x in items]
        conditions = {"original": rows(read_dataset(source)), "legacy": rows(legacy.accepted), "managed": rows(managed)}
        assert conditions["original"] == conditions["legacy"] == conditions["managed"]
        vocab, test_rows = build_vocab(conditions["original"]), rows(read_dataset(test))
        measured = {}
        for name, samples in conditions.items():
            runs = []
            for seed in [17, 29, 43]:
                print(f"Training {size}/{name} seed={seed}", flush=True)
                runs.append(train_once(samples, test_rows, vocab, 4, seed, 6))
            measured[name] = {"runs": runs, "summary": summarize(runs)}
        metric_keys = ("accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std")
        metric_identity = all(measured["original"]["summary"][key] == measured["legacy"]["summary"][key]
                              == measured["managed"]["summary"][key] for key in metric_keys)
        results["sizes"][str(size)] = {"row_identity": True, "metric_identity": metric_identity, "conditions": measured,
            "test_rows": len(test_rows), "legacy_seconds": legacy_seconds, "managed_seconds": managed_seconds}
        (root / "results.json").write_text(json.dumps(results, indent=2))
        assert metric_identity, "Accuracy or macro-F1 changed"
        print(f"PASS {size}: row and training-metric parity", flush=True)


if __name__ == "__main__":
    main()
