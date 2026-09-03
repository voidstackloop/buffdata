#!/usr/bin/env python3
"""Real CLI classification-correctness check: for each of the binary/multi-class/
multi-label datasets already used by the accuracy-recovery benchmarks, sample N
already-labeled rows, strip their labels, run the REAL `buffdata classify` CLI command
against them with real LLM calls, and compare its predictions against the real ground
truth. Answers a different question than the recovery benchmarks: not "does cleaning
help" but "does the classify stage actually assign correct labels."

Requires GEMINI_API_KEY (or whichever --provider you configure) and makes real,
billed API calls -- num_datasets * sample_size rows total, batched by buffdata's own
classifier (see buffdata/optimizers/classifier.py), not one request per row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_scale_matrix import DATASETS as SCALAR_DATASETS, load_normalized  # noqa: E402
from benchmark_multilabel_matrix import DATASETS as MULTILABEL_DATASETS, load_normalized_multilabel  # noqa: E402
from benchmark_multilabel_matrix import _CIVIL_COMMENTS_COLS, _PUBMED_MESH_COLS, _JIGSAW_COLS  # noqa: E402

SAMPLE_SIZE = 20
TEST_SEED = 999


def _class_names_scalar(spec: dict[str, Any], sample_rows: list[dict[str, Any]]) -> list[str]:
    """Real class names for a scalar-label dataset: HF's ClassLabel.names when the
    source feature carries one, else derived from real data (a label_text column, which
    every SetFit-hosted dataset in this benchmark provides) -- never invented.

    `sample_rows` (already-normalized {"text", "label"} rows from load_normalized) never
    carries label_text -- every normalize lambda strips it along with every other
    original column. The label_text fallback below loads its own small RAW sample
    directly, bypassing normalization, specifically to keep that column.
    """
    from datasets import ClassLabel, load_dataset, load_dataset_builder

    builder = load_dataset_builder(spec["hf_id"], spec.get("hf_config")) if spec.get("hf_config") else load_dataset_builder(spec["hf_id"])
    label_col = "intent" if spec["hf_id"] == "clinc/clinc_oos" else "label"
    feature = builder.info.features.get(label_col)
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    # Fall back to the label_text companion column SetFit-hosted datasets provide.
    # Scans incrementally rather than a fixed slice and stops once every declared class
    # is covered -- some of these datasets are sorted by label for long stretches (e.g.
    # SetFit/imdb's first 3000 rows are a single class), so a small fixed sample can
    # silently miss classes; a hard cap still bounds the worst case for very large,
    # very skewed sources.
    expected = spec["classes"]
    max_scan = 200_000
    raw = load_dataset(spec["hf_id"], spec.get("hf_config"), split="train") if spec.get("hf_config") else load_dataset(spec["hf_id"], split="train")
    names: dict[int, str] = {}
    for scanned, row in enumerate(raw):
        if scanned >= max_scan or len(names) >= expected:
            break
        if "label_text" in row and row.get("label") is not None:
            names[int(row["label"])] = str(row["label_text"])
    if not names:
        raise RuntimeError(f"No class names resolvable for {spec['hf_id']} -- no ClassLabel feature and no label_text column.")
    if len(names) < expected:
        raise RuntimeError(
            f"Only found {len(names)}/{expected} class names for {spec['hf_id']} within the first "
            f"{max_scan} rows -- label_text coverage is incomplete, not a hardcoded guess."
        )
    return [names.get(i, f"class_{i}") for i in range(max(names) + 1)]


def _class_names_multilabel(name: str, spec: dict[str, Any]) -> list[str]:
    if name == "civil_comments":
        return _CIVIL_COMMENTS_COLS
    if name == "pubmed_mesh":
        return _PUBMED_MESH_COLS
    if name == "jigsaw_toxicity":
        return _JIGSAW_COLS
    # go_emotions, eurlex: real Sequence(ClassLabel) feature on the "labels" column.
    from datasets import load_dataset_builder

    builder = load_dataset_builder(spec["hf_id"], spec.get("hf_config")) if spec.get("hf_config") else load_dataset_builder(spec["hf_id"])
    feature = builder.info.features["labels"]
    return list(feature.feature.names)


async def check_scalar_dataset(name: str, spec: dict[str, Any], provider: str, model: str, tmp_dir: Path) -> dict[str, Any]:
    import random

    train_split, _test_split = load_normalized(spec)
    rows = list(train_split)
    class_names = _class_names_scalar(spec, rows[:200])
    kind = spec["kind"]

    rng = random.Random(TEST_SEED)
    sample = rng.sample(rows, min(SAMPLE_SIZE, len(rows)))
    input_path = tmp_dir / f"{name}_input.jsonl"
    output_path = tmp_dir / f"{name}_output.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for row in sample:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")

    cli_type = "binary" if kind == "binary" else "multi-class"
    started = time.perf_counter()
    result = subprocess.run(
        [
            "buffdata", "classify", str(input_path), "-o", str(output_path),
            "--type", cli_type, "--classes", ",".join(class_names),
            "--provider", provider, "--model", model,
        ],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0 or not output_path.exists():
        return {"dataset": name, "kind": kind, "error": (result.stderr or result.stdout)[-500:], "elapsed_s": elapsed}

    predicted = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    correct = 0
    errored = 0
    error_samples: list[str] = []
    scored = 0
    for true_row, pred_row in zip(sample, predicted):
        row_error = pred_row.get("_buffdata_metadata", {}).get("classification_error")
        if row_error:
            errored += 1
            if len(error_samples) < 2:
                error_samples.append(row_error[:200])
            continue
        scored += 1
        true_name = class_names[int(true_row["label"])]
        pred_name = pred_row.get("labels") or pred_row.get("label")
        if isinstance(pred_name, list):
            pred_name = pred_name[0] if pred_name else None
        if pred_name == true_name:
            correct += 1
    return {
        "dataset": name, "kind": kind, "sampled": len(sample), "predicted": len(predicted),
        "scored": scored, "errored": errored, "error_samples": error_samples,
        "accuracy": correct / scored if scored else None, "elapsed_s": elapsed,
    }


async def check_multilabel_dataset(name: str, spec: dict[str, Any], provider: str, model: str, tmp_dir: Path) -> dict[str, Any]:
    import random

    train_split, _test_split = load_normalized_multilabel(spec)
    rows = list(train_split)
    class_names = _class_names_multilabel(name, spec)

    rng = random.Random(TEST_SEED)
    sample = rng.sample(rows, min(SAMPLE_SIZE, len(rows)))
    input_path = tmp_dir / f"{name}_input.jsonl"
    output_path = tmp_dir / f"{name}_output.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for row in sample:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")

    started = time.perf_counter()
    result = subprocess.run(
        [
            "buffdata", "classify", str(input_path), "-o", str(output_path),
            "--type", "multi-label", "--classes", ",".join(class_names),
            "--provider", provider, "--model", model,
        ],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0 or not output_path.exists():
        return {"dataset": name, "kind": "multilabel", "error": (result.stderr or result.stdout)[-500:], "elapsed_s": elapsed}

    predicted = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_f1s = []
    errored = 0
    error_samples: list[str] = []
    for true_row, pred_row in zip(sample, predicted):
        row_error = pred_row.get("_buffdata_metadata", {}).get("classification_error")
        if row_error:
            errored += 1
            if len(error_samples) < 2:
                error_samples.append(row_error[:200])
            continue
        true_names = {class_names[i] for i in true_row["label"]}
        pred_names = set(pred_row.get("labels") or [])
        tp = len(true_names & pred_names)
        fp = len(pred_names - true_names)
        fn = len(true_names - pred_names)
        precision = tp / (tp + fp) if (tp + fp) else (1.0 if not true_names else 0.0)
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        label_f1s.append(f1)
    return {
        "dataset": name, "kind": "multilabel", "sampled": len(sample), "predicted": len(predicted),
        "scored": len(label_f1s), "errored": errored, "error_samples": error_samples,
        "per_row_f1_mean": sum(label_f1s) / len(label_f1s) if label_f1s else None, "elapsed_s": elapsed,
    }


async def main(args: argparse.Namespace) -> None:
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    results: list[dict[str, Any]] = []
    scalar_names = args.datasets if args.datasets else list(SCALAR_DATASETS)
    multilabel_names = args.datasets if args.datasets else list(MULTILABEL_DATASETS)

    for name in scalar_names:
        if name not in SCALAR_DATASETS:
            continue
        print(f"=== {name} (scalar) ===", flush=True)
        try:
            outcome = await check_scalar_dataset(name, SCALAR_DATASETS[name], args.provider, args.model, tmp_dir)
        except Exception as exc:
            outcome = {"dataset": name, "kind": SCALAR_DATASETS[name]["kind"], "error": f"{type(exc).__name__}: {exc}"}
        print(f"  -> {outcome}", flush=True)
        results.append(outcome)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for name in multilabel_names:
        if name not in MULTILABEL_DATASETS:
            continue
        print(f"=== {name} (multi-label) ===", flush=True)
        try:
            outcome = await check_multilabel_dataset(name, MULTILABEL_DATASETS[name], args.provider, args.model, tmp_dir)
        except Exception as exc:
            outcome = {"dataset": name, "kind": "multilabel", "error": f"{type(exc).__name__}: {exc}"}
        print(f"  -> {outcome}", flush=True)
        results.append(outcome)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Results: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmarks/results-classification-correctness")
    parser.add_argument("--tmp-dir", default="/tmp/classify_correctness")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument("--datasets", nargs="+", default=None)
    asyncio.run(main(parser.parse_args()))
