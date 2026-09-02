#!/usr/bin/env python3
"""Original-versus-generated accuracy regression and value benchmark.

The preservation panel reproduces the reported classification regression using
matching AG News records from ``train_25k_raw.jsonl`` and the previously generated
``train_25k_clean.jsonl``. The value panel starts from a fixed Hugging Face sample,
adds disclosed deterministic defects, and compares that original contaminated
dataset with BuffData's newly generated optimized output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from buffdata.engine.client import LLMProvider
from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.models.schemas import DatasetItem, PipelineConfig
from benchmark_buffdata import (
    build_vocab,
    make_dirty,
    stratified_rows,
    summarize,
    train_once,
    write_jsonl,
)


class OfflineClient:
    """Pipeline client for stages that must never make a remote request."""

    provider = LLMProvider.GEMINI
    default_model = "offline-no-remote"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def generate_structured_async(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("The local accuracy benchmark attempted a remote LLM call")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


async def optimize(
    rows: list[dict[str, Any]],
    *,
    strict: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = [DatasetItem.from_dict(row) for row in rows]
    result = await OptimizationPipeline(
        PipelineConfig(
            quality_mode="off",
            classification="off",
            scrub_pii=False,
            dedup_method="exact",
            accuracy_contract="strict" if strict else "balanced",
        ),
        client=OfflineClient(),
    ).run(items)
    generated = [
        {"id": item.id, "text": item.get_classification_text(), "label": int(item.labels)}
        for item in result.accepted
    ]
    return generated, result.metrics


def evaluate(
    conditions: dict[str, list[dict[str, Any]]],
    test_rows: list[dict[str, Any]],
    vocab_rows: list[dict[str, Any]],
    seeds: list[int],
    epochs: int,
) -> dict[str, Any]:
    vocab = build_vocab(vocab_rows)
    results: dict[str, Any] = {}
    for name, rows in conditions.items():
        runs = []
        for seed in seeds:
            print(f"Training {name}, seed={seed}...")
            runs.append(train_once(rows, test_rows, vocab, 4, seed, epochs))
        results[name] = {"rows": len(rows), "runs": runs, "summary": summarize(runs)}
    return results


def delta(results: dict[str, Any], left: str, right: str) -> dict[str, float]:
    left_summary = results[left]["summary"]
    right_summary = results[right]["summary"]
    return {
        "accuracy": left_summary["accuracy_mean"] - right_summary["accuracy_mean"],
        "macro_f1": left_summary["macro_f1_mean"] - right_summary["macro_f1_mean"],
    }


def report(payload: dict[str, Any]) -> str:
    preservation = payload["preservation"]
    value = payload["value"]
    lines = [
        "# BuffData original-versus-generated accuracy gate",
        "",
        "Every condition uses the same AG News test rows, vocabulary, PyTorch architecture,",
        "hyperparameters, and three fixed random seeds. No LLM call is made by this benchmark.",
        "",
        "## Regression repair (3,000 train / 2,000 test)",
        "",
        "| Condition | Rows | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, condition in preservation["conditions"].items():
        summary = condition["summary"]
        lines.append(
            f"| {name} | {condition['rows']} | "
            f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} | "
            f"{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} |"
        )
    recovery = preservation["repaired_minus_regressed"]
    control = preservation["repaired_minus_original"]
    lines.extend([
        "",
        f"The repaired generated data recovers **{recovery['accuracy'] * 100:+.2f} accuracy points** "
        f"and finishes {control['accuracy'] * 100:+.2f} points from the matching original control.",
        f"It changes {preservation['changed_rows']['repaired_generated']:,} texts, compared with "
        f"{preservation['changed_rows']['regressed_generated']:,} in the regressed output.",
        "",
        "## Value test: contaminated original vs generated optimized (3,000 source / 2,000 test)",
        "",
        "The original condition contains all 3,000 source rows plus disclosed conflicting-label",
        "duplicates, class-skewing duplicates, and empty rows. BuffData generates the optimized",
        "condition from the exact same input.",
        "",
        "| Condition | Rows | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|",
    ])
    for name, condition in value["conditions"].items():
        summary = condition["summary"]
        lines.append(
            f"| {name} | {condition['rows']} | "
            f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} | "
            f"{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} |"
        )
    gain = value["generated_minus_original"]
    lines.extend([
        "",
        f"Generated optimized data improves accuracy by **{gain['accuracy'] * 100:+.2f} points** "
        f"and macro-F1 by **{gain['macro_f1'] * 100:+.2f} points**.",
        "",
        "## Gates",
        "",
        f"- Regression recovery: **{'PASS' if payload['gates']['regression_recovered'] else 'FAIL'}**",
        f"- Strict original preservation (generated accuracy >= original and identical text): "
        f"**{'PASS' if payload['gates']['original_preserved'] else 'FAIL'}**",
        f"- Contaminated-data value: **{'PASS' if payload['gates']['value_improved'] else 'FAIL'}**",
        "",
        "These results establish behavior for these fixed samples and this lightweight classifier;",
        "they do not promise the same gain on every dataset or model.",
        "",
    ])
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    import torch

    torch.set_num_threads(args.threads)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = {row["id"]: row for row in load_jsonl(Path(args.original_file))}
    regressed_rows = load_jsonl(Path(args.regressed_file))[: args.train_rows]
    missing = [row["id"] for row in regressed_rows if row["id"] not in raw_rows]
    if missing:
        raise ValueError(f"{len(missing)} generated rows do not have matching originals")
    original_rows = [
        {
            "id": row["id"],
            "text": str(raw_rows[row["id"]]["text"]),
            "label": int(raw_rows[row["id"]]["label"]),
        }
        for row in regressed_rows
    ]
    repaired_rows, repaired_metrics = await optimize(original_rows, strict=True)
    regressed_condition = [
        {"id": row["id"], "text": str(row["text"]), "label": int(row["label"])}
        for row in regressed_rows
    ]

    print("Fetching AG News...")
    source = load_dataset("fancyzhx/ag_news")
    test_rows = stratified_rows(source["test"], args.test_rows, 4, ["text"], 200, "balanced")
    preservation_conditions = evaluate(
        {
            "original_dataset": original_rows,
            "regressed_generated": regressed_condition,
            "repaired_generated": repaired_rows,
        },
        test_rows,
        original_rows,
        args.seeds,
        args.epochs,
    )

    clean_source = stratified_rows(source["train"], args.train_rows, 4, ["text"], 100, "balanced")
    original_contaminated, optimizer_input, defects = make_dirty(clean_source, 4, 300)
    generated_optimized, generated_metrics = await optimize(optimizer_input, strict=False)
    value_conditions = evaluate(
        {
            "original_contaminated": original_contaminated,
            "generated_optimized": generated_optimized,
        },
        test_rows,
        clean_source,
        args.seeds,
        args.epochs,
    )

    preservation_recovery = delta(
        preservation_conditions, "repaired_generated", "regressed_generated"
    )
    preservation_control = delta(
        preservation_conditions, "repaired_generated", "original_dataset"
    )
    value_gain = delta(value_conditions, "generated_optimized", "original_contaminated")
    repaired_text_identical = (
        len(repaired_rows) == len(original_rows)
        and all(
            original["id"] == repaired["id"]
            and original["text"] == repaired["text"]
            and original["label"] == repaired["label"]
            for original, repaired in zip(original_rows, repaired_rows)
        )
    )
    payload = {
        "method": {
            "hf_id": "fancyzhx/ag_news",
            "train_rows": args.train_rows,
            "test_rows": args.test_rows,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "model": "EmbeddingBag(mean, 64) + Linear",
            "device": "cpu",
            "shared_vocabulary_per_panel": True,
            "remote_llm_calls": 0,
        },
        "preservation": {
            "conditions": preservation_conditions,
            "repaired_minus_regressed": preservation_recovery,
            "repaired_minus_original": preservation_control,
            "changed_rows": {
                "regressed_generated": sum(
                    a["text"] != b["text"] for a, b in zip(original_rows, regressed_condition)
                ),
                "repaired_generated": sum(
                    a["text"] != b["text"] for a, b in zip(original_rows, repaired_rows)
                ),
            },
            "repaired_text_identical": repaired_text_identical,
            "pipeline_metrics": repaired_metrics,
        },
        "value": {
            "conditions": value_conditions,
            "generated_minus_original": value_gain,
            "defects": defects,
            "pipeline_metrics": generated_metrics,
        },
        "gates": {
            "regression_recovered": preservation_recovery["accuracy"] > 0,
            "original_preserved": (
                preservation_control["accuracy"] >= 0 and repaired_text_identical
            ),
            "value_improved": value_gain["accuracy"] > 0,
        },
    }

    write_jsonl(output_dir / "repaired_generated.jsonl", repaired_rows)
    write_jsonl(output_dir / "generated_optimized.jsonl", generated_optimized)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(report(payload), encoding="utf-8")
    print(f"Results: {output_dir / 'results.json'}")
    print(f"Report:  {output_dir / 'REPORT.md'}")
    if args.enforce_gates and not all(payload["gates"].values()):
        raise SystemExit("One or more accuracy gates failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-file", default="train_25k_raw.jsonl")
    parser.add_argument("--regressed-file", default="train_25k_clean.jsonl")
    parser.add_argument("--output-dir", default="benchmarks/results-regression")
    parser.add_argument("--train-rows", type=int, default=3000)
    parser.add_argument("--test-rows", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--enforce-gates", action=argparse.BooleanOptionalAction, default=True)
    asyncio.run(main(parser.parse_args()))
