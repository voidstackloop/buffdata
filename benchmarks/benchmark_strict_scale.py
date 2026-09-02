#!/usr/bin/env python3
"""Verify strict original-versus-generated parity at 20k and 100k scale."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from benchmark_buffdata import build_vocab, stratified_rows, summarize, train_once
from buffdata.engine.client import LLMProvider
from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.models.schemas import DatasetItem, PipelineConfig


class OfflineClient:
    provider = LLMProvider.GEMINI
    default_model = "offline-no-remote"
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def generate_structured_async(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Strict scale benchmark attempted a remote LLM call")


async def strict_generate(rows: list[dict[str, Any]], size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = [
        DatasetItem.from_dict({"id": f"strict_{size}_{index}", **row})
        for index, row in enumerate(rows)
    ]
    result = await OptimizationPipeline(
        PipelineConfig(
            accuracy_contract="strict",
            classification="multi-class",
            classification_pii_mode="all",
            dedup_method="minhash",
            quality_mode="llm",
        ),
        client=OfflineClient(),
    ).run(items)
    generated = [
        {"text": item.get_classification_text(), "label": int(item.labels)}
        for item in result.accepted
    ]
    return generated, result.metrics


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Strict accuracy contract at 20k and 100k scale",
        "",
        "Each generated condition passed through BuffData strict mode while its config",
        "requested all PII redaction, minhash deduplication, LLM filtering, and forced",
        "reclassification. Strict mode overrode every mutating stage. Original and generated",
        "conditions use the same AG News test set, vocabulary, model, epochs, and seeds.",
        "",
        "| Train rows | Condition | Accuracy | Macro-F1 | Text/label identical |",
        "|---:|---|---:|---:|---:|",
    ]
    for size, result in payload["sizes"].items():
        for condition in ("original", "strict_generated"):
            summary = result["conditions"][condition]["summary"]
            lines.append(
                f"| {int(size):,} | {condition} | "
                f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} | "
                f"{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} | "
                f"{'yes' if result['row_identity'] else 'no'} |"
            )
        lines.append("")
    lines.extend([
        "## Gates",
        "",
        f"- 20k strict parity: **{'PASS' if payload['gates']['20000'] else 'FAIL'}**",
        f"- 100k strict parity: **{'PASS' if payload['gates']['100000'] else 'FAIL'}**",
        "",
        "A passing gate requires the complete valid labeled row sequence, every classification",
        "text, every label, accuracy, and macro-F1 to match the original exactly.",
        "",
    ])
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    import torch

    torch.set_num_threads(args.threads)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_dataset("fancyzhx/ag_news")
    test_rows = stratified_rows(source["test"], args.test_rows, 4, ["text"], 902, "balanced")
    payload: dict[str, Any] = {
        "method": {
            "hf_id": "fancyzhx/ag_news",
            "sizes": args.sizes,
            "test_rows": len(test_rows),
            "epochs": args.epochs,
            "seeds": args.seeds,
            "model": "EmbeddingBag(mean, 64) + Linear",
            "device": "cpu",
            "remote_llm_calls": 0,
        },
        "sizes": {},
        "gates": {},
    }

    for offset, size in enumerate(args.sizes):
        print(f"Preparing strict comparison for {size:,} rows...")
        original = stratified_rows(source["train"], size, 4, ["text"], 700 + offset, "balanced")
        generated, metrics = await strict_generate(original, size)
        row_identity = original == generated
        vocab = build_vocab(original)
        conditions: dict[str, Any] = {}
        for name, rows in (("original", original), ("strict_generated", generated)):
            runs = []
            for seed in args.seeds:
                print(f"Training {size}/{name}, seed={seed}...")
                runs.append(train_once(rows, test_rows, vocab, 4, seed, args.epochs))
            conditions[name] = {"rows": len(rows), "runs": runs, "summary": summarize(runs)}
        original_summary = conditions["original"]["summary"]
        generated_summary = conditions["strict_generated"]["summary"]
        metric_identity = all(
            generated_summary[key] == original_summary[key]
            for key in ("accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std")
        )
        gate = row_identity and metric_identity
        payload["sizes"][str(size)] = {
            "row_identity": row_identity,
            "metric_identity": metric_identity,
            "conditions": conditions,
            "pipeline_metrics": metrics,
        }
        payload["gates"][str(size)] = gate

    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(markdown_report(payload), encoding="utf-8")
    print(f"Results: {output_dir / 'results.json'}")
    print(f"Report:  {output_dir / 'REPORT.md'}")
    if not all(payload["gates"].values()):
        raise SystemExit("A strict scale parity gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[20_000, 100_000])
    parser.add_argument("--test-rows", type=int, default=7_000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", default="benchmarks/results-strict-scale")
    asyncio.run(main(parser.parse_args()))
