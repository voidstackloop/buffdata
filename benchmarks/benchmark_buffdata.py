#!/usr/bin/env python3
"""Reproducible BuffData/PyTorch benchmark on real Hugging Face datasets.

The benchmark deliberately separates two questions:
1. Does BuffData preserve an already-clean dataset?
2. Does BuffData recover performance when realistic pipeline defects are present?

The dirty variants are deterministic derivatives of real records. They add empty
rows, conflicting duplicate labels, and class-skewing exact duplicates. No claim
is made that Hugging Face published those defects in the source datasets.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.models.schemas import DatasetItem, PipelineConfig


DATASETS = {
    "ag_news": {
        "hf_id": "fancyzhx/ag_news",
        "classes": 4,
        "description": "AG News topic classification",
    },
    "imdb": {
        "hf_id": "stanfordnlp/imdb",
        "classes": 2,
        "description": "IMDb sentiment classification",
        "text_fields": ["text"],
    },
    "dbpedia_14": {
        "hf_id": "fancyzhx/dbpedia_14",
        "classes": 14,
        "description": "DBpedia ontology classification",
        "text_fields": ["title", "content"],
    },
    "yelp_polarity": {
        "hf_id": "fancyzhx/yelp_polarity",
        "classes": 2,
        "description": "Yelp review sentiment classification",
        "text_fields": ["text"],
    },
    "emotion": {
        "hf_id": "dair-ai/emotion",
        "config": "split",
        "classes": 6,
        "description": "English emotion classification",
        "text_fields": ["text"],
        "sampling": "random",
    },
}
DATASETS["ag_news"]["text_fields"] = ["text"]
TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


@dataclass
class Condition:
    name: str
    rows: list[dict[str, Any]]
    quality: dict[str, Any]


class EncodedTextDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], vocab: dict[str, int]):
        self.samples: list[tuple[list[int], int]] = []
        for row in rows:
            tokens = tokenize(row.get("text", ""))
            token_ids = [vocab.get(token, 1) for token in tokens] or [1]
            self.samples.append((token_ids, int(row["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.samples[index]


class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, classes: int, dimension: int = 64):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, dimension, mode="mean")
        self.classifier = nn.Linear(dimension, classes)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(tokens, offsets))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def collate(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat: list[int] = []
    offsets: list[int] = []
    labels: list[int] = []
    for tokens, label in batch:
        offsets.append(len(flat))
        flat.extend(tokens)
        labels.append(label)
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def stratified_rows(
    split: Iterable[dict[str, Any]],
    count: int,
    classes: int,
    text_fields: list[str],
    seed: int,
    sampling: str = "balanced",
) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {label: [] for label in range(classes)}
    for row in split:
        label = int(row["label"])
        text = "\n\n".join(str(row.get(field, "")).strip() for field in text_fields).strip()
        buckets[label].append({"text": text, "label": label})
    rng = random.Random(seed)
    if sampling == "random":
        combined = [row for bucket in buckets.values() for row in bucket]
        if len(combined) < count:
            raise ValueError(f"Split only has {len(combined)} rows; requested {count}.")
        rng.shuffle(combined)
        return combined[:count]
    per_class, remainder = divmod(count, classes)
    result: list[dict[str, Any]] = []
    for label in range(classes):
        bucket = buckets[label]
        rng.shuffle(bucket)
        take = per_class + (1 if label < remainder else 0)
        if len(bucket) < take:
            raise ValueError(f"Class {label} only has {len(bucket)} rows; requested {take}.")
        result.extend(bucket[:take])
    rng.shuffle(result)
    return result


def make_dirty(
    clean: list[dict[str, Any]], classes: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Append defects without removing any clean source record."""
    rng = random.Random(seed)
    extras: list[dict[str, Any]] = []
    conflict_count = max(1, round(len(clean) * 0.40))
    skew_count = max(1, round(len(clean) * 0.40))
    empty_count = max(1, round(len(clean) * 0.10))

    conflict_indices = rng.sample(range(len(clean)), conflict_count)
    for index in conflict_indices:
        row = clean[index]
        extras.append({"text": row["text"], "label": (int(row["label"]) + 1) % classes})

    preferred = [row for row in clean if int(row["label"]) == 0]
    for _ in range(skew_count):
        extras.append(dict(rng.choice(preferred)))

    for index in range(empty_count):
        extras.append({"text": "", "label": index % classes})

    optimizer_input = [dict(row) for row in clean] + [dict(row) for row in extras]
    dirty = [dict(row) for row in optimizer_input]
    rng.shuffle(dirty)
    # BuffData keeps the first duplicate. Put originals first so conflicting copies
    # cannot replace authoritative source rows during deterministic deduplication.
    return dirty, optimizer_input, {
        "conflicting_duplicate_rows": conflict_count,
        "class_skew_duplicate_rows": skew_count,
        "empty_rows": empty_count,
    }


async def optimize(
    rows: list[dict[str, Any]],
    *,
    gemini_audit_rows: int = 0,
    gemini_model: str = "gemini-3.5-flash-lite",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = [DatasetItem.from_dict(row) for row in rows]
    config = PipelineConfig(
        provider="gemini",
        model=gemini_model,
        dedup_method="exact",
        scrub_pii=False,
        classification="auto",
        quality_mode="sampled" if gemini_audit_rows else "off",
        quality_sample_size=max(1, gemini_audit_rows),
        quality_audit_batch_size=20,
        concurrency=2,
        max_rpm=5,
    )
    result = await OptimizationPipeline(config).run(items)
    output = [{"text": item.text or "", "label": int(item.labels)} for item in result.accepted]
    quality = {
        "input_rows": len(rows),
        "output_rows": len(output),
        "rejected_rows": len(result.rejected),
        "profile": result.profile.model_dump(mode="json"),
        "stages": result.metrics["stages"],
        "provider": result.metrics["provider"],
        "model": result.metrics["model"],
        "usage": result.metrics["usage"],
    }
    return output, quality


def build_vocab(rows: list[dict[str, Any]], max_size: int = 20_000) -> dict[str, int]:
    counts = Counter(token for row in rows for token in tokenize(row["text"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common(max_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def macro_f1(labels: list[int], predictions: list[int], classes: int) -> float:
    values: list[float] = []
    for label in range(classes):
        tp = sum(y == label and p == label for y, p in zip(labels, predictions))
        fp = sum(y != label and p == label for y, p in zip(labels, predictions))
        fn = sum(y == label and p != label for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values)


def train_once(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    vocab: dict[str, int],
    classes: int,
    seed: int,
    epochs: int,
) -> dict[str, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        EncodedTextDataset(train_rows, vocab), batch_size=64, shuffle=True,
        collate_fn=collate, generator=generator,
    )
    test_loader = DataLoader(
        EncodedTextDataset(test_rows, vocab), batch_size=128, shuffle=False, collate_fn=collate,
    )
    model = FastTextClassifier(len(vocab), classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for tokens, offsets, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, offsets), labels)
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - started

    actual: list[int] = []
    predicted: list[int] = []
    model.eval()
    with torch.no_grad():
        for tokens, offsets, labels in test_loader:
            predicted.extend(model(tokens, offsets).argmax(dim=1).tolist())
            actual.extend(labels.tolist())
    accuracy = sum(y == p for y, p in zip(actual, predicted)) / len(actual)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1(actual, predicted, classes),
        "train_seconds": train_seconds,
    }


def summarize(runs: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("accuracy", "macro_f1", "train_seconds"):
        values = [run[metric] for run in runs]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# BuffData real-dataset PyTorch benchmark",
        "",
        "This benchmark uses fixed, deterministic samples of real Hugging Face records. The dirty",
        "variants are controlled derivatives, not defects attributed to the dataset publishers.",
        "Every condition uses the same vocabulary, architecture, hyperparameters, test set, and seeds.",
        "",
        "| Dataset | Condition | Rows | Accuracy | Macro-F1 | Train seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name, dataset in payload["datasets"].items():
        for condition_name, result in dataset["conditions"].items():
            summary = result["summary"]
            lines.append(
                f"| {dataset_name} | {condition_name} | {result['rows']} | "
                f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} | "
                f"{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} | "
                f"{summary['train_seconds_mean']:.2f} |"
            )
        delta = dataset["dirty_optimized_minus_dirty_raw"]
        lines.extend([
            "",
            f"**{dataset_name} dirty-data recovery:** accuracy {delta['accuracy']:+.4f}; "
            f"macro-F1 {delta['macro_f1']:+.4f}.",
            "",
        ])
        quality = dataset["conditions"]["dirty_optimized"]["quality"]
        score_stage = quality["stages"]["score_refine"]
        average = score_stage.get("average_overall_score")
        average_text = f"{average:.2f}/10" if average is not None else "n/a"
        lines.extend([
            f"BuffData processed {quality['input_rows']} rows and retained {quality['output_rows']}; "
            f"Gemini audit mode was `{score_stage['mode']}` with {score_stage['scored']} sampled scores. "
            f"Mean sampled quality was {average_text}; recorded Gemini usage was "
            f"{quality['usage']['total_tokens']} total tokens across "
            f"{score_stage.get('remote_batches', 0)} batched request(s).",
            "",
        ])
    lines.extend([
        "## Interpretation boundaries",
        "",
        "- The comparison proves behavior for these fixed samples, defects, model, and seeds—not every dataset or model.",
        "- Schema parsing, validation, automatic task detection, and exact deduplication determine row inclusion in this run.",
        "- Gemini performs a sampled semantic audit only; its scores are informational and do not cause the measured training recovery.",
        "- Full per-row Gemini refinement is intentionally excluded because these are already-labeled classification records and sampled audit mode avoids unnecessary cost.",
        "- The dirty benchmark is a controlled stress test; the clean control reveals whether processing clean data changes results.",
        "",
    ])
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    torch.set_num_threads(args.threads)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "method": {
            "train_rows": args.train_rows,
            "test_rows": args.test_rows,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "model": "EmbeddingBag(mean, 64) + Linear",
            "device": "cpu",
            "dirty_recipe": {"conflicting_duplicates": 0.40, "class_skew_duplicates": 0.40, "empty_rows": 0.10},
            "gemini_audit_rows": args.gemini_audit_rows,
            "gemini_model": args.gemini_model,
        },
        "datasets": {},
    }

    selected = args.datasets or list(DATASETS)
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    for dataset_offset, name in enumerate(selected):
        spec = DATASETS[name]
        print(f"Fetching {spec['hf_id']}...")
        source = load_dataset(spec["hf_id"], spec.get("config"))
        test_count = min(args.test_rows, len(source["test"]))
        clean_train = stratified_rows(
            source["train"], args.train_rows, spec["classes"], spec["text_fields"],
            100 + dataset_offset, spec.get("sampling", "balanced")
        )
        clean_test = stratified_rows(
            source["test"], test_count, spec["classes"], spec["text_fields"],
            200 + dataset_offset, spec.get("sampling", "balanced")
        )
        dirty_raw, optimizer_input, defects = make_dirty(
            clean_train, spec["classes"], 300 + dataset_offset
        )
        clean_optimized, clean_quality = await optimize(clean_train)
        dirty_optimized, dirty_quality = await optimize(
            optimizer_input,
            gemini_audit_rows=args.gemini_audit_rows,
            gemini_model=args.gemini_model,
        )

        dataset_dir = output_dir / name
        write_jsonl(dataset_dir / "clean_train.jsonl", clean_train)
        write_jsonl(dataset_dir / "clean_test.jsonl", clean_test)
        write_jsonl(dataset_dir / "dirty_train.jsonl", dirty_raw)
        write_jsonl(dataset_dir / "dirty_optimized.jsonl", dirty_optimized)

        conditions = [
            Condition("clean_raw", clean_train, {"input_rows": len(clean_train)}),
            Condition("clean_optimized", clean_optimized, clean_quality),
            Condition("dirty_raw", dirty_raw, {"input_rows": len(dirty_raw), **defects}),
            Condition("dirty_optimized", dirty_optimized, dirty_quality),
        ]
        vocab = build_vocab(clean_train)
        results: dict[str, Any] = {}
        for condition in conditions:
            runs = []
            for seed in args.seeds:
                print(f"Training {name}/{condition.name}, seed={seed}...")
                runs.append(train_once(condition.rows, clean_test, vocab, spec["classes"], seed, args.epochs))
            results[condition.name] = {
                "rows": len(condition.rows),
                "quality": condition.quality,
                "runs": runs,
                "summary": summarize(runs),
            }
        raw = results["dirty_raw"]["summary"]
        optimized = results["dirty_optimized"]["summary"]
        payload["datasets"][name] = {
            "hf_id": spec["hf_id"],
            "description": spec["description"],
            "conditions": results,
            "dirty_optimized_minus_dirty_raw": {
                "accuracy": optimized["accuracy_mean"] - raw["accuracy_mean"],
                "macro_f1": optimized["macro_f1_mean"] - raw["macro_f1_mean"],
            },
        }

    results_path = output_dir / "results.json"
    report_path = output_dir / "REPORT.md"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(markdown_report(payload), encoding="utf-8")
    print(f"Results: {results_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--train-rows", type=int, default=3000)
    parser.add_argument("--test-rows", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--gemini-audit-rows", type=int, default=0)
    parser.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    asyncio.run(main(parser.parse_args()))
