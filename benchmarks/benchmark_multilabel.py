#!/usr/bin/env python3
"""Multi-label counterpart to benchmark_buffdata.py's core helpers.

Multi-label rows carry a *list* of labels rather than one scalar class, which changes
three things throughout: the classifier needs a sigmoid output (one independent
yes/no per label) instead of a single softmax over mutually-exclusive classes, the
defect-injection logic needs a label-set-aware notion of "conflicting" and "skewed",
and stratified sampling has no single scalar to bucket by -- this uses plain random
sampling instead of balanced-per-class, a deliberate simplification (true multi-label
stratification is a harder, separate problem) documented here rather than silently
assumed.
"""

from __future__ import annotations

import random
import re
import statistics
import time
from collections import Counter
from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from buffdata.engine.pipeline import OptimizationPipeline
from buffdata.models.schemas import DatasetItem, PipelineConfig
from benchmark_buffdata import summarize_data_hygiene

TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def build_vocab(rows: list[dict[str, Any]], max_size: int = 20_000) -> dict[str, int]:
    counts = Counter(token for row in rows for token in tokenize(row["text"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common(max_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def stratified_rows_multilabel(
    split: Iterable[dict[str, Any]],
    count: int,
    text_fields: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Random sample of `count` rows with a normalized {"text", "label": [...]} shape.
    No per-label balancing (see module docstring) -- every row is sampled uniformly.
    """
    all_rows = []
    for row in split:
        labels = row["label"]
        if not isinstance(labels, list):
            labels = [labels]
        text = "\n\n".join(str(row.get(field, "")).strip() for field in text_fields).strip()
        all_rows.append({"text": text, "label": [int(label) for label in labels]})
    if len(all_rows) < count:
        raise ValueError(f"Split only has {len(all_rows)} rows; requested {count}.")
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    return all_rows[:count]


DEFECT_CLEAN = "clean"
DEFECT_CONFLICT = "conflicting_duplicate"
DEFECT_SKEW = "class_skew_duplicate"
DEFECT_EMPTY = "empty"


def _tag_defect(row: dict[str, Any], category: str) -> dict[str, Any]:
    """Stamp a row with which defect category it was injected as -- see
    benchmark_buffdata._tag_defect for why this survives the whole pipeline run."""
    row["_buffdata_metadata"] = {"defect_category": category}
    return row


def make_dirty_multilabel(
    clean: list[dict[str, Any]], num_labels: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Multi-label counterpart to benchmark_buffdata.make_dirty. Same three defect
    categories at the same 40/40/10 proportions, adapted to label *sets*:
    - "conflicting duplicate": same text, but with one label toggled (added if absent,
      removed if present) -- the multi-label analogue of flipping a scalar label.
    - "class-skew duplicate": an exact duplicate of an existing row (text + labels both),
      chosen uniformly at random -- unlike the scalar version's bias toward class 0,
      multi-label has no single natural "majority class" to skew toward.
    - "empty rows": empty text, empty label set.
    """
    rng = random.Random(seed)
    extras: list[dict[str, Any]] = []
    conflict_count = max(1, round(len(clean) * 0.40))
    skew_count = max(1, round(len(clean) * 0.40))
    empty_count = max(1, round(len(clean) * 0.10))

    conflict_indices = rng.sample(range(len(clean)), conflict_count)
    for index in conflict_indices:
        row = clean[index]
        labels = set(row["label"])
        toggle = rng.randrange(num_labels)
        if toggle in labels:
            labels.discard(toggle)
        else:
            labels.add(toggle)
        extras.append(_tag_defect({"text": row["text"], "label": sorted(labels)}, DEFECT_CONFLICT))

    for _ in range(skew_count):
        extras.append(_tag_defect(dict(rng.choice(clean)), DEFECT_SKEW))

    for _ in range(empty_count):
        extras.append(_tag_defect({"text": "", "label": []}, DEFECT_EMPTY))

    clean_tagged = [_tag_defect(dict(row), DEFECT_CLEAN) for row in clean]
    optimizer_input = clean_tagged + [dict(row) for row in extras]
    dirty = [dict(row) for row in optimizer_input]
    rng.shuffle(dirty)
    return dirty, optimizer_input, {
        "conflicting_duplicate_rows": conflict_count,
        "class_skew_duplicate_rows": skew_count,
        "empty_rows": empty_count,
    }


def summarize_defect_breakdown_multilabel(
    accepted: list[DatasetItem], rejected: list[DatasetItem]
) -> dict[str, dict[str, Any]]:
    """Cross-tabulate each injected defect category against what a real
    optimize_multilabel() run actually did with it -- input/retained/lost counts, plus
    which exact pipeline stage caught each lost row."""
    breakdown: dict[str, dict[str, Any]] = {}

    def bucket(category: str) -> dict[str, Any]:
        return breakdown.setdefault(
            category, {"input": 0, "retained": 0, "lost": 0, "lost_by_stage": {}}
        )

    for item in accepted:
        entry = bucket(item.metadata.get("defect_category", "untagged"))
        entry["input"] += 1
        entry["retained"] += 1

    for item in rejected:
        entry = bucket(item.metadata.get("defect_category", "untagged"))
        entry["input"] += 1
        entry["lost"] += 1
        stage = item.metadata.get("rejection", {}).get("stage", "unknown")
        entry["lost_by_stage"][stage] = entry["lost_by_stage"].get(stage, 0) + 1

    return breakdown


async def optimize_multilabel(
    rows: list[dict[str, Any]],
    *,
    gemini_audit_rows: int = 0,
    gemini_model: str = "gemini-3.5-flash-lite",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = [DatasetItem.from_dict(row) for row in rows]
    for item in items:
        item.metadata.setdefault("defect_category", DEFECT_CLEAN)
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
    output = [{"text": item.text or "", "label": [int(x) for x in item.labels]} for item in result.accepted]
    quality = {
        "input_rows": len(rows),
        "output_rows": len(output),
        "rejected_rows": len(result.rejected),
        "profile": result.profile.model_dump(mode="json"),
        "stages": result.metrics["stages"],
        "provider": result.metrics["provider"],
        "model": result.metrics["model"],
        "usage": result.metrics["usage"],
        "defect_breakdown": summarize_defect_breakdown_multilabel(result.accepted, result.rejected),
        "hygiene": summarize_data_hygiene(rows, result.accepted, result.rejected),
    }
    return output, quality


class MultiHotTextDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], vocab: dict[str, int], num_labels: int):
        self.num_labels = num_labels
        self.samples: list[tuple[list[int], list[int]]] = []
        for row in rows:
            tokens = tokenize(row.get("text", ""))
            token_ids = [vocab.get(token, 1) for token in tokens] or [1]
            self.samples.append((token_ids, row["label"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate_multilabel(batch, num_labels: int):
    flat: list[int] = []
    offsets: list[int] = []
    targets = torch.zeros(len(batch), num_labels, dtype=torch.float32)
    for i, (tokens, labels) in enumerate(batch):
        offsets.append(len(flat))
        flat.extend(tokens)
        for label in labels:
            targets[i, label] = 1.0
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        targets,
    )


class MultiLabelTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, dimension: int = 64):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, dimension, mode="mean")
        self.classifier = nn.Linear(dimension, num_labels)

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(tokens, offsets))  # raw logits


def multilabel_macro_f1(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    """Macro-F1 averaged over labels (not samples): matches the spirit of
    benchmark_buffdata.macro_f1, generalized to independent per-label binary decisions."""
    values = []
    num_labels = actual.shape[1]
    for label in range(num_labels):
        y = actual[:, label]
        p = predicted[:, label]
        tp = ((y == 1) & (p == 1)).sum().item()
        fp = ((y == 0) & (p == 1)).sum().item()
        fn = ((y == 1) & (p == 0)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        values.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(values) / len(values)


def subset_accuracy(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    """Fraction of rows whose *entire* predicted label set exactly matches the true
    set -- a strict metric (partial credit is what macro-F1 above captures instead)."""
    exact = (actual == predicted).all(dim=1).float().mean().item()
    return exact


def train_once_multilabel(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    vocab: dict[str, int],
    num_labels: int,
    seed: int,
    epochs: int,
) -> dict[str, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    collate = lambda batch: collate_multilabel(batch, num_labels)  # noqa: E731
    train_loader = DataLoader(
        MultiHotTextDataset(train_rows, vocab, num_labels), batch_size=64, shuffle=True,
        collate_fn=collate, generator=generator,
    )
    test_loader = DataLoader(
        MultiHotTextDataset(test_rows, vocab, num_labels), batch_size=128, shuffle=False, collate_fn=collate,
    )
    model = MultiLabelTextClassifier(len(vocab), num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for tokens, offsets, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, offsets), targets)
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - started

    actual_batches, predicted_batches = [], []
    model.eval()
    with torch.no_grad():
        for tokens, offsets, targets in test_loader:
            logits = model(tokens, offsets)
            predicted_batches.append((torch.sigmoid(logits) > 0.5).float())
            actual_batches.append(targets)
    actual = torch.cat(actual_batches)
    predicted = torch.cat(predicted_batches)
    return {
        "subset_accuracy": subset_accuracy(actual, predicted),
        "macro_f1": multilabel_macro_f1(actual, predicted),
        "train_seconds": train_seconds,
    }


def summarize_multilabel(runs: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("subset_accuracy", "macro_f1", "train_seconds"):
        values = [run[metric] for run in runs]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result
