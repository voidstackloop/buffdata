"""Deterministic PyTorch accuracy gate for labeled classification datasets."""

from __future__ import annotations

from collections import Counter
import random
import re
import statistics
import time
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from buffdata.models.schemas import DatasetItem


TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _validated_rows(items: Iterable[DatasetItem]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for item in items:
        if item.labels is None or isinstance(item.labels, list):
            raise ValueError("Accuracy gain evaluation requires one scalar label per record")
        text = item.get_classification_text().strip()
        if not text:
            continue
        rows.append((text, str(item.labels)))
    if not rows:
        raise ValueError("Accuracy gain evaluation found no labeled text records")
    return rows


def _stratified_cap(rows: list[tuple[str, str]], limit: int, seed: int) -> list[tuple[str, str]]:
    if len(rows) <= limit:
        return rows
    buckets: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        buckets.setdefault(row[1], []).append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    labels = sorted(buckets)
    per_label, remainder = divmod(limit, len(labels))
    selected: list[tuple[str, str]] = []
    for index, label in enumerate(labels):
        selected.extend(buckets[label][:per_label + (1 if index < remainder else 0)])
    rng.shuffle(selected)
    return selected


class _EncodedDataset(Dataset):
    def __init__(self, rows, vocabulary, labels):
        self.samples = []
        for text, label in rows:
            token_ids = [vocabulary.get(token, 1) for token in _tokenize(text)] or [1]
            self.samples.append((token_ids, labels[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _collate(batch):
    tokens: list[int] = []
    offsets: list[int] = []
    labels: list[int] = []
    for token_ids, label in batch:
        offsets.append(len(tokens))
        tokens.extend(token_ids)
        labels.append(label)
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


class _Classifier(nn.Module):
    def __init__(self, vocabulary_size: int, class_count: int, dimension: int = 64):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocabulary_size, dimension, mode="mean")
        self.output = nn.Linear(dimension, class_count)

    def forward(self, tokens, offsets):
        return self.output(self.embedding(tokens, offsets))


def _train_once(train_rows, validation_rows, vocabulary, labels, seed, epochs):
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _EncodedDataset(train_rows, vocabulary, labels),
        batch_size=64,
        shuffle=True,
        collate_fn=_collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        _EncodedDataset(validation_rows, vocabulary, labels),
        batch_size=256,
        shuffle=False,
        collate_fn=_collate,
    )
    model = _Classifier(len(vocabulary), len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for tokens, offsets, actual in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, offsets), actual)
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - started
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for tokens, offsets, actual in validation_loader:
            predictions = model(tokens, offsets).argmax(dim=1)
            correct += int((predictions == actual).sum().item())
            total += len(actual)
    return {"accuracy": correct / total, "train_seconds": train_seconds}


def evaluate_accuracy_gain(
    original: Sequence[DatasetItem],
    candidate: Sequence[DatasetItem],
    validation: Sequence[DatasetItem],
    *,
    seeds: Sequence[int] = (17, 29, 43),
    epochs: int = 6,
    minimum_gain: float = 0.0,
    minimum_relative_gain: float | None = None,
    max_train_rows: int = 100_000,
    max_vocabulary: int = 20_000,
) -> dict[str, Any]:
    """Accept only when candidate accuracy improves by the threshold on every seed.

    ``minimum_gain`` is an absolute accuracy-point floor (e.g. 0.02 == +2 points).
    ``minimum_relative_gain`` instead requires candidate accuracy to be at least
    that fraction higher than original accuracy (e.g. 0.10 == +10% relative), on
    the mean and on every seed. Only one of the two may be set.
    """
    if not seeds:
        raise ValueError("At least one accuracy-gate seed is required")
    if minimum_relative_gain is not None:
        if minimum_gain:
            raise ValueError("Specify either minimum_gain or minimum_relative_gain, not both")
        if minimum_relative_gain < 0:
            raise ValueError("minimum_relative_gain must be zero or positive")
    original_rows = _stratified_cap(_validated_rows(original), max_train_rows, 5101)
    candidate_rows = _stratified_cap(_validated_rows(candidate), max_train_rows, 5101)
    validation_rows = _validated_rows(validation)
    label_names = sorted({label for _, label in original_rows + candidate_rows + validation_rows})
    labels = {label: index for index, label in enumerate(label_names)}
    train_labels = {label for _, label in original_rows} & {label for _, label in candidate_rows}
    validation_labels = {label for _, label in validation_rows}
    if not validation_labels.issubset(train_labels):
        missing = sorted(validation_labels - train_labels)
        raise ValueError(f"Validation labels are missing from a training condition: {missing}")
    counts = Counter(
        token
        for text, _ in original_rows + candidate_rows
        for token in _tokenize(text)
    )
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common(max_vocabulary - 2):
        vocabulary[token] = len(vocabulary)

    original_runs = []
    candidate_runs = []
    deltas = []
    for seed in seeds:
        original_result = _train_once(
            original_rows, validation_rows, vocabulary, labels, int(seed), epochs
        )
        candidate_result = _train_once(
            candidate_rows, validation_rows, vocabulary, labels, int(seed), epochs
        )
        original_runs.append(original_result)
        candidate_runs.append(candidate_result)
        deltas.append(candidate_result["accuracy"] - original_result["accuracy"])
    original_mean = statistics.mean(run["accuracy"] for run in original_runs)
    candidate_mean = statistics.mean(run["accuracy"] for run in candidate_runs)
    mean_gain = candidate_mean - original_mean
    if minimum_relative_gain is not None:
        def meets_relative_gain(candidate_accuracy: float, original_accuracy: float) -> bool:
            # A zero baseline makes the multiplicative threshold degenerate (0 * anything
            # == 0, which candidate_accuracy would always clear). Treat it explicitly:
            # any positive candidate accuracy over a zero baseline is a real improvement.
            if original_accuracy <= 0:
                return candidate_accuracy > 0
            return candidate_accuracy >= original_accuracy * (1 + minimum_relative_gain)

        per_seed_ok = all(
            meets_relative_gain(candidate_runs[index]["accuracy"], original_runs[index]["accuracy"])
            for index in range(len(seeds))
        )
        accepted = meets_relative_gain(candidate_mean, original_mean) and per_seed_ok
        criterion = "candidate accuracy must be at least minimum_relative_gain higher, relative to original, on the mean and every seed"
    else:
        accepted = mean_gain > minimum_gain and all(delta > minimum_gain for delta in deltas)
        criterion = "candidate gain must exceed minimum_gain on the mean and every seed"
    return {
        "accepted": accepted,
        "criterion": criterion,
        "minimum_gain": minimum_gain,
        "minimum_relative_gain": minimum_relative_gain,
        "seeds": list(seeds),
        "epochs": epochs,
        "original_rows_evaluated": len(original_rows),
        "candidate_rows_evaluated": len(candidate_rows),
        "validation_rows": len(validation_rows),
        "original_runs": original_runs,
        "candidate_runs": candidate_runs,
        "per_seed_accuracy_gain": deltas,
        "original_accuracy_mean": original_mean,
        "candidate_accuracy_mean": candidate_mean,
        "accuracy_gain": mean_gain,
    }


def diagnose_candidate_errors(
    candidate: Sequence[DatasetItem],
    validation: Sequence[DatasetItem],
    *,
    seed: int = 17,
    epochs: int = 6,
    max_examples_per_label: int = 5,
    max_vocabulary: int = 20_000,
) -> dict[str, Any]:
    """Train one classifier on ``candidate`` and report which validation labels it
    struggles with, plus a capped sample of the texts it got wrong for each. Used
    to steer the next round of generation toward the classifier's actual weak
    spots instead of generating undirected variations.
    """
    candidate_rows = _validated_rows(candidate)
    validation_rows = _validated_rows(validation)
    label_names = sorted({label for _, label in candidate_rows + validation_rows})
    if len(label_names) < 2:
        raise ValueError("Error diagnosis requires at least two distinct labels")
    labels = {label: index for index, label in enumerate(label_names)}
    index_to_label = {index: label for label, index in labels.items()}

    counts = Counter(token for text, _ in candidate_rows for token in _tokenize(text))
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common(max_vocabulary - 2):
        vocabulary[token] = len(vocabulary)

    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _EncodedDataset(candidate_rows, vocabulary, labels),
        batch_size=64,
        shuffle=True,
        collate_fn=_collate,
        generator=generator,
    )
    model = _Classifier(len(vocabulary), len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for tokens, offsets, actual in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, offsets), actual)
            loss.backward()
            optimizer.step()

    validation_loader = DataLoader(
        _EncodedDataset(validation_rows, vocabulary, labels),
        batch_size=256,
        shuffle=False,
        collate_fn=_collate,
    )
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for tokens, offsets, _ in validation_loader:
            predictions.extend(model(tokens, offsets).argmax(dim=1).tolist())

    correct_by_label: Counter[str] = Counter()
    total_by_label: Counter[str] = Counter()
    misclassified: dict[str, list[str]] = {label: [] for label in label_names}
    for (text, true_label), prediction_index in zip(validation_rows, predictions):
        predicted_label = index_to_label[prediction_index]
        total_by_label[true_label] += 1
        if predicted_label == true_label:
            correct_by_label[true_label] += 1
        elif len(misclassified[true_label]) < max_examples_per_label:
            misclassified[true_label].append(text)

    per_label_accuracy = {
        label: (correct_by_label[label] / total_by_label[label]) if total_by_label[label] else 1.0
        for label in label_names
    }
    return {
        "per_label_accuracy": per_label_accuracy,
        "misclassified_examples": misclassified,
    }
