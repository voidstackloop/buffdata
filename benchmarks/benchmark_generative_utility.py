#!/usr/bin/env python3
"""Downstream clean/raw/dirty/optimized utility comparison for generative datasets.

This complements benchmark_generative_matrix.py's cleaning accuracy with held-out
task metrics. Lightweight deterministic local baselines keep the 10k/30k matrix
reproducible on CPU: nearest-neighbor response retrieval, a linear preference
ranker, and a word-bigram language model. These are benchmark models, not claims
about frontier-model quality.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any
import warnings

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors

from buffdata.models.schemas import DatasetItem

from benchmark_generative_matrix import (
    DATASETS,
    _signature,
    load_source_rows,
    make_dirty,
    normalize_row,
    optimize,
)


TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
EVAL_SPLITS = {
    "ultrachat": "test_sft",
    "ultrafeedback_dpo": "test_prefs",
    "squad_qa": "validation",
    "xsum": "validation",
    "wikitext_103": "validation",
}
PRIMARY_METRICS = {
    "alpaca_sft": ("response_token_f1", True),
    "ultrachat": ("response_token_f1", True),
    "ultrafeedback_dpo": ("preference_accuracy", True),
    "squad_qa": ("answer_token_f1", True),
    "xsum": ("rouge_l_f1", True),
    "wikitext_103": ("next_token_accuracy", True),
}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def exact_match(prediction: str, reference: str) -> float:
    return float(" ".join(tokens(prediction)) == " ".join(tokens(reference)))


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = Counter(tokens(prediction)), Counter(tokens(reference))
    if not predicted and not expected:
        return 1.0
    overlap = sum((predicted & expected).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    left, right = tokens(prediction), tokens(reference)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(left), lcs / len(right)
    return 2 * precision * recall / (precision + recall) if lcs else 0.0


def load_eval_rows(dataset_name: str, count: int, seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    spec = DATASETS[dataset_name]
    split = EVAL_SPLITS[dataset_name]
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    stream = (
        load_dataset(spec["hf_id"], spec["config"], **kwargs)
        if spec.get("config")
        else load_dataset(spec["hf_id"], **kwargs)
    ).shuffle(seed=seed, buffer_size=min(10_000, count * 10))
    rows: list[dict[str, Any]] = []
    for source in stream:
        normalized = normalize_row(dataset_name, source)
        if normalized is None:
            continue
        normalized["id"] = f"{dataset_name}-eval-{len(rows):06d}"
        rows.append(normalized)
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise ValueError(f"{dataset_name} supplied only {len(rows)} valid evaluation rows")
    return rows


def prompt_response(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    prompts, responses = [], []
    for row in rows:
        prompt, response = DatasetItem.from_dict(row).get_prompt_and_response()
        if prompt.strip() and response.strip():
            prompts.append(prompt)
            responses.append(response)
    return prompts, responses


def retrieval_metrics(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    family: str,
) -> dict[str, Any]:
    train_prompts, train_responses = prompt_response(train_rows)
    eval_prompts, eval_responses = prompt_response(eval_rows)
    if not train_prompts or not eval_prompts:
        raise ValueError("retrieval benchmark requires non-empty prompt/response rows")
    vectorizer = HashingVectorizer(
        n_features=2**16,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm="l2",
        lowercase=True,
    )
    train_vectors = vectorizer.transform(train_prompts)
    eval_vectors = vectorizer.transform(eval_prompts)
    model = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    model.fit(train_vectors)
    indices = model.kneighbors(eval_vectors, return_distance=False).ravel()
    predictions = [train_responses[int(index)] for index in indices]
    f1_values = [token_f1(prediction, reference) for prediction, reference in zip(predictions, eval_responses)]
    exact_values = [exact_match(prediction, reference) for prediction, reference in zip(predictions, eval_responses)]
    result: dict[str, Any] = {
        "model": "hashed n-gram nearest-response retrieval",
        "usable_train_rows": len(train_prompts),
        "eval_rows": len(eval_prompts),
        "response_token_f1": statistics.fmean(f1_values),
        "response_exact_match": statistics.fmean(exact_values),
    }
    if family == "extractive QA":
        result["answer_token_f1"] = result.pop("response_token_f1")
        result["answer_exact_match"] = result.pop("response_exact_match")
    elif family == "summarization":
        rouge_values = [
            rouge_l_f1(prediction, reference)
            for prediction, reference in zip(predictions, eval_responses)
        ]
        result["rouge_l_f1"] = statistics.fmean(rouge_values)
    return result


def preference_metrics(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    train_texts: list[str] = []
    labels: list[int] = []
    for row in train_rows:
        item = DatasetItem.from_dict(row)
        train_texts.extend([f"{item.prompt or ''}\n{item.chosen or ''}", f"{item.prompt or ''}\n{item.rejected or ''}"])
        labels.extend([1, 0])
    eval_items = [DatasetItem.from_dict(row) for row in eval_rows]
    vectorizer = HashingVectorizer(
        n_features=2**17,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm="l2",
    )
    train_vectors = vectorizer.transform(train_texts)
    model = SGDClassifier(
        loss="log_loss", alpha=1e-5, max_iter=20, tol=1e-4, random_state=seed, average=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(train_vectors, np.asarray(labels))
    chosen = vectorizer.transform([f"{item.prompt or ''}\n{item.chosen or ''}" for item in eval_items])
    rejected = vectorizer.transform([f"{item.prompt or ''}\n{item.rejected or ''}" for item in eval_items])
    accuracy = float(np.mean(model.decision_function(chosen) > model.decision_function(rejected)))
    return {
        "model": "hashed n-gram linear preference ranker",
        "usable_train_pairs": len(train_rows),
        "eval_pairs": len(eval_items),
        "preference_accuracy": accuracy,
    }


def language_model_metrics(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    context_counts: Counter[str] = Counter()
    bigram_counts: Counter[tuple[str, str]] = Counter()
    vocabulary: set[str] = set()
    for row in train_rows:
        sequence = tokens(str(row.get("text", "")))
        vocabulary.update(sequence)
        context_counts.update(sequence[:-1])
        bigram_counts.update(zip(sequence, sequence[1:]))
    best_next: dict[str, str] = {}
    best_count: dict[str, int] = {}
    for (context, following), count in bigram_counts.items():
        if count > best_count.get(context, -1):
            best_count[context] = count
            best_next[context] = following
    correct = total = 0
    negative_log_likelihood = 0.0
    alpha = 0.1
    vocab_size = max(len(vocabulary), 1)
    for row in eval_rows:
        sequence = tokens(str(row.get("text", "")))
        for context, following in zip(sequence, sequence[1:]):
            correct += best_next.get(context) == following
            total += 1
            probability = (bigram_counts[(context, following)] + alpha) / (
                context_counts[context] + alpha * vocab_size
            )
            negative_log_likelihood -= math.log(probability)
    return {
        "model": "additive-smoothed word bigram language model",
        "usable_train_rows": sum(bool(tokens(str(row.get("text", "")))) for row in train_rows),
        "eval_transitions": total,
        "next_token_accuracy": correct / total if total else 0.0,
        "perplexity": math.exp(negative_log_likelihood / total) if total else None,
    }


def rows_from_items(items: list[DatasetItem]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in items]


def same_training_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        _signature(DatasetItem.from_dict(a)) == _signature(DatasetItem.from_dict(b))
        for a, b in zip(left, right)
    )


def evaluate_condition(
    dataset_name: str,
    rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    family = DATASETS[dataset_name]["family"]
    started = time.perf_counter()
    if dataset_name == "ultrafeedback_dpo":
        result = preference_metrics(rows, eval_rows, seed)
    elif dataset_name == "wikitext_103":
        result = language_model_metrics(rows, eval_rows)
    else:
        result = retrieval_metrics(rows, eval_rows, family)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


async def run_combo(
    dataset_name: str,
    source_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    scale: int,
    duplicate_ratio: float,
    invalid_ratio: float,
    seed: int,
) -> dict[str, Any]:
    clean_raw = copy.deepcopy(source_rows[:scale])
    dirty_raw, injected = make_dirty(dataset_name, clean_raw, duplicate_ratio, invalid_ratio, seed)
    clean_accepted, _, _ = await optimize(copy.deepcopy(clean_raw))
    dirty_accepted, _, _ = await optimize(copy.deepcopy(dirty_raw))
    clean_optimized, dirty_optimized = rows_from_items(clean_accepted), rows_from_items(dirty_accepted)
    conditions = {
        "clean_raw": clean_raw,
        "clean_optimized": clean_optimized,
        "dirty_raw": dirty_raw,
        "dirty_optimized": dirty_optimized,
    }
    evaluated: dict[str, dict[str, Any]] = {}
    for condition, rows in conditions.items():
        reused = next(
            (
                previous
                for previous, previous_rows in conditions.items()
                if previous in evaluated and same_training_rows(rows, previous_rows)
            ),
            None,
        )
        if reused:
            evaluated[condition] = dict(evaluated[reused], reused_from=reused)
        else:
            evaluated[condition] = evaluate_condition(dataset_name, rows, eval_rows, seed)
    primary, higher_is_better = PRIMARY_METRICS[dataset_name]
    return {
        "dataset": dataset_name,
        "family": DATASETS[dataset_name]["family"],
        "scale": scale,
        "eval_rows": len(eval_rows),
        "injected": injected,
        "primary_metric": primary,
        "higher_is_better": higher_is_better,
        "conditions": evaluated,
        "clean_raw_to_optimized": {
            "before": evaluated["clean_raw"][primary],
            "after": evaluated["clean_optimized"][primary],
            "delta": evaluated["clean_optimized"][primary] - evaluated["clean_raw"][primary],
        },
        "dirty_raw_to_optimized": {
            "before": evaluated["dirty_raw"][primary],
            "after": evaluated["dirty_optimized"][primary],
            "delta": evaluated["dirty_optimized"][primary] - evaluated["dirty_raw"][primary],
        },
    }


def render_report(payload: dict[str, Any]) -> str:
    combos = payload["combinations"]
    lines = [
        "# BuffData generative downstream utility",
        "",
        "This report shows the requested before → after held-out performance movement.",
        "The benchmark uses deterministic CPU baselines appropriate to each task family;",
        "it does not relabel cleaning-decision accuracy as downstream model accuracy.",
        "",
        "| Dataset | Scale | Held-out metric | Clean raw → optimized | Δ | Dirty raw → optimized | Δ |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for combo in combos:
        clean, dirty = combo["clean_raw_to_optimized"], combo["dirty_raw_to_optimized"]
        lines.append(
            f"| {combo['dataset']} | {combo['scale']:,} | {combo['primary_metric']} | "
            f"{clean['before']:.4f} → {clean['after']:.4f} | {clean['delta']:+.4f} | "
            f"{dirty['before']:.4f} → {dirty['after']:.4f} | {dirty['delta']:+.4f} |"
        )
    lines.extend([
        "",
        "## Metric definitions",
        "",
        "- Instruction/SFT and chat: token-overlap F1 of a held-out response retrieved from the nearest training prompt.",
        "- Preference/DPO: pairwise accuracy that a linear response ranker scores the chosen answer above the rejected answer.",
        "- Extractive QA: answer token-F1; exact match is also recorded in `utility_results.json`.",
        "- Summarization: ROUGE-L F1; response token-F1 is also recorded.",
        "- Raw text: word-bigram next-token accuracy; perplexity is also recorded and lower is better.",
        "",
        "Every evaluation uses a disjoint fixed holdout. `clean raw → optimized` measures whether",
        "cleaning an unmodified source changes utility. `dirty raw → optimized` measures recovery",
        "from the disclosed 40% exact-duplicate plus 10% invalid-row contamination. The benchmark",
        "models are intentionally small and local so data-condition differences—not API/model drift—",
        "drive the comparison.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--scales", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--eval-rows", type=int, default=250)
    parser.add_argument("--duplicate-ratio", type=float, default=0.40)
    parser.add_argument("--invalid-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results-generative"))
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combinations: list[dict[str, Any]] = []
    for offset, dataset_name in enumerate(args.datasets):
        print(f"Loading {dataset_name} train/eval rows...", flush=True)
        if dataset_name == "alpaca_sft":
            combined = load_source_rows(
                dataset_name, max(args.scales) + args.eval_rows, args.seed + offset
            )
            source_rows = combined[: max(args.scales)]
            eval_rows = [
                {key: value for key, value in row.items() if key != "_buffdata_metadata"}
                for row in combined[max(args.scales) :]
            ]
        else:
            source_rows = load_source_rows(dataset_name, max(args.scales), args.seed + offset)
            eval_rows = load_eval_rows(dataset_name, args.eval_rows, args.seed + 10_000 + offset)
        for scale in args.scales:
            print(f"  Evaluating {scale:,} training rows across four conditions...", flush=True)
            combo = await run_combo(
                dataset_name,
                source_rows,
                eval_rows,
                scale,
                args.duplicate_ratio,
                args.invalid_ratio,
                args.seed + offset * 1000 + scale,
            )
            combinations.append(combo)
            movement = combo["dirty_raw_to_optimized"]
            print(
                f"    {combo['primary_metric']}: {movement['before']:.4f} -> "
                f"{movement['after']:.4f} ({movement['delta']:+.4f})",
                flush=True,
            )
    payload = {
        "benchmark": "buffdata-generative-downstream-utility",
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "combinations": combinations,
    }
    (args.output_dir / "utility_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "UTILITY_REPORT.md").write_text(render_report(payload), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'UTILITY_REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
