#!/usr/bin/env python3
"""Binary vs. multi-class classification benchmark: 10 real binary-labeled datasets and
9 real multi-class-labeled (3+ classes) datasets from Hugging Face, two measurements each.

1. Dirty-data recovery (reuses benchmark_buffdata.py's exact methodology: 40% conflicting-
   label duplicates, 40% class-skew duplicates, 10% empty rows, then BuffData's local
   validate + exact-dedup) at two data scales per dataset, to see whether the recovery
   result documented for a handful of datasets in benchmark_scale_matrix.py holds broadly
   across many more datasets and both classification shapes. No LLM involved.

2. Real classification accuracy: BuffData's own classify stage (DatasetClassifier via
   OptimizationPipeline, classification="binary"/"multi-class", the dataset's real class
   list passed explicitly so this isolates classification accuracy from schema-discovery
   noise) against a held-out labeled sample, using gemini-3.7-flash for every call. This is
   the part that directly answers "how does the CLI's binary vs. multi-class path work."

Every class name comes from the dataset's own HF ClassLabel feature (or a companion
`<col>_text` field when the label column isn't stored as a ClassLabel) -- never hardcoded,
so BuffData's classifier is judged against the same label vocabulary the dataset actually
ships.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_buffdata import (  # noqa: E402
    DEFECT_CLEAN,
    build_vocab,
    make_dirty,
    stratified_rows,
    summarize,
    train_once,
)

from buffdata.engine.client import create_llm_client  # noqa: E402
from buffdata.engine.limiter import AsyncRateLimiter  # noqa: E402
from buffdata.engine.pipeline import OptimizationPipeline  # noqa: E402
from buffdata.models.schemas import DatasetItem, PipelineConfig  # noqa: E402
from buffdata.optimizers.classifier import DatasetClassifier  # noqa: E402

SCALES = [1000, 3000]
TEST_ROWS = 500
CLASSIFY_SAMPLE_SIZE = 40


def _joined(*parts: Optional[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _max_balanced_count(split, classes: int) -> int:
    """The largest row count stratified_rows("balanced") can actually deliver for this
    split without raising -- some real datasets (TREC's famously tiny, imbalanced test
    set in particular) have far fewer rows in their smallest class than an even split of
    a round requested total would need."""
    counts: dict[int, int] = {}
    for label in split["label"]:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < classes or not counts:
        return 0
    return min(counts.values()) * classes


# --- Dataset definitions -------------------------------------------------------------------
# Each: hf_id, optional config, "kind" (binary/multi), text_fields to join, label_col, and
# optional explicit train/test split names (defaults to "train"/"test"; a dataset with only
# one split gets a deterministic 85/15 held-out split instead).

DATASETS: dict[str, dict[str, Any]] = {
    # --- binary (2-class) -------------------------------------------------------------
    "yelp_polarity": {
        "hf_id": "fancyzhx/yelp_polarity", "kind": "binary",
        "text_fields": ["text"], "label_col": "label",
        "description": "Yelp review sentiment polarity",
    },
    "amazon_polarity": {
        "hf_id": "fancyzhx/amazon_polarity", "kind": "binary",
        "text_fields": ["title", "content"], "label_col": "label",
        "description": "Amazon review sentiment polarity",
    },
    "imdb": {
        "hf_id": "stanfordnlp/imdb", "kind": "binary",
        "text_fields": ["text"], "label_col": "label",
        "description": "IMDb movie review sentiment",
    },
    "rotten_tomatoes": {
        "hf_id": "cornell-movie-review-data/rotten_tomatoes", "kind": "binary",
        "text_fields": ["text"], "label_col": "label",
        "description": "Rotten Tomatoes critic-snippet sentiment",
    },
    "sst2": {
        "hf_id": "SetFit/sst2", "kind": "binary",
        "text_fields": ["text"], "label_col": "label",
        "description": "Stanford Sentiment Treebank (binary)",
    },
    "subj": {
        "hf_id": "SetFit/subj", "kind": "binary",
        "text_fields": ["text"], "label_col": "label",
        "description": "Subjective vs. objective sentence classification",
    },
    "tweet_eval_irony": {
        "hf_id": "cardiffnlp/tweet_eval", "config": "irony", "kind": "binary",
        "text_fields": ["text"], "label_col": "label",
        "description": "Tweet irony detection",
    },
    "tweet_eval_hate": {
        "hf_id": "cardiffnlp/tweet_eval", "config": "hate", "kind": "binary",
        "text_fields": ["text"], "label_col": "label",
        "description": "Tweet hate-speech detection",
    },
    "cr": {
        "hf_id": "SetFit/CR", "kind": "binary",
        "text_fields": ["text"], "label_col": "label",
        "description": "Customer review sentiment",
    },
    "amazon_counterfactual": {
        "hf_id": "SetFit/amazon_counterfactual_en", "kind": "binary",
        "text_fields": ["text"], "label_col": "label",
        "description": "Amazon review counterfactual-statement detection",
    },
    # --- multi-class (3+ classes) -------------------------------------------------------
    "ag_news": {
        "hf_id": "fancyzhx/ag_news", "kind": "multi",
        "text_fields": ["text"], "label_col": "label",
        "description": "AG News topic classification (4 classes)",
    },
    "dbpedia_14": {
        "hf_id": "fancyzhx/dbpedia_14", "kind": "multi",
        "text_fields": ["title", "content"], "label_col": "label",
        "description": "DBpedia ontology classification (14 classes)",
    },
    "yahoo_answers_topics": {
        "hf_id": "community-datasets/yahoo_answers_topics", "kind": "multi",
        "text_fields": ["question_title", "question_content", "best_answer"], "label_col": "topic",
        "description": "Yahoo! Answers topic classification (10 classes, noisy)",
    },
    "emotion": {
        "hf_id": "dair-ai/emotion", "kind": "multi",
        "text_fields": ["text"], "label_col": "label",
        "description": "Emotion classification (6 classes)",
    },
    "tweet_eval_emotion": {
        "hf_id": "cardiffnlp/tweet_eval", "config": "emotion", "kind": "multi",
        "text_fields": ["text"], "label_col": "label",
        "description": "Tweet emotion classification (4 classes)",
    },
    "tweet_eval_sentiment": {
        "hf_id": "cardiffnlp/tweet_eval", "config": "sentiment", "kind": "multi",
        "text_fields": ["text"], "label_col": "label",
        "description": "Tweet 3-way sentiment classification",
    },
    "20_newsgroups": {
        "hf_id": "SetFit/20_newsgroups", "kind": "multi",
        "text_fields": ["text"], "label_col": "label",
        "description": "20 Newsgroups topic classification (20 classes)",
    },
    "trec_coarse": {
        "hf_id": "SetFit/TREC-QC", "kind": "multi",
        "text_fields": ["text"], "label_col": "label_coarse",
        "description": "TREC question-type classification (6 coarse classes)",
    },
    "tweet_sentiment_extraction": {
        "hf_id": "SetFit/tweet_sentiment_extraction", "kind": "multi",
        "text_fields": ["text"], "label_col": "label",
        "description": "Tweet 3-way sentiment classification (independent source from tweet_eval_sentiment)",
    },
}


def _resolve_class_names(split, label_col: str) -> list[str]:
    """Official class names from the dataset's own ClassLabel feature, or -- when the
    label column is a plain int (not stored as a ClassLabel) -- derived from whatever
    `<label_col>_text` companion column the dataset ships. Never hardcoded/guessed.
    """
    names = getattr(split.features[label_col], "names", None)
    if names:
        return list(names)
    text_col = f"{label_col}_text"
    if text_col in split.column_names:
        pairs = sorted(set(zip(split[label_col], split[text_col])))
        return [text for _, text in pairs]
    n_classes = len(set(split[label_col]))
    return [str(i) for i in range(n_classes)]


def load_and_normalize(spec: dict[str, Any]):
    from datasets import load_dataset

    ds = load_dataset(spec["hf_id"], spec["config"]) if spec.get("config") else load_dataset(spec["hf_id"])
    train_raw = ds[spec.get("train_split", "train")]
    test_split_name = spec.get("test_split", "test")
    if test_split_name in ds:
        test_raw = ds[test_split_name]
    else:
        split = train_raw.train_test_split(test_size=0.15, seed=42)
        train_raw, test_raw = split["train"], split["test"]

    label_col = spec["label_col"]
    class_names = _resolve_class_names(train_raw, label_col)
    text_fields = spec["text_fields"]

    def normalize_row(row):
        return {"text": _joined(*(str(row.get(f, "")) for f in text_fields)), "label": int(row[label_col])}

    train = train_raw.map(normalize_row, remove_columns=train_raw.column_names)
    test = test_raw.map(normalize_row, remove_columns=test_raw.column_names)
    return train, test, class_names


# --- Real classification-accuracy measurement (real gemini-3.7-flash calls) ----------------

async def measure_classification_accuracy(
    rows: list[dict[str, Any]],
    class_names: list[str],
    mode: str,
    gemini_model: str,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Strips labels from a stratified sample and classifies it exactly the way the
    standalone `buffdata classify` command does -- DatasetClassifier.resolve_schema()
    then classify_batch() directly, not the full 7-stage pipeline, since that command
    bypasses the pipeline's unconditional profiling stage entirely (see cli/main.py's
    classify command). Passing classes explicitly makes resolve_schema a free local
    step (classifier.py checks `task_hint and requested_classes` first), so every real
    network call here is a genuine classify_batch call, nothing else. Compares assigned
    labels against the true ones held out beforehand.
    """
    rng = random.Random(seed)
    by_class: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(row["label"], []).append(row)
    per_class_quota = max(1, sample_size // len(class_names))
    sampled: list[dict[str, Any]] = []
    for label_rows in by_class.values():
        take = list(label_rows)
        rng.shuffle(take)
        sampled.extend(take[:per_class_quota])
    rng.shuffle(sampled)
    sampled = sampled[:sample_size]

    items = [DatasetItem.from_dict({"text": row["text"]}) for row in sampled]
    true_by_id = {item.id: class_names[row["label"]] for item, row in zip(items, sampled)}

    client = create_llm_client("gemini", gemini_model)
    limiter = AsyncRateLimiter(max_rpm=60, concurrency=5)
    classifier = DatasetClassifier(client, limiter, gemini_model)

    started = time.perf_counter()
    schema = await classifier.resolve_schema(items, mode=mode, classes=class_names, sample_size=len(items))
    classified = await classifier.classify_batch(items, schema)
    elapsed = time.perf_counter() - started

    correct = 0
    scored = 0
    confusions: list[dict[str, str]] = []
    for item in classified:
        if item.metadata.get("classification_error"):
            continue
        true_label = true_by_id.get(item.id)
        predicted = item.labels if isinstance(item.labels, str) else None
        if true_label is None or predicted is None:
            continue
        scored += 1
        if predicted == true_label:
            correct += 1
        else:
            confusions.append({"true": true_label, "predicted": predicted})

    return {
        "mode": mode,
        "classes": class_names,
        "sampled": len(sampled),
        "scored": scored,
        "failed_or_unclassified": len(sampled) - scored,
        "correct": correct,
        "accuracy": correct / scored if scored else None,
        "elapsed_seconds": round(elapsed, 1),
        "usage": dict(client.usage),
        "confusions_sample": confusions[:10],
    }


# --- Recovery measurement (local only, reused methodology) ---------------------------------

async def measure_recovery(
    dataset_name: str,
    spec: dict[str, Any],
    scale: int,
    seeds: list[int],
    epochs: int,
    train_split,
    test_split,
    class_names: list[str],
    dataset_offset: int,
) -> dict[str, Any]:
    from benchmark_buffdata import optimize  # local import: heavy (pulls in the full pipeline)

    classes = len(class_names)
    scale = min(scale, _max_balanced_count(train_split, classes))
    test_count = min(TEST_ROWS, _max_balanced_count(test_split, classes))
    clean_train = stratified_rows(train_split, scale, classes, ["text"], 100 + dataset_offset, "balanced")
    clean_test = stratified_rows(test_split, test_count, classes, ["text"], 200 + dataset_offset, "balanced")
    dirty_raw, optimizer_input, defects = make_dirty(clean_train, classes, 300 + dataset_offset)

    clean_optimized, clean_quality = await optimize(clean_train)
    dirty_optimized, dirty_quality = await optimize(optimizer_input)

    conditions = {"clean_raw": clean_train, "clean_optimized": clean_optimized, "dirty_raw": dirty_raw, "dirty_optimized": dirty_optimized}
    vocab = build_vocab(clean_train)
    results: dict[str, Any] = {}
    for name, rows in conditions.items():
        runs = [train_once(rows, clean_test, vocab, classes, seed, epochs) for seed in seeds]
        results[name] = {"rows": len(rows), "runs": runs, "summary": summarize(runs)}
        print(
            f"  [{dataset_name} @ {scale}] {name} acc={results[name]['summary']['accuracy_mean']:.4f}",
            flush=True,
        )

    return {
        "scale_requested": scale,
        "defects": defects,
        "conditions": results,
        "recovery_accuracy": results["dirty_optimized"]["summary"]["accuracy_mean"] - results["dirty_raw"]["summary"]["accuracy_mean"],
        "clean_delta_accuracy": results["clean_optimized"]["summary"]["accuracy_mean"] - results["clean_raw"]["summary"]["accuracy_mean"],
        # Real per-category, per-stage accept/reject cross-tab from the pipeline's own
        # decisions (not the injection ratios) -- how many of each injected defect type
        # were actually caught vs. slipped through, and which stage caught them.
        "dirty_defect_breakdown": dirty_quality.get("defect_breakdown", {}),
        "clean_defect_breakdown": clean_quality.get("defect_breakdown", {}),
        "dirty_hygiene": dirty_quality.get("hygiene", {}),
        "clean_hygiene": clean_quality.get("hygiene", {}),
    }


async def run_dataset(
    name: str, spec: dict[str, Any], scales: list[int], seeds: list[int], epochs: int,
    gemini_model: str, dataset_offset: int, skip_classification: bool = False,
    existing: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    print(f"=== {name} ({spec['hf_id']}{'/' + spec['config'] if spec.get('config') else ''}) ===", flush=True)
    train_split, test_split, class_names = load_and_normalize(spec)
    mode = "binary" if spec["kind"] == "binary" else "multi-class"

    # Recovery is fully deterministic (fixed seeds, local-only) -- reuse a prior run's
    # numbers for this exact scale set rather than repeating ~20 minutes of training that
    # can't produce a different answer, e.g. when resuming tomorrow just for the part
    # that needed today's exhausted Gemini quota to reset.
    existing_recovery = (existing or {}).get("recovery_by_scale") or {}
    if existing_recovery and set(existing_recovery) == {str(s) for s in scales}:
        recovery_by_scale = existing_recovery
        print(f"  [{name}] reusing recovery results from a prior run (same scales)", flush=True)
    else:
        recovery_by_scale = {}
        for scale in scales:
            recovery_by_scale[str(scale)] = await measure_recovery(
                name, spec, scale, seeds, epochs, train_split, test_split, class_names, dataset_offset,
            )

    classification = (existing or {}).get("classification")
    if classification is not None:
        print(f"  [{name}] reusing classification results from a prior run", flush=True)
    elif not skip_classification:
        classification = await measure_classification_accuracy(
            list(train_split), class_names, mode, gemini_model, CLASSIFY_SAMPLE_SIZE, seed=17 + dataset_offset,
        )
        acc_text = f"{classification['accuracy']:.1%}" if classification["accuracy"] is not None else "n/a"
        print(
            f"  [{name}] classification ({mode}, {len(class_names)} classes): "
            f"{classification['correct']}/{classification['scored']} correct "
            f"({acc_text}), {classification['failed_or_unclassified']} failed, {classification['elapsed_seconds']}s",
            flush=True,
        )

    return {
        "dataset": name,
        "kind": spec["kind"],
        "description": spec["description"],
        "hf_id": spec["hf_id"],
        "num_classes": len(class_names),
        "class_names": class_names,
        "recovery_by_scale": recovery_by_scale,
        "classification": classification,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    have_classification = any(r.get("classification") for r in payload["results"])
    lines = [
        "# Binary vs. multi-class classification benchmark",
        "",
        f"Model: `{payload['method']['gemini_model']}` | Scales: {payload['method']['scales']} | "
        f"Seeds: {payload['method']['seeds']} | Classification sample size: {payload['method']['classify_sample_size']}",
        "",
    ]
    if have_classification:
        lines += [
            "## Classification accuracy (real gemini-3.7-flash calls)",
            "",
            "| Dataset | Kind | Classes | Scored | Correct | Accuracy |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in payload["results"]:
            c = r.get("classification")
            if c is None:
                continue
            acc = f"{c['accuracy']:.1%}" if c["accuracy"] is not None else "n/a"
            lines.append(f"| {r['dataset']} | {r['kind']} | {r['num_classes']} | {c['scored']} | {c['correct']} | {acc} |")
        lines.append("")
    else:
        lines += ["## Classification accuracy", "", "Not run this pass (`--skip-classification`).", ""]

    lines += [
        "## Dirty-data recovery (local only, no LLM)",
        "",
        "\"Rows\" is the actual balanced training size used, which can be smaller than the "
        "scale header: `_max_balanced_count` caps every request at `smallest_class_size × "
        "num_classes` so a stratified sample never asks a rare class for more rows than it "
        "has (see trec_coarse below, whose rarest coarse class has only 86 examples -- both "
        "scale requests land on the same capped, and therefore identical, result).",
        "",
    ]
    for scale in payload["method"]["scales"]:
        lines.append(f"### Scale {scale}")
        lines.append("")
        lines.append("| Dataset | Kind | Rows | Dirty raw acc | Dirty optimized acc | Recovery | Clean delta |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in payload["results"]:
            rec = r["recovery_by_scale"].get(str(scale))
            if rec is None:
                continue
            c = rec["conditions"]
            rows = c["clean_raw"]["rows"]
            capped = " †" if rows < scale else ""
            lines.append(
                f"| {r['dataset']} | {r['kind']} | {rows}{capped} | {c['dirty_raw']['summary']['accuracy_mean']:.4f} | "
                f"{c['dirty_optimized']['summary']['accuracy_mean']:.4f} | {rec['recovery_accuracy']:+.4f} | "
                f"{rec['clean_delta_accuracy']:+.4f} |"
            )
        lines.append("")
    lines.append("† capped below the requested scale by the smallest class's available rows.")

    lines += [
        "",
        "## Data hygiene: how many rows were actually retained vs. lost",
        "",
        "Per-category cross-tab of the real pipeline decision for the `dirty_optimized` run at "
        "each scale, taken from `item.metadata` after the run (not the injection ratios) -- "
        "'Lost' is a row the pipeline actually rejected, split by the stage that caught it. "
        "'Clean rows lost' isolates incidental duplicates the source dataset already had before "
        "any defect was injected.",
        "",
    ]
    for scale in payload["method"]["scales"]:
        lines.append(f"### Scale {scale}")
        lines.append("")
        lines.append("| Dataset | Input | Retained | Deleted | Duplicate input rows | Invalid deleted | Duplicate deleted | Defects retained | Clean rows deleted |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in payload["results"]:
            rec = r["recovery_by_scale"].get(str(scale))
            if rec is None:
                continue
            hygiene = rec.get("dirty_hygiene", {})
            breakdown = rec.get("dirty_defect_breakdown", {})
            totals = {"input": 0, "retained": 0, "lost": 0}
            by_stage: dict[str, int] = {}
            for entry in breakdown.values():
                for key in ("input", "retained", "lost"):
                    totals[key] += entry[key]
                for stage, count in entry["lost_by_stage"].items():
                    by_stage[stage] = by_stage.get(stage, 0) + count
            clean_lost = breakdown.get(DEFECT_CLEAN, {}).get("lost", 0)
            lines.append(
                f"| {r['dataset']} | {totals['input']:,} | {totals['retained']:,} | {totals['lost']:,} | "
                f"{hygiene.get('duplicate_rows_in_input', 0):,} | "
                f"{hygiene.get('invalid_rows_deleted', by_stage.get('validate', 0)):,} | "
                f"{hygiene.get('duplicate_rows_deleted', by_stage.get('dedup', 0)):,} | "
                f"{hygiene.get('injected_defects_retained', 0):,} | {clean_lost:,} |"
            )
        lines.append("")
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    report_path = output_dir / "REPORT.md"

    existing_by_name: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        prior = json.loads(results_path.read_text(encoding="utf-8"))
        existing_by_name = {r["dataset"]: r for r in prior.get("results", [])}
        print(f"Resuming: found {len(existing_by_name)} dataset(s) already recorded in {results_path}", flush=True)

    payload: dict[str, Any] = {
        "method": {
            "seeds": args.seeds, "epochs": args.epochs, "scales": args.scales or SCALES,
            "gemini_model": args.gemini_model, "classify_sample_size": CLASSIFY_SAMPLE_SIZE,
        },
        "results": [],
    }
    selected = args.datasets or list(DATASETS)
    for dataset_offset, name in enumerate(selected):
        spec = DATASETS[name]
        started = time.perf_counter()
        combo = await run_dataset(
            name, spec, args.scales or SCALES, args.seeds, args.epochs, args.gemini_model, dataset_offset,
            skip_classification=args.skip_classification, existing=existing_by_name.get(name),
        )
        payload["results"].append(combo)
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report_path.write_text(markdown_report(payload), encoding="utf-8")
        print(f"--- {name} done in {time.perf_counter() - started:.1f}s (checkpoint saved) ---", flush=True)

    print(f"Results: {results_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmarks/results-classification-matrix")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--scales", type=int, nargs="+")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--gemini-model", default="gemini-3.7-flash")
    parser.add_argument(
        "--skip-classification", action="store_true",
        help="Run only the local recovery benchmark; skip every real Gemini classify call (for when quota is exhausted).",
    )
    asyncio.run(main(parser.parse_args()))
