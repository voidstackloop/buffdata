#!/usr/bin/env python3
"""Multi-label accuracy-recovery benchmark: same methodology as
benchmark_scale_matrix.py (deterministic dirty-data injection, then BuffData's local
validate/dedup cleaning, at multiple real-world scales), adapted for multi-label data --
see benchmark_multilabel.py for what specifically changes and why.

Real, verified multi-label datasets (benchmarks/_verify_datasets.py,
_verify_true_classes.py): large (10k+ train rows, except pubmed_mesh at 50k train with
no native test split, handled with a deterministic manual split), genuinely multi-label
(a list of active labels per row, not one label per row), confirmed by inspecting real
schema/data rather than dataset names/descriptions alone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_multilabel import (  # noqa: E402
    build_vocab,
    make_dirty_multilabel,
    optimize_multilabel,
    stratified_rows_multilabel,
    summarize_multilabel,
    train_once_multilabel,
)

SCALES = [10_000, 20_000, 30_000]
TEST_ROWS = 2_000

_GO_EMOTIONS_COLS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]
_CIVIL_COMMENTS_COLS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
_PUBMED_MESH_COLS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M", "N", "Z"]
_JIGSAW_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _multi_hot_from_columns(columns: list[str], threshold: float = 0.5):
    def normalize(row: dict[str, Any]) -> dict[str, Any]:
        labels = [i for i, col in enumerate(columns) if float(row.get(col, 0) or 0) >= threshold]
        return {"text": row.get("text") or row.get("comment_text") or "", "label": labels}
    return normalize


DATASETS: dict[str, dict[str, Any]] = {
    "go_emotions": {
        "hf_id": "google-research-datasets/go_emotions", "hf_config": "simplified",
        "num_labels": 28, "description": "Reddit comment emotion classification (28 emotions, GoEmotions)",
        "normalize": lambda row: {"text": row["text"], "label": row["labels"]},
    },
    "civil_comments": {
        "hf_id": "google/civil_comments", "num_labels": 7,
        "description": "Civil Comments toxicity sub-type classification (7 types, >=0.5 crowd-annotation threshold)",
        "normalize": _multi_hot_from_columns(_CIVIL_COMMENTS_COLS, threshold=0.5),
    },
    "pubmed_mesh": {
        "hf_id": "owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH", "num_labels": 14,
        "description": "PubMed abstract MeSH root-category classification (14 categories)",
        "normalize": lambda row: {
            "text": " ".join(filter(None, [row.get("Title"), row.get("abstractText")])),
            "label": [i for i, col in enumerate(_PUBMED_MESH_COLS) if int(row.get(col, 0) or 0) == 1],
        },
        "no_native_test_split": True,
    },
    "jigsaw_toxicity": {
        "hf_id": "Arsive/toxicity_classification_jigsaw", "num_labels": 6,
        "description": "Jigsaw toxic-comment sub-type classification (6 types)",
        "normalize": _multi_hot_from_columns(_JIGSAW_COLS, threshold=0.5),
    },
    "eurlex": {
        "hf_id": "coastalcph/lex_glue", "hf_config": "eurlex", "num_labels": 100,
        "description": "EU legal document topic classification (100 possible EuroVoc concepts, LexGLUE)",
        "normalize": lambda row: {"text": row["text"], "label": row["labels"]},
    },
}


def load_normalized_multilabel(spec: dict[str, Any]):
    from datasets import load_dataset

    source = load_dataset(spec["hf_id"], spec["hf_config"]) if spec.get("hf_config") else load_dataset(spec["hf_id"])
    fn: Callable[[dict], dict] = spec["normalize"]
    normalize_row = lambda row: fn(row)  # noqa: E731

    if spec.get("no_native_test_split"):
        # Deterministic 85/15 split of the one available split -- documented here rather
        # than silently assumed, since it means "train" below isn't literally everything
        # the source dataset calls "train".
        full = source["train"].map(normalize_row, remove_columns=source["train"].column_names)
        split = full.train_test_split(test_size=0.15, seed=42)
        return split["train"], split["test"]

    eval_split_name = spec.get("eval_split", "test")
    train = source["train"].map(normalize_row, remove_columns=source["train"].column_names)
    test = source[eval_split_name].map(normalize_row, remove_columns=source[eval_split_name].column_names)
    return train, test


async def run_combo(
    dataset_name: str,
    spec: dict[str, Any],
    scale: int,
    seeds: list[int],
    epochs: int,
    train_split,
    test_split,
    dataset_offset: int,
    gemini_audit_rows: int = 0,
    gemini_model: str = "gemini-3.7-flash",
) -> dict[str, Any]:
    num_labels = spec["num_labels"]
    clean_train = stratified_rows_multilabel(train_split, scale, ["text"], 100 + dataset_offset)
    clean_test = stratified_rows_multilabel(test_split, min(TEST_ROWS, len(test_split)), ["text"], 200 + dataset_offset)
    dirty_raw, optimizer_input, defects = make_dirty_multilabel(clean_train, num_labels, 300 + dataset_offset)

    clean_optimized, clean_quality = await optimize_multilabel(clean_train)
    dirty_optimized, dirty_quality = await optimize_multilabel(
        optimizer_input, gemini_audit_rows=gemini_audit_rows, gemini_model=gemini_model,
    )
    if gemini_audit_rows:
        score_stage = dirty_quality["stages"].get("score_refine", {})
        print(
            f"  [{dataset_name} @ {scale}] Gemini audit: {score_stage.get('scored', 0)} rows scored, "
            f"avg {score_stage.get('average_overall_score')}/10, "
            f"{dirty_quality['usage']['total_tokens']} tokens",
            flush=True,
        )

    conditions = {
        "clean_raw": clean_train,
        "clean_optimized": clean_optimized,
        "dirty_raw": dirty_raw,
        "dirty_optimized": dirty_optimized,
    }
    vocab = build_vocab(clean_train)
    results: dict[str, Any] = {}
    for name, rows in conditions.items():
        runs = []
        for seed in seeds:
            started = time.perf_counter()
            runs.append(train_once_multilabel(rows, clean_test, vocab, num_labels, seed, epochs))
            print(
                f"  [{dataset_name} @ {scale}] {name} seed={seed} "
                f"subset_acc={runs[-1]['subset_accuracy']:.4f} macro_f1={runs[-1]['macro_f1']:.4f} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
        results[name] = {"rows": len(rows), "runs": runs, "summary": summarize_multilabel(runs)}

    dirty_recovery = {
        "subset_accuracy": results["dirty_optimized"]["summary"]["subset_accuracy_mean"] - results["dirty_raw"]["summary"]["subset_accuracy_mean"],
        "macro_f1": results["dirty_optimized"]["summary"]["macro_f1_mean"] - results["dirty_raw"]["summary"]["macro_f1_mean"],
    }
    clean_delta = {
        "subset_accuracy": results["clean_optimized"]["summary"]["subset_accuracy_mean"] - results["clean_raw"]["summary"]["subset_accuracy_mean"],
    }
    dirty_input_rows = len(optimizer_input)
    dirty_val_rej = dirty_quality["stages"].get("validate", {})
    dirty_dedup_rej = dirty_quality["stages"].get("dedup", {})
    return {
        "dataset": dataset_name,
        "scale": scale,
        "num_labels": num_labels,
        "defects": defects,
        "conditions": results,
        "dirty_optimized_minus_dirty_raw": dirty_recovery,
        "clean_optimized_minus_clean_raw": clean_delta,
        "dirty_stage_breakdown": {
            "input_rows": dirty_input_rows,
            "output_rows": dirty_quality["output_rows"],
            "validate": dirty_val_rej,
            "dedup": dirty_dedup_rej,
        },
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# BuffData multi-label accuracy-recovery matrix",
        "",
        f"Seeds: {payload['method']['seeds']} | Epochs: {payload['method']['epochs']} | "
        f"Test rows/dataset: {payload['method']['test_rows']} | "
        "Metrics: subset accuracy (exact label-set match) and macro-F1 (per-label, "
        "partial-credit) -- see benchmark_multilabel.py.",
        "",
        "| Dataset | Scale | Labels | Dirty raw subset-acc | Dirty optimized subset-acc | Recovery (subset-acc) | Recovery (macro-F1) | Clean delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for combo in payload["results"]:
        c = combo["conditions"]
        lines.append(
            f"| {combo['dataset']} | {combo['scale']:,} | {combo['num_labels']} | "
            f"{c['dirty_raw']['summary']['subset_accuracy_mean']:.4f} | "
            f"{c['dirty_optimized']['summary']['subset_accuracy_mean']:.4f} | "
            f"{combo['dirty_optimized_minus_dirty_raw']['subset_accuracy']:+.4f} | "
            f"{combo['dirty_optimized_minus_dirty_raw']['macro_f1']:+.4f} | "
            f"{combo['clean_optimized_minus_clean_raw']['subset_accuracy']:+.4f} |"
        )
    lines += [
        "",
        "'Dirty raw' trains on the source data plus disclosed conflicting-label duplicates "
        "(one label toggled, 40%), class-skew duplicates (exact row copies, 40%), and empty "
        "rows (10%). 'Dirty optimized' is BuffData's cleaned output from that exact "
        "contaminated input (validate + exact dedup only). 'Clean delta' is a control: it "
        "should stay near zero, showing BuffData does not damage already-clean data.",
    ]
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    report_path = output_dir / "REPORT.md"

    payload: dict[str, Any] = {
        "method": {"seeds": args.seeds, "epochs": args.epochs, "test_rows": TEST_ROWS, "scales": SCALES},
        "results": [],
    }

    skipped: list[dict[str, Any]] = []
    selected = args.datasets or list(DATASETS)
    for dataset_offset, name in enumerate(selected):
        spec = DATASETS[name]
        print(f"=== {name} ({spec['hf_id']}) ===", flush=True)
        train_split, test_split = load_normalized_multilabel(spec)
        for scale in (args.scales or SCALES):
            started = time.perf_counter()
            try:
                combo = await run_combo(
                    name, spec, scale, args.seeds, args.epochs, train_split, test_split, dataset_offset,
                    gemini_audit_rows=args.gemini_audit_rows, gemini_model=args.gemini_model,
                )
            except ValueError as exc:
                print(f"  [{name} @ {scale}] SKIPPED -- {exc}", flush=True)
                skipped.append({"dataset": name, "scale": scale, "reason": str(exc)})
                (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
                continue
            payload["results"].append(combo)
            results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            report_path.write_text(markdown_report(payload), encoding="utf-8")
            print(f"--- {name} @ {scale} done in {time.perf_counter() - started:.1f}s (checkpoint saved) ---", flush=True)

    print(f"Results: {results_path}")
    print(f"Report:  {report_path}")
    if skipped:
        print(f"Skipped {len(skipped)} dataset@scale combination(s) that didn't fit -- see {output_dir / 'skipped.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmarks/results-multilabel-matrix")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--scales", type=int, nargs="+", choices=SCALES)
    parser.add_argument(
        "--gemini-audit-rows", type=int, default=0,
        help="Sample-audit N rows of the dirty-optimized condition per combo with Gemini (needs GEMINI_API_KEY); 0 stays fully offline",
    )
    parser.add_argument("--gemini-model", default="gemini-3.7-flash")
    asyncio.run(main(parser.parse_args()))
