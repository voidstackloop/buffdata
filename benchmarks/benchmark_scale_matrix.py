#!/usr/bin/env python3
"""5-dataset x 3-scale (10k / 25k / 50k) real-world accuracy-recovery benchmark.

Reuses the exact, proven, zero-remote-call methodology from benchmark_buffdata.py
(deterministic dirty-data injection: 40% conflicting-label duplicates, 40%
class-skew duplicates, 10% empty rows -- then BuffData's local validate/dedup
cleaning) at larger scale and across five real Hugging Face classification
datasets, to answer one question with real numbers: is BuffData's cleaning
worth using on messy real-world data, at realistic training-set sizes?

By default no LLM/API key is used: DatasetProfiler skips the remote call entirely
for already-labeled data (see buffdata/engine/profiler.py), and PII scrubbing and
quality scoring stay off, matching benchmark_buffdata.py's own defaults. Pass
--gemini-audit-rows N (needs GEMINI_API_KEY) to additionally have Gemini
sample-audit N rows of the *dirty* condition's cleaned output per combo in
`quality_mode="sampled"` -- the same real, structured-batch LLM call the CLI's
`optimize` command makes, informational only, never deciding which rows survive
the local validate/dedup stages that produce the accuracy numbers below.
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
from benchmark_buffdata import (  # noqa: E402
    build_vocab,
    make_dirty,
    optimize,
    stratified_rows,
    summarize,
    train_once,
    write_jsonl,
)

SCALES = [10_000, 20_000, 30_000]
TEST_ROWS = 2_000


def _joined(*parts: str | None) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _pair(col1: str, col2: str):
    return lambda row: {"text": _joined(row.get(col1), row.get(col2)), "label": row["label"]}


def _single(col: str = "text", label_col: str = "label"):
    return lambda row: {"text": row[col], "label": row[label_col]}


# Verified for real against the HF Hub (buffdata/../_verify_datasets.py): schema, split
# size (>=10k train rows), and actual class cardinality (many HF dataset cards understate
# or omit this -- e.g. hate_speech18 and hate_speech_offensive looked binary from their
# names but are genuinely 4-class and 3-class respectively, confirmed by sampling real
# label values, not by trusting metadata alone). Near-duplicate re-hosts of the same
# underlying corpus under a different HF namespace (SetFit/ag_news vs fancyzhx/ag_news,
# SetFit/emotion vs dair-ai/emotion, etc.) are deliberately NOT double-counted here --
# each entry below is a genuinely distinct source corpus.
DATASETS: dict[str, dict[str, Any]] = {
    # --- binary (2 classes) ---
    "imdb": {
        "hf_id": "stanfordnlp/imdb", "classes": 2, "kind": "binary",
        "description": "IMDB movie review sentiment",
        "normalize": _single("text", "label"),
    },
    "yelp_polarity": {
        "hf_id": "fancyzhx/yelp_polarity", "classes": 2, "kind": "binary",
        "description": "Yelp review sentiment polarity",
        "normalize": _single("text", "label"),
    },
    "amazon_polarity": {
        "hf_id": "fancyzhx/amazon_polarity", "classes": 2, "kind": "binary",
        "description": "Amazon review sentiment polarity",
        "normalize": _pair("title", "content"),
    },
    "sst2": {
        "hf_id": "stanfordnlp/sst2", "classes": 2, "kind": "binary",
        "description": "Stanford Sentiment Treebank v2, binary",
        "normalize": _single("sentence", "label"),
        # GLUE's real "test" split labels are hidden (-1 sentinel, for the official
        # leaderboard) -- "validation" is the split with real labels. Confirmed by
        # sampling actual label values, not assumed from the split name.
        "eval_split": "validation",
    },
    "enron_spam": {
        "hf_id": "SetFit/enron_spam", "classes": 2, "kind": "binary",
        "description": "Enron email spam/ham",
        "normalize": _single("text", "label"),
    },
    "toxic_conversations": {
        "hf_id": "SetFit/toxic_conversations_50k", "classes": 2, "kind": "binary",
        "description": "Jigsaw toxic conversation comments",
        "normalize": _single("text", "label"),
    },
    "insincere_questions": {
        "hf_id": "SetFit/insincere-questions", "classes": 2, "kind": "binary",
        "description": "Quora insincere-question detection",
        "normalize": _single("text", "label"),
    },
    "ade_corpus": {
        "hf_id": "SetFit/ade_corpus_v2_classification", "classes": 2, "kind": "binary",
        "description": "Adverse Drug Event mention detection in medical text",
        "normalize": _single("text", "label"),
    },
    "qqp": {
        "hf_id": "SetFit/qqp", "classes": 2, "kind": "binary",
        "description": "Quora Question Pairs duplicate detection",
        "normalize": _pair("text1", "text2"),
        "eval_split": "validation",  # "test" labels are GLUE's hidden -1 sentinel
    },
    "qnli": {
        "hf_id": "SetFit/qnli", "classes": 2, "kind": "binary",
        "description": "Question-answering NLI entailment",
        "normalize": _pair("text1", "text2"),
        "eval_split": "validation",  # "test" labels are GLUE's hidden -1 sentinel
    },

    # --- multi-class (>2 classes, single label per row) ---
    "ag_news": {
        "hf_id": "fancyzhx/ag_news", "classes": 4, "kind": "multiclass",
        "description": "AG News topic classification",
        "normalize": _single("text", "label"),
    },
    "dbpedia_14": {
        "hf_id": "fancyzhx/dbpedia_14", "classes": 14, "kind": "multiclass",
        "description": "DBpedia ontology classification",
        "normalize": _pair("title", "content"),
    },
    "yahoo_answers_topics": {
        "hf_id": "community-datasets/yahoo_answers_topics", "classes": 10, "kind": "multiclass",
        "description": "Yahoo! Answers topic classification, noisy user-generated Q&A",
        "normalize": lambda row: {
            "text": _joined(row.get("question_title"), row.get("question_content"), row.get("best_answer")),
            "label": row["topic"],
        },
    },
    "emotion": {
        "hf_id": "dair-ai/emotion", "classes": 6, "kind": "multiclass",
        "description": "Twitter emotion classification",
        "normalize": _single("text", "label"),
    },
    "newsgroups_20": {
        "hf_id": "SetFit/20_newsgroups", "classes": 20, "kind": "multiclass",
        "description": "20 Newsgroups topic classification",
        "normalize": _single("text", "label"),
    },
    "yelp_review_full": {
        "hf_id": "SetFit/yelp_review_full", "classes": 5, "kind": "multiclass",
        "description": "Yelp review star rating (1-5)",
        "normalize": _single("text", "label"),
    },
    "student_questions": {
        "hf_id": "SetFit/student-question-categories", "classes": 4, "kind": "multiclass",
        "description": "Student question subject-area categories",
        "normalize": _single("text", "label"),
    },
    "amazon_massive_scenario": {
        # classes verified against the FULL train+test label set (benchmarks/_verify_true_classes.py),
        # not a partial sample -- a 3000-row sample only showed 14 of the real 18 classes.
        "hf_id": "SetFit/amazon_massive_scenario_en-US", "classes": 18, "kind": "multiclass",
        "description": "Amazon MASSIVE assistant-query scenario classification (English)",
        "normalize": _single("text", "label"),
    },
    "amazon_massive_intent": {
        # same caution: a partial sample showed 39 of the real 60 classes.
        "hf_id": "SetFit/amazon_massive_intent_en-US", "classes": 60, "kind": "multiclass",
        "description": "Amazon MASSIVE assistant-query intent classification (English) -- same corpus as amazon_massive_scenario, a genuinely different label task",
        "normalize": _single("text", "label"),
    },
    "amazon_reviews_multi_en": {
        "hf_id": "SetFit/amazon_reviews_multi_en", "classes": 5, "kind": "multiclass",
        "description": "Amazon multilingual reviews, English subset, star rating",
        "normalize": _single("text", "label"),
    },
    "xglue_news": {
        "hf_id": "SetFit/xglue_nc", "classes": 10, "kind": "multiclass",
        "description": "XGLUE news classification",
        "normalize": _single("text", "label"),
    },
    "tweet_sentiment_extraction": {
        "hf_id": "mteb/tweet_sentiment_extraction", "classes": 3, "kind": "multiclass",
        "description": "Tweet sentiment (negative/neutral/positive)",
        "normalize": _single("text", "label"),
    },
    "tweet_sentiment": {
        "hf_id": "cardiffnlp/tweet_eval", "hf_config": "sentiment", "classes": 3, "kind": "multiclass",
        "description": "Tweet sentiment, TweetEval benchmark (distinct corpus from tweet_sentiment_extraction)",
        "normalize": _single("text", "label"),
    },
    "tweet_emoji": {
        "hf_id": "cardiffnlp/tweet_eval", "hf_config": "emoji", "classes": 20, "kind": "multiclass",
        "description": "Tweet-to-emoji prediction",
        "normalize": _single("text", "label"),
    },
    "hate_speech_offensive": {
        "hf_id": "SetFit/hate_speech_offensive", "classes": 3, "kind": "multiclass",
        "description": "Hate/offensive/neither tweet classification (3-class, confirmed by sampling)",
        "normalize": _single("text", "label"),
    },
    "clinc_oos": {
        "hf_id": "clinc/clinc_oos", "hf_config": "plus", "classes": 151, "kind": "multiclass",
        "description": "CLINC150 intent classification plus out-of-scope queries",
        "normalize": _single("text", "intent"),
    },
    "mnli": {
        "hf_id": "SetFit/mnli", "classes": 3, "kind": "multiclass",
        "description": "MultiNLI entailment/neutral/contradiction",
        "normalize": _pair("text1", "text2"),
        "eval_split": "validation",  # "test" labels are GLUE's hidden -1 sentinel
    },
}

MULTILABEL_DATASETS: dict[str, dict[str, Any]] = {
    # Populated separately once the multi-label training/injection engine exists --
    # see benchmark_multilabel_matrix.py. Real, verified candidates already identified:
    # go_emotions (simplified + raw), civil_comments, PubMed MeSH, Jigsaw toxicity
    # (Arsive), lex_glue/eurlex -- 6 datasets, most needing a one-hot-columns-to-label-list
    # transform this dict's plain per-row normalize lambdas don't support yet.
}


def load_normalized(spec: dict[str, Any]):
    from datasets import load_dataset

    source = load_dataset(spec["hf_id"], spec["hf_config"]) if spec.get("hf_config") else load_dataset(spec["hf_id"])
    fn: Callable[[dict], dict] = spec["normalize"]
    normalize_row = lambda row: fn(row)  # noqa: E731
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
    gemini_model: str = "gemini-3.5-flash-lite",
) -> dict[str, Any]:
    classes = spec["classes"]
    clean_train = stratified_rows(train_split, scale, classes, ["text"], 100 + dataset_offset, "balanced")
    clean_test = stratified_rows(test_split, min(TEST_ROWS, len(test_split)), classes, ["text"], 200 + dataset_offset, "balanced")
    dirty_raw, optimizer_input, defects = make_dirty(clean_train, classes, 300 + dataset_offset)

    clean_optimized, clean_quality = await optimize(clean_train)
    dirty_optimized, dirty_quality = await optimize(
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
            runs.append(train_once(rows, clean_test, vocab, classes, seed, epochs))
            print(
                f"  [{dataset_name} @ {scale}] {name} seed={seed} "
                f"acc={runs[-1]['accuracy']:.4f} ({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
        results[name] = {"rows": len(rows), "runs": runs, "summary": summarize(runs)}

    dirty_recovery = {
        "accuracy": results["dirty_optimized"]["summary"]["accuracy_mean"] - results["dirty_raw"]["summary"]["accuracy_mean"],
        "macro_f1": results["dirty_optimized"]["summary"]["macro_f1_mean"] - results["dirty_raw"]["summary"]["macro_f1_mean"],
    }
    clean_delta = {
        "accuracy": results["clean_optimized"]["summary"]["accuracy_mean"] - results["clean_raw"]["summary"]["accuracy_mean"],
    }
    gemini_audit = None
    if gemini_audit_rows:
        score_stage = dirty_quality["stages"].get("score_refine", {})
        gemini_audit = {
            "rows_scored": score_stage.get("scored", 0),
            "average_overall_score": score_stage.get("average_overall_score"),
            "remote_batches": score_stage.get("remote_batches", 0),
            "model": dirty_quality["model"],
            "usage": dirty_quality["usage"],
        }
    # Per-stage accepted/rejected breakdown from the real OptimizationPipeline run --
    # which stage (validate vs. dedup) actually caught each defect category, not an
    # inference from the injection ratios.
    dirty_stage_breakdown = {
        "input_rows": dirty_quality["input_rows"],
        "output_rows": dirty_quality["output_rows"],
        "rejected_rows": dirty_quality["rejected_rows"],
        "validate": dirty_quality["stages"].get("validate"),
        "dedup": dirty_quality["stages"].get("dedup"),
    }
    clean_stage_breakdown = {
        "input_rows": clean_quality["input_rows"],
        "output_rows": clean_quality["output_rows"],
        "rejected_rows": clean_quality["rejected_rows"],
        "validate": clean_quality["stages"].get("validate"),
        "dedup": clean_quality["stages"].get("dedup"),
    }
    return {
        "dataset": dataset_name,
        "scale": scale,
        "defects": defects,
        "conditions": results,
        "dirty_optimized_minus_dirty_raw": dirty_recovery,
        "clean_optimized_minus_clean_raw": clean_delta,
        "gemini_audit": gemini_audit,
        "dirty_stage_breakdown": dirty_stage_breakdown,
        "clean_stage_breakdown": clean_stage_breakdown,
    }


def markdown_report(payload: dict[str, Any]) -> str:
    audited = any(combo.get("gemini_audit") for combo in payload["results"])
    header = (
        "| Dataset | Scale | Dirty raw acc | Dirty optimized acc | Recovery | Clean raw acc | Clean optimized acc | Clean delta |"
        + (" Gemini audit (dirty-optimized) |" if audited else "")
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|" + (":---|" if audited else "")
    lines = [
        "# BuffData 5-dataset x 3-scale accuracy-recovery matrix",
        "",
        f"Seeds: {payload['method']['seeds']} | Epochs: {payload['method']['epochs']} | "
        f"Test rows/dataset: {payload['method']['test_rows']}"
        + (" | Gemini sample-audits the dirty-optimized condition only." if audited else " | No LLM/API calls used."),
        "",
        header,
        divider,
    ]
    for combo in payload["results"]:
        c = combo["conditions"]
        row = (
            f"| {combo['dataset']} | {combo['scale']:,} | "
            f"{c['dirty_raw']['summary']['accuracy_mean']:.4f} | "
            f"{c['dirty_optimized']['summary']['accuracy_mean']:.4f} | "
            f"{combo['dirty_optimized_minus_dirty_raw']['accuracy']:+.4f} | "
            f"{c['clean_raw']['summary']['accuracy_mean']:.4f} | "
            f"{c['clean_optimized']['summary']['accuracy_mean']:.4f} | "
            f"{combo['clean_optimized_minus_clean_raw']['accuracy']:+.4f} |"
        )
        if audited:
            audit = combo.get("gemini_audit")
            if audit:
                score = audit["average_overall_score"]
                score_text = f"{score:.2f}/10" if score is not None else "n/a"
                row += f" {audit['rows_scored']} rows, {score_text}, {audit['usage']['total_tokens']} tokens |"
            else:
                row += " - |"
        lines.append(row)
    lines += [
        "",
        "'Dirty raw' trains on the source data plus disclosed conflicting-label duplicates (40%), "
        "class-skew duplicates (40%), and empty rows (10%). 'Dirty optimized' is BuffData's cleaned "
        "output from that exact contaminated input (validate + exact dedup only -- Gemini's sample "
        "audit, when enabled, is informational and does not decide which rows are kept). "
        "'Clean delta' is a control: it should stay near zero, showing BuffData does not damage "
        "already-clean data.",
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
        train_split, test_split = load_normalized(spec)
        for scale in (args.scales or SCALES):
            started = time.perf_counter()
            try:
                combo = await run_combo(
                    name, spec, scale, args.seeds, args.epochs, train_split, test_split, dataset_offset,
                    gemini_audit_rows=args.gemini_audit_rows, gemini_model=args.gemini_model,
                )
            except ValueError as exc:
                # A real dataset can be smaller than a requested scale, or too
                # class-imbalanced to support a balanced sample at that size --
                # stratified_rows raises ValueError rather than silently upsampling or
                # returning a smaller-than-requested result. Skip just this one
                # combination and keep the rest of the run going rather than losing
                # hours of already-completed work over one dataset@scale that doesn't fit.
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
    parser.add_argument("--output-dir", default="benchmarks/results-scale-matrix")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--scales", type=int, nargs="+", choices=SCALES)
    parser.add_argument(
        "--gemini-audit-rows", type=int, default=0,
        help="Sample-audit N rows of the dirty-optimized condition per combo with Gemini (needs GEMINI_API_KEY); 0 stays fully offline",
    )
    parser.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    asyncio.run(main(parser.parse_args()))
