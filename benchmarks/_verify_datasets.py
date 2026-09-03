#!/usr/bin/env python3
"""Verify candidate Hugging Face datasets for the binary/multi-class/multi-label CLI
classification benchmark, without downloading full data -- uses
load_dataset_builder().info to inspect schema and split sizes.
"""
from __future__ import annotations

import json

from datasets import ClassLabel, Sequence, Value, load_dataset_builder

LABEL_COL_CANDIDATES = ("label", "labels", "topic", "answer", "category", "class")

# (hf_id, config_or_None, expected_type)
CANDIDATES = [
    # --- binary (SetFit re-hosts + originals known to load) ---
    ("stanfordnlp/imdb", None, "binary"),
    ("fancyzhx/yelp_polarity", None, "binary"),
    ("fancyzhx/amazon_polarity", None, "binary"),
    ("stanfordnlp/sst2", None, "binary"),
    ("SetFit/sst2", None, "binary"),
    ("SetFit/imdb", None, "binary"),
    ("SetFit/amazon_polarity", None, "binary"),
    ("SetFit/enron_spam", None, "binary"),
    ("SetFit/subj", None, "binary"),
    ("SetFit/CR", None, "binary"),
    ("SetFit/SentEval-CR", None, "binary"),
    ("SetFit/mrpc", None, "binary"),
    ("SetFit/qqp", None, "binary"),
    ("SetFit/qnli", None, "binary"),
    ("SetFit/rte", None, "binary"),
    ("SetFit/wnli", None, "binary"),
    ("SetFit/hate_speech18", None, "binary"),
    ("SetFit/hate_speech_offensive", None, "binary"),
    ("SetFit/ethos_binary", None, "binary"),
    ("SetFit/toxic_conversations", None, "binary"),
    ("SetFit/toxic_conversations_50k", None, "binary"),
    ("SetFit/insincere-questions", None, "binary"),
    ("SetFit/ade_corpus_v2_classification", None, "binary"),
    ("SetFit/onestop_english", None, "multiclass"),
    ("SetFit/wsc_fixed", None, "binary"),
    ("nyu-mll/glue", "sst2", "binary"),
    ("nyu-mll/glue", "qqp", "binary"),
    ("cardiffnlp/tweet_eval", "offensive", "binary"),
    ("cardiffnlp/tweet_eval", "emotion", "multiclass"),
    ("cornell-movie-review-data/rotten_tomatoes", None, "binary"),
    # --- multi-class ---
    ("fancyzhx/ag_news", None, "multiclass"),
    ("fancyzhx/dbpedia_14", None, "multiclass"),
    ("community-datasets/yahoo_answers_topics", None, "multiclass"),
    ("dair-ai/emotion", None, "multiclass"),
    ("SetFit/20_newsgroups", None, "multiclass"),
    ("SetFit/emotion", None, "multiclass"),
    ("SetFit/ag_news", None, "multiclass"),
    ("SetFit/bbc-news", None, "multiclass"),
    ("SetFit/sst5", None, "multiclass"),
    ("SetFit/yelp_review_full", None, "multiclass"),
    ("SetFit/TREC-QC", None, "multiclass"),
    ("SetFit/student-question-categories", None, "multiclass"),
    ("SetFit/tweet_eval_stance", None, "multiclass"),
    ("SetFit/amazon_massive_scenario_en-US", None, "multiclass"),
    ("SetFit/amazon_massive_intent_en-US", None, "multiclass"),
    ("SetFit/amazon_reviews_multi_en", None, "multiclass"),
    ("SetFit/ethos", None, "multiclass"),
    # --- multi-label ---
    ("google-research-datasets/go_emotions", "simplified", "multilabel"),
    ("google-research-datasets/go_emotions", "raw", "multilabel"),
    ("SetFit/go_emotions", None, "multilabel"),
    ("argilla/go_emotions_multi-label", None, "multilabel"),
    ("owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH", None, "multilabel"),
    ("google/civil_comments", None, "multilabel"),
    ("google/jigsaw_toxicity_pred", None, "multilabel"),
    ("mteb/toxic_conversations_50k", None, "binary"),
    ("Arsive/toxicity_classification_jigsaw", None, "multilabel"),
    ("SetFit/ethos", "multilabel", "multilabel"),
]

results = []
for hf_id, config, expected in CANDIDATES:
    entry = {"hf_id": hf_id, "config": config, "expected": expected}
    try:
        builder = load_dataset_builder(hf_id, config) if config else load_dataset_builder(hf_id)
        info = builder.info
        features = info.features or {}
        splits = info.splits
        train_split = None
        if splits:
            for name in ("train", "training"):
                if name in splits:
                    train_split = name
                    break
        train_rows = splits[train_split].num_examples if train_split else None
        entry["train_rows"] = train_rows
        entry["columns"] = list(features.keys())

        label_col = None
        label_type = None  # "single" or "multi"
        num_classes = None
        for name in LABEL_COL_CANDIDATES:
            if name in features:
                feat = features[name]
                label_col = name
                if isinstance(feat, ClassLabel):
                    label_type, num_classes = "single", feat.num_classes
                elif isinstance(feat, Sequence) and isinstance(feat.feature, ClassLabel):
                    label_type, num_classes = "multi", feat.feature.num_classes
                elif isinstance(feat, Sequence):
                    label_type = "multi"
                elif isinstance(feat, (Value,)) and feat.dtype in ("int64", "int32", "bool"):
                    label_type = "single"  # plain int/bool label column (e.g. SetFit style)
                break
        entry["label_col"] = label_col
        entry["label_type"] = label_type
        entry["num_classes"] = num_classes
        entry["ok"] = bool(train_rows and train_rows >= 10000 and label_col)
        entry["error"] = None
    except Exception as exc:
        entry["ok"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"[:150]
    results.append(entry)
    status = "OK" if entry.get("ok") else "FAIL"
    print(f"[{status}] {hf_id} ({config}) expected={expected} -> "
          f"rows={entry.get('train_rows')} label_col={entry.get('label_col')} "
          f"label_type={entry.get('label_type')} classes={entry.get('num_classes')} "
          f"err={entry.get('error')}", flush=True)

with open("benchmarks/_dataset_verification.json", "w") as f:
    json.dump(results, f, indent=2)

ok = [r for r in results if r["ok"]]
print(f"\n{len(ok)}/{len(results)} candidates verified OK")
for kind in ("binary", "multiclass", "multilabel"):
    matching = [r for r in ok if r["expected"] == kind]
    print(f"  {kind}: {len(matching)} verified")
