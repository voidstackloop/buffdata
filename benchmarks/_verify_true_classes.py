#!/usr/bin/env python3
"""Get the TRUE full-dataset class count for every entry in DATASETS (benchmark_scale_matrix.py),
using dataset.unique() on the full train split -- not a partial sample, which can miss rare
classes (exactly what caused the amazon_massive_scenario bug: a 3000-row sample only showed
14 of the real 17+ classes)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_scale_matrix import DATASETS, load_normalized

for name, spec in DATASETS.items():
    try:
        train, test = load_normalized(spec)
        train_labels = set(train.unique("label"))
        test_labels = set(test.unique("label"))
        all_labels = train_labels | test_labels
        true_n = len(all_labels)
        declared_n = spec["classes"]
        min_label, max_label = min(all_labels), max(all_labels)
        contiguous = all_labels == set(range(min_label, max_label + 1))
        status = "OK" if true_n == declared_n and min_label == 0 and contiguous else "MISMATCH"
        print(f"[{status}] {name}: declared={declared_n} true={true_n} range=[{min_label},{max_label}] contiguous={contiguous} train_rows={len(train)}", flush=True)
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
