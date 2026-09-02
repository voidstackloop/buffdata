# Reproducible real-dataset benchmark

This benchmark downloads fixed stratified slices from `fancyzhx/ag_news` and
`stanfordnlp/imdb`, generates disclosed dirty variants, applies BuffData's local
validation/task-detection/exact-deduplication stages, and trains identical PyTorch
classifiers across three fixed seeds.

```bash
cd ~/projects/buffdata
source .venv/bin/activate
python benchmarks/benchmark_buffdata.py
```

Outputs are written to `benchmarks/results/`, including the exact JSONL inputs,
per-seed metrics in `results.json`, and a concise `REPORT.md`.

For the larger Gemini-audited suite (12,000 training records per dataset):

```bash
python benchmarks/benchmark_buffdata.py \
  --datasets dbpedia_14 yelp_polarity emotion \
  --train-rows 12000 --test-rows 2000 --epochs 6 \
  --gemini-audit-rows 20 --output-dir benchmarks/results-large
```

`--gemini-audit-rows` uses BuffData's scalable batched-sample quality mode. Validation,
deduplication, and task detection still process every row; Gemini judges only a
deterministic representative sample (20 rows per structured request) and its token
usage is recorded in the report.

The local benchmark does not require an API key. After `GEMINI_API_KEY` is set,
the provider-backed optimizer can be tested separately on generative datasets;
it is deliberately not mixed into this classification benchmark because LLM
scoring/refinement and deterministic validation/deduplication test different claims.

## Accuracy regression gate

The regression benchmark explicitly compares matching original records, the
previously generated output that lost accuracy, and the repaired generated output.
It also compares a contaminated original dataset with the output BuffData generates
from that same input. Both panels use 3,000+ training rows, the same 2,000-row test
set, shared vocabulary, and fixed seeds.

```bash
python benchmarks/benchmark_accuracy_regression.py
```

The command exits unsuccessfully if the repair does not recover accuracy, if the
generated clean control is not text-identical to and at least as accurate as the
original, or if BuffData fails to improve the contaminated-data baseline.

After the small gate passes, reproduce the 20k+ and 100k+ scale runs with:

```bash
python benchmarks/benchmark_buffdata.py \
  --datasets ag_news --train-rows 20000 --test-rows 4000 --epochs 6 \
  --gemini-audit-rows 20 --output-dir benchmarks/results-20k

python benchmarks/benchmark_buffdata.py \
  --datasets ag_news --train-rows 100000 --test-rows 7000 --epochs 6 \
  --gemini-audit-rows 20 --output-dir benchmarks/results-100k
```

The reports are stored in `benchmarks/results-20k/REPORT.md` and
`benchmarks/results-100k/REPORT.md`; per-seed metrics and provider token usage are
stored beside each report in `results.json`.

For the strict clean-data parity proof at both scales:

```bash
python benchmarks/benchmark_strict_scale.py
```

This trains original and strict-generated conditions at 20,000 and 100,000 rows.
The run fails unless row order, classification text, labels, accuracy, and macro-F1
are identical. Results are written to `benchmarks/results-strict-scale/`.
