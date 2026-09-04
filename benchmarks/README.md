# Reproducible real-dataset benchmark

## Non-classification matrix

The generative matrix complements the label-classification suites with six large,
real Hugging Face datasets covering instruction/SFT, multi-turn chat, preference/DPO,
extractive QA, summarization, and raw language-modeling text. It compares every
unmodified source slice with BuffData's output, then appends 40% exact duplicates and
10% invalid records and repeats the comparison at 10,000 and 30,000 source rows.

```bash
python benchmarks/benchmark_generative_matrix.py \
  --scales 10000 30000 \
  --output-dir benchmarks/results-generative
```

The generated `REPORT.md` and `results.json` include rows retained/deleted, source
rows lost or changed, exact duplicates found and removed, invalid rows removed,
defect leakage, cleaning-decision accuracy, deletion precision/recall/F1, exact
output-content accuracy, clean-vs-dirty output parity, character preservation,
per-stage rejections, and throughput. Because these datasets are open-ended rather
than class-labeled, accuracy is defined against the clean optimized control instead
of inventing an inapplicable classifier label. LLM quality scoring and classification
are disabled, and provider token usage must remain zero.

For downstream before → after movement on disjoint held-out rows, run:

```bash
python benchmarks/benchmark_generative_utility.py \
  --scales 10000 30000 --eval-rows 250 \
  --output-dir benchmarks/results-generative
```

This writes `UTILITY_REPORT.md` and `utility_results.json`. Metrics are task-specific:
response token-F1 for instruction/chat, preference accuracy for DPO, answer exact
match/F1 for QA, ROUGE-L for summarization, and next-token accuracy/perplexity for
raw text. Each report shows clean raw → clean optimized and dirty raw → dirty optimized.

This benchmark downloads fixed stratified slices from **19 Hugging Face datasets**,
generates disclosed dirty variants, applies BuffData's local validation/task-detection/
exact-deduplication stages, and trains identical PyTorch classifiers across three fixed
seeds. The catalog spans sentiment, topic, emotion, hate/irony, question type, and
subjectivity tasks with 2 to 20 classes.

| Family | Datasets |
|---|---|
| Binary | `imdb`, `yelp_polarity`, `amazon_polarity`, `rotten_tomatoes`, `sst2`, `subj`, `tweet_eval_irony`, `tweet_eval_hate`, `cr`, `amazon_counterfactual` |
| Multi-class | `ag_news`, `dbpedia_14`, `emotion`, `tweet_eval_sentiment`, `yahoo_answers_topics`, `tweet_eval_emotion`, `newsgroups_20`, `trec_coarse`, `tweet_sentiment_extraction` |

```bash
cd ~/projects/buffdata
source .venv/bin/activate
python benchmarks/benchmark_buffdata.py
```

The default run is local after the Hugging Face downloads complete; it makes no LLM
API calls. To run a smaller slice while developing:

```bash
python benchmarks/benchmark_buffdata.py \
  --datasets ag_news imdb tweet_eval_sentiment \
  --train-rows 1000 --test-rows 500 --epochs 2
```

Outputs are written to `benchmarks/results/`, including the exact JSONL inputs,
per-seed metrics in `results.json`, and a concise `REPORT.md`. The report and JSON now
include both model-quality metrics and explicit data-hygiene accounting:

- input, retained, and deleted row counts plus deletion rate;
- duplicate rows and duplicate groups present in the pipeline input;
- conflicting-label duplicate rows/groups;
- invalid rows deleted by validation and duplicates deleted by exact dedup;
- clean source rows accidentally deleted;
- injected defects removed versus retained, with a per-category/per-stage cross-tab;
- unique-content and empty-text counts, plus exact rejection-reason totals.

`duplicate_rows_in_input` follows BuffData's own exact-dedup equivalence rule: trimmed
classification text, independent of label. `duplicate_rows_deleted` is the observed
dedup-stage result, so the report distinguishes duplicates that exist from rows that
were actually removed.

For the larger Gemini-audited suite (12,000 training records per dataset):

```bash
python benchmarks/benchmark_buffdata.py \
  --datasets dbpedia_14 yelp_polarity emotion \
  --train-rows 12000 --test-rows 2000 --epochs 6 \
  --gemini-audit-rows 20 --audit-provider gemini \
  --output-dir benchmarks/results-large
```

`--gemini-audit-rows` uses BuffData's scalable batched-sample quality mode. Validation,
deduplication, and task detection still process every row; Gemini judges only a
deterministic representative sample (20 rows per structured request) and its token
usage is recorded in the report.

The local benchmark does not require an API key. Setting `--gemini-audit-rows` above
zero enables the optional provider-backed quality audit and requires the corresponding
provider credential. Audit scoring is informational: deterministic validation and
deduplication remain the only stages deciding the row-retention numbers.

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
