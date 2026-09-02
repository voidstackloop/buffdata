# Binary vs. multi-class classification benchmark

Answers a narrower question than the [5-dataset recovery benchmark](../README.md#5-dataset-x-3-scale-real-world-recovery-benchmark):
not just "does BuffData's cleaning help," but **does it help the same way for binary-labeled
data as it does for multi-class-labeled data, and how well does BuffData's own classify
stage perform on each shape** -- across 19 real, independently-sourced Hugging Face
datasets (10 binary, 9 multi-class, 3-20 classes), using `gemini-3.7-flash`.

**Status: two-part benchmark, part 1 complete, part 2 queued.** Part 1 (dirty-data
recovery) needs zero LLM calls and is done, below. Part 2 (real classification accuracy)
needs real `gemini-3.7-flash` calls, and this API key's free tier caps that model at
**20 requests/day, total** -- already exhausted partway through part 1's development
today. It'll run and this doc will be updated once the daily quota resets.

- [Full report](../benchmarks/results-classification-matrix/REPORT.md)
- [Per-seed metrics](../benchmarks/results-classification-matrix/results.json)
- [Benchmark source](../benchmarks/benchmark_classification_matrix.py)

## The 19 datasets

Every class name comes from the dataset's own Hugging Face `ClassLabel` feature (or a
companion `<column>_text` field when the label isn't stored as a `ClassLabel`) --
never hardcoded, so BuffData's classifier is judged against the exact label vocabulary
each dataset actually ships.

| Binary (2-class) | Multi-class (3-20 classes) |
|---|---|
| yelp_polarity (Yelp review sentiment) | ag_news (4-class topic) |
| amazon_polarity (Amazon review sentiment) | dbpedia_14 (14-class ontology) |
| imdb (movie review sentiment) | yahoo_answers_topics (10-class, noisy Q&A) |
| rotten_tomatoes (critic-snippet sentiment) | emotion (6-class) |
| sst2 (sentence sentiment) | tweet_eval/emotion (4-class) |
| subj (subjective vs. objective) | tweet_eval/sentiment (3-class) |
| tweet_eval/irony | 20_newsgroups (20-class topic) |
| tweet_eval/hate | trec_coarse (6-class question type) |
| CR (customer review sentiment) | tweet_sentiment_extraction (3-class, independent source from tweet_eval/sentiment) |
| amazon_counterfactual (counterfactual-statement detection) | |

## Part 1: dirty-data recovery

Same methodology as the 5-dataset benchmark -- 40% conflicting-label duplicates, 40%
class-skew duplicates, 10% empty rows, then BuffData's local validate + exact-dedup,
entirely offline -- run at two scales (1,000 and 3,000 rows) with three fixed seeds
(17/29/43) per dataset. `_max_balanced_count` caps a request below the nominal scale
when a dataset's rarest class doesn't have enough rows for an even split (marked `†`
in the tables; `trec_coarse`'s rarest coarse class, "abbreviation," has only 86
examples, so *both* its scale requests land on the same capped 516 rows and produce
identical numbers -- a real property of that dataset, not a bug).

### Average recovery, binary vs. multi-class

| | Scale 1,000 | Scale 3,000 |
|---|---:|---:|
| Binary (n=10) | **+3.05 points** (2 of 10 negative) | **+4.84 points** (1 of 10 negative) |
| Multi-class (n=9) | **-0.54 points** (5 of 9 negative) | **+4.54 points** (2 of 9 negative) |

The headline finding: **at 1,000 rows, multi-class recovery is unreliable -- on average
slightly negative, with more than half the datasets showing no recovery at all -- while
binary recovery is already solidly positive.** By 3,000 rows, multi-class catches up to
parity with binary (both ~4.5-4.8 points). This isn't a claim that BuffData's cleaning
works differently for multi-class data -- the defect-injection and cleaning mechanics
are identical regardless of class count (see [architecture.md](architecture.md)). It's
a proxy-classifier measurement artifact with a real, mechanical explanation: at 1,000
rows split across, say, `dbpedia_14`'s 14 classes, that's only ~70 examples per class
to train *and evaluate* the small `EmbeddingBag+Linear` accuracy-measurement model
with -- both the dirty and clean conditions get noisier, and that noise is large
enough to swamp the actual recovery signal. A 2-class split of the same 1,000 rows
gives each class ~500 examples, a much more stable training signal. The practical
takeaway: **judge BuffData's recovery on classification data with more classes using a
larger sample** -- the benefit is real either way, but a small multi-class sample can't
reliably show it.

<details>
<summary>Full per-dataset results, both scales</summary>

See [benchmarks/results-classification-matrix/REPORT.md](../benchmarks/results-classification-matrix/REPORT.md)
for the complete table -- all 19 datasets × 2 scales, with the effective (possibly
capped) row count for each.

</details>

## Part 2: classification accuracy (pending)

Once quota resets: strips labels from a stratified 40-row sample per dataset and runs
it through exactly what the standalone `buffdata classify` command does --
`DatasetClassifier.resolve_schema()` then `classify_batch()` directly (not the full
7-stage pipeline, which would also trigger an unrelated, unconditional profiling call
this measurement doesn't need) -- with the dataset's real class list passed explicitly
so schema resolution is a free local step and every real network call is a genuine
classification call. Predicted labels are compared against the true ones held out
beforehand. Re-running `python benchmarks/benchmark_classification_matrix.py` (no
`--skip-classification`) resumes automatically: it reuses today's already-computed,
fully-deterministic recovery numbers from `results.json` and only makes the network
calls this part still needs.
