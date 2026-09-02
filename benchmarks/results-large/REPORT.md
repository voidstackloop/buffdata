# BuffData real-dataset PyTorch benchmark

This benchmark uses fixed, deterministic samples of real Hugging Face records. The dirty
variants are controlled derivatives, not defects attributed to the dataset publishers.
Every condition uses the same vocabulary, architecture, hyperparameters, test set, and seeds.

| Dataset | Condition | Rows | Accuracy | Macro-F1 | Train seconds |
|---|---:|---:|---:|---:|---:|
| dbpedia_14 | clean_raw | 12000 | 0.9668 ± 0.0013 | 0.9669 ± 0.0013 | 4.09 |
| dbpedia_14 | clean_optimized | 12000 | 0.9668 ± 0.0013 | 0.9669 ± 0.0013 | 4.42 |
| dbpedia_14 | dirty_raw | 22800 | 0.8697 ± 0.0308 | 0.8692 ± 0.0315 | 8.42 |
| dbpedia_14 | dirty_optimized | 12000 | 0.9668 ± 0.0013 | 0.9669 ± 0.0013 | 4.13 |

**dbpedia_14 dirty-data recovery:** accuracy +0.0972; macro-F1 +0.0977.

BuffData processed 22800 rows and retained 12000; Gemini audit mode was `sampled` with 20 sampled scores. Mean sampled quality was 10.00/10; recorded Gemini usage was 6717 total tokens across 1 batched request(s).

| yelp_polarity | clean_raw | 12000 | 0.9060 ± 0.0023 | 0.9060 ± 0.0023 | 6.09 |
| yelp_polarity | clean_optimized | 12000 | 0.9060 ± 0.0023 | 0.9060 ± 0.0023 | 6.58 |
| yelp_polarity | dirty_raw | 22800 | 0.8353 ± 0.0165 | 0.8336 ± 0.0178 | 12.07 |
| yelp_polarity | dirty_optimized | 12000 | 0.9060 ± 0.0023 | 0.9060 ± 0.0023 | 6.33 |

**yelp_polarity dirty-data recovery:** accuracy +0.0707; macro-F1 +0.0724.

BuffData processed 22800 rows and retained 12000; Gemini audit mode was `sampled` with 20 sampled scores. Mean sampled quality was 10.00/10; recorded Gemini usage was 10125 total tokens across 1 batched request(s).

| emotion | clean_raw | 12000 | 0.8360 ± 0.0035 | 0.7789 ± 0.0080 | 1.53 |
| emotion | clean_optimized | 11978 | 0.8377 ± 0.0008 | 0.7805 ± 0.0022 | 1.61 |
| emotion | dirty_raw | 22800 | 0.7123 ± 0.0106 | 0.6350 ± 0.0113 | 2.90 |
| emotion | dirty_optimized | 11978 | 0.8377 ± 0.0008 | 0.7805 ± 0.0022 | 1.53 |

**emotion dirty-data recovery:** accuracy +0.1253; macro-F1 +0.1455.

BuffData processed 22800 rows and retained 11978; Gemini audit mode was `sampled` with 20 sampled scores. Mean sampled quality was 9.00/10; recorded Gemini usage was 4114 total tokens across 1 batched request(s).

## Interpretation boundaries

- The comparison proves behavior for these fixed samples, defects, model, and seeds—not every dataset or model.
- Schema parsing, validation, automatic task detection, and exact deduplication determine row inclusion in this run.
- Gemini performs a sampled semantic audit only; its scores are informational and do not cause the measured training recovery.
- Full per-row Gemini refinement is intentionally excluded because these are already-labeled classification records and sampled audit mode avoids unnecessary cost.
- The dirty benchmark is a controlled stress test; the clean control reveals whether processing clean data changes results.
