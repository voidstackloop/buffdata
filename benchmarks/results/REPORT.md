# BuffData real-dataset PyTorch benchmark

This benchmark uses fixed, stratified samples of real Hugging Face records. The dirty
variants are controlled derivatives, not defects attributed to the dataset publishers.
Every condition uses the same vocabulary, architecture, hyperparameters, test set, and seeds.

| Dataset | Condition | Rows | Accuracy | Macro-F1 | Train seconds |
|---|---:|---:|---:|---:|---:|
| ag_news | clean_raw | 3000 | 0.8390 ± 0.0052 | 0.8381 ± 0.0043 | 0.79 |
| ag_news | clean_optimized | 3000 | 0.8390 ± 0.0052 | 0.8381 ± 0.0043 | 0.76 |
| ag_news | dirty_raw | 5700 | 0.7530 ± 0.0148 | 0.7523 ± 0.0138 | 1.37 |
| ag_news | dirty_optimized | 3000 | 0.8390 ± 0.0052 | 0.8381 ± 0.0043 | 0.71 |

**ag_news dirty-data recovery:** accuracy +0.0860; macro-F1 +0.0858.

| imdb | clean_raw | 3000 | 0.8307 ± 0.0059 | 0.8306 ± 0.0059 | 1.50 |
| imdb | clean_optimized | 3000 | 0.8307 ± 0.0059 | 0.8306 ± 0.0059 | 1.47 |
| imdb | dirty_raw | 5700 | 0.7603 ± 0.0274 | 0.7557 ± 0.0314 | 2.73 |
| imdb | dirty_optimized | 3000 | 0.8307 ± 0.0059 | 0.8306 ± 0.0059 | 1.51 |

**imdb dirty-data recovery:** accuracy +0.0703; macro-F1 +0.0749.

## Interpretation boundaries

- The comparison proves behavior for these fixed samples, defects, model, and seeds—not every dataset or model.
- The local run validates BuffData schema parsing, validation, automatic task detection, and exact deduplication.
- Gemini scoring/refinement is intentionally excluded until a GEMINI_API_KEY is configured.
- The dirty benchmark is a controlled stress test; the clean control reveals whether processing clean data changes results.
