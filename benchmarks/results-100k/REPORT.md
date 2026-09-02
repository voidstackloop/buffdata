# BuffData real-dataset PyTorch benchmark

This benchmark uses fixed, deterministic samples of real Hugging Face records. The dirty
variants are controlled derivatives, not defects attributed to the dataset publishers.
Every condition uses the same vocabulary, architecture, hyperparameters, test set, and seeds.

| Dataset | Condition | Rows | Accuracy | Macro-F1 | Train seconds |
|---|---:|---:|---:|---:|---:|
| ag_news | clean_raw | 100000 | 0.8938 ± 0.0014 | 0.8936 ± 0.0014 | 34.61 |
| ag_news | clean_optimized | 99981 | 0.8942 ± 0.0027 | 0.8942 ± 0.0027 | 40.86 |
| ag_news | dirty_raw | 190000 | 0.8339 ± 0.0209 | 0.8342 ± 0.0209 | 80.15 |
| ag_news | dirty_optimized | 99981 | 0.8942 ± 0.0027 | 0.8942 ± 0.0027 | 42.26 |

**ag_news dirty-data recovery:** accuracy +0.0603; macro-F1 +0.0600.

BuffData processed 190000 rows and retained 99981; Gemini audit mode was `sampled` with 20 sampled scores. Mean sampled quality was 10.00/10; recorded Gemini usage was 5561 total tokens across 1 batched request(s).

## Interpretation boundaries

- The comparison proves behavior for these fixed samples, defects, model, and seeds—not every dataset or model.
- Schema parsing, validation, automatic task detection, and exact deduplication determine row inclusion in this run.
- Gemini performs a sampled semantic audit only; its scores are informational and do not cause the measured training recovery.
- Full per-row Gemini refinement is intentionally excluded because these are already-labeled classification records and sampled audit mode avoids unnecessary cost.
- The dirty benchmark is a controlled stress test; the clean control reveals whether processing clean data changes results.
