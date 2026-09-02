# BuffData real-dataset PyTorch benchmark

This benchmark uses fixed, deterministic samples of real Hugging Face records. The dirty
variants are controlled derivatives, not defects attributed to the dataset publishers.
Every condition uses the same vocabulary, architecture, hyperparameters, test set, and seeds.

| Dataset | Condition | Rows | Accuracy | Macro-F1 | Train seconds |
|---|---:|---:|---:|---:|---:|
| ag_news | clean_raw | 20000 | 0.8780 ± 0.0022 | 0.8781 ± 0.0023 | 6.24 |
| ag_news | clean_optimized | 20000 | 0.8780 ± 0.0022 | 0.8781 ± 0.0023 | 6.37 |
| ag_news | dirty_raw | 38000 | 0.8014 ± 0.0068 | 0.8015 ± 0.0067 | 11.83 |
| ag_news | dirty_optimized | 20000 | 0.8780 ± 0.0022 | 0.8781 ± 0.0023 | 6.60 |

**ag_news dirty-data recovery:** accuracy +0.0766; macro-F1 +0.0766.

BuffData processed 38000 rows and retained 20000; Gemini audit mode was `sampled` with 20 sampled scores. Mean sampled quality was 10.00/10; recorded Gemini usage was 5534 total tokens across 1 batched request(s).

## Interpretation boundaries

- The comparison proves behavior for these fixed samples, defects, model, and seeds—not every dataset or model.
- Schema parsing, validation, automatic task detection, and exact deduplication determine row inclusion in this run.
- Gemini performs a sampled semantic audit only; its scores are informational and do not cause the measured training recovery.
- Full per-row Gemini refinement is intentionally excluded because these are already-labeled classification records and sampled audit mode avoids unnecessary cost.
- The dirty benchmark is a controlled stress test; the clean control reveals whether processing clean data changes results.
