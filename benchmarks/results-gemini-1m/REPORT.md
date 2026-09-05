# gemini benchmark

Provider: `gemini` | Model: `gemini-3.7-flash` | Base URL: `-` | Seeds: [17, 29] | Epochs: 2 |
Audit rows: 20 | Classify sample: 20

## Recovery (dirty_optimized vs dirty_raw)

| Dataset | Classes | Dirty raw | Dirty optimized | Recovery | Clean delta |
|---|---|---:|---:|---:|---:|
| amazon_polarity | 2 | 0.8419 | 0.9070 | +0.0651 | +0.0000 |

## Provider usage per dataset

| Dataset | Audit tokens (in/out/total) | Audit score | Classify tokens |
|---|---|---|---|
| amazon_polarity | 4183/2293/7596 | 8.98 | 3637 |
