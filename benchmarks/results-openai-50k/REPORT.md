# OpenAI benchmark

Provider: `openai` | Model: `gpt-5.6-sol` | Base URL: `-` | Seeds: [17, 29] | Epochs: 2 |
Audit rows: 0 | Classify sample: 20

## Recovery (dirty_optimized vs dirty_raw)

| Dataset | Classes | Dirty raw | Dirty optimized | Recovery | Clean delta |
|---|---|---:|---:|---:|---:|
| ag_news | 4 | 0.8775 | 0.9055 | +0.0280 | -0.0009 |

## Provider usage per dataset

| Dataset | Audit tokens (in/out/total) | Audit score | Classify tokens |
|---|---|---|---|
| ag_news | 0/0/0 | None | 0 |
