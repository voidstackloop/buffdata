# OpenAI benchmark

Provider: `openai` | Model: `gpt-5.6-sol` | Base URL: `-` | Seeds: [17, 29] | Epochs: 2 |
Audit rows: 0 | Classify sample: 20

## Recovery (dirty_optimized vs dirty_raw)

| Dataset | Classes | Dirty raw | Dirty optimized | Recovery | Clean delta |
|---|---|---:|---:|---:|---:|
| ag_news | 4 | 0.8642 | 0.9020 | +0.0377 | +0.0000 |
| imdb | 2 | 0.7565 | 0.8655 | +0.1090 | +0.0015 |

## Provider usage per dataset

| Dataset | Audit tokens (in/out/total) | Audit score | Classify tokens |
|---|---|---|---|
| ag_news | 0/0/0 | None | 0 |
| imdb | 0/0/0 | None | 0 |
