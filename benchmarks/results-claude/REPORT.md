# Claude benchmark (Anthropic)

Provider: `anthropic` | Model: `claude-sonnet-5` | Base URL: `-` | Seeds: [17, 29] | Epochs: 2 |
Audit rows: 0 | Classify sample: 20

## Recovery (Claude-audited dirty_optimized vs dirty_raw)

| Dataset | Classes | Dirty raw | Dirty optimized | Recovery | Clean delta |
|---|---|---:|---:|---:|---:|
| ag_news | 4 | 0.6590 | 0.6180 | -0.0410 | +0.0000 |
| imdb | 2 | 0.5920 | 0.6850 | +0.0930 | +0.0000 |

## Claude usage per dataset

| Dataset | Audit tokens (in/out/total) | Audit score | Classify tokens |
|---|---|---|---|
| ag_news | 0/0/0 | None | 0 |
| imdb | 0/0/0 | None | 0 |
