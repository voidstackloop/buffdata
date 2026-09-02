# BuffData 5-dataset x 3-scale accuracy-recovery matrix

Seeds: [17] | Epochs: 2 | Test rows/dataset: 2000 | No LLM/API calls used.

| Dataset | Scale | Dirty raw acc | Dirty optimized acc | Recovery | Clean raw acc | Clean optimized acc | Clean delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| ag_news | 10,000 | 0.7850 | 0.8800 | +0.0950 | 0.8800 | 0.8800 | +0.0000 |

'Dirty raw' trains on the source data plus disclosed conflicting-label duplicates (40%), class-skew duplicates (40%), and empty rows (10%). 'Dirty optimized' is BuffData's cleaned output from that exact contaminated input (validate + exact dedup only, no LLM calls). 'Clean delta' is a control: it should stay near zero, showing BuffData does not damage already-clean data.