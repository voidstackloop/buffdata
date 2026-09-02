# BuffData 5-dataset x 3-scale accuracy-recovery matrix

Seeds: [17, 29, 43] | Epochs: 6 | Test rows/dataset: 2000 | No LLM/API calls used.

| Dataset | Scale | Dirty raw acc | Dirty optimized acc | Recovery | Clean raw acc | Clean optimized acc | Clean delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| ag_news | 10,000 | 0.7788 | 0.8753 | +0.0965 | 0.8753 | 0.8753 | +0.0000 |
| ag_news | 25,000 | 0.7965 | 0.8883 | +0.0918 | 0.8883 | 0.8883 | +0.0000 |
| ag_news | 50,000 | 0.8183 | 0.8863 | +0.0680 | 0.8873 | 0.8863 | -0.0010 |
| dbpedia_14 | 10,000 | 0.8622 | 0.9600 | +0.0978 | 0.9600 | 0.9600 | +0.0000 |
| dbpedia_14 | 25,000 | 0.8615 | 0.9688 | +0.1073 | 0.9688 | 0.9688 | +0.0000 |
| dbpedia_14 | 50,000 | 0.8667 | 0.9703 | +0.1037 | 0.9703 | 0.9703 | +0.0000 |
| yelp_polarity | 10,000 | 0.8132 | 0.8885 | +0.0753 | 0.8885 | 0.8885 | +0.0000 |
| yelp_polarity | 25,000 | 0.8233 | 0.8983 | +0.0750 | 0.8983 | 0.8983 | +0.0000 |
| yelp_polarity | 50,000 | 0.8542 | 0.9015 | +0.0473 | 0.9015 | 0.9015 | +0.0000 |
| amazon_polarity | 10,000 | 0.7905 | 0.8593 | +0.0688 | 0.8593 | 0.8593 | +0.0000 |
| amazon_polarity | 25,000 | 0.7918 | 0.8588 | +0.0670 | 0.8588 | 0.8588 | +0.0000 |
| amazon_polarity | 50,000 | 0.8252 | 0.8770 | +0.0518 | 0.8770 | 0.8770 | +0.0000 |
| yahoo_answers_topics | 10,000 | 0.5367 | 0.6292 | +0.0925 | 0.6292 | 0.6292 | +0.0000 |
| yahoo_answers_topics | 25,000 | 0.5253 | 0.6253 | +0.1000 | 0.6253 | 0.6253 | +0.0000 |
| yahoo_answers_topics | 50,000 | 0.5450 | 0.6227 | +0.0777 | 0.6227 | 0.6227 | +0.0000 |

'Dirty raw' trains on the source data plus disclosed conflicting-label duplicates (40%), class-skew duplicates (40%), and empty rows (10%). 'Dirty optimized' is BuffData's cleaned output from that exact contaminated input (validate + exact dedup only, no LLM calls). 'Clean delta' is a control: it should stay near zero, showing BuffData does not damage already-clean data.