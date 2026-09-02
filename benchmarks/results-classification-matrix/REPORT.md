# Binary vs. multi-class classification benchmark

Model: `gemini-3.7-flash` | Scales: [1000, 3000] | Seeds: [17, 29, 43] | Classification sample size: 40

## Classification accuracy

Not run this pass (`--skip-classification`).

## Dirty-data recovery (local only, no LLM)

"Rows" is the actual balanced training size used, which can be smaller than the scale header: `_max_balanced_count` caps every request at `smallest_class_size × num_classes` so a stratified sample never asks a rare class for more rows than it has (see trec_coarse below, whose rarest coarse class has only 86 examples -- both scale requests land on the same capped, and therefore identical, result).

### Scale 1000

| Dataset | Kind | Rows | Dirty raw acc | Dirty optimized acc | Recovery | Clean delta |
|---|---|---:|---:|---:|---:|---:|
| yelp_polarity | binary | 1000 | 0.7840 | 0.8387 | +0.0547 | +0.0000 |
| amazon_polarity | binary | 1000 | 0.7133 | 0.7660 | +0.0527 | +0.0000 |
| imdb | binary | 1000 | 0.7433 | 0.8147 | +0.0713 | +0.0000 |
| rotten_tomatoes | binary | 1000 | 0.6467 | 0.6513 | +0.0047 | +0.0000 |
| sst2 | binary | 1000 | 0.6540 | 0.6480 | -0.0060 | +0.0000 |
| subj | binary | 1000 | 0.7840 | 0.8300 | +0.0460 | +0.0000 |
| tweet_eval_irony | binary | 1000 | 0.5827 | 0.6160 | +0.0333 | +0.0000 |
| tweet_eval_hate | binary | 1000 | 0.5353 | 0.5220 | -0.0133 | +0.0047 |
| cr | binary | 1000 | 0.7132 | 0.7451 | +0.0319 | +0.0000 |
| amazon_counterfactual | binary | 1000 | 0.7873 | 0.8173 | +0.0300 | +0.0000 |
| ag_news | multi | 1000 | 0.6873 | 0.7193 | +0.0320 | +0.0000 |
| dbpedia_14 | multi | 1000 | 0.7940 | 0.7733 | -0.0207 | +0.0000 |
| yahoo_answers_topics | multi | 1000 | 0.4373 | 0.4153 | -0.0220 | +0.0000 |
| emotion | multi | 1000 | 0.4175 | 0.4015 | -0.0160 | +0.0000 |
| tweet_eval_emotion | multi | 1000 | 0.4390 | 0.4289 | -0.0102 | +0.0000 |
| tweet_eval_sentiment | multi | 1000 | 0.4407 | 0.4440 | +0.0033 | +0.0000 |
| 20_newsgroups | multi | 1000 | 0.3173 | 0.3247 | +0.0073 | -0.0087 |
| trec_coarse | multi | 516 † | 0.6975 | 0.6728 | -0.0247 | +0.0000 |
| tweet_sentiment_extraction | multi | 1000 | 0.5120 | 0.5140 | +0.0020 | +0.0000 |

### Scale 3000

| Dataset | Kind | Rows | Dirty raw acc | Dirty optimized acc | Recovery | Clean delta |
|---|---|---:|---:|---:|---:|---:|
| yelp_polarity | binary | 3000 | 0.8027 | 0.8727 | +0.0700 | +0.0000 |
| amazon_polarity | binary | 3000 | 0.7773 | 0.8233 | +0.0460 | +0.0000 |
| imdb | binary | 3000 | 0.7680 | 0.8607 | +0.0927 | +0.0047 |
| rotten_tomatoes | binary | 3000 | 0.6713 | 0.7013 | +0.0300 | +0.0000 |
| sst2 | binary | 3000 | 0.7020 | 0.7520 | +0.0500 | -0.0020 |
| subj | binary | 3000 | 0.8260 | 0.9067 | +0.0807 | +0.0000 |
| tweet_eval_irony | binary | 2834 † | 0.5787 | 0.6053 | +0.0267 | +0.0000 |
| tweet_eval_hate | binary | 3000 | 0.5813 | 0.5647 | -0.0167 | -0.0093 |
| cr | binary | 2458 † | 0.7145 | 0.7708 | +0.0564 | +0.0110 |
| amazon_counterfactual | binary | 1054 † | 0.7633 | 0.8120 | +0.0487 | +0.0000 |
| ag_news | multi | 3000 | 0.7420 | 0.8120 | +0.0700 | +0.0000 |
| dbpedia_14 | multi | 3000 | 0.8233 | 0.9173 | +0.0940 | +0.0000 |
| yahoo_answers_topics | multi | 3000 | 0.5040 | 0.5500 | +0.0460 | +0.0000 |
| emotion | multi | 3000 | 0.5976 | 0.6961 | +0.0985 | -0.0118 |
| tweet_eval_emotion | multi | 1176 † | 0.4797 | 0.4661 | -0.0136 | -0.0257 |
| tweet_eval_sentiment | multi | 3000 | 0.4627 | 0.4687 | +0.0060 | +0.0000 |
| 20_newsgroups | multi | 3000 | 0.4647 | 0.5373 | +0.0727 | -0.0127 |
| trec_coarse | multi | 516 † | 0.6975 | 0.6728 | -0.0247 | +0.0000 |
| tweet_sentiment_extraction | multi | 3000 | 0.5480 | 0.6073 | +0.0593 | +0.0027 |

† capped below the requested scale by the smallest class's available rows.