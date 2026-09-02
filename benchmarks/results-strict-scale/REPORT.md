# Strict accuracy contract at 20k and 100k scale

Each generated condition passed through BuffData strict mode while its config
requested all PII redaction, minhash deduplication, LLM filtering, and forced
reclassification. Strict mode overrode every mutating stage. Original and generated
conditions use the same AG News test set, vocabulary, model, epochs, and seeds.

| Train rows | Condition | Accuracy | Macro-F1 | Text/label identical |
|---:|---|---:|---:|---:|
| 20,000 | original | 0.8811 ± 0.0016 | 0.8811 ± 0.0016 | yes |
| 20,000 | strict_generated | 0.8811 ± 0.0016 | 0.8811 ± 0.0016 | yes |

| 100,000 | original | 0.8964 ± 0.0011 | 0.8962 ± 0.0011 | yes |
| 100,000 | strict_generated | 0.8964 ± 0.0011 | 0.8962 ± 0.0011 | yes |

## Gates

- 20k strict parity: **PASS**
- 100k strict parity: **PASS**

A passing gate requires the complete valid labeled row sequence, every classification
text, every label, accuracy, and macro-F1 to match the original exactly.
