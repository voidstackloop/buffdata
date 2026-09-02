# BuffData original-versus-generated accuracy gate

Every condition uses the same AG News test rows, vocabulary, PyTorch architecture,
hyperparameters, and three fixed random seeds. No LLM call is made by this benchmark.

## Regression repair (3,000 train / 2,000 test)

| Condition | Rows | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| original_dataset | 3000 | 0.8315 ± 0.0074 | 0.8307 ± 0.0076 |
| regressed_generated | 3000 | 0.8070 ± 0.0040 | 0.8061 ± 0.0037 |
| repaired_generated | 3000 | 0.8315 ± 0.0074 | 0.8307 ± 0.0076 |

The repaired generated data recovers **+2.45 accuracy points** and finishes +0.00 points from the matching original control.
It changes 0 texts, compared with 2,767 in the regressed output.

## Value test: contaminated original vs generated optimized (3,000 source / 2,000 test)

The original condition contains all 3,000 source rows plus disclosed conflicting-label
duplicates, class-skewing duplicates, and empty rows. BuffData generates the optimized
condition from the exact same input.

| Condition | Rows | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| original_contaminated | 5700 | 0.7515 ± 0.0088 | 0.7509 ± 0.0077 |
| generated_optimized | 3000 | 0.8425 ± 0.0088 | 0.8415 ± 0.0089 |

Generated optimized data improves accuracy by **+9.10 points** and macro-F1 by **+9.06 points**.

## Gates

- Regression recovery: **PASS**
- Strict original preservation (generated accuracy >= original and identical text): **PASS**
- Contaminated-data value: **PASS**

These results establish behavior for these fixed samples and this lightweight classifier;
they do not promise the same gain on every dataset or model.
