# Stage86 - Stage84F External Error Audit

## Decision

`STAGE86_EXTERNAL_ERROR_AUDIT_READY`

## Next direction

`external_entitlement_recovery_without_NE_safety_expansion`

## Summary

| stage   | decision                           | source_predictions                                                                                                                    |    n | stage73_reference                                                                                                                    | stage84f_observed                                                                                                                                                                                                                   | error_counts                                                                                                                                                                 | derived_error_counts                                                                                                                                                                                                      | diagnosis_rules                                                                                                                                                                                                                              | next_direction                                            | primary_policy       | rejected_policy         |
|:--------|:-----------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|-----:|:-------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|:---------------------|:------------------------|
| Stage86 | STAGE86_EXTERNAL_ERROR_AUDIT_READY | results/stage84f_stage83c_vitaminc_external_exact_ood_schema_run_20260703_033759/stage84f_stage83c_vitaminc_external_predictions.json | 1000 | {"external_acc": 0.353, "external_macro_f1_all3": 0.3262, "prediction_counts": {"NOT_ENTITLED": 388, "REFUTE": 219, "SUPPORT": 393}} | {"external_acc": 0.32600000500679016, "external_macro_f1_all3": 0.3066716953259303, "prediction_counts": {"NOT_ENTITLED": 439, "REFUTE": 202, "SUPPORT": 359}, "gold_counts": {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500}} | {"false_NE_on_SUPPORT": 205, "false_NE_on_REFUTE": 162, "false_SUPPORT_on_NE": 42, "false_REFUTE_on_NE": 31, "false_SUPPORT_on_REFUTE": 128, "false_REFUTE_on_SUPPORT": 106} | {"false_NE_total": 367, "false_entitlement_total": 73, "polarity_error_total": 234, "false_SUPPORT_total": 170, "false_REFUTE_total": 137, "true_SUPPORT_correct": 189, "true_REFUTE_correct": 65, "true_NE_correct": 72} | ["dominant_error=false_NE_entitlement_suppression", "support_recall_suppression_ge_refute_recall_suppression", "NE_prediction_count_increased_vs_stage73", "SUPPORT_prediction_count_decreased_vs_stage73", "external_regression_confirmed"] | external_entitlement_recovery_without_NE_safety_expansion | KEEP_STAGE71_PRIMARY | DO_NOT_PROMOTE_STAGE83C |

## Confusion matrix

| gold\pred    |   REFUTE |   NOT_ENTITLED |   SUPPORT |
|:-------------|---------:|---------------:|----------:|
| REFUTE       |       65 |            162 |       128 |
| NOT_ENTITLED |       31 |             72 |        42 |
| SUPPORT      |      106 |            205 |       189 |

## Per-label metrics

| label        |   support |   pred_count |   tp |   fp |   fn |   precision |   recall |       f1 |
|:-------------|----------:|-------------:|-----:|-----:|-----:|------------:|---------:|---------:|
| REFUTE       |       355 |          202 |   65 |  137 |  290 |    0.321782 | 0.183099 | 0.233393 |
| NOT_ENTITLED |       145 |          439 |   72 |  367 |   73 |    0.164009 | 0.496552 | 0.246575 |
| SUPPORT      |       500 |          359 |  189 |  170 |  311 |    0.526462 | 0.378    | 0.440047 |

## Error breakdown

| error_type              |   count |
|:------------------------|--------:|
| false_NE_on_SUPPORT     |     205 |
| false_NE_on_REFUTE      |     162 |
| false_SUPPORT_on_NE     |      42 |
| false_REFUTE_on_NE      |      31 |
| false_SUPPORT_on_REFUTE |     128 |
| false_REFUTE_on_SUPPORT |     106 |
| false_NE_total          |     367 |
| false_entitlement_total |      73 |
| polarity_error_total    |     234 |
| false_SUPPORT_total     |     170 |
| false_REFUTE_total      |     137 |
| true_SUPPORT_correct    |     189 |
| true_REFUTE_correct     |      65 |
| true_NE_correct         |      72 |
