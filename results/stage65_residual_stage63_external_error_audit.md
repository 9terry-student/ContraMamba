# Stage65 — Residual Stage63 External Error Audit

## Decision

`STAGE65_RESIDUAL_STAGE63_EXTERNAL_ERROR_AUDIT_READY`

## Summary

| stage   | decision                                            | run_dir                                                               | prediction_file                                                                                                                                                                            |   row_count |   correct_count |   error_count |   accuracy_from_predictions |   stage63_report_accuracy |   stage63_report_macro_f1 | stage63_dataset_decision              | stage63_aggregate_decision                      | major_error_type    |   false_ne_total |   polarity_error_total |   false_entitlement_total | recommended_focus                                                                                                                                                                                                                                                                                                                                                                           | recommended_next_stage                   | allowed_claim                                                                                                                                   | forbidden_claim                                                                                           |
|:--------|:----------------------------------------------------|:----------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------:|----------------:|--------------:|----------------------------:|--------------------------:|--------------------------:|:--------------------------------------|:------------------------------------------------|:--------------------|-----------------:|-----------------------:|--------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
| Stage65 | STAGE65_RESIDUAL_STAGE63_EXTERNAL_ERROR_AUDIT_READY | results/stage63_bridge_enabled_vitaminc_external_eval_20260702_060044 | results/stage63_bridge_enabled_vitaminc_external_eval_20260702_060044/external/stage63_bridge_enabled_vitaminc_stage43b1_vitaminc_validation_sample1000_external_factver_predictions.jsonl |        1000 |             322 |           678 |                       0.322 |                     0.322 |                  0.315319 | STAGE43C0_EXTERNAL_FACTVER_INCOMPLETE | STAGE43C0_EXTERNAL_FACTVER_AGGREGATE_INCOMPLETE | false_NE_on_SUPPORT |              411 |                    203 |                        64 | ['increase entitlement-positive bridge coverage for gold SUPPORT/REFUTE still predicted NOT_ENTITLED', 'add polarity-disambiguating bridge rows for REFUTE/SUPPORT confusion', 'add stricter NE/frame-mismatch bridge rows to prevent false entitlement', 'inspect temporal/date-heavy residuals', 'inspect numeric/entity-attribute residuals', 'inspect comparative predicate residuals'] | Stage66 residual-bridge expansion design | Stage65 identifies the dominant remaining Stage63 external errors after bridge training; it does not introduce new training or external tuning. | Do not claim solved external generalization. Stage63 remains incomplete and residual errors remain large. |

## Confusion matrix long

| gold_label   | final_pred   |   count |
|:-------------|:-------------|--------:|
| NOT_ENTITLED | NOT_ENTITLED |      81 |
| NOT_ENTITLED | REFUTE       |      33 |
| NOT_ENTITLED | SUPPORT      |      31 |
| REFUTE       | NOT_ENTITLED |     166 |
| REFUTE       | REFUTE       |      85 |
| REFUTE       | SUPPORT      |     104 |
| SUPPORT      | NOT_ENTITLED |     245 |
| SUPPORT      | REFUTE       |      99 |
| SUPPORT      | SUPPORT      |     156 |

## Error type counts

| error_type                       |   count |   rate_total |   rate_errors_only |
|:---------------------------------|--------:|-------------:|-------------------:|
| correct                          |     322 |        0.322 |          0.474926  |
| false_NE_on_SUPPORT              |     245 |        0.245 |          0.361357  |
| false_NE_on_REFUTE               |     166 |        0.166 |          0.244838  |
| wrong_polarity_REFUTE_to_SUPPORT |     104 |        0.104 |          0.153392  |
| wrong_polarity_SUPPORT_to_REFUTE |      99 |        0.099 |          0.146018  |
| false_REFUTE_on_NE               |      33 |        0.033 |          0.0486726 |
| false_SUPPORT_on_NE              |      31 |        0.031 |          0.0457227 |

## Gold recall audit

| gold_label   |   gold_count |   correct |   recall |   false_NE_count |   false_NE_rate |
|:-------------|-------------:|----------:|---------:|-----------------:|----------------:|
| NOT_ENTITLED |          145 |        81 | 0.558621 |              nan |      nan        |
| REFUTE       |          355 |        85 | 0.239437 |              166 |        0.467606 |
| SUPPORT      |          500 |       156 | 0.312    |              245 |        0.49     |

## Prediction precision audit

| pred_label   |   pred_count |   correct |   precision |
|:-------------|-------------:|----------:|------------:|
| NOT_ENTITLED |          492 |        81 |    0.164634 |
| REFUTE       |          217 |        85 |    0.391705 |
| SUPPORT      |          291 |       156 |    0.536082 |

## Feature audit by error type

| error_type                       |   count | accuracy_bucket   |   has_digit_rate |   has_year_rate |   has_temporal_word_rate |   has_comparison_word_rate |   has_negation_rate |   avg_claim_len |   avg_evidence_len |
|:---------------------------------|--------:|:------------------|-----------------:|----------------:|-------------------------:|---------------------------:|--------------------:|----------------:|-------------------:|
| correct                          |     322 | correct           |         0.692547 |        0.391304 |                 0.298137 |                   0.381988 |          0.0434783  |         14.1739 |            27.4938 |
| false_NE_on_SUPPORT              |     245 | error             |         0.689796 |        0.444898 |                 0.367347 |                   0.2      |          0.0326531  |         11.3714 |            30.5878 |
| false_NE_on_REFUTE               |     166 | error             |         0.76506  |        0.487952 |                 0.343373 |                   0.295181 |          0.0180723  |         10.253  |            28.5181 |
| wrong_polarity_REFUTE_to_SUPPORT |     104 | error             |         0.75     |        0.509615 |                 0.423077 |                   0.288462 |          0.00961538 |         11.4423 |            28.6346 |
| wrong_polarity_SUPPORT_to_REFUTE |      99 | error             |         0.838384 |        0.333333 |                 0.282828 |                   0.606061 |          0.0505051  |         15.0101 |            27.0606 |
| false_REFUTE_on_NE               |      33 | error             |         0.909091 |        0.606061 |                 0.393939 |                   0.515152 |          0.0606061  |         17.7273 |            28.3333 |
| false_SUPPORT_on_NE              |      31 | error             |         0.645161 |        0.387097 |                 0.322581 |                   0.193548 |          0.0967742  |         18.5484 |            31.8387 |

## Residual samples

See `results/stage65_residual_stage63_external_error_samples.csv`.

## Recommended focus

- increase entitlement-positive bridge coverage for gold SUPPORT/REFUTE still predicted NOT_ENTITLED
- add polarity-disambiguating bridge rows for REFUTE/SUPPORT confusion
- add stricter NE/frame-mismatch bridge rows to prevent false entitlement
- inspect temporal/date-heavy residuals
- inspect numeric/entity-attribute residuals
- inspect comparative predicate residuals

## Allowed claim

Stage65 identifies the dominant remaining Stage63 external errors after bridge training; it does not introduce new training or external tuning.

## Forbidden claim

Do not claim solved external generalization. Stage63 remains incomplete and residual errors remain large.

## Recommended next stage

Stage66 residual-bridge expansion design
