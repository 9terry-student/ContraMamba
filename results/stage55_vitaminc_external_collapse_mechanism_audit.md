# Stage55 — VitaminC External Collapse Mechanism Audit

## Decision

`STAGE55_VITAMINC_EXTERNAL_COLLAPSE_MECHANISM_AUDIT_READY`

## Sources

- Run dir: `results/stage53a_vitaminc_frozen_external_eval_20260702_045300`
- External dir: `results/stage53a_vitaminc_frozen_external_eval_20260702_045300/external`
- Report: `results/stage53a_vitaminc_frozen_external_eval_20260702_045300/external/stage53a_vitaminc_frozen_stage43b1_vitaminc_validation_sample1000_external_factver_report.json`
- Predictions: `results/stage53a_vitaminc_frozen_external_eval_20260702_045300/external/stage53a_vitaminc_frozen_stage43b1_vitaminc_validation_sample1000_external_factver_predictions.jsonl`

## Decision / safety

| aggregate_decision                          | dataset_decision                  |   row_count |   base_accuracy |   composed_accuracy |   base_macro_f1 |   composed_macro_f1 |   delta_accuracy |   delta_macro_f1 |   introduced_unsafe_SUPPORT_count |   introduced_REFUTE_to_SUPPORT_count |   introduced_SUPPORT_to_REFUTE_count |   changed_row_count |   recovery_fired_count |   blocker_fired_count |
|:--------------------------------------------|:----------------------------------|------------:|----------------:|--------------------:|----------------:|--------------------:|-----------------:|-----------------:|----------------------------------:|-------------------------------------:|-------------------------------------:|--------------------:|-----------------------:|----------------------:|
| STAGE43C0_EXTERNAL_FACTVER_AGGREGATE_UNSAFE | STAGE43C0_EXTERNAL_FACTVER_UNSAFE |        1000 |           0.156 |               0.156 |        0.115986 |            0.116717 |                0 |         0.000731 |                                 1 |                                    0 |                                    0 |                   2 |                      0 |                     0 |

## Label / template sanity

| label_id_to_name                                     | name_to_label_id                               | external_uses_same_label_mapping_as_dev   | external_uses_same_prediction_path_as_dev   | external_tokenizer_source   | controlled_dev_tokenizer_source   |   external_max_length |   controlled_dev_max_length |
|:-----------------------------------------------------|:-----------------------------------------------|:------------------------------------------|:--------------------------------------------|:----------------------------|:----------------------------------|----------------------:|----------------------------:|
| {'0': 'REFUTE', '1': 'NOT_ENTITLED', '2': 'SUPPORT'} | {'NOT_ENTITLED': 1, 'REFUTE': 0, 'SUPPORT': 2} | True                                      | True                                        | state-spaces/mamba-130m-hf  | state-spaces/mamba-130m-hf        |                   128 |                         128 |

## Distribution collapse

| label        |   gold_count |   base_pred_count |   composed_pred_count |   base_pred_minus_gold |   composed_pred_minus_gold |   base_pred_rate |   composed_pred_rate |   gold_rate |
|:-------------|-------------:|------------------:|----------------------:|-----------------------:|---------------------------:|-----------------:|---------------------:|------------:|
| REFUTE       |          355 |                33 |                    33 |                   -322 |                       -322 |            0.033 |                0.033 |       0.355 |
| NOT_ENTITLED |          145 |               946 |                   944 |                    801 |                        799 |            0.946 |                0.944 |       0.145 |
| SUPPORT      |          500 |                21 |                    23 |                   -479 |                       -477 |            0.021 |                0.023 |       0.5   |

## Per-gold NOT_ENTITLED collapse

| gold    |   total |   base_to_NOT_ENTITLED |   base_to_NOT_ENTITLED_rate |   composed_to_NOT_ENTITLED |   composed_to_NOT_ENTITLED_rate |
|:--------|--------:|-----------------------:|----------------------------:|---------------------------:|--------------------------------:|
| SUPPORT |     500 |                    472 |                    0.944    |                        471 |                        0.942    |
| REFUTE  |     355 |                    343 |                    0.966197 |                        343 |                        0.966197 |

## Per-label metrics

| mode     | label        |   precision |   recall |       f1 |   support |
|:---------|:-------------|------------:|---------:|---------:|----------:|
| base     | SUPPORT      |    0.761905 | 0.032    | 0.06142  |       500 |
| base     | REFUTE       |    0.272727 | 0.025352 | 0.046392 |       355 |
| base     | NOT_ENTITLED |    0.138478 | 0.903448 | 0.240147 |       145 |
| composed | SUPPORT      |    0.73913  | 0.034    | 0.06501  |       500 |
| composed | REFUTE       |    0.272727 | 0.025352 | 0.046392 |       355 |
| composed | NOT_ENTITLED |    0.137712 | 0.896552 | 0.238751 |       145 |

## Token / truncation audit

|   external_max_length |   external_truncation_count |   external_truncation_rate |   token_min |   token_p25 |   token_median |   token_p75 |   token_p90 |   token_p95 |   token_max |
|----------------------:|----------------------------:|---------------------------:|------------:|------------:|---------------:|------------:|------------:|------------:|------------:|
|                   128 |                          68 |                      0.068 |          16 |          37 |             48 |          60 |          78 |       91.05 |         278 |

## Confidence / entropy audit

|   entropy_min |   entropy_p25 |   entropy_median |   entropy_p75 |   entropy_p90 |   entropy_p95 |   entropy_max |   maxprob_min |   maxprob_p25 |   maxprob_median |   maxprob_p75 |   maxprob_p90 |   maxprob_p95 |   maxprob_max |
|--------------:|--------------:|-----------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|-----------------:|--------------:|--------------:|--------------:|--------------:|
|        0.0676 |      0.939423 |         0.939463 |      0.940835 |      0.963018 |       1.01094 |       1.07393 |      0.387999 |      0.608436 |          0.60966 |      0.609695 |      0.609697 |      0.609697 |       0.98911 |

## Composer effect audit

| composer_mode      | prediction_source                  |   composer_available_row_count |   composer_unavailable_row_count |   changed_row_count |   changed_to_SUPPORT_count |   changed_to_REFUTE_count |   changed_to_NOT_ENTITLED_count |   recovery_fired_count |   blocker_fired_count |   stage43c2_composer_application_count |   stage43c2_composer_blocked_count |
|:-------------------|:-----------------------------------|-------------------------------:|---------------------------------:|--------------------:|---------------------------:|--------------------------:|--------------------------------:|-----------------------:|----------------------:|---------------------------------------:|-----------------------------------:|
| safe_structured_v2 | post_training_in_memory_best_state |                           1000 |                                0 |                   2 |                          2 |                         0 |                               0 |                      0 |                     0 |                                      2 |                                  0 |

## Mechanism verdict

| mechanism                           | verdict             | evidence                                                                                                                                                                    |
|:------------------------------------|:--------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| label_mapping_mismatch              | unlikely            | external_uses_same_label_mapping_as_dev=True                                                                                                                                |
| prediction_path_mismatch            | unlikely            | external_uses_same_prediction_path_as_dev=True                                                                                                                              |
| truncation_or_max_length            | unlikely_as_primary | external_truncation_rate=0.068                                                                                                                                              |
| external_domain_not_entitled_bias   | primary_failure     | base NOT_ENTITLED pred rate=0.946, gold NOT_ENTITLED rate=0.145, base NOT_ENTITLED count=946/1000                                                                           |
| composer_or_stage45c_recovery       | not_primary         | changed_row_count=2, recovery_fired_count=0, blocker_fired_count=0                                                                                                          |
| controlled_to_vitaminc_domain_shift | likely              | Internal controlled training used synthetic/controlled examples; VitaminC external examples contain open-domain lexical, numeric, temporal, and entity paraphrase patterns. |

## Interpretation

Primary conclusion: VitaminC failure is primarily a base prediction NOT_ENTITLED collapse under external-domain transfer.

Not primary causes:
- Stage45C recovery is not primary because recovery_fired_count=0.
- Composer is not primary because changed_row_count is very small.
- Label mapping mismatch is unlikely if same label mapping is confirmed.
- Truncation is unlikely as primary if truncation rate remains low.

## Leakage policy

- External data used for training: `False`
- External data used for checkpoint selection: `False`
- External data used for threshold tuning: `False`
- External data used for diagnosis only: `True`

## Next stage

Stage56 should be a non-training diagnostic or small controlled bridge design, not composer threshold tuning on VitaminC.
