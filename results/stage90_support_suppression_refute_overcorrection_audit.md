# Stage90 - SUPPORT Suppression / REFUTE Overcorrection Audit

## Decision

`STAGE90_SUPPORT_SUPPRESSION_REFUTE_OVERCORRECTION_AUDIT_READY`

## Summary

| stage   | decision                                                      | stage73_prediction_file                                                                                                                                          | stage88c_prediction_file                                                                                                                             | match_mode   |   matched_n | stage73_metrics                                                                                                                                                                                                                                                                                                                                                                                                | stage88c_metrics                                                                                                                                                                                                                                                                                                                                                                                                | critical_transition_counts                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | diagnosis_rules                                                                                                                                                                                                  | interpretation                                                                                                                                                                                                                                                   | next_direction                                                                              | primary_policy       |
|:--------|:--------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:-------------|------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:---------------------|
| Stage90 | STAGE90_SUPPORT_SUPPRESSION_REFUTE_OVERCORRECTION_AUDIT_READY | results/stage73_retry_stage57_stage66_bridge_vitaminc_external_diag_run_20260702_080534/external_vitaminc/external_probe_stage73_retry_vitaminc_predictions.json | results/stage88c_stage57_stage66_stage88a_vitaminc_external_run_20260703_043753/stage88c_stage57_stage66_stage88a_vitaminc_external_predictions.json | key          |        1000 | {"accuracy": 0.353, "macro_f1_all3": 0.3261787057160285, "prediction_counts": {"SUPPORT": 393, "REFUTE": 219, "NOT_ENTITLED": 388}, "errors": {"false_NE_on_SUPPORT": 178, "false_NE_on_REFUTE": 145, "false_SUPPORT_on_NE": 39, "false_REFUTE_on_NE": 41, "false_SUPPORT_on_REFUTE": 138, "false_REFUTE_on_SUPPORT": 106, "false_NE_total": 323, "false_entitlement_total": 80, "polarity_error_total": 244}} | {"accuracy": 0.338, "macro_f1_all3": 0.32945768332771147, "prediction_counts": {"SUPPORT": 294, "REFUTE": 283, "NOT_ENTITLED": 423}, "errors": {"false_NE_on_SUPPORT": 197, "false_NE_on_REFUTE": 152, "false_SUPPORT_on_NE": 31, "false_REFUTE_on_NE": 40, "false_SUPPORT_on_REFUTE": 101, "false_REFUTE_on_SUPPORT": 141, "false_NE_total": 349, "false_entitlement_total": 71, "polarity_error_total": 242}} | {"stage73_correct_stage88c_wrong_total": 97, "stage73_wrong_stage88c_correct_total": 82, "support_losses_stage73_support_to_stage88c_non_support": 79, "support_stage73_support_to_stage88c_ne": 34, "support_stage73_support_to_stage88c_refute": 45, "support_recoveries_stage73_wrong_to_stage88c_support": 25, "refute_gains_stage73_wrong_to_stage88c_refute": 45, "refute_gains_from_stage73_ne": 7, "refute_gains_from_stage73_support": 38, "new_refute_errors_on_support_not_refute_in_stage73": 50, "new_ne_errors_on_support_not_ne_in_stage73": 34} | ["REFUTE_prediction_mass_increased", "SUPPORT_prediction_mass_decreased", "SUPPORT_to_REFUTE_polarity_error_increased", "SUPPORT_to_NE_suppression_increased", "REFUTE_recovery_trades_off_with_SUPPORT_recall"] | Stage88C improves REFUTE recovery and macro-F1 but over-rotates prediction mass from SUPPORT toward REFUTE/NOT_ENTITLED. The next branch should preserve SUPPORT recall while retaining REFUTE gains, rather than adding more balanced/generic entitlement rows. | Stage91 support-preserving recovery design; avoid generic REFUTE-heavy entitlement pressure | KEEP_STAGE71_PRIMARY |

## Critical transition counts

| metric                                                 |   count |
|:-------------------------------------------------------|--------:|
| stage73_correct_stage88c_wrong_total                   |      97 |
| stage73_wrong_stage88c_correct_total                   |      82 |
| support_losses_stage73_support_to_stage88c_non_support |      79 |
| support_stage73_support_to_stage88c_ne                 |      34 |
| support_stage73_support_to_stage88c_refute             |      45 |
| support_recoveries_stage73_wrong_to_stage88c_support   |      25 |
| refute_gains_stage73_wrong_to_stage88c_refute          |      45 |
| refute_gains_from_stage73_ne                           |       7 |
| refute_gains_from_stage73_support                      |      38 |
| new_refute_errors_on_support_not_refute_in_stage73     |      50 |
| new_ne_errors_on_support_not_ne_in_stage73             |      34 |

## Error delta

| error_type              |   stage73 |   stage88c |   delta_stage88c_minus_stage73 |
|:------------------------|----------:|-----------:|-------------------------------:|
| false_NE_on_REFUTE      |       145 |        152 |                              7 |
| false_NE_on_SUPPORT     |       178 |        197 |                             19 |
| false_NE_total          |       323 |        349 |                             26 |
| false_REFUTE_on_NE      |        41 |         40 |                             -1 |
| false_REFUTE_on_SUPPORT |       106 |        141 |                             35 |
| false_SUPPORT_on_NE     |        39 |         31 |                             -8 |
| false_SUPPORT_on_REFUTE |       138 |        101 |                            -37 |
| false_entitlement_total |        80 |         71 |                             -9 |
| polarity_error_total    |       244 |        242 |                             -2 |

## Per-label delta

| label        |   support_stage73 |   pred_count_stage73 |   tp_stage73 |   fp_stage73 |   fn_stage73 |   precision_stage73 |   recall_stage73 |   f1_stage73 |   support_stage88c |   pred_count_stage88c |   tp_stage88c |   fp_stage88c |   fn_stage88c |   precision_stage88c |   recall_stage88c |   f1_stage88c |   precision_delta |   recall_delta |   f1_delta |   pred_count_delta |   tp_delta |   fp_delta |   fn_delta |
|:-------------|------------------:|---------------------:|-------------:|-------------:|-------------:|--------------------:|-----------------:|-------------:|-------------------:|----------------------:|--------------:|--------------:|--------------:|---------------------:|------------------:|--------------:|------------------:|---------------:|-----------:|-------------------:|-----------:|-----------:|-----------:|
| REFUTE       |               355 |                  219 |           72 |          147 |          283 |            0.328767 |         0.202817 |     0.250871 |                355 |                   283 |           102 |           181 |           253 |             0.360424 |          0.287324 |      0.319749 |        0.0316569  |       0.084507 |  0.0688781 |                 64 |         30 |         34 |        -30 |
| NOT_ENTITLED |               145 |                  388 |           65 |          323 |           80 |            0.167526 |         0.448276 |     0.243902 |                145 |                   423 |            74 |           349 |            71 |             0.174941 |          0.510345 |      0.260563 |        0.00741513 |       0.062069 |  0.0166609 |                 35 |          9 |         26 |         -9 |
| SUPPORT      |               500 |                  393 |          216 |          177 |          284 |            0.549618 |         0.432    |     0.483763 |                500 |                   294 |           162 |           132 |           338 |             0.55102  |          0.324    |      0.40806  |        0.00140209 |      -0.108    | -0.0757021 |                -99 |        -54 |        -45 |         54 |

## Prediction transition

| stage73_pred   | stage88c_pred   |   count |
|:---------------|:----------------|--------:|
| NOT_ENTITLED   | NOT_ENTITLED    |     357 |
| SUPPORT        | SUPPORT         |     245 |
| REFUTE         | REFUTE          |     181 |
| SUPPORT        | REFUTE          |      89 |
| SUPPORT        | NOT_ENTITLED    |      59 |
| REFUTE         | SUPPORT         |      31 |
| NOT_ENTITLED   | SUPPORT         |      18 |
| NOT_ENTITLED   | REFUTE          |      13 |
| REFUTE         | NOT_ENTITLED    |       7 |

## Gold-conditioned transition

| gold         | stage73_pred   | stage88c_pred   |   count |
|:-------------|:---------------|:----------------|--------:|
| NOT_ENTITLED | NOT_ENTITLED   | NOT_ENTITLED    |      62 |
| NOT_ENTITLED | REFUTE         | REFUTE          |      33 |
| NOT_ENTITLED | SUPPORT        | SUPPORT         |      24 |
| NOT_ENTITLED | SUPPORT        | NOT_ENTITLED    |       9 |
| NOT_ENTITLED | SUPPORT        | REFUTE          |       6 |
| NOT_ENTITLED | REFUTE         | SUPPORT         |       5 |
| NOT_ENTITLED | REFUTE         | NOT_ENTITLED    |       3 |
| NOT_ENTITLED | NOT_ENTITLED   | SUPPORT         |       2 |
| NOT_ENTITLED | NOT_ENTITLED   | REFUTE          |       1 |
| REFUTE       | NOT_ENTITLED   | NOT_ENTITLED    |     132 |
| REFUTE       | SUPPORT        | SUPPORT         |      84 |
| REFUTE       | REFUTE         | REFUTE          |      57 |
| REFUTE       | SUPPORT        | REFUTE          |      38 |
| REFUTE       | SUPPORT        | NOT_ENTITLED    |      16 |
| REFUTE       | REFUTE         | SUPPORT         |      11 |
| REFUTE       | NOT_ENTITLED   | REFUTE          |       7 |
| REFUTE       | NOT_ENTITLED   | SUPPORT         |       6 |
| REFUTE       | REFUTE         | NOT_ENTITLED    |       4 |
| SUPPORT      | NOT_ENTITLED   | NOT_ENTITLED    |     163 |
| SUPPORT      | SUPPORT        | SUPPORT         |     137 |
| SUPPORT      | REFUTE         | REFUTE          |      91 |
| SUPPORT      | SUPPORT        | REFUTE          |      45 |
| SUPPORT      | SUPPORT        | NOT_ENTITLED    |      34 |
| SUPPORT      | REFUTE         | SUPPORT         |      15 |
| SUPPORT      | NOT_ENTITLED   | SUPPORT         |      10 |
| SUPPORT      | NOT_ENTITLED   | REFUTE          |       5 |

## Diagnosis rules

| diagnosis_rule                                 |
|:-----------------------------------------------|
| REFUTE_prediction_mass_increased               |
| SUPPORT_prediction_mass_decreased              |
| SUPPORT_to_REFUTE_polarity_error_increased     |
| SUPPORT_to_NE_suppression_increased            |
| REFUTE_recovery_trades_off_with_SUPPORT_recall |
