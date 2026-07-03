# Stage87A - Stage71 vs Stage84F External Prediction Diff Audit

## Decision

`STAGE87A_STAGE71_VS_STAGE84F_EXTERNAL_DIFF_AUDIT_READY`

## Next direction

`Stage87B_design_external_entitlement_recovery_without_NE_safety`

## Summary

| stage    | decision                                               | stage73_prediction_file                                                                                                                                          | stage84f_prediction_file                                                                                                              | match_mode   |   matched_n | stage73_metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | stage84f_metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | critical_transition_counts                                                                                                                                                        | diagnosis_rules                                                                                                       | next_direction                                                  | primary_policy       |
|:---------|:-------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|:-------------|------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|:---------------------|
| Stage87A | STAGE87A_STAGE71_VS_STAGE84F_EXTERNAL_DIFF_AUDIT_READY | results/stage73_retry_stage57_stage66_bridge_vitaminc_external_diag_run_20260702_080534/external_vitaminc/external_probe_stage73_retry_vitaminc_predictions.json | results/stage84f_stage83c_vitaminc_external_exact_ood_schema_run_20260703_033759/stage84f_stage83c_vitaminc_external_predictions.json | key          |        1000 | {"n": 1000, "accuracy": 0.353, "macro_f1_all3": 0.3261787057160285, "gold_counts": {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500}, "prediction_counts": {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393}, "errors": {"false_NE_on_SUPPORT": 178, "false_NE_on_REFUTE": 145, "false_SUPPORT_on_NE": 39, "false_REFUTE_on_NE": 41, "false_SUPPORT_on_REFUTE": 138, "false_REFUTE_on_SUPPORT": 106, "false_NE_total": 323, "false_entitlement_total": 80, "polarity_error_total": 244}} | {"n": 1000, "accuracy": 0.326, "macro_f1_all3": 0.3066716953259303, "gold_counts": {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500}, "prediction_counts": {"REFUTE": 202, "NOT_ENTITLED": 439, "SUPPORT": 359}, "errors": {"false_NE_on_SUPPORT": 205, "false_NE_on_REFUTE": 162, "false_SUPPORT_on_NE": 42, "false_REFUTE_on_NE": 31, "false_SUPPORT_on_REFUTE": 128, "false_REFUTE_on_SUPPORT": 106, "false_NE_total": 367, "false_entitlement_total": 73, "polarity_error_total": 234}} | {"stage73_correct_stage84f_wrong_total": 83, "stage73_wrong_stage84f_correct_total": 56, "stage73_correct_stage84f_false_ne_total": 48, "stage73_entitled_stage84f_ne_total": 89} | ["false_NE_increased_vs_stage71", "NE_prediction_mass_increased", "external_regression_confirmed_by_prediction_diff"] | Stage87B_design_external_entitlement_recovery_without_NE_safety | KEEP_STAGE71_PRIMARY |

## Stage73 confusion

| gold\pred    |   REFUTE |   NOT_ENTITLED |   SUPPORT |
|:-------------|---------:|---------------:|----------:|
| REFUTE       |       72 |            145 |       138 |
| NOT_ENTITLED |       41 |             65 |        39 |
| SUPPORT      |      106 |            178 |       216 |

## Stage84F confusion

| gold\pred    |   REFUTE |   NOT_ENTITLED |   SUPPORT |
|:-------------|---------:|---------------:|----------:|
| REFUTE       |       65 |            162 |       128 |
| NOT_ENTITLED |       31 |             72 |        42 |
| SUPPORT      |      106 |            205 |       189 |

## Error delta

| error_type              |   stage73 |   stage84f |   delta_stage84f_minus_stage73 |
|:------------------------|----------:|-----------:|-------------------------------:|
| false_NE_on_REFUTE      |       145 |        162 |                             17 |
| false_NE_on_SUPPORT     |       178 |        205 |                             27 |
| false_NE_total          |       323 |        367 |                             44 |
| false_REFUTE_on_NE      |        41 |         31 |                            -10 |
| false_REFUTE_on_SUPPORT |       106 |        106 |                              0 |
| false_SUPPORT_on_NE     |        39 |         42 |                              3 |
| false_SUPPORT_on_REFUTE |       138 |        128 |                            -10 |
| false_entitlement_total |        80 |         73 |                             -7 |
| polarity_error_total    |       244 |        234 |                            -10 |

## Prediction transition

| stage73_pred   | stage84f_pred   |   count |
|:---------------|:----------------|--------:|
| NOT_ENTITLED   | NOT_ENTITLED    |     350 |
| SUPPORT        | SUPPORT         |     296 |
| REFUTE         | REFUTE          |     165 |
| SUPPORT        | NOT_ENTITLED    |      73 |
| REFUTE         | SUPPORT         |      38 |
| NOT_ENTITLED   | SUPPORT         |      25 |
| SUPPORT        | REFUTE          |      24 |
| REFUTE         | NOT_ENTITLED    |      16 |
| NOT_ENTITLED   | REFUTE          |      13 |

## Gold-conditioned transition

| gold         | stage73_pred   | stage84f_pred   |   count |
|:-------------|:---------------|:----------------|--------:|
| NOT_ENTITLED | NOT_ENTITLED   | NOT_ENTITLED    |      59 |
| NOT_ENTITLED | SUPPORT        | SUPPORT         |      29 |
| NOT_ENTITLED | REFUTE         | REFUTE          |      27 |
| NOT_ENTITLED | REFUTE         | SUPPORT         |       8 |
| NOT_ENTITLED | SUPPORT        | NOT_ENTITLED    |       7 |
| NOT_ENTITLED | REFUTE         | NOT_ENTITLED    |       6 |
| NOT_ENTITLED | NOT_ENTITLED   | SUPPORT         |       5 |
| NOT_ENTITLED | SUPPORT        | REFUTE          |       3 |
| NOT_ENTITLED | NOT_ENTITLED   | REFUTE          |       1 |
| REFUTE       | NOT_ENTITLED   | NOT_ENTITLED    |     130 |
| REFUTE       | SUPPORT        | SUPPORT         |     107 |
| REFUTE       | REFUTE         | REFUTE          |      51 |
| REFUTE       | SUPPORT        | NOT_ENTITLED    |      25 |
| REFUTE       | REFUTE         | SUPPORT         |      14 |
| REFUTE       | NOT_ENTITLED   | REFUTE          |       8 |
| REFUTE       | REFUTE         | NOT_ENTITLED    |       7 |
| REFUTE       | NOT_ENTITLED   | SUPPORT         |       7 |
| REFUTE       | SUPPORT        | REFUTE          |       6 |
| SUPPORT      | NOT_ENTITLED   | NOT_ENTITLED    |     161 |
| SUPPORT      | SUPPORT        | SUPPORT         |     160 |
| SUPPORT      | REFUTE         | REFUTE          |      87 |
| SUPPORT      | SUPPORT        | NOT_ENTITLED    |      41 |
| SUPPORT      | REFUTE         | SUPPORT         |      16 |
| SUPPORT      | SUPPORT        | REFUTE          |      15 |
| SUPPORT      | NOT_ENTITLED   | SUPPORT         |      13 |
| SUPPORT      | NOT_ENTITLED   | REFUTE          |       4 |
| SUPPORT      | REFUTE         | NOT_ENTITLED    |       3 |
