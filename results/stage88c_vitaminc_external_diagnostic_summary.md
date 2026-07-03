# Stage88C - VitaminC External Diagnostic

## Decision

`STAGE88C_VITAMINC_EXTERNAL_DIAGNOSTIC_READY`

## Candidate status

`CLEAN_OK_EXTERNAL_MIXED_OR_FAILED`

## Summary

| stage    | decision                                    | candidate_status                  | run_dir                                                                         | bridge_config                                                                |   stage88c_best_epoch |   stage88c_clean_dev_acc |   stage88c_clean_dev_macro_f1 | stage88c_clean_prediction_counts                    |   stage88c_external_acc |   stage88c_external_macro_f1_all3 | stage88c_external_prediction_counts                  | stage88c_external_gold_counts                        | stage88c_external_errors                                                                                                                                                                                                                                        |   stage71_clean_dev_acc |   stage71_clean_dev_macro_f1 |   stage73_external_acc |   stage73_external_macro_f1_all3 | stage73_external_prediction_counts                   | stage73_external_errors                                                                                                                    |   stage84f_external_acc |   stage84f_external_macro_f1_all3 | stage84f_external_prediction_counts                  | stage84f_external_errors                                                                                                                   |   stage88c_minus_stage73_external_acc |   stage88c_minus_stage73_external_macro_f1 |   stage88c_minus_stage84f_false_NE_total |   stage88c_minus_stage73_false_NE_total |   elapsed_min | recommended_next_stage   |
|:---------|:--------------------------------------------|:----------------------------------|:--------------------------------------------------------------------------------|:-----------------------------------------------------------------------------|----------------------:|-------------------------:|------------------------------:|:----------------------------------------------------|------------------------:|----------------------------------:|:-----------------------------------------------------|:-----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------:|-----------------------------:|-----------------------:|---------------------------------:|:-----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|------------------------:|----------------------------------:|:-----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------:|-------------------------------------------:|-----------------------------------------:|----------------------------------------:|--------------:|:-------------------------|
| Stage88C | STAGE88C_VITAMINC_EXTERNAL_DIAGNOSTIC_READY | CLEAN_OK_EXTERNAL_MIXED_OR_FAILED | results/stage88c_stage57_stage66_stage88a_vitaminc_external_run_20260703_043753 | Stage57 + Stage66 + Stage88A; no Stage83A; VitaminC external diagnostic only |                   171 |                 0.976389 |                      0.965904 | {"NOT_ENTITLED": 523, "REFUTE": 90, "SUPPORT": 107} |                   0.338 |                          0.329458 | {"REFUTE": 283, "NOT_ENTITLED": 423, "SUPPORT": 294} | {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500} | {"false_NE_on_SUPPORT": 197, "false_NE_on_REFUTE": 152, "false_SUPPORT_on_NE": 31, "false_REFUTE_on_NE": 40, "false_SUPPORT_on_REFUTE": 101, "false_REFUTE_on_SUPPORT": 141, "false_NE_total": 349, "false_entitlement_total": 71, "polarity_error_total": 242} |                   0.975 |                     0.964047 |                  0.353 |                         0.326179 | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} | {"false_NE_on_SUPPORT": 178, "false_NE_on_REFUTE": 145, "false_NE_total": 323, "false_entitlement_total": 80, "polarity_error_total": 244} |                   0.326 |                          0.306672 | {"REFUTE": 202, "NOT_ENTITLED": 439, "SUPPORT": 359} | {"false_NE_on_SUPPORT": 205, "false_NE_on_REFUTE": 162, "false_NE_total": 367, "false_entitlement_total": 73, "polarity_error_total": 234} |                                -0.015 |                                 0.00327898 |                                      -18 |                                      26 |        7.0683 | Stage89 decision report  |

## Comparison

| run                                  |   external_acc |   external_macro_f1_all3 | prediction_counts                                    |   false_NE_total |   false_entitlement_total |   polarity_error_total |
|:-------------------------------------|---------------:|-------------------------:|:-----------------------------------------------------|-----------------:|--------------------------:|-----------------------:|
| Stage73/Stage71 primary external     |          0.353 |                 0.326179 | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} |              323 |                        80 |                    244 |
| Stage84F/Stage83C rejected external  |          0.326 |                 0.306672 | {"REFUTE": 202, "NOT_ENTITLED": 439, "SUPPORT": 359} |              367 |                        73 |                    234 |
| Stage88C/Stage88A candidate external |          0.338 |                 0.329458 | {"REFUTE": 283, "NOT_ENTITLED": 423, "SUPPORT": 294} |              349 |                        71 |                    242 |

## Metadata checks

| check                               | pass   |
|:------------------------------------|:-------|
| returncode_zero                     | True   |
| train_report_exists                 | True   |
| ood_report_exists                   | True   |
| ood_predictions_exists              | True   |
| stage57_rows_520                    | True   |
| stage66_rows_720                    | True   |
| stage88a_rows_360_via_stage80a_slot | True   |
| combined_bridge_1600                | True   |
| final_train_4480                    | True   |
| external_n_1000                     | True   |

## Clean checks

| check                              | pass   |
|:-----------------------------------|:-------|
| clean_acc_ge_stage71_minus_0p003   | True   |
| clean_macro_ge_stage71_minus_0p003 | True   |

## External checks

| check                                 | pass   |
|:--------------------------------------|:-------|
| external_acc_ge_stage73               | False  |
| external_macro_ge_stage73             | True   |
| false_NE_total_lt_stage84f            | True   |
| false_NE_total_lte_stage73            | False  |
| false_entitlement_lte_stage73_plus_10 | True   |

## Confusion

| gold\pred    |   REFUTE |   NOT_ENTITLED |   SUPPORT |
|:-------------|---------:|---------------:|----------:|
| REFUTE       |      102 |            152 |       101 |
| NOT_ENTITLED |       40 |             74 |        31 |
| SUPPORT      |      141 |            197 |       162 |

## Per-label

| label        |   support |   pred_count |   tp |   fp |   fn |   precision |   recall |       f1 |
|:-------------|----------:|-------------:|-----:|-----:|-----:|------------:|---------:|---------:|
| REFUTE       |       355 |          283 |  102 |  181 |  253 |    0.360424 | 0.287324 | 0.319749 |
| NOT_ENTITLED |       145 |          423 |   74 |  349 |   71 |    0.174941 | 0.510345 | 0.260563 |
| SUPPORT      |       500 |          294 |  162 |  132 |  338 |    0.55102  | 0.324    | 0.40806  |
