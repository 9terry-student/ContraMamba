# Stage92C - VitaminC External Diagnostic

## Decision

`STAGE92C_VITAMINC_EXTERNAL_DIAGNOSTIC_READY`

## Candidate status

`CLEAN_OK_EXTERNAL_MIXED_OR_FAILED`

## Summary

| stage    | decision                                    | candidate_status                  | run_dir                                                                         | bridge_config                                                                             |   stage92c_best_epoch |   stage92c_clean_dev_acc |   stage92c_clean_dev_macro_f1 | stage92c_clean_prediction_counts                    |   stage92c_external_acc |   stage92c_external_macro_f1_all3 | stage92c_external_prediction_counts                  | stage92c_external_gold_counts                        | stage92c_external_errors                                                                                                                                                                                                                                        |   stage92c_support_recall |   stage92c_refute_recall |   stage92c_ne_recall |   stage71_clean_dev_acc |   stage71_clean_dev_macro_f1 |   stage73_external_acc |   stage73_external_macro_f1_all3 | stage73_external_prediction_counts                   | stage73_external_errors                                                                                                                                                                                    |   stage73_support_recall |   stage73_refute_recall |   stage88c_external_acc |   stage88c_external_macro_f1_all3 | stage88c_external_prediction_counts                  | stage88c_external_errors                                                                                                                                                                                   |   stage88c_support_recall |   stage88c_refute_recall |   stage92c_minus_stage73_external_acc |   stage92c_minus_stage73_external_macro_f1 |   stage92c_minus_stage73_false_NE_total |   stage92c_minus_stage88c_false_NE_total |   stage92c_minus_stage73_support_recall |   stage92c_minus_stage88c_support_recall |   elapsed_min | recommended_next_stage   |
|:---------|:--------------------------------------------|:----------------------------------|:--------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|----------------------:|-------------------------:|------------------------------:|:----------------------------------------------------|------------------------:|----------------------------------:|:-----------------------------------------------------|:-----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------:|-------------------------:|---------------------:|------------------------:|-----------------------------:|-----------------------:|---------------------------------:|:-----------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------:|------------------------:|------------------------:|----------------------------------:|:-----------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------:|-------------------------:|--------------------------------------:|-------------------------------------------:|----------------------------------------:|-----------------------------------------:|----------------------------------------:|-----------------------------------------:|--------------:|:-------------------------|
| Stage92C | STAGE92C_VITAMINC_EXTERNAL_DIAGNOSTIC_READY | CLEAN_OK_EXTERNAL_MIXED_OR_FAILED | results/stage92c_stage57_stage66_stage92a_vitaminc_external_run_20260703_052044 | Stage57 + Stage66 + Stage92A; no Stage83A; no Stage88A; VitaminC external diagnostic only |                   171 |                 0.977778 |                      0.967777 | {"NOT_ENTITLED": 524, "REFUTE": 90, "SUPPORT": 106} |                   0.351 |                          0.330837 | {"REFUTE": 223, "NOT_ENTITLED": 412, "SUPPORT": 365} | {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500} | {"false_NE_on_SUPPORT": 192, "false_NE_on_REFUTE": 153, "false_SUPPORT_on_NE": 45, "false_REFUTE_on_NE": 33, "false_SUPPORT_on_REFUTE": 119, "false_REFUTE_on_SUPPORT": 107, "false_NE_total": 345, "false_entitlement_total": 78, "polarity_error_total": 226} |                     0.402 |                 0.233803 |             0.462069 |                   0.975 |                     0.964047 |                  0.353 |                         0.326179 | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} | {"false_NE_on_SUPPORT": 178, "false_NE_on_REFUTE": 145, "false_NE_total": 323, "false_entitlement_total": 80, "polarity_error_total": 244, "false_REFUTE_on_SUPPORT": 106, "false_SUPPORT_on_REFUTE": 138} |                    0.432 |                0.202817 |                   0.338 |                          0.329458 | {"REFUTE": 283, "NOT_ENTITLED": 423, "SUPPORT": 294} | {"false_NE_on_SUPPORT": 197, "false_NE_on_REFUTE": 152, "false_NE_total": 349, "false_entitlement_total": 71, "polarity_error_total": 242, "false_REFUTE_on_SUPPORT": 141, "false_SUPPORT_on_REFUTE": 101} |                     0.324 |                 0.287324 |                                -0.002 |                                  0.0046585 |                                      22 |                                       -4 |                                   -0.03 |                                    0.078 |        6.6745 | Stage93 decision report  |

## Comparison

| run                                  |   external_acc |   external_macro_f1_all3 | prediction_counts                                    |   support_recall |   refute_recall |   false_NE_total |   false_entitlement_total |   false_REFUTE_on_SUPPORT |   false_SUPPORT_on_REFUTE |
|:-------------------------------------|---------------:|-------------------------:|:-----------------------------------------------------|-----------------:|----------------:|-----------------:|--------------------------:|--------------------------:|--------------------------:|
| Stage73/Stage71 primary external     |          0.353 |                 0.326179 | {"REFUTE": 219, "NOT_ENTITLED": 388, "SUPPORT": 393} |            0.432 |        0.202817 |              323 |                        80 |                       106 |                       138 |
| Stage88C/Stage88A mixed external     |          0.338 |                 0.329458 | {"REFUTE": 283, "NOT_ENTITLED": 423, "SUPPORT": 294} |            0.324 |        0.287324 |              349 |                        71 |                       141 |                       101 |
| Stage92C/Stage92A candidate external |          0.351 |                 0.330837 | {"REFUTE": 223, "NOT_ENTITLED": 412, "SUPPORT": 365} |            0.402 |        0.233803 |              345 |                        78 |                       107 |                       119 |

## Metadata checks

| check                               | pass   |
|:------------------------------------|:-------|
| returncode_zero                     | True   |
| train_report_exists                 | True   |
| ood_report_exists                   | True   |
| ood_predictions_exists              | True   |
| stage57_rows_520                    | True   |
| stage66_rows_720                    | True   |
| stage92a_rows_240_via_stage80a_slot | True   |
| combined_bridge_1480                | True   |
| final_train_4360                    | True   |
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
| false_NE_total_lte_stage73            | False  |
| false_entitlement_lte_stage73_plus_10 | True   |
| support_recall_ge_stage73_minus_0p02  | False  |
| support_recall_gt_stage88c            | True   |
| false_REFUTE_on_SUPPORT_lt_stage88c   | True   |

## Confusion

| gold\pred    |   REFUTE |   NOT_ENTITLED |   SUPPORT |
|:-------------|---------:|---------------:|----------:|
| REFUTE       |       83 |            153 |       119 |
| NOT_ENTITLED |       33 |             67 |        45 |
| SUPPORT      |      107 |            192 |       201 |

## Per-label

| label        |   support |   pred_count |   tp |   fp |   fn |   precision |   recall |       f1 |
|:-------------|----------:|-------------:|-----:|-----:|-----:|------------:|---------:|---------:|
| REFUTE       |       355 |          223 |   83 |  140 |  272 |    0.372197 | 0.233803 | 0.287197 |
| NOT_ENTITLED |       145 |          412 |   67 |  345 |   78 |    0.162621 | 0.462069 | 0.240575 |
| SUPPORT      |       500 |          365 |  201 |  164 |  299 |    0.550685 | 0.402    | 0.46474  |
