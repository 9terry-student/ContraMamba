# Stage97C - Embedded External Eval Recovery

## Decision

`STAGE97C_EMBEDDED_EXTERNAL_EVAL_RECOVERED`

## Candidate status

`TOPLINE_AND_EMBEDDED_BREAKDOWN_AVAILABLE`

## Summary

| stage    | decision                                  | candidate_status                         | run_dir                                                                                  | external_eval_name                                        | external_report_storage                 | separate_external_prediction_file_found   |   stage97c_clean_best_epoch |   stage97c_clean_dev_acc |   stage97c_clean_dev_macro_f1 |   stage97b_clean_dev_acc |   stage97b_clean_dev_macro_f1 |   stage97c_external_n |   stage97c_external_acc |   stage97c_external_macro_f1_all3 |   stage97c_false_SUPPORT_logged |   stage73_external_acc |   stage73_external_macro_f1_all3 |   stage92c_external_acc |   stage92c_external_macro_f1_all3 |   stage97c_minus_stage73_acc |   stage97c_minus_stage73_macro |   stage97c_minus_stage92c_acc |   stage97c_minus_stage92c_macro | full_breakdown_available   | recommended_next_stage                                                                                                            |
|:---------|:------------------------------------------|:-----------------------------------------|:-----------------------------------------------------------------------------------------|:----------------------------------------------------------|:----------------------------------------|:------------------------------------------|----------------------------:|-------------------------:|------------------------------:|-------------------------:|------------------------------:|----------------------:|------------------------:|----------------------------------:|--------------------------------:|-----------------------:|---------------------------------:|------------------------:|----------------------------------:|-----------------------------:|-------------------------------:|------------------------------:|--------------------------------:|:---------------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| Stage97C | STAGE97C_EMBEDDED_EXTERNAL_EVAL_RECOVERED | TOPLINE_AND_EMBEDDED_BREAKDOWN_AVAILABLE | results/stage97c_stage57_stage66_stage92a_stage97a_vitaminc_external_run_20260703_061803 | stage84e2_vitaminc_validation_sample1000_exact_ood_schema | embedded_in_train_report_external_evals | False                                     |                         200 |                 0.976389 |                      0.965617 |                 0.976389 |                      0.965617 |                  1000 |                   0.355 |                             0.343 |                             136 |                  0.353 |                         0.326179 |                   0.351 |                          0.330837 |                        0.002 |                      0.0168213 |                         0.004 |                        0.012163 | True                       | Stage98 embedded-external decision/audit; promotion still requires exact false_NE/error breakdown or rerun with prediction export |

## Embedded metrics

| metric               | value                                                |
|:---------------------|:-----------------------------------------------------|
| n                    | 1000                                                 |
| accuracy             | 0.355                                                |
| macro_f1_all3        | 0.343                                                |
| false_SUPPORT_logged | 136                                                  |
| prediction_counts    | {"NOT_ENTITLED": 394, "REFUTE": 296, "SUPPORT": 310} |
| gold_counts          | {"NOT_ENTITLED": 145, "REFUTE": 355, "SUPPORT": 500} |

## Top-line checks

| check                      | pass   |
|:---------------------------|:-------|
| external_n_1000            | True   |
| external_acc_ge_stage73    | True   |
| external_macro_ge_stage73  | True   |
| external_acc_ge_stage92c   | True   |
| external_macro_ge_stage92c | True   |
