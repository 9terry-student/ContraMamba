# Stage80G - Stage80A Clean-Dev Decision Report

## Decision

`STAGE80G_STAGE80A_CLEAN_DEV_DECISION_REPORT_READY`

## Stage80A decision

`STOP_BEFORE_EXTERNAL_DIAGNOSTIC`

## Summary

| stage    | decision                                          | stage80a_decision               | decision_reason                                                                                                                                                                                                                |   stage71_acc |   stage71_macro_f1 |   stage75f_acc |   stage75f_macro_f1 |   stage80f_acc |   stage80f_macro_f1 |   stage80f_minus_stage71_acc |   stage80f_minus_stage71_macro_f1 |   stage80f_minus_stage75f_acc |   stage80f_minus_stage75f_macro_f1 |   strict_macro_preservation_threshold | stage80f_macro_preservation_vs_stage71_pass   |   stage80f_macro_preservation_margin_vs_stage71 | training_executed   | external_eval_executed   | recommended_next_stage                                                                   |
|:---------|:--------------------------------------------------|:--------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|-------------------:|---------------:|--------------------:|---------------:|--------------------:|-----------------------------:|----------------------------------:|------------------------------:|-----------------------------------:|--------------------------------------:|:----------------------------------------------|------------------------------------------------:|:--------------------|:-------------------------|:-----------------------------------------------------------------------------------------|
| Stage80G | STAGE80G_STAGE80A_CLEAN_DEV_DECISION_REPORT_READY | STOP_BEFORE_EXTERNAL_DIAGNOSTIC | Stage80A integration is technically valid, but Stage80F clean-dev macro-F1 falls below Stage71 by more than the strict tolerance. Do not spend the next run on external diagnostic by default; keep Stage71_retry2 as primary. |         0.975 |           0.964047 |       0.973611 |            0.962205 |       0.970833 |            0.958564 |                  -0.00416666 |                       -0.00548313 |                   -0.00277776 |                        -0.00364087 |                                -0.005 | False                                         |                                    -0.000483132 | False               | False                    | Stage83 design smaller polarity-only or NE-safety-only ablation, or stop Stage80A branch |

## Checks

| check                          | pass   |
|:-------------------------------|:-------|
| stage71_exists                 | True   |
| stage75f_exists                | True   |
| stage80f_exists                | True   |
| stage80f_summary_exists        | True   |
| stage80e_static_ready          | True   |
| stage80f_execution_zero        | True   |
| stage80f_stage80a_used_500     | True   |
| stage80f_stage75_full_not_used | True   |
| decision_nonempty              | True   |

## Signals

| signal                                       | value   |
|:---------------------------------------------|:--------|
| stage80e_static_ready                        | True    |
| stage80f_execution_returncode_zero           | True    |
| stage80f_metadata_valid                      | True    |
| stage80f_decision_needs_review               | True    |
| stage80f_acc_above_0p97                      | True    |
| stage80f_macro_above_0p95                    | True    |
| stage80f_macro_preservation_vs_stage71_pass  | False   |
| stage80f_macro_preservation_vs_stage75f_pass | True    |
| stage80f_stage75_full_not_used               | True    |
| stage80f_stage80a_used_500                   | True    |

## Run comparison

| stage                          | source_path                                                                                                                                           |   best_epoch |   selected_epoch |   best_dev_acc |   best_dev_macro_f1 |   pred_NOT_ENTITLED |   pred_REFUTE |   pred_SUPPORT |   stage57_rows |   stage66_rows |   stage75_rows |   stage80a_rows |   combined_bridge_rows |   final_train_row_count_expected | pairwise_source       |   pairwise_bridge_rows_excluded | clean_dev_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   |
|:-------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|-------------:|-----------------:|---------------:|--------------------:|--------------------:|--------------:|---------------:|---------------:|---------------:|---------------:|----------------:|-----------------------:|---------------------------------:|:----------------------|--------------------------------:|:-------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|
| Stage71_retry2_primary         | results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage57_stage66_train_report.json            |          151 |              151 |       0.975    |            0.964047 |                 522 |            90 |            108 |            520 |            720 |              0 |               0 |                   1240 |                              nan | clean_main_train_only |                            1240 | True                                 | False                             | False                                        | False            |
| Stage75F_full_bridge           | results/stage75f_stage57_stage66_stage75_bridge_enabled_frozen_recovery_run_20260703_002555/stage75f_stage57_stage66_stage75_train_report.json        |          184 |              184 |       0.973611 |            0.962205 |                 521 |            90 |            109 |            520 |            720 |           1020 |               0 |                   2260 |                              nan | clean_main_train_only |                            2260 | True                                 | False                             | False                                        | False            |
| Stage80F_conservative_Stage80A | results/stage80f_stage57_stage66_stage80a_conservative_bridge_frozen_recovery_run_20260703_013106/stage80f_stage57_stage66_stage80a_train_report.json |          180 |              180 |       0.970833 |            0.958564 |                 519 |            90 |            111 |            520 |            720 |              0 |             500 |                   1740 |                             4620 | clean_main_train_only |                            1740 | True                                 | False                             | False                                        | False            |

## Delta comparison

| comparison                                              |   acc_delta |   macro_f1_delta |   pred_NE_delta |   pred_REFUTE_delta |   pred_SUPPORT_delta |
|:--------------------------------------------------------|------------:|-----------------:|----------------:|--------------------:|---------------------:|
| Stage75F_full_bridge - Stage71_retry2_primary           | -0.00138891 |      -0.00184226 |              -1 |                   0 |                    1 |
| Stage80F_conservative_Stage80A - Stage71_retry2_primary | -0.00416666 |      -0.00548313 |              -3 |                   0 |                    3 |
| Stage80F_conservative_Stage80A - Stage75F_full_bridge   | -0.00277776 |      -0.00364087 |              -2 |                   0 |                    2 |

## Recommendations

| recommendation                                              | rationale                                                                                                           |
|:------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|
| Do not promote Stage80A conservative bridge to primary.     | Stage80F failed strict clean-dev macro preservation versus Stage71.                                                 |
| Do not run Stage81 external diagnostic by default.          | External diagnostic is lower priority when the candidate already loses clean-dev macro beyond tolerance.            |
| Keep Stage71_retry2 as primary.                             | Stage71 still has the best clean-dev macro-F1 among Stage71, Stage75F, and Stage80F.                                |
| Keep Stage80F as a valid negative result.                   | The runner integration and metadata are valid; the failure is model/result-level, not execution-level.              |
| Next useful branch is ablation, not full Stage80A external. | Stage80A combined polarity+NE safety may be too heavy; test smaller polarity-only or NE-safety-only only if needed. |

## Decision reason

Stage80A integration is technically valid, but Stage80F clean-dev macro-F1 falls below Stage71 by more than the strict tolerance. Do not spend the next run on external diagnostic by default; keep Stage71_retry2 as primary.

## Recommended next stage

Stage83 design smaller polarity-only or NE-safety-only ablation, or stop Stage80A branch
