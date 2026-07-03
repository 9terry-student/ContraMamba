# Stage83D - A3 NE-Safety-Only Clean-Dev Decision Report

## Decision

`STAGE83D_A3_NE_SAFETY_ONLY_CLEAN_DEV_DECISION_READY`

## Stage83C decision

`ALLOW_EXTERNAL_DIAGNOSTIC`

## Candidate status

`CLEAN_DEV_PRIMARY_CANDIDATE_PENDING_EXTERNAL`

## Summary

| stage    | decision                                            | stage83c_decision         | candidate_status                             | decision_reason                                                                                                                                                                                                                                     |   stage71_acc |   stage71_macro_f1 |   stage83c_acc |   stage83c_macro_f1 |   stage83c_minus_stage71_acc |   stage83c_minus_stage71_macro_f1 |   stage83c_minus_stage80f_macro_f1 |   stage83c_combined_bridge_rows |   stage83c_final_train_row_count_expected |   stage83c_stage83a_rows | external_eval_executed   | training_executed   | recommended_next_stage                            |
|:---------|:----------------------------------------------------|:--------------------------|:---------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------:|-------------------:|---------------:|--------------------:|-----------------------------:|----------------------------------:|-----------------------------------:|--------------------------------:|------------------------------------------:|-------------------------:|:-------------------------|:--------------------|:--------------------------------------------------|
| Stage83D | STAGE83D_A3_NE_SAFETY_ONLY_CLEAN_DEV_DECISION_READY | ALLOW_EXTERNAL_DIAGNOSTIC | CLEAN_DEV_PRIMARY_CANDIDATE_PENDING_EXTERNAL | Stage83C A3 NE-safety-only bridge improves clean-dev accuracy and macro-F1 over Stage71, while preserving bridge isolation and excluding bridge rows from pairwise loss. External diagnostic is now warranted before promoting it as final primary. |         0.975 |           0.964047 |       0.979167 |            0.969664 |                   0.00416666 |                        0.00561689 |                             0.0111 |                            1400 |                                      4280 |                      160 | False                    | False               | Stage84 VitaminC external diagnostic for Stage83C |

## Checks

| check                         | pass   |
|:------------------------------|:-------|
| stage83c_summary_exists       | True   |
| stage83c_train_report_exists  | True   |
| stage83b_static_ready         | True   |
| stage83c_run_ready            | True   |
| stage83c_metadata_ok          | True   |
| stage83c_preservation_ok      | True   |
| stage83c_beats_stage71_acc    | True   |
| stage83c_beats_stage71_macro  | True   |
| stage83c_combined_bridge_1400 | True   |
| stage83c_final_train_4280     | True   |
| stage83c_external_not_used    | True   |
| stage83c_time_swap_false      | True   |
| decision_nonempty             | True   |

## Signals

| signal                              | value   |
|:------------------------------------|:--------|
| stage83c_run_ready                  | True    |
| stage83c_returncode_zero            | True    |
| stage83c_metadata_ok                | True    |
| stage83c_preservation_ok            | True    |
| stage83b_static_ready               | True    |
| stage83a_generation_ready           | True    |
| stage83_branch_allowed_a3           | True    |
| stage80g_stopped_full_stage80a      | True    |
| stage79_revised_stage75_not_default | True    |
| stage83c_beats_stage71_acc          | True    |
| stage83c_beats_stage71_macro        | True    |
| stage83c_beats_stage80f_macro       | True    |
| stage83c_no_stage75_full_bridge     | True    |
| stage83c_stage83a_rows_160          | True    |
| stage83c_combined_bridge_1400       | True    |
| stage83c_final_train_4280           | True    |
| stage83c_external_not_used          | True    |
| stage83c_time_swap_false            | True    |

## Run comparison

| stage                         | source_path                                                                                                                                           |   best_epoch |   selected_epoch |   best_dev_acc |   best_dev_macro_f1 |   pred_NOT_ENTITLED |   pred_REFUTE |   pred_SUPPORT |   stage57_rows |   stage66_rows |   stage75_rows |   stage80a_or_stage83a_rows |   combined_bridge_rows |   final_train_row_count_expected | pairwise_source       |   pairwise_bridge_rows_excluded | clean_dev_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   |
|:------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|-------------:|-----------------:|---------------:|--------------------:|--------------------:|--------------:|---------------:|---------------:|---------------:|---------------:|----------------------------:|-----------------------:|---------------------------------:|:----------------------|--------------------------------:|:-------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|
| Stage71_retry2_primary        | results/stage71_retry2_stage57_stage66_bridge_enabled_frozen_recovery_run_20260702_072827/stage71_retry2_stage57_stage66_train_report.json            |          151 |              151 |       0.975    |            0.964047 |                 522 |            90 |            108 |            520 |            720 |              0 |                           0 |                   1240 |                              nan | clean_main_train_only |                            1240 | True                                 | False                             | False                                        | False            |
| Stage75F_full_stage75_bridge  | results/stage75f_stage57_stage66_stage75_bridge_enabled_frozen_recovery_run_20260703_002555/stage75f_stage57_stage66_stage75_train_report.json        |          184 |              184 |       0.973611 |            0.962205 |                 521 |            90 |            109 |            520 |            720 |           1020 |                           0 |                   2260 |                              nan | clean_main_train_only |                            2260 | True                                 | False                             | False                                        | False            |
| Stage80F_full_stage80a_bridge | results/stage80f_stage57_stage66_stage80a_conservative_bridge_frozen_recovery_run_20260703_013106/stage80f_stage57_stage66_stage80a_train_report.json |          180 |              180 |       0.970833 |            0.958564 |                 519 |            90 |            111 |            520 |            720 |              0 |                         500 |                   1740 |                             4620 | clean_main_train_only |                            1740 | True                                 | False                             | False                                        | False            |
| Stage83C_a3_ne_safety_only    | results/stage83c_stage57_stage66_stage83a_ne_safety_only_frozen_recovery_run_20260703_020152/stage83c_stage57_stage66_stage83a_train_report.json      |          158 |              158 |       0.979167 |            0.969664 |                 525 |            90 |            105 |            520 |            720 |              0 |                         160 |                   1400 |                             4280 | clean_main_train_only |                            1400 | True                                 | False                             | False                                        | False            |

## Delta comparison

| comparison                                                 |   acc_delta |   macro_f1_delta |   pred_NE_delta |   pred_REFUTE_delta |   pred_SUPPORT_delta |
|:-----------------------------------------------------------|------------:|-----------------:|----------------:|--------------------:|---------------------:|
| Stage75F_full_stage75_bridge - Stage71_retry2_primary      | -0.00138891 |      -0.00184226 |              -1 |                   0 |                    1 |
| Stage80F_full_stage80a_bridge - Stage71_retry2_primary     | -0.00416666 |      -0.00548313 |              -3 |                   0 |                    3 |
| Stage83C_a3_ne_safety_only - Stage71_retry2_primary        |  0.00416666 |       0.00561689 |               3 |                   0 |                   -3 |
| Stage83C_a3_ne_safety_only - Stage75F_full_stage75_bridge  |  0.00555557 |       0.00745915 |               4 |                   0 |                   -4 |
| Stage83C_a3_ne_safety_only - Stage80F_full_stage80a_bridge |  0.00833333 |       0.0111     |               6 |                   0 |                   -6 |

## Recommendations

| recommendation                                              | rationale                                                                                  |
|:------------------------------------------------------------|:-------------------------------------------------------------------------------------------|
| Allow Stage84 external diagnostic for Stage83C.             | Stage83C beats Stage71 on clean-dev acc and macro-F1 and has valid metadata isolation.     |
| Do not promote Stage83C as final primary yet.               | External VitaminC behavior has not been measured for this candidate.                       |
| Keep Stage71 as current primary until Stage84 is evaluated. | Stage71 is the currently external-diagnosed primary; Stage83C is clean-dev candidate only. |
| Do not revive full Stage80A or Stage75 full bridge.         | Both were already rejected/not-default by prior decision reports.                          |

## Decision reason

Stage83C A3 NE-safety-only bridge improves clean-dev accuracy and macro-F1 over Stage71, while preserving bridge isolation and excluding bridge rows from pairwise loss. External diagnostic is now warranted before promoting it as final primary.

## Recommended next stage

Stage84 VitaminC external diagnostic for Stage83C
