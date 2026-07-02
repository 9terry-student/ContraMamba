# Stage62 — Stage51 vs Stage61 Internal Comparison

## Decision

`STAGE62_STAGE51_VS_STAGE61_INTERNAL_COMPARISON_READY`

## Comparison table

| stage                                  | run_dir                                                            |   best_epoch | selected_epoch   |   best_dev_acc |   best_dev_macro_f1 | stage57_bridge_train_enabled   | stage57_bridge_train_mode   |   stage57_bridge_train_row_count | stage57_bridge_train_only   | stage57_bridge_used_for_dev   | stage57_bridge_used_for_checkpoint_selection   | stage57_external_data_used_for_training   | stage57_external_metrics_used_for_threshold_tuning   | time_swap_used_in_main_clean_data   |
|:---------------------------------------|:-------------------------------------------------------------------|-------------:|:-----------------|---------------:|--------------------:|:-------------------------------|:----------------------------|---------------------------------:|:----------------------------|:------------------------------|:-----------------------------------------------|:------------------------------------------|:-----------------------------------------------------|:------------------------------------|
| Stage51_baseline_frozen_recovery       | results/stage51_frozen_recovery_run_20260702_044005                |           80 |                  |       0.973611 |            0.962855 | False                          | none                        |                                0 | False                       | False                         | False                                          | False                                     | False                                                | False                               |
| Stage61_bridge_enabled_frozen_recovery | results/stage61_bridge_enabled_frozen_recovery_run_20260702_055007 |          191 |                  |       0.970833 |            0.958808 | True                           | append_train_only           |                              520 | True                        | False                         | False                                          | False                                     | False                                                | False                               |

## Delta summary

| stage   | decision                                             | baseline                         | bridge_enabled                         |   stage51_best_dev_acc |   stage61_best_dev_acc |   delta_dev_acc |   stage51_best_dev_macro_f1 |   stage61_best_dev_macro_f1 |   delta_dev_macro_f1 |   bridge_row_count | bridge_train_only   | bridge_used_for_dev   | bridge_used_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   | clean_dev_preservation_verdict   | interpretation                                                                                                                                                                                                   |
|:--------|:-----------------------------------------------------|:---------------------------------|:---------------------------------------|-----------------------:|-----------------------:|----------------:|----------------------------:|----------------------------:|---------------------:|-------------------:|:--------------------|:----------------------|:---------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|:---------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage62 | STAGE62_STAGE51_VS_STAGE61_INTERNAL_COMPARISON_READY | Stage51_baseline_frozen_recovery | Stage61_bridge_enabled_frozen_recovery |               0.973611 |               0.970833 |     -0.00277776 |                    0.962855 |                    0.958808 |          -0.00404618 |                520 | True                | False                 | False                                  | False                             | False                                        | False            | PRESERVED_WITH_SMALL_DROP        | Stage61 bridge-enabled training preserved clean-dev performance within a small-drop band relative to Stage51. This supports moving to external diagnostic rerun, while not claiming external generalization yet. |

## Safety checks

| check                                         | value   | pass   |
|:----------------------------------------------|:--------|:-------|
| bridge_train_only                             | True    | True   |
| bridge_not_used_for_dev                       | False   | True   |
| bridge_not_used_for_checkpoint_selection      | False   | True   |
| no_external_data_used_for_training            | False   | True   |
| no_external_metrics_used_for_threshold_tuning | False   | True   |
| time_swap_absent                              | False   | True   |

## Bridge label counts

| label        |   count |
|:-------------|--------:|
| NOT_ENTITLED |     200 |
| REFUTE       |     160 |
| SUPPORT      |     160 |

## Bridge family counts

| family                     |   count |
|:---------------------------|--------:|
| distractor_evidence_bridge |      40 |
| entity_attribute_bridge    |     120 |
| lexical_paraphrase_bridge  |     120 |
| numeric_comparison_bridge  |     120 |
| temporal_comparison_bridge |     120 |

## Interpretation

Stage61 bridge-enabled training preserved clean-dev performance within a small-drop band relative to Stage51. This supports moving to external diagnostic rerun, while not claiming external generalization yet.

## Recommended next stage

Stage63 bridge-enabled VitaminC external diagnostic rerun
