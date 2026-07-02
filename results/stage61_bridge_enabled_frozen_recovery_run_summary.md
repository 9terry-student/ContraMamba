# Stage61 — Bridge-enabled Frozen Recovery Run Summary

## Compact summary

| stage   | decision                                                 | run_dir                                                            |   best_epoch | selected_epoch   |   best_dev_acc |   best_dev_macro_f1 | stage57_bridge_train_enabled   | stage57_bridge_train_mode   |   stage57_bridge_train_row_count | stage57_bridge_train_only   | stage57_bridge_used_for_dev   | stage57_bridge_used_for_checkpoint_selection   | stage57_external_data_used_for_training   | stage57_external_metrics_used_for_threshold_tuning   | time_swap_used_in_main_clean_data   | stage45c_enabled   |   stage45c_support_recovery_weight |   stage45c_entitled_ne_penalty_weight |
|:--------|:---------------------------------------------------------|:-------------------------------------------------------------------|-------------:|:-----------------|---------------:|--------------------:|:-------------------------------|:----------------------------|---------------------------------:|:----------------------------|:------------------------------|:-----------------------------------------------|:------------------------------------------|:-----------------------------------------------------|:------------------------------------|:-------------------|-----------------------------------:|--------------------------------------:|
| Stage61 | STAGE61_BRIDGE_ENABLED_FROZEN_RECOVERY_RUN_SUMMARY_READY | results/stage61_bridge_enabled_frozen_recovery_run_20260702_055007 |          191 |                  |       0.970833 |            0.958808 | True                           | append_train_only           |                              520 | True                        | False                         | False                                          | False                                     | False                                                | False                               | True               |                                0.1 |                                   0.1 |

## Distributions

_empty_

## Per-label metrics

_empty_

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

## Leakage policy

- Stage57 bridge train only: `True`
- Stage57 bridge used for dev: `False`
- Stage57 bridge used for checkpoint selection: `False`
- Stage57 external data used for training: `False`
- Stage57 external metrics used for threshold tuning: `False`
- time_swap used in main clean data: `False`
