# Stage69 — Stage57 + Stage66 Bridge Integration Plan

## Decision

`STAGE69_STAGE57_STAGE66_BRIDGE_INTEGRATION_PLAN_READY`

## Summary

| stage   | decision                                              | main_data                                     |   main_row_count | stage57_bridge_data                           |   stage57_bridge_row_count | stage66_bridge_data                |   stage66_bridge_row_count |   combined_bridge_row_count |   stage57_ratio_vs_main |   stage66_ratio_vs_main |   combined_bridge_ratio_vs_main | primary_integration_mode          | clean_dev_for_checkpoint_selection   | stage57_used_for_dev   | stage66_used_for_dev   | stage57_used_for_checkpoint_selection   | stage66_used_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   | recommended_next_stage                                                   |
|:--------|:------------------------------------------------------|:----------------------------------------------|-----------------:|:----------------------------------------------|---------------------------:|:-----------------------------------|---------------------------:|----------------------------:|------------------------:|------------------------:|--------------------------------:|:----------------------------------|:-------------------------------------|:-----------------------|:-----------------------|:----------------------------------------|:----------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|:-------------------------------------------------------------------------|
| Stage69 | STAGE69_STAGE57_STAGE66_BRIDGE_INTEGRATION_PLAN_READY | data/controlled_v5_v3_without_time_swap.jsonl |             3600 | data/stage57_nonleaking_external_bridge.jsonl |                        520 | data/stage66_residual_bridge.jsonl |                        720 |                        1240 |                0.144444 |                     0.2 |                        0.344444 | stage57_stage66_append_train_only | True                                 | False                  | False                  | False                                   | False                                   | False                             | False                                        | False            | Stage70 runner patch for Stage57 + Stage66 train-only bridge integration |

## Checks

| check                           | pass   |
|:--------------------------------|:-------|
| main_clean_exists               | True   |
| stage57_exists                  | True   |
| stage66_exists                  | True   |
| main_row_count_ok               | True   |
| stage57_row_count_ok            | True   |
| stage66_row_count_ok            | True   |
| combined_bridge_row_count_ok    | True   |
| stage57_label_counts_ok         | True   |
| stage66_label_counts_ok         | True   |
| combined_bridge_label_counts_ok | True   |
| stage68_ready                   | True   |
| stage67_ready                   | True   |
| stage66_design_ready            | True   |
| no_stage57_stage66_id_overlap   | True   |
| no_stage57_stage66_pair_overlap | True   |
| no_main_stage57_pair_overlap    | True   |
| no_main_stage66_pair_overlap    | True   |

## Bridge label plan

| label        |   stage57_count |   stage66_count |   combined_bridge_count |   main_plus_bridge_count |
|:-------------|----------------:|----------------:|------------------------:|-------------------------:|
| NOT_ENTITLED |             200 |              40 |                     240 |                     2940 |
| REFUTE       |             160 |             320 |                     480 |                      930 |
| SUPPORT      |             160 |             360 |                     520 |                      970 |

## Bridge ratio plan

|   main_rows |   stage57_rows |   stage66_rows |   combined_bridge_rows |   stage57_ratio_vs_main |   stage66_ratio_vs_main |   combined_bridge_ratio_vs_main |
|------------:|---------------:|---------------:|-----------------------:|------------------------:|------------------------:|--------------------------------:|
|        3600 |            520 |            720 |                   1240 |                0.144444 |                     0.2 |                        0.344444 |

## Integration modes

| mode                              | description                                                                                   | train_data                                                         | dev_data                  |   bridge_rows | purpose                                                               |
|:----------------------------------|:----------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|:--------------------------|--------------:|:----------------------------------------------------------------------|
| baseline_no_bridge                | Stage51-compatible frozen recovery baseline without Stage57 or Stage66 bridge rows.           | clean main train split only                                        | clean main dev split only |             0 | Reference internal baseline.                                          |
| stage57_only_append_train_only    | Stage61-compatible Stage57 train-only bridge integration.                                     | clean main train split + Stage57 bridge rows                       | clean main dev split only |           520 | Reference bridge improvement baseline.                                |
| stage57_stage66_append_train_only | Primary Stage70 mode: append Stage57 and Stage66 bridge rows to train only after clean split. | clean main train split + Stage57 bridge rows + Stage66 bridge rows | clean main dev split only |          1240 | Test residual bridge expansion while preserving clean-dev evaluation. |

## Runner patch requirements

| requirement                                        | detail                                                                                           |
|:---------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| Add optional CLI --stage66-bridge-train-jsonl      | Path to data/stage66_residual_bridge.jsonl. Default None.                                        |
| Add optional CLI --stage66-bridge-train-mode       | Choices: none, append_train_only. Default none.                                                  |
| Keep existing Stage57 flags unchanged              | Do not break --stage57-bridge-train-jsonl or --stage57-bridge-train-mode.                        |
| Split clean main data before appending bridge rows | Stage57/Stage66 bridge rows must never enter dev or checkpoint selection.                        |
| Report separate and combined bridge metadata       | Include Stage57 counts, Stage66 counts, combined bridge counts, family counts, and label counts. |
| Hard fail forbidden external/time_swap paths       | Stage66 must be synthetic-only; no VitaminC/Stage63/Stage65 text, no time_swap.                  |
| Preserve default behavior                          | If both bridge modes are none, output must match pre-Stage60 default behavior.                   |

## Overlap checks

| overlap_check                |   count |
|:-----------------------------|--------:|
| stage57_stage66_id_overlap   |       0 |
| main_stage57_id_overlap      |       0 |
| main_stage66_id_overlap      |       0 |
| stage57_stage66_pair_overlap |       0 |
| main_stage57_pair_overlap    |       0 |
| main_stage66_pair_overlap    |       0 |

## Stage57 family counts

| stage57_family             |   count |
|:---------------------------|--------:|
| distractor_evidence_bridge |      40 |
| entity_attribute_bridge    |     120 |
| lexical_paraphrase_bridge  |     120 |
| numeric_comparison_bridge  |     120 |
| temporal_comparison_bridge |     120 |

## Stage66 family counts

| stage66_family                      |   count |
|:------------------------------------|--------:|
| numeric_temporal_comparison_bridge  |     120 |
| polarity_disambiguation_bridge      |     200 |
| refute_entitlement_recovery_bridge  |     160 |
| strict_ne_frame_safety_bridge       |      40 |
| support_entitlement_recovery_bridge |     200 |

## Recommended next stage

Stage70 runner patch for Stage57 + Stage66 train-only bridge integration
