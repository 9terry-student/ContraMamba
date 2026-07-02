# Stage69 — Stage57 + Stage66 Bridge Integration Plan

## Decision

`STAGE69_STAGE57_STAGE66_BRIDGE_INTEGRATION_PLAN_READY`

## Summary

| stage   | decision                                              |   main_row_count |   stage57_bridge_row_count |   stage66_bridge_row_count |   combined_bridge_row_count |   stage57_ratio_vs_main |   stage66_ratio_vs_main |   combined_bridge_ratio_vs_main | primary_integration_mode          | clean_dev_for_checkpoint_selection   | stage57_used_for_dev   | stage66_used_for_dev   | stage57_used_for_checkpoint_selection   | stage66_used_for_checkpoint_selection   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   | polarity_encode_compatible   | recommended_next_stage              |
|:--------|:------------------------------------------------------|-----------------:|---------------------------:|---------------------------:|----------------------------:|------------------------:|------------------------:|--------------------------------:|:----------------------------------|:-------------------------------------|:-----------------------|:-----------------------|:----------------------------------------|:----------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|:-----------------------------|:------------------------------------|
| Stage69 | STAGE69_STAGE57_STAGE66_BRIDGE_INTEGRATION_PLAN_READY |             3600 |                        520 |                        720 |                        1240 |                0.144444 |                     0.2 |                        0.344444 | stage57_stage66_append_train_only | True                                 | False                  | False                  | False                                   | False                                   | False                             | False                                        | False            | True                         | Stage70 runner static check refresh |

## Checks

| check                           | pass   |
|:--------------------------------|:-------|
| main_row_count_ok               | True   |
| stage57_row_count_ok            | True   |
| stage66_row_count_ok            | True   |
| combined_bridge_row_count_ok    | True   |
| stage57_label_counts_ok         | True   |
| stage66_label_counts_ok         | True   |
| combined_bridge_label_counts_ok | True   |
| stage68_ready                   | True   |
| stage67_ready                   | True   |
| stage71a_ready                  | True   |
| polarity_encode_compatible      | True   |
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
