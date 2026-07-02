# Stage59 — Bridge Integration Plan

## Decision

`STAGE59_BRIDGE_INTEGRATION_PLAN_READY`

## Summary

| stage   | decision                              | main_data                                     | bridge_data                                   | source_stage56                                                  | source_stage57_audit                                  | source_stage58_audit                             |   main_row_count |   bridge_row_count |   bridge_ratio_vs_main | stage58_decision                  | primary_integration_mode    | clean_dev_for_checkpoint_selection   | external_data_used_for_training   | external_data_used_for_checkpoint_selection   | external_data_used_for_threshold_tuning   | time_swap_used   | composer_threshold_tuning   | recommended_next_stage                                   |
|:--------|:--------------------------------------|:----------------------------------------------|:----------------------------------------------|:----------------------------------------------------------------|:------------------------------------------------------|:-------------------------------------------------|-----------------:|-------------------:|-----------------------:|:----------------------------------|:----------------------------|:-------------------------------------|:----------------------------------|:----------------------------------------------|:------------------------------------------|:-----------------|:----------------------------|:---------------------------------------------------------|
| Stage59 | STAGE59_BRIDGE_INTEGRATION_PLAN_READY | data/controlled_v5_v3_without_time_swap.jsonl | data/stage57_nonleaking_external_bridge.jsonl | results/stage56_nonleaking_external_transfer_bridge_design.json | results/stage57_nonleaking_external_bridge_audit.json | results/stage58_bridge_dataset_static_audit.json |             3600 |                520 |               0.144444 | STAGE58_BRIDGE_STATIC_AUDIT_READY | bridge_train_only_append_1x | True                                 | False                             | False                                         | False                                     | False            | False                       | Stage60 runner patch for optional train-only bridge data |

## Label plan

| label        |   main_count |   bridge_count |   combined_if_train_append |   bridge_to_main_ratio |
|:-------------|-------------:|---------------:|---------------------------:|-----------------------:|
| NOT_ENTITLED |         2700 |            200 |                       2900 |              0.0740741 |
| REFUTE       |          450 |            160 |                        610 |              0.355556  |
| SUPPORT      |          450 |            160 |                        610 |              0.355556  |

## Bridge family counts

| bridge_family              |   count |
|:---------------------------|--------:|
| distractor_evidence_bridge |      40 |
| entity_attribute_bridge    |     120 |
| lexical_paraphrase_bridge  |     120 |
| numeric_comparison_bridge  |     120 |
| temporal_comparison_bridge |     120 |

## Bridge family x label

| bridge_family              | final_label   |   label |   count |
|:---------------------------|:--------------|--------:|--------:|
| distractor_evidence_bridge | NOT_ENTITLED  |       1 |      40 |
| entity_attribute_bridge    | REFUTE        |       0 |      40 |
| entity_attribute_bridge    | NOT_ENTITLED  |       1 |      40 |
| entity_attribute_bridge    | SUPPORT       |       2 |      40 |
| lexical_paraphrase_bridge  | REFUTE        |       0 |      40 |
| lexical_paraphrase_bridge  | NOT_ENTITLED  |       1 |      40 |
| lexical_paraphrase_bridge  | SUPPORT       |       2 |      40 |
| numeric_comparison_bridge  | REFUTE        |       0 |      40 |
| numeric_comparison_bridge  | NOT_ENTITLED  |       1 |      40 |
| numeric_comparison_bridge  | SUPPORT       |       2 |      40 |
| temporal_comparison_bridge | REFUTE        |       0 |      40 |
| temporal_comparison_bridge | NOT_ENTITLED  |       1 |      40 |
| temporal_comparison_bridge | SUPPORT       |       2 |      40 |

## Integration modes

| mode                              | description                                                                                               | main_data                                     | bridge_data                                   | clean_dev_for_selection   | bridge_train_only   | recommended_for                | bridge_sampling                    |
|:----------------------------------|:----------------------------------------------------------------------------------------------------------|:----------------------------------------------|:----------------------------------------------|:--------------------------|:--------------------|:-------------------------------|:-----------------------------------|
| baseline_no_bridge                | Stage51/52 frozen recovery baseline. No bridge data.                                                      | data/controlled_v5_v3_without_time_swap.jsonl |                                               | True                      | False               | reference_control              | nan                                |
| bridge_train_only_append_1x       | Split clean main data first, then append all Stage57 bridge rows to train only.                           | data/controlled_v5_v3_without_time_swap.jsonl | data/stage57_nonleaking_external_bridge.jsonl | True                      | True                | primary_stage60_implementation | use_each_bridge_row_once_per_epoch |
| bridge_train_only_family_balanced | Split clean main data first, then append Stage57 bridge rows with family-balanced sampler if implemented. | data/controlled_v5_v3_without_time_swap.jsonl | data/stage57_nonleaking_external_bridge.jsonl | True                      | True                | secondary_diagnostic_only      | family_balanced_optional           |

## Guardrails

| guardrail                    | requirement                                                                                                            |
|:-----------------------------|:-----------------------------------------------------------------------------------------------------------------------|
| clean_dev_selection_only     | Checkpoint selection must remain on clean main dev split, not bridge dev and not external data.                        |
| bridge_train_only            | Stage57 bridge rows may be appended to train split only after the clean train/dev split is created.                    |
| no_external_tuning           | VitaminC/Climate-FEVER metrics must not be used for loss weights, thresholds, checkpoint selection, or early stopping. |
| time_swap_excluded           | Do not use data/controlled_v5_v3.jsonl time_swap rows in main classification training.                                 |
| no_composer_threshold_tuning | Do not tune composer thresholds on Stage53A/55 external failures.                                                      |
| report_all_three_axes        | Future run report must separately show clean dev, controlled pairwise diagnostics, and external diagnostic.            |

## Stage60 implementation requirements

| requirement                                                          | purpose                                                                                           |
|:---------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|
| Add CLI arg --stage57-bridge-train-jsonl                             | Optional path to Stage57 bridge data.                                                             |
| Add CLI arg --stage57-bridge-train-mode                              | Allowed values: none, append_train_only. Default none.                                            |
| Split clean main data before appending bridge rows                   | Prevent bridge rows from entering clean dev/checkpoint selection.                                 |
| Tag report fields                                                    | Record bridge row count, bridge label counts, bridge family counts, and bridge train-only policy. |
| Hard fail if bridge path references VitaminC/Climate-FEVER/time_swap | Protect non-leaking bridge policy.                                                                |

## Leakage policy

- VitaminC text used for training: `False`
- VitaminC labels used for training: `False`
- Climate-FEVER used for training: `False`
- External metrics used for threshold tuning: `False`
- Stage57 bridge train only: `True`
- Clean dev selection only: `True`
- time_swap used: `False`

## Recommended next stage

Stage60: runner patch for optional train-only bridge data.
