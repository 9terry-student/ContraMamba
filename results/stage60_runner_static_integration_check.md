# Stage60 — Runner Static Integration Check

## Decision

`STAGE60_RUNNER_STATIC_INTEGRATION_READY`

## Summary

| stage   | decision                                | script                                  | help_flags_ok   | source_checks_ok   | required_paths_ok   | stage58_stage59_metadata_ok   |   bridge_row_count | training_executed   | external_eval_executed   | next_stage                                 |
|:--------|:----------------------------------------|:----------------------------------------|:----------------|:-------------------|:--------------------|:------------------------------|-------------------:|:--------------------|:-------------------------|:-------------------------------------------|
| Stage60 | STAGE60_RUNNER_STATIC_INTEGRATION_READY | scripts/train_controlled_v6b_minimal.py | True            | True               | True                | True                          |                520 | False               | False                    | Stage61 bridge-enabled frozen recovery run |

## Help flag check

| flag                         | present_in_help   |
|:-----------------------------|:------------------|
| --stage57-bridge-train-jsonl | True              |
| --stage57-bridge-train-mode  | True              |

## Source static checks

| check                                    | pattern                                            | pass   |
|:-----------------------------------------|:---------------------------------------------------|:-------|
| arg_stage57_bridge_train_jsonl           | stage57_bridge_train_jsonl                         | True   |
| arg_stage57_bridge_train_mode            | stage57_bridge_train_mode                          | True   |
| append_train_only_mode                   | append_train_only                                  | True   |
| stage57_report_metadata                  | stage57_bridge_train_row_count                     | True   |
| stage57_train_only_metadata              | stage57_bridge_train_only                          | True   |
| stage57_dev_false_metadata               | stage57_bridge_used_for_dev                        | True   |
| stage57_checkpoint_false_metadata        | stage57_bridge_used_for_checkpoint_selection       | True   |
| stage57_external_training_false_metadata | stage57_external_data_used_for_training            | True   |
| stage57_threshold_false_metadata         | stage57_external_metrics_used_for_threshold_tuning | True   |
| forbidden_source_guard_vitaminc          | vitaminc                                           | True   |
| forbidden_source_guard_time_swap         | time_swap                                          | True   |
| console_stage60_marker                   | [stage60]                                          | True   |

## Required path checks

| name                | path                                             | exists   |    size |
|:--------------------|:-------------------------------------------------|:---------|--------:|
| main_clean_data     | data/controlled_v5_v3_without_time_swap.jsonl    | True     | 1879999 |
| stage57_bridge_data | data/stage57_nonleaking_external_bridge.jsonl    | True     |  374520 |
| stage58_audit       | results/stage58_bridge_dataset_static_audit.json | True     |   18326 |
| stage59_plan        | results/stage59_bridge_integration_plan.json     | True     |    7441 |

## Stage58/59/57 metadata checks

| check                     | value                                 | pass   |
|:--------------------------|:--------------------------------------|:-------|
| stage58_ready             | STAGE58_BRIDGE_STATIC_AUDIT_READY     | True   |
| stage59_ready             | STAGE59_BRIDGE_INTEGRATION_PLAN_READY | True   |
| stage59_primary_mode      | bridge_train_only_append_1x           | True   |
| stage57_row_count         | 520                                   | True   |
| stage57_generation_source | ['synthetic_nonleaking_bridge']       | True   |
| stage57_leakage_policy    | ['no_vitaminc_text_or_labels_used']   | True   |

## Stage57 bridge label counts

| final_label   |   label |   count |
|:--------------|--------:|--------:|
| REFUTE        |       0 |     160 |
| NOT_ENTITLED  |       1 |     200 |
| SUPPORT       |       2 |     160 |

## Stage57 bridge family counts

| stage57_bridge_family      |   count |
|:---------------------------|--------:|
| distractor_evidence_bridge |      40 |
| entity_attribute_bridge    |     120 |
| lexical_paraphrase_bridge  |     120 |
| numeric_comparison_bridge  |     120 |
| temporal_comparison_bridge |     120 |

## Stage57 bridge family x label

| stage57_bridge_family      | final_label   |   label |   count |
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

## Leakage policy

- VitaminC text used for training: `False`
- VitaminC labels used for training: `False`
- Climate-FEVER used for training: `False`
- External metrics used for threshold tuning: `False`
- Stage57 bridge train-only policy: `True`
- Clean dev selection-only policy: `True`
- time_swap used: `False`

## Note

Static integration check only. No training or external evaluation was executed.
