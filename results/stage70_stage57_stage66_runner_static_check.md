# Stage70 — Stage57 + Stage66 Runner Static Check

## Decision

`STAGE70_STAGE57_STAGE66_RUNNER_STATIC_CHECK_READY`

## Summary

| stage   | decision                                          | script                                  | help_flags_ok   | source_checks_ok   | data_checks_ok   |   main_row_count |   stage57_bridge_row_count |   stage66_bridge_row_count |   combined_bridge_row_count | primary_mode                      | training_executed   | external_eval_executed   | next_stage                                                                          |
|:--------|:--------------------------------------------------|:----------------------------------------|:----------------|:-------------------|:-----------------|-----------------:|---------------------------:|---------------------------:|----------------------------:|:----------------------------------|:--------------------|:-------------------------|:------------------------------------------------------------------------------------|
| Stage70 | STAGE70_STAGE57_STAGE66_RUNNER_STATIC_CHECK_READY | scripts/train_controlled_v6b_minimal.py | True            | True               | True             |             3600 |                        520 |                        720 |                        1240 | stage57_stage66_append_train_only | False               | False                    | Stage71 bridge-enabled frozen recovery run with Stage57 + Stage66 train-only append |

## Help flag check

| flag                         | present_in_help   |
|:-----------------------------|:------------------|
| --stage57-bridge-train-jsonl | True              |
| --stage57-bridge-train-mode  | True              |
| --stage66-bridge-train-jsonl | True              |
| --stage66-bridge-train-mode  | True              |

## Source static checks

| check                                             | patterns                                                                                  | pass   |
|:--------------------------------------------------|:------------------------------------------------------------------------------------------|:-------|
| arg_stage57_bridge_train_jsonl_preserved          | ['stage57_bridge_train_jsonl']                                                            | True   |
| arg_stage57_bridge_train_mode_preserved           | ['stage57_bridge_train_mode']                                                             | True   |
| arg_stage66_bridge_train_jsonl_added              | ['stage66_bridge_train_jsonl']                                                            | True   |
| arg_stage66_bridge_train_mode_added               | ['stage66_bridge_train_mode']                                                             | True   |
| append_train_only_mode_present                    | ['append_train_only']                                                                     | True   |
| stage66_bridge_enabled_report                     | ['stage66_bridge_enabled']                                                                | True   |
| stage66_row_count_report                          | ['stage66_bridge_row_count', 'stage66_bridge_train_row_count']                            | True   |
| stage66_label_counts_report                       | ['stage66_bridge_label_counts']                                                           | True   |
| stage66_family_counts_report                      | ['stage66_bridge_family_counts']                                                          | True   |
| stage66_train_only_report                         | ['stage66_bridge_train_only']                                                             | True   |
| stage66_dev_false_report                          | ['stage66_used_for_dev', 'stage66_bridge_used_for_dev']                                   | True   |
| stage66_checkpoint_false_report                   | ['stage66_used_for_checkpoint_selection', 'stage66_bridge_used_for_checkpoint_selection'] | True   |
| combined_bridge_enabled_report                    | ['combined_bridge_enabled']                                                               | True   |
| combined_bridge_row_count_report                  | ['combined_bridge_row_count']                                                             | True   |
| combined_bridge_label_counts_report               | ['combined_bridge_label_counts']                                                          | True   |
| combined_bridge_train_only_report                 | ['combined_bridge_train_only']                                                            | True   |
| bridge_sources_enabled_report                     | ['bridge_sources_enabled']                                                                | True   |
| clean_dev_checkpoint_selection_report             | ['clean_dev_for_checkpoint_selection']                                                    | True   |
| external_training_false_report                    | ['external_data_used_for_training']                                                       | True   |
| external_threshold_false_report                   | ['external_metrics_used_for_threshold_tuning']                                            | True   |
| time_swap_false_report_or_guard                   | ['time_swap_used', 'time_swap']                                                           | True   |
| forbidden_path_guard_marker_present_vitaminc      | ['vitaminc']                                                                              | True   |
| forbidden_path_guard_marker_present_vitamin-c     | ['vitamin-c']                                                                             | True   |
| forbidden_path_guard_marker_present_climate_fever | ['climate_fever']                                                                         | True   |
| forbidden_path_guard_marker_present_climate-fever | ['climate-fever']                                                                         | True   |
| forbidden_path_guard_marker_present_feverous      | ['feverous']                                                                              | True   |
| forbidden_path_guard_marker_present_time_swap     | ['time_swap']                                                                             | True   |
| forbidden_path_guard_marker_present_stage43       | ['stage43']                                                                               | True   |
| forbidden_path_guard_marker_present_stage53       | ['stage53']                                                                               | True   |
| forbidden_path_guard_marker_present_stage55       | ['stage55']                                                                               | True   |
| forbidden_path_guard_marker_present_stage63       | ['stage63']                                                                               | True   |
| forbidden_path_guard_marker_present_stage65       | ['stage65']                                                                               | True   |

## Data/report checks

| check                                   | pass   |
|:----------------------------------------|:-------|
| main_row_count_ok                       | True   |
| stage57_row_count_ok                    | True   |
| stage66_row_count_ok                    | True   |
| combined_bridge_row_count_ok            | True   |
| stage57_label_counts_ok                 | True   |
| stage66_label_counts_ok                 | True   |
| combined_bridge_label_counts_ok         | True   |
| stage66_family_counts_ok                | True   |
| stage69_ready                           | True   |
| stage68_ready                           | True   |
| stage67_ready                           | True   |
| stage69_primary_mode_ok                 | True   |
| stage69_clean_dev_selection_ok          | True   |
| stage69_no_external_training_ok         | True   |
| stage69_no_external_threshold_tuning_ok | True   |
| stage69_no_time_swap_ok                 | True   |

## Bridge label counts

| label        |   stage57_count |   stage66_count |   combined_bridge_count |   expected_combined |
|:-------------|----------------:|----------------:|------------------------:|--------------------:|
| NOT_ENTITLED |             200 |              40 |                     240 |                 240 |
| REFUTE       |             160 |             320 |                     480 |                 480 |
| SUPPORT      |             160 |             360 |                     520 |                 520 |

## Stage57 family counts

| stage57_family             |   count |
|:---------------------------|--------:|
| distractor_evidence_bridge |      40 |
| entity_attribute_bridge    |     120 |
| lexical_paraphrase_bridge  |     120 |
| numeric_comparison_bridge  |     120 |
| temporal_comparison_bridge |     120 |

## Stage66 family counts

| stage66_family                      |   count |   expected |
|:------------------------------------|--------:|-----------:|
| numeric_temporal_comparison_bridge  |     120 |        120 |
| polarity_disambiguation_bridge      |     200 |        200 |
| refute_entitlement_recovery_bridge  |     160 |        160 |
| strict_ne_frame_safety_bridge       |      40 |         40 |
| support_entitlement_recovery_bridge |     200 |        200 |

## Execution policy

No training, no external evaluation, no smoke run, no full run.
Only py_compile and --help/static source/data checks were executed.

## Recommended next stage

Stage71 bridge-enabled frozen recovery run with Stage57 + Stage66 train-only append
