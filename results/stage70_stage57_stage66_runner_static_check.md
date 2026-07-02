# Stage70 — Stage57 + Stage66 Runner Static Check

## Decision

`STAGE70_STAGE57_STAGE66_RUNNER_STATIC_CHECK_READY`

## Summary

| stage   | decision                                          | script                                  | help_flags_ok   | source_checks_ok   | data_checks_ok   |   main_row_count |   stage57_bridge_row_count |   stage66_bridge_row_count |   combined_bridge_row_count | primary_mode                      | stage66_encode_label_tensors_ok   | training_executed   | external_eval_executed   | next_stage                                                                              |
|:--------|:--------------------------------------------------|:----------------------------------------|:----------------|:-------------------|:-----------------|-----------------:|---------------------------:|---------------------------:|----------------------------:|:----------------------------------|:----------------------------------|:--------------------|:-------------------------|:----------------------------------------------------------------------------------------|
| Stage70 | STAGE70_STAGE57_STAGE66_RUNNER_STATIC_CHECK_READY | scripts/train_controlled_v6b_minimal.py | True            | True               | True             |             3600 |                        520 |                        720 |                        1240 | stage57_stage66_append_train_only | True                              | False               | False                    | Stage71 retry bridge-enabled frozen recovery run with encoder-compatible Stage66 bridge |

## Help flag check

| flag                         | present_in_help   |
|:-----------------------------|:------------------|
| --stage57-bridge-train-jsonl | True              |
| --stage57-bridge-train-mode  | True              |
| --stage66-bridge-train-jsonl | True              |
| --stage66-bridge-train-mode  | True              |

## Source static checks

| check                                      | pass   |
|:-------------------------------------------|:-------|
| stage57_bridge_train_jsonl                 | True   |
| stage57_bridge_train_mode                  | True   |
| stage66_bridge_train_jsonl                 | True   |
| stage66_bridge_train_mode                  | True   |
| append_train_only                          | True   |
| stage66_bridge_enabled                     | True   |
| stage66_bridge_row_count                   | True   |
| stage66_bridge_label_counts                | True   |
| stage66_bridge_family_counts               | True   |
| stage66_bridge_train_only                  | True   |
| stage66_used_for_dev                       | True   |
| stage66_used_for_checkpoint_selection      | True   |
| combined_bridge_enabled                    | True   |
| combined_bridge_row_count                  | True   |
| combined_bridge_label_counts               | True   |
| combined_bridge_train_only                 | True   |
| bridge_sources_enabled                     | True   |
| clean_dev_for_checkpoint_selection         | True   |
| external_data_used_for_training            | True   |
| external_metrics_used_for_threshold_tuning | True   |
| time_swap_used                             | True   |

## Data/report checks

| check                           | pass   |
|:--------------------------------|:-------|
| help_flags_ok                   | True   |
| source_checks_ok                | True   |
| stage69_ready                   | True   |
| stage68_ready                   | True   |
| stage71a_ready                  | True   |
| stage66_encode_label_tensors_ok | True   |
| combined_bridge_row_count_ok    | True   |
| combined_bridge_label_counts_ok | True   |

## Bridge label counts

| label        |   stage57_count |   stage66_count |   combined_bridge_count |   expected_combined |
|:-------------|----------------:|----------------:|------------------------:|--------------------:|
| NOT_ENTITLED |             200 |              40 |                     240 |                 240 |
| REFUTE       |             160 |             320 |                     480 |                 480 |
| SUPPORT      |             160 |             360 |                     520 |                 520 |

## Stage66 polarity label counts

| polarity_label   |   count |   encoder_id | encoder_compatible   |
|:-----------------|--------:|-------------:|:---------------------|
| NONE             |      40 |            0 | True                 |
| REFUTE           |     320 |            1 | True                 |
| SUPPORT          |     360 |            2 | True                 |
