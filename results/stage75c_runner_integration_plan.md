# Stage75C — Runner Integration Plan

## Decision

`STAGE75C_RUNNER_INTEGRATION_PLAN_READY`

## Summary

| stage    | decision                               | purpose                                                                         | main_clean_jsonl                              | stage57_bridge_jsonl                          | stage66_bridge_jsonl               | stage75_bridge_jsonl                        |   main_clean_row_count |   clean_train_estimated_row_count |   clean_dev_estimated_row_count |   stage57_row_count |   stage66_row_count |   stage75_row_count |   combined_bridge_row_count_stage57_stage66_stage75 |   final_train_row_count_if_appended | stage75_label_counts                                | combined_bridge_label_counts                          | integration_mode                          | clean_dev_for_checkpoint_selection   | stage75_used_for_dev   | stage75_used_for_checkpoint_selection   | stage75_excluded_from_intervention_pairwise_loss   | external_data_used_for_training   | external_metrics_used_for_threshold_tuning   | time_swap_used   | training_executed   | external_eval_executed   | recommended_next_stage                                            |
|:---------|:---------------------------------------|:--------------------------------------------------------------------------------|:----------------------------------------------|:----------------------------------------------|:-----------------------------------|:--------------------------------------------|-----------------------:|----------------------------------:|--------------------------------:|--------------------:|--------------------:|--------------------:|----------------------------------------------------:|------------------------------------:|:----------------------------------------------------|:------------------------------------------------------|:------------------------------------------|:-------------------------------------|:-----------------------|:----------------------------------------|:---------------------------------------------------|:----------------------------------|:---------------------------------------------|:-----------------|:--------------------|:-------------------------|:------------------------------------------------------------------|
| Stage75C | STAGE75C_RUNNER_INTEGRATION_PLAN_READY | Plan runner integration for Stage75 train-only bridge append before code patch. | data/controlled_v5_v3_without_time_swap.jsonl | data/stage57_nonleaking_external_bridge.jsonl | data/stage66_residual_bridge.jsonl | data/stage75_targeted_residual_bridge.jsonl |                   3600 |                              2880 |                             720 |                 520 |                 720 |                1020 |                                                2260 |                                5140 | {"SUPPORT": 480, "REFUTE": 460, "NOT_ENTITLED": 80} | {"SUPPORT": 1000, "REFUTE": 940, "NOT_ENTITLED": 320} | stage57_stage66_stage75_append_train_only | True                                 | False                  | False                                   | True                                               | False                             | False                                        | False            | False               | False                    | Stage75D patch runner to support Stage75 train-only bridge append |

## Row ratios

| scope                                   |   row_count |   ratio_to_main_clean |   ratio_to_clean_train |
|:----------------------------------------|------------:|----------------------:|-----------------------:|
| main_clean_total                        |        3600 |              1        |               1.25     |
| clean_train_estimated                   |        2880 |              0.8      |               1        |
| clean_dev_estimated                     |         720 |              0.2      |               0.25     |
| stage57_bridge                          |         520 |              0.144444 |               0.180556 |
| stage66_bridge                          |         720 |              0.2      |               0.25     |
| stage75_bridge                          |        1020 |              0.283333 |               0.354167 |
| stage57_stage66_stage75_combined_bridge |        2260 |              0.627778 |               0.784722 |
| final_train_if_appended                 |        5140 |              1.42778  |               1.78472  |

## Label plan

| label        |   stage57 |   stage66 |   stage75 |   combined_bridge |
|:-------------|----------:|----------:|----------:|------------------:|
| SUPPORT      |       160 |       360 |       480 |              1000 |
| REFUTE       |       160 |       320 |       460 |               940 |
| NOT_ENTITLED |       200 |        40 |        80 |               320 |

## Stage75 family counts

| source   | family                                  |   count |
|:---------|:----------------------------------------|--------:|
| stage75  | lexical_type_polarity_disambiguation_v2 |     220 |
| stage75  | numeric_temporal_polarity_comparison_v2 |     260 |
| stage75  | refute_entitlement_direct_recovery_v2   |     220 |
| stage75  | strict_ne_external_style_safety_v2      |      80 |
| stage75  | support_entitlement_direct_recovery_v2  |     240 |

## Integration rules

| rule                                       | value   | reason                                                                               |
|:-------------------------------------------|:--------|:-------------------------------------------------------------------------------------|
| append_after_clean_split                   | True    | Preserve clean controlled dev for checkpoint selection.                              |
| train_only                                 | True    | Stage75 bridge must not enter dev/checkpoint selection.                              |
| exclude_from_intervention_pairwise_loss    | True    | Stage75 rows are standalone synthetic bridge rows and intentionally have no pair_id. |
| external_data_used_for_training            | False   | Stage75 is synthetic-only; Stage74 used only aggregate residual counts.              |
| external_metrics_used_for_threshold_tuning | False   | No threshold or selection tuning from VitaminC metrics.                              |
| time_swap_used                             | False   | Main data remains controlled_v5_v3_without_time_swap.jsonl.                          |

## Runner requirements

| item         | requirement                                                                             |
|:-------------|:----------------------------------------------------------------------------------------|
| new CLI flag | --stage75-bridge-train-jsonl                                                            |
| new CLI flag | --stage75-bridge-train-mode {none,append_train_only}                                    |
| metadata     | stage75_bridge_enabled, row_count, label_counts, family_counts                          |
| metadata     | stage75_bridge_train_only=True, used_for_dev=False, used_for_checkpoint_selection=False |
| metadata     | combined_bridge_row_count=2260 for Stage57+66+75                                        |
| metadata     | stage75_excluded_from_intervention_pairwise_loss=True                                   |
| metadata     | intervention_pairwise_loss_stage75_row_count_excluded=1020                              |
| metadata     | intervention_pairwise_loss_bridge_row_count_excluded=2260                               |
| preservation | clean_dev_for_checkpoint_selection=True                                                 |

## Checks

| check                                   | pass   |
|:----------------------------------------|:-------|
| stage75a_ready                          | True   |
| stage75b_generation_ready               | True   |
| stage75b_static_ready                   | True   |
| stage75_row_count_1020                  | True   |
| stage75_support_480                     | True   |
| stage75_refute_460                      | True   |
| stage75_ne_80                           | True   |
| stage57_row_count_520                   | True   |
| stage66_row_count_720                   | True   |
| combined_bridge_2260                    | True   |
| clean_main_3600                         | True   |
| clean_train_estimated_2880              | True   |
| clean_dev_estimated_720                 | True   |
| stage75_has_no_pair_id                  | True   |
| stage75_synthetic_only                  | True   |
| stage75_external_text_false             | True   |
| stage75_external_label_false            | True   |
| stage71c_pairwise_exclusion_prior_ready | True   |
| stage72_preservation_prior_ready        | True   |

## Interpretation

Stage75 should be integrated exactly like Stage57/Stage66: append after the clean train/dev split, train-only, not used for dev or checkpoint selection, and excluded from intervention pairwise loss. The clean dev set remains the sole checkpoint-selection basis.

## Recommended next stage

Stage75D patch runner to support Stage75 train-only bridge append
