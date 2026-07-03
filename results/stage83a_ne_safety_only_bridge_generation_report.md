# Stage83A - NE-Safety-Only Bridge Generation Report

## Decision

`STAGE83A_NE_SAFETY_ONLY_BRIDGE_READY`

## Summary

| stage    | decision                             | purpose                                                                   | source_jsonl                                      | output_jsonl                              | target_family                     |   source_rows |   output_rows | output_label_counts   | output_family_counts                       | sha256                                                           | training_executed   | external_eval_executed   | recommended_next_stage                                    |
|:---------|:-------------------------------------|:--------------------------------------------------------------------------|:--------------------------------------------------|:------------------------------------------|:----------------------------------|--------------:|--------------:|:----------------------|:-------------------------------------------|:-----------------------------------------------------------------|:--------------------|:-------------------------|:----------------------------------------------------------|
| Stage83A | STAGE83A_NE_SAFETY_ONLY_BRIDGE_READY | Create A3 NE-safety-only bridge subset from Stage80A conservative bridge. | data/stage80a_conservative_stage75v2_bridge.jsonl | data/stage83a_ne_safety_only_bridge.jsonl | strict_ne_false_support_safety_v2 |           500 |           160 | {"NOT_ENTITLED": 160} | {"strict_ne_false_support_safety_v2": 160} | 23c43c8fb6e9a696937854bf4b05bb7fd1f75923e6526bb0490df31c7a9e0b7d | False               | False                    | Stage83B static runner check for A3 NE-safety-only bridge |

## Checks

| check                                | pass   |
|:-------------------------------------|:-------|
| stage83_ready                        | True   |
| stage83_allows_a3                    | True   |
| source_rows_500                      | True   |
| source_has_target_family_160         | True   |
| subset_rows_160                      | True   |
| subset_all_target_family             | True   |
| subset_all_not_entitled              | True   |
| subset_duplicate_ids_zero            | True   |
| subset_duplicate_claim_evidence_zero | True   |
| subset_no_pair_id                    | True   |
| subset_synthetic_only_true           | True   |
| subset_external_text_false           | True   |
| subset_external_label_false          | True   |
| subset_has_stage83a_metadata         | True   |
| deterministic_serialization          | True   |
| runner_stage80a_flags_available      | True   |
| runner_stage80a_loader_available     | True   |

## Source vs subset counts

| scope           |   rows | label_counts                                         | family_counts                                                                                                                                            |
|:----------------|-------:|:-----------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage80A_source |    500 | {"SUPPORT": 170, "REFUTE": 170, "NOT_ENTITLED": 160} | {"numeric_temporal_polarity_repair_v2_conservative": 180, "lexical_type_polarity_repair_v2_conservative": 160, "strict_ne_false_support_safety_v2": 160} |
| Stage83A_subset |    160 | {"NOT_ENTITLED": 160}                                | {"strict_ne_false_support_safety_v2": 160}                                                                                                               |

## Recommended next stage

Stage83B static runner check for A3 NE-safety-only bridge
