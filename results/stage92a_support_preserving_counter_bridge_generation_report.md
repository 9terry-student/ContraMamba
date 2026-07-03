# Stage92A - SUPPORT-Preserving Counter-Bridge Generation

## Decision

`STAGE92A_SUPPORT_PRESERVING_COUNTER_BRIDGE_READY`

## Summary

| stage    | decision                                         | out                                                   |   n | label_counts                                       | family_counts                                                                                                                                                                                      | polarity_counts                            | repair                                                                                                                                                                                                                                                                                                                                      | target                                                                                           | non_leakage_policy                                                                                        | intended_next_stage                                                                                                     | overlap_by_file                                                                                                                                                                                                                                                                                                                                                                                       | checks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|:---------|:-------------------------------------------------|:------------------------------------------------------|----:|:---------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage92A | STAGE92A_SUPPORT_PRESERVING_COUNTER_BRIDGE_READY | data/stage92a_support_preserving_counter_bridge.jsonl | 240 | {"SUPPORT": 120, "REFUTE": 60, "NOT_ENTITLED": 60} | {"support_preserving_entity_role_distractor": 60, "support_preserving_numeric_distractor": 60, "support_preserving_temporal_distractor": 60, "support_preserving_lexical_location_distractor": 60} | {"SUPPORT": 120, "REFUTE": 60, "NONE": 60} | {"initial_unique_pairs": 240, "initial_duplicate_groups": 0, "initial_duplicate_surplus": 0, "duplicate_repairs": 0, "overlap_repairs": 10, "total_marker_appends": 10, "final_unique_pairs": 240, "final_duplicate_groups": 0, "method": "neutral evidence disambiguator until each row is unique and non-overlapping with checked files"} | Protect SUPPORT recall against REFUTE/NE overcorrection while retaining guarded REFUTE recovery. | Synthetic/internal rows only; VitaminC external examples are overlap-checked but not used as bridge rows. | Stage92B clean-dev frozen recovery run with Stage57+Stage66+Stage92A only; no Stage83A; no Stage88A; no external first. | {"data/controlled_v5_v3_without_time_swap.jsonl": 0, "data/stage57_nonleaking_external_bridge.jsonl": 0, "data/stage66_residual_bridge.jsonl": 0, "data/stage83a_ne_safety_only_bridge.jsonl": 0, "data/stage88a_balanced_entitlement_recovery_bridge.jsonl": 0, "data/stage43b1_vitaminc_validation_sample1000.jsonl": 0, "data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl": 0} | {"row_count_240": true, "support_120": true, "refute_60": true, "not_entitled_60": true, "support_ratio_0p50": true, "refute_ratio_0p25": true, "ne_ratio_0p25": true, "ids_unique": true, "pair_ids_unique": true, "claim_evidence_pairs_unique": true, "duplicate_groups_zero": true, "no_exact_overlap_any_checked_file": true, "encode_label_tensors_ok": true, "encode_mamba_records_ok": true, "synthetic_internal_only": true, "vitaminc_external_not_used_as_training_source": true} |

## Label counts

| label        |   count |
|:-------------|--------:|
| NOT_ENTITLED |      60 |
| REFUTE       |      60 |
| SUPPORT      |     120 |

## Family counts

| family                                         |   count |
|:-----------------------------------------------|--------:|
| support_preserving_entity_role_distractor      |      60 |
| support_preserving_lexical_location_distractor |      60 |
| support_preserving_numeric_distractor          |      60 |
| support_preserving_temporal_distractor         |      60 |

## Repair

|   initial_unique_pairs |   initial_duplicate_groups |   initial_duplicate_surplus |   duplicate_repairs |   overlap_repairs |   total_marker_appends |   final_unique_pairs |   final_duplicate_groups | method                                                                                         |
|-----------------------:|---------------------------:|----------------------------:|--------------------:|------------------:|-----------------------:|---------------------:|-------------------------:|:-----------------------------------------------------------------------------------------------|
|                    240 |                          0 |                           0 |                   0 |                10 |                     10 |                  240 |                        0 | neutral evidence disambiguator until each row is unique and non-overlapping with checked files |

## Exact-overlap checks

| file                                                                 |   exact_pair_overlap |
|:---------------------------------------------------------------------|---------------------:|
| data/controlled_v5_v3_without_time_swap.jsonl                        |                    0 |
| data/stage57_nonleaking_external_bridge.jsonl                        |                    0 |
| data/stage66_residual_bridge.jsonl                                   |                    0 |
| data/stage83a_ne_safety_only_bridge.jsonl                            |                    0 |
| data/stage88a_balanced_entitlement_recovery_bridge.jsonl             |                    0 |
| data/stage43b1_vitaminc_validation_sample1000.jsonl                  |                    0 |
| data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl |                    0 |

## Checks

| check                                         | pass   |
|:----------------------------------------------|:-------|
| row_count_240                                 | True   |
| support_120                                   | True   |
| refute_60                                     | True   |
| not_entitled_60                               | True   |
| support_ratio_0p50                            | True   |
| refute_ratio_0p25                             | True   |
| ne_ratio_0p25                                 | True   |
| ids_unique                                    | True   |
| pair_ids_unique                               | True   |
| claim_evidence_pairs_unique                   | True   |
| duplicate_groups_zero                         | True   |
| no_exact_overlap_any_checked_file             | True   |
| encode_label_tensors_ok                       | True   |
| encode_mamba_records_ok                       | True   |
| synthetic_internal_only                       | True   |
| vitaminc_external_not_used_as_training_source | True   |
