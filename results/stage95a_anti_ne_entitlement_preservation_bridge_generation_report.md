# Stage95A - Anti-NE Entitlement Preservation Bridge

## Decision

`STAGE95A_ANTI_NE_ENTITLEMENT_PRESERVATION_BRIDGE_READY`

## Summary

| stage    | decision                                               | stage95a_out                                                | combined_stage80a_out                                       |   stage95a_n | stage95a_label_counts                             | stage95a_family_counts                                                                                                                                                                                                                                                           | stage95a_polarity_counts                  |   combined_n | combined_label_counts                                | combined_intervention_counts                                                                                 | repair                                                                                                                                                                                      | target                                                                                                                               | source_policy                                                                                                                               | intended_next_stage                                                                                                                                        |   expected_stage95b_bridge_count |   expected_stage95b_final_train_count | overlap_by_file                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | checks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:---------|:-------------------------------------------------------|:------------------------------------------------------------|:------------------------------------------------------------|-------------:|:--------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------|-------------:|:-----------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------:|--------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage95A | STAGE95A_ANTI_NE_ENTITLEMENT_PRESERVATION_BRIDGE_READY | data/stage95a_anti_ne_entitlement_preservation_bridge.jsonl | data/stage95a_stage92a_combined_support_antine_bridge.jsonl |          160 | {"SUPPORT": 72, "REFUTE": 48, "NOT_ENTITLED": 40} | {"explicit_sufficient_support_with_ne_distractor": 36, "explicit_sufficient_refute_with_ne_distractor": 24, "support_exact_predicate_coverage_after_background": 36, "refute_exact_predicate_coverage_after_background": 24, "matched_missing_decisive_predicate_guardrail": 40} | {"SUPPORT": 72, "REFUTE": 48, "NONE": 40} |          400 | {"SUPPORT": 192, "REFUTE": 108, "NOT_ENTITLED": 100} | {"stage92a_support_preserving_counter_bridge": 240, "stage95a_anti_ne_entitlement_preservation_bridge": 160} | {"duplicate_repairs": 0, "overlap_repairs": 0, "total_marker_appends": 0, "method": "neutral evidence disambiguator until Stage95A rows are unique and non-overlapping with checked files"} | Reduce residual false_NE absorption under sufficient SUPPORT/REFUTE evidence without reintroducing SUPPORT-to-REFUTE overcorrection. | Synthetic/internal rows only. Stage94 diagnostic examples and VitaminC external rows are checked for overlap but not used as training rows. | Stage95B clean-dev frozen recovery run with Stage57 + Stage66 + combined Stage92A+Stage95A via stage80a slot; no Stage83A; no Stage88A; no external first. |                             1640 |                                  4520 | {"data/controlled_v5_v3_without_time_swap.jsonl": 0, "data/stage57_nonleaking_external_bridge.jsonl": 0, "data/stage66_residual_bridge.jsonl": 0, "data/stage83a_ne_safety_only_bridge.jsonl": 0, "data/stage88a_balanced_entitlement_recovery_bridge.jsonl": 0, "data/stage92a_support_preserving_counter_bridge.jsonl": 0, "data/stage43b1_vitaminc_validation_sample1000.jsonl": 0, "data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl": 0, "results/stage94_new_false_ne_stage73_non_ne_stage92c_ne_examples.jsonl": 0, "results/stage94_repaired_false_ne_stage73_ne_stage92c_non_ne_examples.jsonl": 0, "results/stage94_persistent_false_ne_stage73_stage92c_both_ne_examples.jsonl": 0, "results/stage94_stage92c_false_ne_all_examples.jsonl": 0} | {"stage95a_row_count_160": true, "stage95a_support_72": true, "stage95a_refute_48": true, "stage95a_not_entitled_40": true, "stage95a_pairs_unique": true, "stage95a_no_exact_overlap_any_checked_file": true, "stage95a_synthetic_internal_only": true, "stage95a_no_vitaminc_source": true, "combined_row_count_400": true, "combined_contains_stage92a_240": true, "combined_contains_stage95a_160": true, "combined_pairs_unique": true, "combined_support_192": true, "combined_refute_108": true, "combined_ne_100": true, "encode_label_tensors_stage95a_ok": true, "encode_label_tensors_combined_ok": true, "encode_mamba_records_stage95a_ok": true, "encode_mamba_records_combined_ok": true} |

## Stage95A label counts

| label        |   count |
|:-------------|--------:|
| NOT_ENTITLED |      40 |
| REFUTE       |      48 |
| SUPPORT      |      72 |

## Stage95A family counts

| family                                            |   count |
|:--------------------------------------------------|--------:|
| explicit_sufficient_refute_with_ne_distractor     |      24 |
| explicit_sufficient_support_with_ne_distractor    |      36 |
| matched_missing_decisive_predicate_guardrail      |      40 |
| refute_exact_predicate_coverage_after_background  |      24 |
| support_exact_predicate_coverage_after_background |      36 |

## Combined Stage92A+Stage95A label counts

| label        |   count |
|:-------------|--------:|
| NOT_ENTITLED |     100 |
| REFUTE       |     108 |
| SUPPORT      |     192 |

## Repair

|   duplicate_repairs |   overlap_repairs |   total_marker_appends | method                                                                                               |
|--------------------:|------------------:|-----------------------:|:-----------------------------------------------------------------------------------------------------|
|                   0 |                 0 |                      0 | neutral evidence disambiguator until Stage95A rows are unique and non-overlapping with checked files |

## Exact-overlap checks

| file                                                                        |   exact_pair_overlap |
|:----------------------------------------------------------------------------|---------------------:|
| data/controlled_v5_v3_without_time_swap.jsonl                               |                    0 |
| data/stage57_nonleaking_external_bridge.jsonl                               |                    0 |
| data/stage66_residual_bridge.jsonl                                          |                    0 |
| data/stage83a_ne_safety_only_bridge.jsonl                                   |                    0 |
| data/stage88a_balanced_entitlement_recovery_bridge.jsonl                    |                    0 |
| data/stage92a_support_preserving_counter_bridge.jsonl                       |                    0 |
| data/stage43b1_vitaminc_validation_sample1000.jsonl                         |                    0 |
| data/stage84e2_vitaminc_validation_sample1000_exact_ood_schema.jsonl        |                    0 |
| results/stage94_new_false_ne_stage73_non_ne_stage92c_ne_examples.jsonl      |                    0 |
| results/stage94_repaired_false_ne_stage73_ne_stage92c_non_ne_examples.jsonl |                    0 |
| results/stage94_persistent_false_ne_stage73_stage92c_both_ne_examples.jsonl |                    0 |
| results/stage94_stage92c_false_ne_all_examples.jsonl                        |                    0 |

## Checks

| check                                      | pass   |
|:-------------------------------------------|:-------|
| stage95a_row_count_160                     | True   |
| stage95a_support_72                        | True   |
| stage95a_refute_48                         | True   |
| stage95a_not_entitled_40                   | True   |
| stage95a_pairs_unique                      | True   |
| stage95a_no_exact_overlap_any_checked_file | True   |
| stage95a_synthetic_internal_only           | True   |
| stage95a_no_vitaminc_source                | True   |
| combined_row_count_400                     | True   |
| combined_contains_stage92a_240             | True   |
| combined_contains_stage95a_160             | True   |
| combined_pairs_unique                      | True   |
| combined_support_192                       | True   |
| combined_refute_108                        | True   |
| combined_ne_100                            | True   |
| encode_label_tensors_stage95a_ok           | True   |
| encode_label_tensors_combined_ok           | True   |
| encode_mamba_records_stage95a_ok           | True   |
| encode_mamba_records_combined_ok           | True   |
