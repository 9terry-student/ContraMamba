# Stage88A - Balanced Entitlement Recovery Bridge Generation

## Decision

`STAGE88A_BALANCED_ENTITLEMENT_RECOVERY_BRIDGE_READY`

## Summary

| stage    | decision                                            | out                                                      |   n | label_counts                                        | family_counts                                                                                                                                                                                                                                              | polarity_counts                             | duplicate_repair                                                                                               | target                                                                            | non_leakage_policy                                                                          | intended_next_stage                                                                | checks                                                                                                                                                                                                                                                                                                                                                                                            |
|:---------|:----------------------------------------------------|:---------------------------------------------------------|----:|:----------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stage88A | STAGE88A_BALANCED_ENTITLEMENT_RECOVERY_BRIDGE_READY | data/stage88a_balanced_entitlement_recovery_bridge.jsonl | 360 | {"SUPPORT": 150, "REFUTE": 150, "NOT_ENTITLED": 60} | {"explicit_entity_role_entitlement": 120, "explicit_numeric_entitlement": 80, "explicit_temporal_entitlement": 60, "lexical_paraphrase_entitlement": 40, "matched_missing_predicate_guardrail": 30, "matched_insufficient_numeric_temporal_guardrail": 30} | {"SUPPORT": 150, "REFUTE": 150, "NONE": 60} | {"before_duplicate_groups": 26, "before_duplicate_surplus_rows": 26, "after_unique_claim_evidence_pairs": 360} | Recover SUPPORT/REFUTE entitlement recall without adding NE-safety-only pressure. | Synthetic/internal rows only; VitaminC external examples are not used as train bridge rows. | Stage88B clean-dev frozen recovery run with Stage57+Stage66+Stage88A, no Stage83A. | {"row_count_360": true, "support_150": true, "refute_150": true, "not_entitled_60": true, "ne_ratio_le_0p20": true, "ids_unique": true, "pair_ids_unique": true, "claim_evidence_pairs_unique": true, "no_exact_train_overlap": true, "no_exact_external_overlap": true, "encode_label_tensors_ok": true, "encode_mamba_records_ok": true, "vitaminc_external_not_used_as_training_source": true} |

## Label counts

| label        |   count |
|:-------------|--------:|
| NOT_ENTITLED |      60 |
| REFUTE       |     150 |
| SUPPORT      |     150 |

## Family counts

| family                                          |   count |
|:------------------------------------------------|--------:|
| explicit_entity_role_entitlement                |     120 |
| explicit_numeric_entitlement                    |      80 |
| explicit_temporal_entitlement                   |      60 |
| lexical_paraphrase_entitlement                  |      40 |
| matched_insufficient_numeric_temporal_guardrail |      30 |
| matched_missing_predicate_guardrail             |      30 |

## Checks

| check                                         | pass   |
|:----------------------------------------------|:-------|
| row_count_360                                 | True   |
| support_150                                   | True   |
| refute_150                                    | True   |
| not_entitled_60                               | True   |
| ne_ratio_le_0p20                              | True   |
| ids_unique                                    | True   |
| pair_ids_unique                               | True   |
| claim_evidence_pairs_unique                   | True   |
| no_exact_train_overlap                        | True   |
| no_exact_external_overlap                     | True   |
| encode_label_tensors_ok                       | True   |
| encode_mamba_records_ok                       | True   |
| vitaminc_external_not_used_as_training_source | True   |
