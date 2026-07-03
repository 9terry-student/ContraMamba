# Stage83 - Stage80A Branch Ablation Decision Report

## Decision

`STAGE83_STAGE80A_BRANCH_ABLATION_DECISION_READY`

## Branch decision

`STOP_FULL_STAGE80A_ALLOW_A3_NE_SAFETY_ONLY_CLEAN_DEV_ABLATION`

## Summary

| stage   | decision                                        | branch_decision                                               | decision_reason                                                                                                                                                                 | stage80g_stage80a_decision      |   stage80f_acc |   stage80f_macro_f1 |   stage80f_minus_stage71_macro_f1 |   stage80f_macro_preservation_margin_vs_stage71 |   full_stage80a_rows |   a1_polarity_only_rows |   a3_ne_safety_only_rows | a3_target_output_jsonl                    | training_executed   | external_eval_executed   | recommended_next_stage                          |
|:--------|:------------------------------------------------|:--------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------|---------------:|--------------------:|----------------------------------:|------------------------------------------------:|---------------------:|------------------------:|-------------------------:|:------------------------------------------|:--------------------|:-------------------------|:------------------------------------------------|
| Stage83 | STAGE83_STAGE80A_BRANCH_ABLATION_DECISION_READY | STOP_FULL_STAGE80A_ALLOW_A3_NE_SAFETY_ONLY_CLEAN_DEV_ABLATION | Full Stage80A should stop before external diagnostic. If continuing the branch, only the smallest NE-safety-only subset is justified because it avoids additional SUPPORT rows. | STOP_BEFORE_EXTERNAL_DIAGNOSTIC |       0.970833 |            0.958564 |                       -0.00548313 |                                    -0.000483132 |                  500 |                     340 |                      160 | data/stage83a_ne_safety_only_bridge.jsonl | False               | False                    | Stage83A create A3 NE-safety-only subset bridge |

## Checks

| check                                      | pass   |
|:-------------------------------------------|:-------|
| stage80g_ready                             | True   |
| stage80g_stop_before_external              | True   |
| stage80f_execution_valid                   | True   |
| stage80f_clean_dev_macro_failed_vs_stage71 | True   |
| a1_rows_340                                | True   |
| a3_rows_160                                | True   |
| a3_all_ne                                  | True   |
| a3_no_duplicates                           | True   |
| a3_synthetic_only                          | True   |
| decision_nonempty                          | True   |

## Signals

| signal                                     | value   |
|:-------------------------------------------|:--------|
| stage80g_ready                             | True    |
| stage80g_stop_before_external              | True    |
| stage80f_execution_valid                   | True    |
| stage80f_stage75_full_unused               | True    |
| stage80f_stage80a_used_500                 | True    |
| stage80f_clean_dev_macro_failed_vs_stage71 | True    |
| stage80f_macro_margin_negative             | True    |
| stage75_full_already_revise_not_default    | True    |
| stage78_false_support_increased            | True    |
| a1_polarity_only_rows_340                  | True    |
| a3_ne_safety_only_rows_160                 | True    |

## Candidate decisions

| candidate             | decision                                   |   rows | reason                                                                                                        | next_action                                                                                   |
|:----------------------|:-------------------------------------------|-------:|:--------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
| full_stage80a_500     | REJECT_AS_PRIMARY_AND_SKIP_EXTERNAL        |    500 | Stage80F failed strict clean-dev macro preservation versus Stage71.                                           | Do not run external diagnostic.                                                               |
| A1_polarity_only_340  | DENY_BY_DEFAULT                            |    340 | A1 removes NE safety and keeps only SUPPORT/REFUTE polarity rows; likely still pressures SUPPORT/entitlement. | Do not run unless explicitly exploring polarity-only failure mode.                            |
| A3_ne_safety_only_160 | ALLOW_ONE_CLEAN_DEV_ABLATION_IF_CONTINUING |    160 | Smallest subset; directly targets false SUPPORT / over-entitlement without adding SUPPORT rows.               | Create subset JSONL, static check it, then run clean-dev only before any external diagnostic. |

## Subset stats

| candidate         |   rows | label_counts                    | family_counts                                                                                                  |   duplicate_ids |   duplicate_claim_evidence_pairs | synthetic_only_all_true   | external_text_used_any   | external_label_used_any   | pair_id_any   |
|:------------------|-------:|:--------------------------------|:---------------------------------------------------------------------------------------------------------------|----------------:|---------------------------------:|:--------------------------|:-------------------------|:--------------------------|:--------------|
| A1_polarity_only  |    340 | {"SUPPORT": 170, "REFUTE": 170} | {"numeric_temporal_polarity_repair_v2_conservative": 180, "lexical_type_polarity_repair_v2_conservative": 160} |               0 |                                0 | True                      | False                    | False                     | False         |
| A3_ne_safety_only |    160 | {"NOT_ENTITLED": 160}           | {"strict_ne_false_support_safety_v2": 160}                                                                     |               0 |                                0 | True                      | False                    | False                     | False         |

## Recommendations

| recommendation                                               | rationale                                                                          |
|:-------------------------------------------------------------|:-----------------------------------------------------------------------------------|
| Keep Stage71_retry2 as primary.                              | Stage80G showed Stage71 has the best clean-dev macro-F1 among compared candidates. |
| Do not run Stage81 external for full Stage80A.               | Full Stage80A already failed strict clean-dev preservation.                        |
| Do not run A1 polarity-only by default.                      | It contains 170 SUPPORT and 170 REFUTE rows with no NE safety backstop.            |
| Allow only A3 NE-safety-only as a single clean-dev ablation. | It has 160 NOT_ENTITLED rows and directly targets false SUPPORT pressure.          |
| External diagnostic remains blocked until clean-dev passes.  | No VitaminC run should be spent unless A3 preserves clean-dev versus Stage71.      |

## Decision reason

Full Stage80A should stop before external diagnostic. If continuing the branch, only the smallest NE-safety-only subset is justified because it avoids additional SUPPORT rows.

## Recommended next stage

Stage83A create A3 NE-safety-only subset bridge
