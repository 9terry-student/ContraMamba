# Stage94 - false_NE Residual Audit and Stage95A Design

## Decision

`STAGE94_FALSE_NE_RESIDUAL_AUDIT_DESIGN_READY`

## Summary

| stage   | decision                                     | basis_stage93_decision                          | basis_stage93_candidate_status         | match_mode   |   matched_n | false_ne_partition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | diagnosis_rules                                                                                                                                                                                                                                                             | main_diagnosis                                                                                                                      | recommended_next_stage                                                               | stage95a_design                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | diagnostic_example_files                                                                                                                                                                                                                                                                                                                                                                                                     | source_policy                                                               |
|:--------|:---------------------------------------------|:------------------------------------------------|:---------------------------------------|:-------------|------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|
| Stage94 | STAGE94_FALSE_NE_RESIDUAL_AUDIT_DESIGN_READY | STAGE93_REJECT_STAGE92C_AS_PRIMARY_KEEP_STAGE71 | NEAR_MISS_SUPPORT_RECOVERY_NOT_PRIMARY | key          |        1000 | {"stage73_false_NE_total": 323, "stage88c_false_NE_total": 349, "stage92c_false_NE_total": 345, "stage92c_minus_stage73_false_NE_total": 22, "new_false_NE_vs_stage73": 49, "repaired_false_NE_vs_stage73": 27, "persistent_false_NE_stage73_stage92c": 296, "new_minus_repaired_vs_stage73": 22, "new_false_NE_vs_stage73_by_gold": {"REFUTE": 21, "SUPPORT": 28}, "repaired_false_NE_vs_stage73_by_gold": {"SUPPORT": 14, "REFUTE": 13}, "persistent_false_NE_stage73_stage92c_by_gold": {"SUPPORT": 164, "REFUTE": 132}, "stage92c_false_NE_by_gold": {"SUPPORT": 192, "REFUTE": 153}, "new_false_NE_vs_stage88c": 27, "repaired_false_NE_vs_stage88c": 31, "new_minus_repaired_vs_stage88c": -4} | ["stage92c_has_residual_false_NE_excess_vs_stage73", "new_false_NE_vs_stage73_exceeds_repaired_false_NE", "support_false_NE_is_larger_absolute_component", "refute_false_NE_remains_material", "next_bridge_should_target_entitled_sufficiency_not_generic_label_pressure"] | Stage92C repaired SUPPORT-to-REFUTE overcorrection but still has residual false_NE absorption for entitled SUPPORT/REFUTE examples. | Stage95A anti-NE entitlement preservation bridge generation + encoder preflight only | {"proposed_stage": "Stage95A", "name": "anti_NE_entitlement_preservation_bridge", "purpose": "Reduce residual false_NE absorption in Stage92C without reintroducing Stage88C-style SUPPORT-to-REFUTE overcorrection.", "recommended_distribution": {"SUPPORT": 72, "REFUTE": 48, "NOT_ENTITLED": 40, "total": 160}, "why_small": "Stage92C is already a near-miss; large generic pressure can break clean-dev or false-entitlement guardrails.", "bridge_composition": [{"family": "explicit_sufficient_support_with_ne_distractor", "label": "SUPPORT", "rows": 36, "description": "Evidence explicitly states the claim while also mentioning irrelevant/missing contextual facts that should not trigger NOT_ENTITLED."}, {"family": "explicit_sufficient_refute_with_ne_distractor", "label": "REFUTE", "rows": 24, "description": "Evidence explicitly contradicts the claim while also including incomplete peripheral context; decisive contradiction should prevent NE absorption."}, {"family": "support_exact_predicate_coverage_after_background", "label": "SUPPORT", "rows": 36, "description": "Longer evidence begins with background/uncertain context but later gives exact predicate coverage for SUPPORT."}, {"family": "refute_exact_predicate_coverage_after_background", "label": "REFUTE", "rows": 24, "description": "Longer evidence begins with background/uncertain context but later gives exact predicate contradiction for REFUTE."}, {"family": "matched_missing_decisive_predicate_guardrail", "label": "NOT_ENTITLED", "rows": 40, "description": "Matched cases where claim/evidence share frame/entity but the decisive predicate/value/date/role is genuinely absent."}], "required_training_combo_for_stage95b": "Stage57 + Stage66 + Stage92A + Stage95A only; no Stage83A; no Stage88A.", "forbidden": ["Do not use VitaminC external examples as training rows.", "Do not add Stage83A NE-safety rows.", "Do not add generic SUPPORT-only pressure.", "Do not add REFUTE-heavy pressure.", "Do not tune using external labels."], "promotion_criteria": {"clean_dev": "clean-dev macro_f1 >= Stage71 macro_f1 - 0.003", "external_acc": ">= Stage73 external accuracy 0.353", "external_macro": ">= Stage73 external macro_f1_all3 0.3261787", "false_NE_total": "<= Stage73 false_NE_total 323", "support_recall": ">= Stage73 support recall 0.432 - 0.02", "false_entitlement_total": "<= Stage73 false_entitlement_total + 10", "false_REFUTE_on_SUPPORT": "remain near Stage73; avoid Stage88C regression"}} | {"new_false_ne_vs_stage73": "results/stage94_new_false_ne_stage73_non_ne_stage92c_ne_examples.jsonl", "repaired_false_ne_vs_stage73": "results/stage94_repaired_false_ne_stage73_ne_stage92c_non_ne_examples.jsonl", "persistent_false_ne_stage73_stage92c": "results/stage94_persistent_false_ne_stage73_stage92c_both_ne_examples.jsonl", "stage92c_false_ne_all": "results/stage94_stage92c_false_ne_all_examples.jsonl"} | All exported examples are diagnostic only and not allowed as training data. |

## false_NE partition

| metric                                       | value                           |
|:---------------------------------------------|:--------------------------------|
| stage73_false_NE_total                       | 323                             |
| stage88c_false_NE_total                      | 349                             |
| stage92c_false_NE_total                      | 345                             |
| stage92c_minus_stage73_false_NE_total        | 22                              |
| new_false_NE_vs_stage73                      | 49                              |
| repaired_false_NE_vs_stage73                 | 27                              |
| persistent_false_NE_stage73_stage92c         | 296                             |
| new_minus_repaired_vs_stage73                | 22                              |
| new_false_NE_vs_stage73_by_gold              | {"REFUTE": 21, "SUPPORT": 28}   |
| repaired_false_NE_vs_stage73_by_gold         | {"SUPPORT": 14, "REFUTE": 13}   |
| persistent_false_NE_stage73_stage92c_by_gold | {"SUPPORT": 164, "REFUTE": 132} |
| stage92c_false_NE_by_gold                    | {"SUPPORT": 192, "REFUTE": 153} |
| new_false_NE_vs_stage88c                     | 27                              |
| repaired_false_NE_vs_stage88c                | 31                              |
| new_minus_repaired_vs_stage88c               | -4                              |

## Diagnosis rules

| rule                                                                      |
|:--------------------------------------------------------------------------|
| stage92c_has_residual_false_NE_excess_vs_stage73                          |
| new_false_NE_vs_stage73_exceeds_repaired_false_NE                         |
| support_false_NE_is_larger_absolute_component                             |
| refute_false_NE_remains_material                                          |
| next_bridge_should_target_entitled_sufficiency_not_generic_label_pressure |

## Stage73 -> Stage92C transitions

| subset        | gold    | stage73_pred   | stage92c_pred   |   count |
|:--------------|:--------|:---------------|:----------------|--------:|
| gold_entitled | SUPPORT | SUPPORT        | SUPPORT         |     169 |
| gold_entitled | SUPPORT | NOT_ENTITLED   | NOT_ENTITLED    |     164 |
| gold_entitled | REFUTE  | NOT_ENTITLED   | NOT_ENTITLED    |     132 |
| gold_entitled | REFUTE  | SUPPORT        | SUPPORT         |     103 |
| gold_entitled | SUPPORT | REFUTE         | REFUTE          |      80 |
| gold_entitled | REFUTE  | REFUTE         | REFUTE          |      58 |
| gold_entitled | SUPPORT | SUPPORT        | NOT_ENTITLED    |      25 |
| gold_entitled | SUPPORT | REFUTE         | SUPPORT         |      23 |
| gold_entitled | SUPPORT | SUPPORT        | REFUTE          |      22 |
| gold_entitled | REFUTE  | SUPPORT        | REFUTE          |      18 |
| gold_entitled | REFUTE  | SUPPORT        | NOT_ENTITLED    |      17 |
| gold_entitled | REFUTE  | REFUTE         | SUPPORT         |      10 |
| gold_entitled | SUPPORT | NOT_ENTITLED   | SUPPORT         |       9 |
| gold_entitled | REFUTE  | NOT_ENTITLED   | REFUTE          |       7 |
| gold_entitled | REFUTE  | NOT_ENTITLED   | SUPPORT         |       6 |
| gold_entitled | SUPPORT | NOT_ENTITLED   | REFUTE          |       5 |
| gold_entitled | REFUTE  | REFUTE         | NOT_ENTITLED    |       4 |
| gold_entitled | SUPPORT | REFUTE         | NOT_ENTITLED    |       3 |
| gold_support  | SUPPORT | SUPPORT        | SUPPORT         |     169 |
| gold_support  | SUPPORT | NOT_ENTITLED   | NOT_ENTITLED    |     164 |
| gold_support  | SUPPORT | REFUTE         | REFUTE          |      80 |
| gold_support  | SUPPORT | SUPPORT        | NOT_ENTITLED    |      25 |
| gold_support  | SUPPORT | REFUTE         | SUPPORT         |      23 |
| gold_support  | SUPPORT | SUPPORT        | REFUTE          |      22 |
| gold_support  | SUPPORT | NOT_ENTITLED   | SUPPORT         |       9 |
| gold_support  | SUPPORT | NOT_ENTITLED   | REFUTE          |       5 |
| gold_support  | SUPPORT | REFUTE         | NOT_ENTITLED    |       3 |
| gold_refute   | REFUTE  | NOT_ENTITLED   | NOT_ENTITLED    |     132 |
| gold_refute   | REFUTE  | SUPPORT        | SUPPORT         |     103 |
| gold_refute   | REFUTE  | REFUTE         | REFUTE          |      58 |
| gold_refute   | REFUTE  | SUPPORT        | REFUTE          |      18 |
| gold_refute   | REFUTE  | SUPPORT        | NOT_ENTITLED    |      17 |
| gold_refute   | REFUTE  | REFUTE         | SUPPORT         |      10 |
| gold_refute   | REFUTE  | NOT_ENTITLED   | REFUTE          |       7 |
| gold_refute   | REFUTE  | NOT_ENTITLED   | SUPPORT         |       6 |
| gold_refute   | REFUTE  | REFUTE         | NOT_ENTITLED    |       4 |

## Stage88C -> Stage92C transitions

| subset        | gold    | stage88c_pred   | stage92c_pred   |   count |
|:--------------|:--------|:----------------|:----------------|--------:|
| gold_entitled | SUPPORT | NOT_ENTITLED    | NOT_ENTITLED    |     176 |
| gold_entitled | REFUTE  | NOT_ENTITLED    | NOT_ENTITLED    |     142 |
| gold_entitled | SUPPORT | SUPPORT         | SUPPORT         |     140 |
| gold_entitled | SUPPORT | REFUTE          | REFUTE          |      85 |
| gold_entitled | REFUTE  | SUPPORT         | SUPPORT         |      80 |
| gold_entitled | REFUTE  | REFUTE          | REFUTE          |      67 |
| gold_entitled | SUPPORT | REFUTE          | SUPPORT         |      47 |
| gold_entitled | REFUTE  | REFUTE          | SUPPORT         |      31 |
| gold_entitled | SUPPORT | SUPPORT         | REFUTE          |      15 |
| gold_entitled | REFUTE  | SUPPORT         | REFUTE          |      14 |
| gold_entitled | SUPPORT | NOT_ENTITLED    | SUPPORT         |      14 |
| gold_entitled | SUPPORT | REFUTE          | NOT_ENTITLED    |       9 |
| gold_entitled | REFUTE  | NOT_ENTITLED    | SUPPORT         |       8 |
| gold_entitled | REFUTE  | SUPPORT         | NOT_ENTITLED    |       7 |
| gold_entitled | SUPPORT | NOT_ENTITLED    | REFUTE          |       7 |
| gold_entitled | SUPPORT | SUPPORT         | NOT_ENTITLED    |       7 |
| gold_entitled | REFUTE  | REFUTE          | NOT_ENTITLED    |       4 |
| gold_entitled | REFUTE  | NOT_ENTITLED    | REFUTE          |       2 |
| gold_support  | SUPPORT | NOT_ENTITLED    | NOT_ENTITLED    |     176 |
| gold_support  | SUPPORT | SUPPORT         | SUPPORT         |     140 |
| gold_support  | SUPPORT | REFUTE          | REFUTE          |      85 |
| gold_support  | SUPPORT | REFUTE          | SUPPORT         |      47 |
| gold_support  | SUPPORT | SUPPORT         | REFUTE          |      15 |
| gold_support  | SUPPORT | NOT_ENTITLED    | SUPPORT         |      14 |
| gold_support  | SUPPORT | REFUTE          | NOT_ENTITLED    |       9 |
| gold_support  | SUPPORT | NOT_ENTITLED    | REFUTE          |       7 |
| gold_support  | SUPPORT | SUPPORT         | NOT_ENTITLED    |       7 |
| gold_refute   | REFUTE  | NOT_ENTITLED    | NOT_ENTITLED    |     142 |
| gold_refute   | REFUTE  | SUPPORT         | SUPPORT         |      80 |
| gold_refute   | REFUTE  | REFUTE          | REFUTE          |      67 |
| gold_refute   | REFUTE  | REFUTE          | SUPPORT         |      31 |
| gold_refute   | REFUTE  | SUPPORT         | REFUTE          |      14 |
| gold_refute   | REFUTE  | NOT_ENTITLED    | SUPPORT         |       8 |
| gold_refute   | REFUTE  | SUPPORT         | NOT_ENTITLED    |       7 |
| gold_refute   | REFUTE  | REFUTE          | NOT_ENTITLED    |       4 |
| gold_refute   | REFUTE  | NOT_ENTITLED    | REFUTE          |       2 |

## Lexical feature diagnostics

| group                                | feature           |   count |      rate |   n |
|:-------------------------------------|:------------------|--------:|----------:|----:|
| stage92c_false_NE_all                | has_number        |     234 | 0.678261  | 345 |
| stage92c_false_NE_all                | has_date_word     |     102 | 0.295652  | 345 |
| stage92c_false_NE_all                | has_negation      |      14 | 0.0405797 | 345 |
| stage92c_false_NE_all                | has_contrast      |      21 | 0.0608696 | 345 |
| stage92c_false_NE_all                | has_role_word     |      16 | 0.0463768 | 345 |
| stage92c_false_NE_all                | has_location_word |      22 | 0.0637681 | 345 |
| new_false_NE_vs_stage73              | has_number        |      35 | 0.714286  |  49 |
| new_false_NE_vs_stage73              | has_date_word     |      14 | 0.285714  |  49 |
| new_false_NE_vs_stage73              | has_negation      |       1 | 0.0204082 |  49 |
| new_false_NE_vs_stage73              | has_contrast      |       3 | 0.0612245 |  49 |
| new_false_NE_vs_stage73              | has_role_word     |       0 | 0         |  49 |
| new_false_NE_vs_stage73              | has_location_word |       4 | 0.0816327 |  49 |
| persistent_false_NE_stage73_stage92c | has_number        |     199 | 0.672297  | 296 |
| persistent_false_NE_stage73_stage92c | has_date_word     |      88 | 0.297297  | 296 |
| persistent_false_NE_stage73_stage92c | has_negation      |      13 | 0.0439189 | 296 |
| persistent_false_NE_stage73_stage92c | has_contrast      |      18 | 0.0608108 | 296 |
| persistent_false_NE_stage73_stage92c | has_role_word     |      16 | 0.0540541 | 296 |
| persistent_false_NE_stage73_stage92c | has_location_word |      18 | 0.0608108 | 296 |
| repaired_false_NE_vs_stage73         | has_number        |      21 | 0.777778  |  27 |
| repaired_false_NE_vs_stage73         | has_date_word     |      11 | 0.407407  |  27 |
| repaired_false_NE_vs_stage73         | has_negation      |       1 | 0.037037  |  27 |
| repaired_false_NE_vs_stage73         | has_contrast      |       1 | 0.037037  |  27 |
| repaired_false_NE_vs_stage73         | has_role_word     |       0 | 0         |  27 |
| repaired_false_NE_vs_stage73         | has_location_word |       2 | 0.0740741 |  27 |

## Stage95A bridge composition

| family                                            | label        |   rows | description                                                                                                                                        |
|:--------------------------------------------------|:-------------|-------:|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| explicit_sufficient_support_with_ne_distractor    | SUPPORT      |     36 | Evidence explicitly states the claim while also mentioning irrelevant/missing contextual facts that should not trigger NOT_ENTITLED.               |
| explicit_sufficient_refute_with_ne_distractor     | REFUTE       |     24 | Evidence explicitly contradicts the claim while also including incomplete peripheral context; decisive contradiction should prevent NE absorption. |
| support_exact_predicate_coverage_after_background | SUPPORT      |     36 | Longer evidence begins with background/uncertain context but later gives exact predicate coverage for SUPPORT.                                     |
| refute_exact_predicate_coverage_after_background  | REFUTE       |     24 | Longer evidence begins with background/uncertain context but later gives exact predicate contradiction for REFUTE.                                 |
| matched_missing_decisive_predicate_guardrail      | NOT_ENTITLED |     40 | Matched cases where claim/evidence share frame/entity but the decisive predicate/value/date/role is genuinely absent.                              |
