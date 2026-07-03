# Stage80A - Conservative Stage75v2 Design Plan

## Decision

`STAGE80A_CONSERVATIVE_STAGE75V2_DESIGN_READY`

## Summary

| stage    | decision                                     | purpose                                                                                  | stage75_source_decision   | planned_bridge_name                    | planned_output_jsonl                              |   planned_total_rows | planned_label_counts                                 |   planned_family_count | main_rule                                                                           |   stage77_minus_stage73_accuracy |   stage77_minus_stage73_macro_f1 |   stage78_minus_stage74_false_ne_total |   stage78_minus_stage74_polarity_error_total |   stage78_minus_stage74_false_entitlement_total |   stage78_minus_stage74_false_support_total |   stage78_minus_stage74_false_refute_total | training_executed   | external_eval_executed   | recommended_next_stage                                     |
|:---------|:---------------------------------------------|:-----------------------------------------------------------------------------------------|:--------------------------|:---------------------------------------|:--------------------------------------------------|---------------------:|:-----------------------------------------------------|-----------------------:|:------------------------------------------------------------------------------------|---------------------------------:|---------------------------------:|---------------------------------------:|---------------------------------------------:|------------------------------------------------:|--------------------------------------------:|-------------------------------------------:|:--------------------|:-------------------------|:-----------------------------------------------------------|
| Stage80A | STAGE80A_CONSERVATIVE_STAGE75V2_DESIGN_READY | Design a conservative Stage75v2 revision after Stage79 rejected full Stage75 as default. | REVISE_NOT_DEFAULT        | stage80a_conservative_stage75v2_bridge | data/stage80a_conservative_stage75v2_bridge.jsonl |                  500 | {"SUPPORT": 170, "REFUTE": 170, "NOT_ENTITLED": 160} |                      3 | Keep balanced polarity repair; drop broad SUPPORT recovery; add NE safety backstop. |                            0.004 |                           -0.008 |                                      1 |                                          -11 |                                               6 |                                          19 |                                        -24 | False               | False                    | Stage80B implement conservative Stage75v2 bridge generator |

## Planned families

| family                                           | include   |   planned_rows |   SUPPORT |   REFUTE |   NOT_ENTITLED | purpose                                                                                                      | targets                                                            |
|:-------------------------------------------------|:----------|---------------:|----------:|---------:|---------------:|:-------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
| numeric_temporal_polarity_repair_v2_conservative | True      |            180 |        90 |       90 |              0 | Retain Stage75's useful polarity repair signal while keeping support/refute balanced.                        | wrong_polarity_REFUTE_to_SUPPORT, wrong_polarity_SUPPORT_to_REFUTE |
| lexical_type_polarity_repair_v2_conservative     | True      |            160 |        80 |       80 |              0 | Retain type/predicate polarity disambiguation but reduce volume from Stage75.                                | wrong_polarity_REFUTE_to_SUPPORT, wrong_polarity_SUPPORT_to_REFUTE |
| strict_ne_false_support_safety_v2                | True      |            160 |         0 |        0 |            160 | Counter the observed false SUPPORT and false entitlement increase after Stage75.                             | false_SUPPORT_on_NE, false_entitlement_total                       |
| support_entitlement_direct_recovery_v2           | False     |              0 |         0 |        0 |              0 | Dropped because broad SUPPORT recovery likely contributed to false SUPPORT increase.                         | false_NE_on_SUPPORT                                                |
| refute_entitlement_direct_recovery_v2            | False     |              0 |         0 |        0 |              0 | Dropped from v2 main bridge; false REFUTE already improved, but direct recovery may destabilize entitlement. | false_NE_on_REFUTE                                                 |

## Ablation plan

| ablation                                | description                                                  |   stage75v2_rows | expected_use                                                       |
|:----------------------------------------|:-------------------------------------------------------------|-----------------:|:-------------------------------------------------------------------|
| A0_primary                              | Stage57+Stage66 only; current primary equals Stage71_retry2. |                0 | baseline/reference                                                 |
| A1_conservative_polarity_only           | Use only the two balanced polarity repair families.          |              340 | test whether polarity gains survive without SUPPORT over-recovery. |
| A2_conservative_polarity_plus_ne_safety | Use full Stage80A v2 plan: polarity repair plus NE safety.   |              500 | main candidate if A1 still produces false SUPPORT.                 |
| A3_ne_safety_only                       | Use only strict_ne_false_support_safety_v2.                  |              160 | diagnostic: isolate effect on false SUPPORT / false entitlement.   |

## Rationale

| finding                        | evidence                                                             | design_response                                                  |
|:-------------------------------|:---------------------------------------------------------------------|:-----------------------------------------------------------------|
| Clean-dev preservation passed. | Stage76: Stage75F macro-F1 drop versus Stage71 was within tolerance. | No need to reject all Stage75-derived ideas.                     |
| External macro-F1 dropped.     | Stage77 - Stage73 macro-F1 = -0.008.                                 | Do not keep Stage75 full bridge as default.                      |
| Polarity error improved.       | Stage78 - Stage74 polarity_error_total = -11.                        | Preserve balanced polarity repair families.                      |
| False REFUTE improved.         | Stage78 - Stage74 false_refute_total = -24.                          | Keep polarity repair, but do not over-train direct recovery.     |
| False SUPPORT worsened.        | Stage78 - Stage74 false_support_total = +19.                         | Drop broad SUPPORT recovery and add explicit NE safety backstop. |
| False NE did not improve.      | Stage78 - Stage74 false_ne_total = +1.                               | Do not spend v2 capacity on broad entitlement recovery.          |

## Checks

| check                                     | pass   |
|:------------------------------------------|:-------|
| stage79_ready                             | True   |
| stage75_decision_revise_not_default       | True   |
| clean_dev_preserved                       | True   |
| external_macro_dropped_vs_stage73         | True   |
| external_acc_slightly_improved_vs_stage73 | True   |
| polarity_error_reduced                    | True   |
| false_refute_reduced                      | True   |
| false_support_increased                   | True   |
| false_entitlement_increased               | True   |
| planned_rows_500                          | True   |
| planned_rows_less_than_stage75_1020       | True   |
| support_rows_not_dominant                 | True   |
| ne_safety_rows_present                    | True   |
| dropped_broad_support_recovery            | True   |

## Recommended next stage

Stage80B implement conservative Stage75v2 bridge generator
