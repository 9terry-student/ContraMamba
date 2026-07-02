# Stage75A — Targeted Bridge Design Plan

## Decision

`STAGE75A_TARGETED_BRIDGE_DESIGN_READY`

## Summary

| stage    | decision                              | source_stage74                                     | design_basis                                                                                                | planned_bridge_name              | planned_output_jsonl                        |   planned_row_count | planned_label_mix                                   | target_priority_order                                                                                                                             | training_executed   | data_generated   | external_eval_executed   | recommended_next_stage                                          |
|:---------|:--------------------------------------|:---------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:---------------------------------|:--------------------------------------------|--------------------:|:----------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:-----------------|:-------------------------|:----------------------------------------------------------------|
| Stage75A | STAGE75A_TARGETED_BRIDGE_DESIGN_READY | results/stage74_residual_external_error_audit.json | Stage74 residual external error audit aggregate counts only; no VitaminC text/labels used as training rows. | stage75_targeted_residual_bridge | data/stage75_targeted_residual_bridge.jsonl |                1020 | {"SUPPORT": 480, "REFUTE": 460, "NOT_ENTITLED": 80} | ["false_NE_on_SUPPORT", "false_NE_on_REFUTE", "wrong_polarity_REFUTE_to_SUPPORT", "wrong_polarity_SUPPORT_to_REFUTE", "small NE safety backstop"] | False               | False            | False                    | Stage75B implement synthetic targeted residual bridge generator |

## Stage74 residual signal

| signal                  |   value |   rate |
|:------------------------|--------:|-------:|
| row_count               |    1000 |  1     |
| error_total             |     647 |  0.647 |
| false_ne_total          |     323 |  0.323 |
| polarity_error_total    |     244 |  0.244 |
| false_entitlement_total |      80 |  0.08  |
| false_support_total     |     177 |  0.177 |
| false_refute_total      |     147 |  0.147 |

## Family plan

| family                                  | target_error                                                        |   stage74_count |   planned_rows |   SUPPORT |   REFUTE |   NOT_ENTITLED | purpose                                                                                       |
|:----------------------------------------|:--------------------------------------------------------------------|----------------:|---------------:|----------:|---------:|---------------:|:----------------------------------------------------------------------------------------------|
| support_entitlement_direct_recovery_v2  | false_NE_on_SUPPORT                                                 |             178 |            240 |       240 |        0 |              0 | Recover obvious SUPPORT cases that are currently over-abstained as NOT_ENTITLED.              |
| refute_entitlement_direct_recovery_v2   | false_NE_on_REFUTE                                                  |             145 |            220 |         0 |      220 |              0 | Recover obvious REFUTE cases that are currently over-abstained as NOT_ENTITLED.               |
| numeric_temporal_polarity_comparison_v2 | wrong_polarity_SUPPORT_to_REFUTE + wrong_polarity_REFUTE_to_SUPPORT |             244 |            260 |       130 |      130 |              0 | Reduce polarity flips on before/after, greater/less, at-least/under, exact-count comparisons. |
| lexical_type_polarity_disambiguation_v2 | wrong_polarity_SUPPORT_to_REFUTE + wrong_polarity_REFUTE_to_SUPPORT |             244 |            220 |       110 |      110 |              0 | Reduce polarity flips where lexical overlap hides type mismatch or direct equivalence.        |
| strict_ne_external_style_safety_v2      | false_SUPPORT_on_NE + false_REFUTE_on_NE                            |              80 |             80 |         0 |        0 |             80 | Preserve NE safety without overcorrecting into more false_NE.                                 |

## Planned label mix

| label        |   planned_rows |      rate |
|:-------------|---------------:|----------:|
| SUPPORT      |            480 | 0.470588  |
| REFUTE       |            460 | 0.45098   |
| NOT_ENTITLED |             80 | 0.0784314 |

## Checks

| check                                  | pass   |
|:---------------------------------------|:-------|
| stage74_ready                          | True   |
| false_ne_is_largest_residual_bucket    | True   |
| polarity_error_second_priority         | True   |
| ne_safety_small_by_design              | True   |
| positive_entitled_skew_by_design       | True   |
| no_external_text_training_policy       | True   |
| stage75_total_rows_1020                | True   |
| stage75_support_refute_balanced_enough | True   |

## Design interpretation

Stage74 shows that the dominant residual failure is not excessive false entitlement. The largest bucket is false_NE_total, followed by polarity_error_total. Therefore Stage75 should add mostly SUPPORT/REFUTE entitlement-recovery and polarity-disambiguation bridge rows, with only a small NOT_ENTITLED safety backstop.

No VitaminC text or labels should be copied into training. Stage74 is used only for aggregate residual targeting.

## Recommended next stage

Stage75B implement synthetic targeted residual bridge generator
