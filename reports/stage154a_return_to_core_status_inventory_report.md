# Stage154-A Return-to-Core Status Inventory

## Summary decision

`STAGE154A_RETURN_TO_CORE_STATUS_INVENTORY_READY`

Stage154-A records that Stage153-A completed the shadow-diagnostics soft freeze. The project should now return to ContraMamba core/generalization/architecture work, with the next step focused on identifying the dominant remaining model-internal error family.

## Why Stage154-A exists

Stage145 through Stage153 produced useful shadow diagnostics for false-SUPPORT failure families, then deliberately paused deterministic route semantic expansion. Stage154-A exists to preserve that status, separate stable diagnostic assets from unresolved core questions, and choose the next non-rule-patch bottleneck decision point.

The intended direction is not another final-prediction guard. The final model direction should be structured internal judgment, not deterministic rule override.

## Stable assets after soft freeze

The main clean data policy remains stable:

| Asset | Status |
| --- | --- |
| Main clean data | `data/controlled_v5_v3_without_time_swap.jsonl` |
| `time_swap` clean training use | forbidden |
| `time_swap` allowed use | auxiliary diagnostic/evaluation only |

The active safety posture is unchanged:

| Constraint | Value |
| --- | --- |
| Shadow-only | true |
| Diagnostic-only | true |
| Source predictions mutated | false |
| Final logits modified | false |
| Final predictions modified | false |
| Training modified | false |
| Checkpoint selection modified | false |
| External data used for training | false |
| Threshold used for model selection | false |

## Shadow diagnostics status

`text_loc_disjoint_v2` remains a useful strong shadow diagnostic. It is soft-frozen, shadow-only, diagnostic-only, and blocked from final integration.

`route_order_reversal_v2` remains a useful conservative shadow diagnostic. It is also soft-frozen, shadow-only, diagnostic-only, and blocked from final integration.

These diagnostics may remain available for audit and report-only analysis, but they must not be used for training, checkpoint selection, final logits, final predictions, or external threshold tuning.

## Recent diagnostic evidence

`text_loc_disjoint_v2` passed Stage146-A with decision `STAGE146A_V2_EXTERNAL_GENERALIZATION_PASS`.

| Metric | Value |
| --- | ---: |
| n_rows | 33000 |
| delta_false_support | -53 |
| delta_false_ne | 3 |
| feature_false_support_tp | 53 |
| feature_correct_support_fp | 0 |
| macro_f1_delta | 0.0017308314825622562 |

`route_order_reversal_v2` was retained through Stage150-B and Stage152-A. Stage150-B decision: `STAGE150B_ROUTE_ORDER_V2_TARGETED_RETAINED_BROADER_FP_REMOVED`. Stage152-A decision: `STAGE152A_ROUTE_ORDER_HARD_EXPANSION_CONSERVATIVE_PASS_WITH_BOUNDARY_MISSES`.

| Finding | Status |
| --- | --- |
| Explicit route reversal | covered |
| Targeted delta_false_support | -24 |
| Targeted delta_false_ne | 0 |
| Broader n_rows | 33000 |
| Broader n_changed_total | 0 |
| Broader feature_correct_support_fp | 0 |
| Hard expansion delta_false_support | -30 |
| Hard expansion delta_false_ne | 0 |
| Hard expansion feature_correct_support_fp | 0 |
| Expected boundary misses | 12 |
| Implicit multi-hop route semantics | backlog boundary miss |
| Final integration | blocked |

## Unresolved core questions

- What is the current best clean/dev architecture after the shadow diagnostic sequence?
- Which external/generalization failure mode is now dominant after location and route/order diagnostics?
- Is the main bottleneck still false SUPPORT, false NOT_ENTITLED, support recall, external fact-verification transfer, or prefix/artifact sensitivity?
- Does vNext routing/entitlement need architecture-level modification rather than more rule diagnostics?
- Which scalar exports best explain current external errors?

## Non-rule-patch bottleneck candidates

| Candidate | Status | Reason |
| --- | --- | --- |
| External fact-verification transfer | likely_priority | Earlier external VitaminC/Climate-FEVER diagnostics remained weak relative to clean/dev despite later shadow diagnostics. |
| Entitlement calibration | likely_priority | Core ContraMamba goal depends on learned entitlement/compositional sufficiency rather than external rule overrides. |
| Support recall vs false SUPPORT tradeoff | likely_priority | Several stages show tension between suppressing false SUPPORT and preserving correct SUPPORT. |
| Prefix artifact and core localization | monitor | Stage120-122 explored prefix/core localization; may need consolidation before new architecture changes. |
| Implicit temporal or route semantics | backlog | Would require broader semantic parsing and negative controls; not current priority. |

The failures most suitable for core architecture treatment are external fact-verification transfer, entitlement calibration, and the support-recall versus false-SUPPORT tradeoff. These are model-internal bottlenecks because they concern learned sufficiency, compositional judgment, calibration, and generalization rather than isolated lexical or route-pattern defects.

## Recommended Stage155-A

Stage155-A should run or synthesize a post-freeze core error inventory over the current best prediction exports.

Recommended default: create a post-freeze core error taxonomy that excludes shadow-rule integration and identifies the dominant remaining model-internal failure mode.

Candidate outputs:

- Dominant remaining error family.
- SUPPORT/NOT_ENTITLED/REFUTE confusion profile.
- Scalar/error correlation table.
- Recommendation for next architecture or evaluation stage.

## Safety constraints

Stage155-A must not integrate shadow diagnostics into final predictions, train on external data, tune thresholds on external labels, or reopen route/order expansion unless evidence shows that route/order semantics are dominant again.

The project state is: shadow diagnostics stabilized; core/generalization work should resume. The current best action is to identify the dominant non-rule-patch error bottleneck.
