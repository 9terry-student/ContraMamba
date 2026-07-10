# Stage147-C Route/Order Targeted Diagnostic Synthesis

## Summary decision

Decision: `STAGE147C_ROUTE_ORDER_TARGETED_DIAGNOSTIC_PASS`

Stage147-B targeted diagnostic passed. The Stage147 route/order shadow analyzer produced analyzer decision `STAGE147_ROUTE_ORDER_SHADOW_CANDIDATE_PASS` on the targeted synthetic diagnostic, with 24 SUPPORT-to-NOT-ENTITLED shadow changes, false SUPPORT reduced from 24 to 0, and no false-NOT-ENTITLED increase.

The analyzer remains shadow-only and is not final integration ready.

## Why route/order is separate from text-location disjoint

Route/order reversal is different from set-based text-location disjointness. A claim such as `from Dublin to Cork` and evidence such as `from Cork to Dublin` mention the same location set, so `text_loc_disjoint_v2` should not trigger. The mismatch is in source/destination roles, which requires a separate directional route diagnostic.

The Stage147 analyzer is separate from `text_loc_disjoint_v2` and should remain separate until broader evidence supports any integration path.

## Stage147-B aggregate result

- Stage147-B decision: `STAGE147B_ROUTE_ORDER_TARGETED_DIAGNOSTIC_PASS`
- Analyzer decision: `STAGE147_ROUTE_ORDER_SHADOW_CANDIDATE_PASS`
- Input directory: `reports/stage147b_route_order_targeted_diagnostic_20260710_044934`
- Run directory: `reports/stage147b_route_order_targeted_diagnostic_20260710_044934/stage147_route_order_shadow_run`
- Rows: 76
- Changed predictions: 24
- SUPPORT to NOT-ENTITLED: 24
- False SUPPORT: 24 -> 0
- Delta false SUPPORT: -24
- False NOT-ENTITLED: 0 -> 0
- Delta false NOT-ENTITLED: 0
- Feature false SUPPORT TP: 24
- Feature correct SUPPORT FP: 0
- Feature false SUPPORT FN: 0
- Feature precision among SUPPORT predictions: 1.0
- Feature recall among SUPPORT predictions: 1.0
- Accuracy: 0.6842105263157895 -> 1.0
- Macro F1: 0.35000000000000003 -> 0.6666666666666666
- Delta macro F1: 0.3166666666666666

Route reversal, alias route reversal, and lowercase route reversal were caught.

## Case-level results

| Case | Assessment |
| --- | --- |
| `route_reversal_ne` | pass |
| `alias_route_reversal_ne` | pass |
| `lowercase_route_reversal_ne` | pass |
| `same_route_support` | safe |
| `same_route_with_extra_distractor_support` | safe |
| `nondirectional_between_support` | safe |
| `directional_vs_between_support` | safe |
| `org_endpoint_reversal_support` | safe |
| `alias_same_route_support` | safe |
| `one_sided_route_support` | safe |
| `non_support_prediction_safety` | safe |

Same-route, alias same-route, between, directional-vs-between, org-endpoint, one-sided route, and non-SUPPORT controls were safe.

## Policy input safety

The `route_order_reversal_v1` policy uses claim text, evidence text, the original prediction, deterministic route rules, deterministic alias rules, and deterministic organization-like endpoint rules. It does not use intervention type, slot mismatch target, gold labels, diagnostic family labels, file path heuristics, or row id heuristics.

The analyzer is shadow-only and diagnostic-only. It did not mutate source predictions, final logits, final predictions, training, checkpoint selection, Stage128 guard behavior, Stage15 behavior, external training data, or threshold/model-selection behavior.

## Interpretation

Stage147-B confirms that the route/order gap is addressed on the targeted synthetic diagnostic. The policy catches same-location-set reversals where source and destination roles are swapped, including alias and lowercase endpoint variants.

The targeted controls also show no harm on same-route support, alias same-route support, non-directional `between` cases, directional-vs-between cases, organization endpoint controls, one-sided route cases, or non-SUPPORT predictions.

This is a targeted synthetic diagnostic pass. It is not evidence of final integration readiness.

## Remaining limitations

- Only targeted synthetic controls have been evaluated.
- Broader external/general prediction export behavior is untested.
- Route extraction is deterministic and pattern-based.
- Complex route language, implicit directionality, multi-hop routes, and non-English routes may be missed.
- The analyzer remains shadow-only and should not be used as final prediction override.

## Stage148 recommendation

Stage148 should test broader/general prediction exports or harder route/order controls.

Recommended actions:

- Run the route/order analyzer on existing prediction JSONLs to estimate natural trigger rate.
- Create harder multi-hop and implicit route-order controls.
- Keep route/order analysis separate from `text_loc_disjoint_v2`.
- Keep all analysis shadow-only.

Avoid automatic final prediction override, training loss integration, checkpoint selection using route/order outputs, and premature merging of route/order into `text_loc_disjoint_v2`.
