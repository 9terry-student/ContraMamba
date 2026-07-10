# Stage153-A Shadow Diagnostics Soft Freeze and Return-to-Core Decision

## Summary decision

`STAGE153A_SHADOW_DIAGNOSTICS_SOFT_FREEZE_RETURN_TO_CORE`

This is a soft freeze, not a permanent closure. `text_loc_disjoint_v2` remains an active strong shadow diagnostic, and `route_order_reversal_v2` remains valid as a conservative shadow diagnostic. Further route semantic expansion is paused, final integration remains blocked, and the next stage should return to ContraMamba core/generalization/architecture work.

## What soft freeze means

Soft freeze means the current shadow diagnostics stay available for audit and report-only scans, but they are not promoted into final logits, final predictions, training loss, checkpoint selection, or deployment guards.

The freeze is not permanent. It preserves a clear reopen path if future external/core evaluation shows that location-disjoint or route-semantic failures are again dominant enough to justify targeted diagnostic work.

## Active diagnostics after freeze

| Diagnostic | Status after Stage153-A | Allowed use | Integration status |
| --- | --- | --- | --- |
| `text_loc_disjoint_v2` | `strong_shadow_diagnostic_soft_frozen` | Shadow-only audit, report-only analysis, future prediction JSONL scans | blocked |
| `route_order_reversal_v2` | `conservative_shadow_diagnostic_soft_frozen` | Shadow-only audit, targeted route/order stress tests, future prediction JSONL scans when explicit route reversal is suspected | blocked |

`text_loc_disjoint_v2` and `route_order_reversal_v2` should remain separate. The first targets disjoint location-like spans; the second targets explicit source/destination route reversal.

## Evidence from Stage150-B

Stage150-B validated `route_order_reversal_v2` with decision `STAGE150B_ROUTE_ORDER_V2_TARGETED_RETAINED_BROADER_FP_REMOVED`.

| Evidence | Value |
| --- | ---: |
| targeted_delta_false_support | -24 |
| targeted_delta_false_ne | 0 |
| targeted_feature_correct_support_fp | 0 |
| broader_n_rows | 33000 |
| broader_n_changed_total | 0 |
| broader_delta_false_ne | 0 |
| broader_feature_correct_support_fp | 0 |

The known title/self-loop false positive was fixed: broader v2 removed the `Side to Side` issue while avoiding false-NE increase.

## Evidence from Stage152-A

Stage152-A passed with decision `STAGE152A_ROUTE_ORDER_HARD_EXPANSION_CONSERVATIVE_PASS_WITH_BOUNDARY_MISSES`.

| Evidence | Value |
| --- | ---: |
| n_rows | 99 |
| delta_false_support | -30 |
| delta_false_ne | 0 |
| feature_false_support_tp | 30 |
| feature_correct_support_fp | 0 |
| feature_false_support_fn | 12 |
| precision | 1.0 |
| recall | 0.7142857142857143 |

Required trigger checks passed for direct, alias, lowercase, and via route reversal. Required safety checks passed for same-route, alias same-route, between/nondirectional, directional-vs-between, organization endpoint, title/self-loop, quoted title, one-sided route, non-SUPPORT, and return-trip controls.

Remaining implicit/multi-hop route misses are backlog items, not current failures.

## Why not expand route semantics now

`route_order_reversal_v2` solves explicit source/destination reversal. It intentionally does not solve implicit depart/arrive semantics, start/end origin-destination semantics, multi-hop itinerary order swaps, or context-dependent itinerary semantics.

Extending those now would turn the project toward deterministic route semantic parser engineering. That risks scope creep and false positives, and it distracts from the core ContraMamba goal: structured epistemic judgment through frame, predicate, sufficiency, polarity, and entitlement.

## Reopen conditions

Reopen `text_loc_disjoint_v2` work if new external/generalization evidence shows location-disjoint false SUPPORT remains dominant, new negative controls expose systematic false positives, or a registry update requires harmonized interfaces.

Reopen `route_order_reversal_v2` work if external/core evaluation shows route semantic errors are a major remaining false-SUPPORT source, implicit depart/arrive or multi-hop route errors become frequent enough to justify a separate semantic analyzer, new negative controls are available before expansion, or the project explicitly branches into semantic itinerary parsing.

## Return-to-core plan

Recommended next focus: ContraMamba core/generalization/architecture work.

Priorities:

- External/generalization diagnostics.
- Core model error taxonomy.
- Architecture-level entitlement and compositional failure analysis.
- Diagnostic registry maintenance only as needed.

Deprioritized:

- Deterministic route semantic parser expansion.
- Final prediction override guards.
- Rule-based training loss integration.

## Global safety policy

All current shadow diagnostics remain shadow-only and diagnostic-only. They must not mutate source predictions, modify final logits, modify final predictions, change training, drive checkpoint selection, enable Stage128/Stage15 behavior, use external data for training, or tune thresholds for model selection.

Final integration remains blocked.

## Stage154 recommendation

Stage154 should return to core ContraMamba evaluation/generalization planning.

Recommended actions:

- Summarize current clean/dev/external status.
- Identify the next non-rule-patch bottleneck.
- Design the next architecture/evaluation stage around model-internal judgment structure rather than external rule overrides.

Avoid more route/order regex expansion unless reopen conditions are met, final prediction override, training loss integration from shadow diagnostics, and checkpoint selection using shadow diagnostic outputs.
