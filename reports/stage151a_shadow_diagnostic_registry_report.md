# Stage151-A Shadow Diagnostic Registry Report

## Summary decision

`STAGE151A_SHADOW_DIAGNOSTIC_REGISTRY_CREATED`

Stage151-A created a static registry for ContraMamba shadow-only diagnostics. The registry distinguishes active recommended shadow diagnostics from historical or superseded analyzers and restates integration blocks.

## Active diagnostics registered

| Diagnostic | Script | Status | Final integration |
| --- | --- | --- | --- |
| `text_loc_disjoint_v2` | `scripts/analyze_stage145_text_location_guard_shadow_v2.py` | `strong_shadow_diagnostic` | blocked |
| `route_order_reversal_v2` | `scripts/analyze_stage150_route_order_shadow_v2.py` | `conservative_shadow_diagnostic` | blocked |

`text_loc_disjoint_v2` is recommended for report-only scans of false SUPPORT risk caused by disjoint claim/evidence location-like spans. Stage146-A passed with `delta_false_support = -53`, `delta_false_ne = +3`, `feature_false_support_tp = 53`, and `feature_correct_support_fp = 0`.

`route_order_reversal_v2` is recommended as a conservative optional diagnostic for explicit source/destination reversal. Stage150-B passed: targeted behavior was retained, and the Stage148-A `Side to Side` self-loop/title false positive was removed. The broader route/order result remains `NO_EFFECT`, not a false-support improvement.

## Historical/superseded diagnostics

| Diagnostic | Script | Status | Registry role |
| --- | --- | --- | --- |
| `text_loc_disjoint_raw_stage142` | `scripts/analyze_stage142_text_location_guard_shadow.py` | `historical_baseline` | historical comparison only |
| `route_order_reversal_v1_stage147` | `scripts/analyze_stage147_route_order_shadow.py` | `historical_baseline_superseded_by_v2` | historical comparison only |

Stage142 raw text-location logic was superseded after Stage144 exposed alias and organization false-positive risk. Stage147 route/order v1 was superseded after Stage148-A exposed the `Side to Side` self-loop/title false positive.

## Safety policy

All registered tools are shadow-only and diagnostic-only. They must not mutate source predictions, modify final logits, modify final predictions, alter training, drive checkpoint selection, enable Stage128/Stage15 behavior, use external data for training, or tune thresholds for model selection.

## Workflow impact

Run `text_loc_disjoint_v2` when investigating false SUPPORT due to claim/evidence location mismatch. Run `route_order_reversal_v2` for route/order targeted diagnostics or prediction JSONLs where explicit `from A to B` reversal is suspected.

Do not use these analyzers for training selection, final prediction mutation, final deployment guards, or automatic final prediction override. `text_loc_disjoint_v2` and `route_order_reversal_v2` should remain separate.

## Stage152 recommendation

Stage152-A should be hard route/order expansion:

- Create multi-hop route/order controls.
- Create implicit direction controls.
- Create via/return/back-to controls.
- Create additional title/quoted phrase controls.
- Evaluate `route_order_reversal_v2` conservatively.
