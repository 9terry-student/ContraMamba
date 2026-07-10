# Stage150-C Route/Order v2 Validation Synthesis

## Summary decision

`STAGE150C_ROUTE_ORDER_V2_TARGETED_RETAINED_BROADER_FP_REMOVED`

Stage150-B passed. `route_order_reversal_v2` retained the targeted route/order diagnostic behavior and removed the broader false positive exposed by Stage148-A. It should remain a conservative shadow-only diagnostic, and final integration remains blocked.

## Why v2 was needed

Stage148-A found that Stage147 v1 could misread the title phrase `Side to Side` as a route. The claim was `Side to Side was released after June 2 , 2016 .`, and the evidence described `'' Side to Side ''` as a song by Ariana Grande. V1 extracted the phrase as a degenerate `side -> side` route, creating a self-loop/title false positive.

Stage150-A created `scripts/analyze_stage150_route_order_shadow_v2.py` as a separate analyzer with policy `route_order_reversal_v2`. It did not replace Stage147 v1, Stage145 text-location guard v2, or Stage142.

## Targeted validation

The targeted Stage147-B diagnostic run passed with decision `STAGE150_ROUTE_ORDER_SHADOW_V2_CANDIDATE_PASS`.

| Metric | Value |
| --- | ---: |
| n_rows | 76 |
| n_changed_total | 24 |
| delta_false_support | -24 |
| delta_false_ne | 0 |
| feature_false_support_tp | 24 |
| feature_correct_support_fp | 0 |
| feature_false_support_fn | 0 |
| feature_precision | 1.0 |
| feature_recall | 1.0 |

Targeted checks passed for route reversal, alias reversal, lowercase known-location reversal, same-route safety, alias same-route safety, `between` safety, organization endpoint safety, and non-SUPPORT safety. Targeted route/order diagnostic behavior was retained.

## Broader validation

The broader seven-file scan completed with decision `STAGE150_ROUTE_ORDER_SHADOW_V2_NO_EFFECT`.

| Metric | Value |
| --- | ---: |
| n_files | 7 |
| n_rows | 33000 |
| n_changed_total | 0 |
| delta_false_support | 0 |
| delta_false_ne | 0 |
| feature_false_support_tp | 0 |
| feature_correct_support_fp | 0 |
| feature_false_support_fn | 229 |
| rows_self_loop_routes_rejected | 2 |
| rows_title_like_routes_rejected | 10 |
| total_self_loop_routes_rejected | 4 |
| total_title_like_routes_rejected | 12 |

The broader result is `NO_EFFECT`, not a false-support improvement. No natural true-positive route reversal gain was observed in the available exports, but there was also no harmful trigger, no correct-SUPPORT false positive, and no false-NE increase.

## Comparison to Stage148-A failure

Stage148-A failed with decision `STAGE148A_ROUTE_ORDER_BROADER_HARM_OR_NO_FS_GAIN`: `delta_false_support = 0`, `delta_false_ne = +1`, and `feature_correct_support_fp = 1`.

Stage150-B removed that broader false positive: `n_changed_total = 0`, `delta_false_ne = 0`, and `feature_correct_support_fp = 0`. The `Side to Side` self-loop/title failure is addressed by the v2 self-loop and title/context guards.

## v2 refinements

- Self-loop guard: enabled.
- Title/context guard: enabled.
- Conservative bare `A to B` handling: enabled by policy design, with broad bare extraction disabled by default.
- Organization endpoint guard: retained.
- Alias canonicalization: retained.

## Policy input safety

The policy uses claim text, evidence text, original prediction labels, deterministic route rules, deterministic alias rules, deterministic organization-like endpoint rules, deterministic self-loop rejection, and deterministic title/context guards.

It does not use gold labels for policy decisions, intervention type, slot mismatch targets, diagnostic family metadata, file path heuristics, or row id heuristics.

## Interpretation

Stage150-B passes. The targeted synthetic behavior is retained, and the broader Stage148-A false positive is removed. Because the broader scan produced no natural true-positive route reversal gain, `route_order_reversal_v2` is ready only as a conservative shadow diagnostic.

`route_order_reversal_v2` should remain shadow-only. Final prediction override remains blocked.

## Remaining limitations

- Broader scan produced no natural true-positive route reversal gain.
- Route extraction remains deterministic and pattern-based.
- Complex route language, implicit directionality, multi-hop routes, and context-dependent route semantics remain untested.
- The analyzer should remain shadow-only.
- Final prediction override remains blocked.

## Stage151 recommendation

Stage151 should unify reporting or registry of shadow diagnostics, or run a larger external route/order stress expansion.

Recommended actions:

- Register `route_order_reversal_v2` as a conservative optional shadow diagnostic.
- Keep it separate from `text_loc_disjoint_v2`.
- Optionally create harder multi-hop or implicit route controls.
- Do not integrate into final logits or final predictions.

Avoid automatic final prediction override, training loss integration, checkpoint selection using route/order outputs, and merging route/order into `text_loc_disjoint_v2`.
