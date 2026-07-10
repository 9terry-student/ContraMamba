# Stage148-B Route/Order Broader Scan Synthesis

## Summary decision

Decision: `STAGE148B_ROUTE_ORDER_BROADER_SCAN_EXPOSES_NATURAL_FALSE_POSITIVE_RISK`

Stage148-A did not pass as a broader/general shadow diagnostic. The broader scan produced one route/order trigger, and that trigger was harmful: it introduced one correct-SUPPORT false positive as a shadow SUPPORT-to-NOT-ENTITLED change.

There was no false-support reduction. Final integration remains blocked.

## Why Stage148-A was needed

Stage147-B showed that `route_order_reversal_v1` works on targeted synthetic route/order controls. It caught same-location-set source/destination reversals, including alias and lowercase variants, while preserving same-route, between, organization-endpoint, one-sided-route, and non-SUPPORT controls.

Stage148-A was needed to test whether the same policy behaves safely on broader existing prediction exports. That broader scan exposed the current boundary: the policy remains useful as a targeted diagnostic, but it is not ready as a broader/general shadow diagnostic.

## Aggregate result

- Stage148-A decision: `STAGE148A_ROUTE_ORDER_BROADER_HARM_OR_NO_FS_GAIN`
- Analyzer decision: `STAGE147_ROUTE_ORDER_SHADOW_HARM_OR_NO_FS_GAIN`
- Run directory: `reports/stage148a_route_order_broader_run_20260710_045357`
- Summary JSON: `reports/stage148a_route_order_broader_summary/stage148a_route_order_broader_summary_report.json`
- Files: 7
- Rows: 33,000
- Changed predictions: 1
- False SUPPORT: 229 -> 229
- Delta false SUPPORT: 0
- False NOT-ENTITLED: 2879 -> 2880
- Delta false NOT-ENTITLED: +1
- Feature false SUPPORT TP: 0
- Feature correct SUPPORT FP: 1
- Feature false SUPPORT FN: 229
- Macro F1: 0.7075607576618963 -> 0.7074803026145983
- Delta macro F1: -0.00008045504729803366
- Min per-file delta macro F1: -0.0008103246190390712
- Support rows with route reversal: 1

The broader scan produced one trigger, no false-support gain, and one false-positive cost.

## Controlled diagnostic subset

- Files: 5
- Rows: 31,000
- Changed predictions: 0
- Support rows with route reversal: 0
- Delta false SUPPORT: 0
- Delta false NOT-ENTITLED: 0
- Feature false SUPPORT TP: 0
- Feature correct SUPPORT FP: 0
- Assessment: `no_natural_route_triggers`

The controlled diagnostic subset had no route/order triggers.

## External/factver subset

- Files: 2
- Rows: 2,000
- Changed predictions: 1
- Support rows with route reversal: 1
- Delta false SUPPORT: 0
- Delta false NOT-ENTITLED: +1
- Feature false SUPPORT TP: 0
- Feature correct SUPPORT FP: 1
- Assessment: `one_harmful_false_positive_trigger`

The external/factver subset had one false-positive trigger.

## Comparison to Stage147-B targeted pass

Stage147-B targeted diagnostic remains valid. It reduced false SUPPORT by 24, introduced no false NOT-ENTITLED increase, and had `feature_correct_support_fp = 0`.

Stage148-A did not reproduce that benefit on broader natural/export behavior. It found one trigger, but that trigger was a correct-SUPPORT false positive and produced no false-support reduction.

This comparison defines the policy boundary: `route_order_reversal_v1` should remain targeted-only until false-positive forensics are complete.

## Policy input safety

The `route_order_reversal_v1` policy uses claim text, evidence text, the original prediction, deterministic route rules, deterministic alias rules, and deterministic organization-like endpoint rules. It does not use intervention type, slot mismatch target, gold labels, diagnostic family labels, file path heuristics, or row id heuristics.

The analyzer is shadow-only and diagnostic-only. It did not mutate source predictions, final logits, final predictions, training, checkpoint selection, Stage128 guard behavior, Stage15 behavior, external training data, or threshold/model-selection behavior.

## Interpretation

The route/order gap is still addressed on targeted synthetic examples, so Stage147-B is not invalidated. However, Stage148-A shows that broader/general use is blocked because the only natural route/order trigger was harmful.

Natural route/order reversal triggers appear rare in the current broader prediction exports. The controlled diagnostic files had no triggers, while the external/factver subset had one false-positive trigger. `route_order_reversal_v1` is therefore a targeted diagnostic only for now, not a broad external/general shadow candidate and not final integration ready.

## Remaining limitations

- Only one natural route/order trigger was found in the broader scan.
- The only broader trigger was a correct-SUPPORT false positive.
- No false-SUPPORT gain was observed in broader files.
- Route extraction is deterministic and pattern-based.
- Complex route language, implicit directionality, multi-hop routes, and context-dependent route semantics remain risky.
- The analyzer should remain targeted-only until false-positive forensics are complete.

## Stage149 recommendation

Stage149 should inspect the single Stage148-A false-positive route/order trigger and determine whether the analyzer needs a conservative patch or should remain targeted-only.

Recommended actions:

- Read the Stage148-A changed example.
- Classify the false positive cause.
- Check whether the route extractor confused non-route text, same-route paraphrase, event ordering, or entity movement context.
- If the cause is patchable, create a stricter route/order v2 policy.
- If the cause is semantic/contextual, keep `route_order_reversal_v1` targeted-only.

Avoid automatic final prediction override, training loss integration, checkpoint selection using route/order outputs, and broad route/order deployment before false-positive forensics.
