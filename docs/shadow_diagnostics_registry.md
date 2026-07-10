# ContraMamba Shadow Diagnostic Registry

## Purpose

This registry documents current ContraMamba shadow-only diagnostics, their evidence, allowed use, limitations, and integration blocks. All listed tools are shadow-only. None may modify final logits or final predictions. None may be used for training, checkpoint selection, or external threshold tuning.

`text_loc_disjoint_v2` and `route_order_reversal_v2` should remain separate diagnostics because they target different failure modes: disjoint location evidence versus same-endpoint source/destination reversal.

## Global safety policy

| Policy | Status |
| --- | --- |
| Shadow-only | true |
| Diagnostic-only | true |
| Source predictions mutated | false |
| Final logits modified | false |
| Final predictions modified | false |
| Training modified | false |
| Checkpoint selection modified | false |
| Stage128 guard enabled | false |
| Stage15 used | false |
| External data used for training | false |
| Threshold used for model selection | false |

## Active diagnostic: `text_loc_disjoint_v2`

| Field | Value |
| --- | --- |
| Script | `scripts/analyze_stage145_text_location_guard_shadow_v2.py` |
| Status | `strong_shadow_diagnostic` |
| Purpose | Detect SUPPORT predictions with disjoint claim/evidence location-like spans after deterministic alias and organization-like guarding. |
| Final integration | blocked |

Evidence:

| Metric | Value |
| --- | ---: |
| Stage146-A decision | `STAGE146A_V2_EXTERNAL_GENERALIZATION_PASS` |
| n_files | 7 |
| n_rows | 33000 |
| delta_false_support | -53 |
| delta_false_ne | 3 |
| feature_false_support_tp | 53 |
| feature_correct_support_fp | 0 |
| macro_f1_delta | 0.0017308314825622562 |
| controlled_diagnostic_delta_false_support | -49 |
| controlled_diagnostic_delta_false_ne | 0 |
| external_or_factver_delta_false_support | -4 |
| external_or_factver_delta_false_ne | 3 |

Allowed use:

- Optional shadow diagnostic.
- Report-only analysis.
- Future prediction JSONL scans.

Forbidden use:

- Final prediction override.
- Training loss integration.
- Checkpoint selection.
- External threshold tuning.

Limitations:

- Small external/factver false-NE tradeoff.
- Deterministic extraction is incomplete.
- Does not handle same-location-set route/order reversal.

## Active diagnostic: `route_order_reversal_v2`

| Field | Value |
| --- | --- |
| Script | `scripts/analyze_stage150_route_order_shadow_v2.py` |
| Status | `conservative_shadow_diagnostic` |
| Purpose | Detect explicit source/destination reversal in SUPPORT predictions using conservative deterministic route extraction. |
| Final integration | blocked |

Evidence:

| Metric | Value |
| --- | ---: |
| Stage150-B decision | `STAGE150B_ROUTE_ORDER_V2_TARGETED_RETAINED_BROADER_FP_REMOVED` |
| Targeted decision | `STAGE150_ROUTE_ORDER_SHADOW_V2_CANDIDATE_PASS` |
| targeted_n_rows | 76 |
| targeted_delta_false_support | -24 |
| targeted_delta_false_ne | 0 |
| targeted_feature_false_support_tp | 24 |
| targeted_feature_correct_support_fp | 0 |
| targeted_precision | 1.0 |
| targeted_recall | 1.0 |
| Broader decision | `STAGE150_ROUTE_ORDER_SHADOW_V2_NO_EFFECT` |
| broader_n_files | 7 |
| broader_n_rows | 33000 |
| broader_n_changed_total | 0 |
| broader_delta_false_support | 0 |
| broader_delta_false_ne | 0 |
| broader_feature_correct_support_fp | 0 |
| rows_self_loop_routes_rejected | 2 |
| rows_title_like_routes_rejected | 10 |
| total_self_loop_routes_rejected | 4 |
| total_title_like_routes_rejected | 12 |

Allowed use:

- Optional conservative shadow diagnostic.
- Targeted route/order stress tests.
- Future prediction JSONL scans.

Forbidden use:

- Final prediction override.
- Training loss integration.
- Checkpoint selection.
- Merging into `text_loc_disjoint_v2`.

Limitations:

- Broader scan produced no natural true-positive route/order gain.
- Deterministic route extraction is pattern-based.
- Complex route language and implicit/multi-hop directionality remain untested.

## Historical/superseded diagnostics

| Diagnostic | Script | Status | Reason | Allowed use | Forbidden use |
| --- | --- | --- | --- | --- | --- |
| `text_loc_disjoint_raw_stage142` | `scripts/analyze_stage142_text_location_guard_shadow.py` | `historical_baseline` | Stage144 negative controls exposed alias and organization false-positive risk; superseded by `text_loc_disjoint_v2`. | Historical comparison only. | Active primary diagnostic use; final prediction override. |
| `route_order_reversal_v1_stage147` | `scripts/analyze_stage147_route_order_shadow.py` | `historical_baseline_superseded_by_v2` | Stage148-A broader scan exposed `Side to Side` self-loop/title false positive; superseded by `route_order_reversal_v2`. | Historical comparison only. | Broader/general active diagnostic use; final prediction override. |

## Recommended workflow

Run `text_loc_disjoint_v2` on prediction JSONLs when investigating false SUPPORT due to claim/evidence location mismatch, especially on controlled diagnostic and external/factver prediction exports.

Run `route_order_reversal_v2` on route/order targeted diagnostics and on prediction JSONLs when explicit `from A to B` route reversal is suspected.

Do not run either analyzer for training selection, final model prediction mutation, or final deployment guards.

## Forbidden uses

- Automatic final prediction override.
- Final logit modification.
- Final prediction modification.
- Training loss integration.
- Checkpoint selection.
- External threshold tuning.
- Merging `route_order_reversal_v2` into `text_loc_disjoint_v2`.

## Next planned diagnostic expansion

Stage152-A should be hard route/order expansion:

- Create multi-hop route/order controls.
- Create implicit direction controls.
- Create via/return/back-to controls.
- Create additional title/quoted phrase controls.
- Evaluate `route_order_reversal_v2` conservatively.
