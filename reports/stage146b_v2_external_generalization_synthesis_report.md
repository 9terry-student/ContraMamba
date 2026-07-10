# Stage146-B v2 External Generalization Synthesis

## Summary decision

Decision: `STAGE146B_V2_EXTERNAL_GENERALIZATION_PASS_STRONG_SHADOW_DIAGNOSTIC`

Stage146-A passed on the currently available external/general prediction JSONLs. The `text_loc_disjoint_v2` analyzer remains a strong shadow diagnostic, not a final prediction override. Final integration remains blocked because external evidence is still limited and the external/factver subset retains a small false-NOT-ENTITLED tradeoff.

## Stage146-A setup

Stage146-A expanded the v2 text-location disjointness policy to existing external/general prediction JSONLs.

- Run directory: `reports/stage146a_v2_external_generalization_run_20260710_043908`
- Summary JSON: `reports/stage146a_v2_external_generalization_summary/stage146a_v2_external_generalization_summary_report.json`
- Stage146-A decision: `STAGE146A_V2_EXTERNAL_GENERALIZATION_PASS`
- Analyzer decision: `STAGE145_TEXT_LOCATION_GUARD_V2_SHADOW_CANDIDATE_MILD_TRADEOFF`
- Files evaluated: 7
- Rows evaluated: 33,000

## Aggregate result

Across all selected Stage146-A files, v2 changed 53 predictions from SUPPORT to NOT-ENTITLED. Aggregate false SUPPORT decreased by 53, while false NOT-ENTITLED increased by 3.

- `n_changed_total = 53`
- `delta_false_support = -53`
- `delta_false_ne = +3`
- `feature_false_support_tp = 53`
- `feature_correct_support_fp = 0`
- `feature_false_support_fn = 176`
- `feature_precision_for_false_support_among_support_preds = 1.0`
- `feature_recall_for_false_support_among_support_preds = 0.2314410480349345`
- `macro_f1_before = 0.7075607576618963`
- `macro_f1_after = 0.7092915891444586`
- `delta_macro_f1 = 0.0017308314825622562`
- `min_per_file_delta_macro_f1 = 0.0`

The aggregate behavior satisfies the Stage146-A pass condition and supports treating v2 as a strong shadow diagnostic.

## Controlled diagnostic subset

The controlled diagnostic subset remained clean under v2.

- Files: 5
- Rows: 31,000
- Changed predictions: 49
- False SUPPORT: -49
- False NOT-ENTITLED: 0
- `feature_correct_support_fp = 0`
- Assessment: `passes_without_false_ne_cost`

This means the controlled diagnostic subset had false SUPPORT -49 and false NE 0.

## External/factver subset

The external/factver subset showed a small tradeoff while preserving aggregate pass behavior.

- Files: 2
- Rows: 2,000
- Changed predictions: 4
- False SUPPORT: -4
- False NOT-ENTITLED: +3
- `feature_correct_support_fp = 0`
- Assessment: `small_tradeoff_but_aggregate_pass`

This means the external/factver subset had false SUPPORT -4 and false NE +3.

## Policy input safety

The v2 policy uses claim text, evidence text, the original prediction, deterministic alias rules, and deterministic organization-like rules. It does not use intervention type, slot mismatch target, gold labels, diagnostic family labels, file path heuristics, or row id heuristics.

The analyzer is shadow-only and diagnostic-only. It did not mutate source predictions, final logits, final predictions, training, checkpoint selection, Stage128 guard behavior, Stage15 behavior, or threshold/model-selection behavior.

## Interpretation

Stage146-B confirms that the v2 negative-control failure fixed in Stage145-B remains compatible with broader external/generalization behavior in the currently available Stage146-A exports. The controlled diagnostic subset is clean, the external/factver subset has a small false-NE tradeoff, and the aggregate metrics still pass.

Because `feature_correct_support_fp = 0`, the observed v2 changes did not incorrectly flag correct SUPPORT predictions in this run. However, v2 remains a strong shadow diagnostic rather than a deployable final override.

## Remaining limitations

- Only two external/factver files were available in the selected Stage146-A run.
- The external/factver subset still had `delta_false_ne = +3`.
- Route/order mismatch remains outside the set-based location-disjoint policy.
- Alias and organization rules are deterministic and incomplete.
- The analyzer remains shadow-only and should not be used as a final prediction override.

Final integration remains blocked.

## Stage147 recommendation

Stage147 should either expand external coverage further or create a separate route/order diagnostic.

If more external prediction exports are available, run another external expansion. Otherwise, start a route/order mismatch diagnostic separately. Stage147 should avoid automatic final prediction override, training loss integration, checkpoint selection using v2 outputs, and any claim of final deployment readiness.
