# Stage145-C Text Location v2 Synthesis

## 1. Summary decision

Decision: `STAGE145C_TEXT_LOC_DISJOINT_V2_NEGATIVE_CONTROLS_PASS_BROADER_RETAINED`

Stage145-C synthesizes the Stage145-B evidence for `text_loc_disjoint_v2`. The refined v2 policy is a strong shadow candidate: it fixes the Stage144 alias and organization-like false-positive failure while retaining the broader 7-file false-support reduction. It is not ready for final integration.

## 2. Why v2 was needed

Stage142's raw `text_loc_disjoint` policy caught explicit location mismatch among SUPPORT predictions, but Stage144-A showed that the raw span-set comparison was too brittle for negative controls.

The main false-positive modes were alias and surface-form matches such as New York City vs New York, NYC vs New York City, Los Angeles vs LA, and San Francisco vs SF. It also over-triggered on organization-like spans such as Beacon Harbor Labs vs Beacon Harbor Laboratory, Oxford Research Center vs Oxford Research Centre, and Falcon Ridge Institute vs Falcon Ridge Initiative.

Stage145-A introduced `text_loc_disjoint_v2` with deterministic alias/canonicalization rules, conservative organization-like span blocking, known lowercase location support, and shadow-only behavior.

## 3. Stage144 negative-control result

Stage145-B ran v2 on the Stage144-A negative controls and produced:

| metric | value |
| --- | --- |
| decision | `STAGE145_TEXT_LOCATION_GUARD_V2_SHADOW_CANDIDATE_PASS` |
| n_rows | 97 |
| n_changed_total | 16 |
| delta_false_support | -16 |
| delta_false_ne | 0 |
| feature_false_support_tp | 16 |
| feature_correct_support_fp | 0 |
| feature_false_support_fn | 6 |
| feature_precision | 1.0 |
| feature_recall | 0.7272727272727273 |
| macro_f1_before | 0.3775067750677507 |
| macro_f1_after | 0.609674892283588 |
| delta_macro_f1 | 0.23216811721583724 |

v2 fixes the Stage144 alias/organization false-positive failure. Alias-surface SUPPORT rows had `n_changed_total = 0` and `delta_false_ne = 0`. Organization-like SUPPORT rows were blocked by the organization span guard, also with `n_changed_total = 0` and `delta_false_ne = 0`.

The policy still catches explicit mismatches: `simple_location_mismatch_ne` moved false SUPPORT from 12 to 0, and `lowercase_location_mismatch_ne` moved false SUPPORT from 4 to 0. Route reversal remains missed because the policy is set-based and sees overlapping locations.

## 4. Broader 7-file retention result

Stage145-B also ran v2 on the broader Stage141/142 7-file stress set:

| metric | value |
| --- | --- |
| decision | `STAGE145_TEXT_LOCATION_GUARD_V2_SHADOW_CANDIDATE_MILD_TRADEOFF` |
| n_files | 7 |
| n_rows | 33000 |
| n_changed_total | 53 |
| delta_false_support | -53 |
| delta_false_ne | 3 |
| feature_false_support_tp | 53 |
| feature_correct_support_fp | 0 |
| feature_false_support_fn | 176 |
| macro_f1_before | 0.7075607576618963 |
| macro_f1_after | 0.7092915891444586 |
| delta_macro_f1 | 0.0017308314825622562 |
| min_per_file_delta_macro_f1 | 0.0 |

v2 preserves the broader 7-file false-support reduction of -53 with `feature_correct_support_fp = 0`. Stage123-B diagnostic variants retain the false-support reduction with `delta_false_ne = 0`; Stage53 VitaminC has no effect; Stage63 VitaminC has `delta_false_support = -4`, a small false-NE tradeoff of +3, and slightly positive macro-F1.

## 5. Comparison against raw Stage144 baseline

The raw Stage144-A baseline had `delta_false_support = -12`, but it also introduced `delta_false_ne = +14` on negative controls. Its feature precision was only 0.46153846153846156, with 14 correct SUPPORT rows incorrectly changed.

v2 improves negative-control performance from false NE +14 to false NE 0. It also improves feature precision to 1.0 on the negative-control run while increasing false-support reduction from -12 to -16. The tradeoff is that relation/order-aware cases, especially route reversal, remain outside the policy.

## 6. Policy input safety

The v2 policy uses only claim text, evidence text, the original prediction label, deterministic alias rules, and deterministic organization-like rules.

It does not use intervention type, slot mismatch targets, gold labels for policy decisions, diagnostic family metadata, file path heuristics, or row id heuristics.

Safety status:

| field | value |
| --- | --- |
| shadow_only | true |
| diagnostic_only | true |
| source_predictions_mutated | false |
| final_logits_modified | false |
| final_predictions_modified | false |
| training_modified | false |
| checkpoint_selection_modified | false |
| stage128_guard_enabled | false |
| stage15_used | false |
| external_data_used_for_training | false |
| threshold_used_for_model_selection | false |

## 7. Interpretation

`text_loc_disjoint_v2` is a strong shadow candidate. It preserves the useful explicit-location mismatch behavior from Stage142/Stage141 stress testing while removing the Stage144-A alias and organization-like false-positive pattern.

The evidence supports continued shadow analysis, not deployment. External evidence remains limited, Stage63 still has a small false-NE tradeoff of +3, and route reversal remains missed and should be handled separately.

## 8. Remaining limitations

- Route reversal remains missed because the policy is set-based rather than relation/order-aware.
- External validation remains limited.
- Stage63 still shows a small false-NE tradeoff of +3.
- Alias and organization rules are deterministic and incomplete.
- The analyzer remains shadow-only and should not be used as a final prediction override.

## 9. Stage146 recommendation

Recommended Stage146 goal: run v2 on additional external/general prediction exports or create a relation/order-aware diagnostic for route reversal.

Recommended actions:

- Keep Stage145 v2 shadow-only.
- Apply v2 to future external prediction exports.
- Optionally design a separate route/order mismatch diagnostic rather than overloading `text_loc_disjoint_v2`.
- Do not integrate v2 into final logits or final predictions yet.

Avoid automatic final prediction override, training loss integration, checkpoint selection using v2 outputs, and claiming final deployment readiness.
