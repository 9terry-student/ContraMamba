# Stage140-B Deployable Location Signal Synthesis

## 1. Summary decision

Decision: `STAGE140B_DEPLOYABLE_TEXT_LOCATION_SIGNAL_CANDIDATE_SAFE_ON_DIAGNOSTIC`.

Stage140-B synthesizes Stage140-A as a report-only result. The deployable text-location signal is promising on the diagnostic, but it is still not enough for final-logit integration or final-prediction integration.

No training code, model code, export behavior, final logits, final predictions, checkpoint selection, or threshold-based model selection is modified by this report.

## 2. Why Stage140-A matters

Stage139-B showed that controlled `intervention_type=location_swap` provided a safe upper bound, but that signal was not deployable because it depended on controlled metadata.

Stage140-A tested deployable location-mismatch policies using only claim text, evidence text, and exported scalar fields where applicable. It explicitly did not use `intervention_type`, `slot_mismatch_target`, gold labels, or diagnostic family metadata as policy inputs.

The Stage140-A base diagnostic had 6,200 rows, accuracy `0.997741935483871`, macro F1 `0.9963904828740083`, 14 false SUPPORTs, and zero false-NE errors. Its decision was `STAGE140A_DEPLOYABLE_TEXT_LOCATION_SIGNAL_CANDIDATE_SAFE`.

## 3. Best deployable candidate policy

The best policy was `text_loc_disjoint`.

`text_loc_disjoint` uses claim/evidence text only. It checks SUPPORT predictions for disjoint extracted location spans between the claim and evidence, then shadows those predictions to NOT_ENTITLED for diagnostic analysis.

The policy did not use `intervention_type`, `slot_mismatch_target`, gold labels, or diagnostic family metadata.

On this diagnostic, `text_loc_disjoint` changed 14 rows. It removed all 14 false SUPPORTs with zero false-NE cost on this diagnostic:

| Metric | Before | After |
| --- | ---: | ---: |
| False SUPPORT | 14 | 0 |
| False NE | 0 | 0 |
| Macro F1 | 0.9963904828740083 | 1.0 |
| SUPPORT recall | 1.0 | 1.0 |
| SUPPORT precision | 0.9815303430079155 | 1.0 |

Assessment: `candidate_safe_on_this_diagnostic`.

## 4. Feature quality

Among SUPPORT predictions, `loc_disjoint` flagged 14 rows. All 14 were false SUPPORT true positives, with zero correct SUPPORT false positives and zero false SUPPORT false negatives. Precision and recall for false SUPPORT detection among SUPPORT predictions were both `1.0`.

`loc_any_diff` matched the same useful profile on this diagnostic: 14 flagged rows, 14 false SUPPORT true positives, zero correct SUPPORT false positives, and zero false SUPPORT false negatives.

Broader nonempty-location features were too broad. `loc_both_nonempty` flagged all 758 SUPPORT predictions, catching the same 14 false SUPPORTs but also flagging 744 correct SUPPORT rows, for precision `0.0184697` despite recall `1.0`.

## 5. Audit observation

After-the-fact audit showed that the 14 changed rows were all `location_swap` cases.

That audit group field was not used by the policy. The deployable policy used claim and evidence text only.

Representative pattern: claim location Dublin versus evidence location Cork, with the original prediction SUPPORT, gold NOT_ENTITLED, and shadow prediction NOT_ENTITLED.

## 6. Safety and leakage policy

Stage140-B is report-only. It does not modify final logits, final predictions, training, model code, export behavior, checkpoint selection, or thresholds used for model selection.

The Stage140-A result remains diagnostic-only and shadow-only. It is not model-consistent final guard evidence, and it should not be treated as external generalization evidence.

There is no policy leakage from controlled metadata: `text_loc_disjoint` did not use `intervention_type`, `slot_mismatch_target`, gold labels, or diagnostic family metadata.

## 7. Stage141 recommendation

Stage141 should stress-test `text_loc_disjoint` and related deployable location heuristics on broader existing prediction exports and negative controls.

Recommended checks:

- Apply `text_loc_disjoint` to existing non-collapsed prediction JSONLs beyond Stage122 core localization.
- Measure false positive risk on correct SUPPORT rows with same-location paraphrases.
- Inspect multi-location and person/title false extraction cases.
- Keep all analysis shadow-only before any final-prediction integration.

Avoid:

- Routing `text_loc_disjoint` into final logits now.
- Treating Stage140-A as external generalization evidence.
- Using `intervention_type` or `slot_mismatch_target` as deployable signals.
- Repeating blind short training reruns.
