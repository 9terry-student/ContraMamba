# Stage138-B Cross-Run Slot Guard Synthesis

## Summary decision

Decision: `STAGE138B_CROSSRUN_SLOT_GUARD_GLOBAL_THRESHOLD_TRADEOFF_TOO_HIGH`

Stage138-B concludes that `slot_mismatch_prob` is not suitable as a global SUPPORT-to-NOT_ENTITLED guard in the current form. The cross-run diagnostic merge shows that removing false SUPPORT with a global threshold requires an unacceptable increase in false NOT_ENTITLED predictions and a large macro-F1 drop.

## Stage138-A setup and diagnostic-only status

Stage138-A performed a cross-run diagnostic merge:

- Historical Stage123-B `core_only` predictions supplied the non-collapsed predictions.
- Stage137-B fresh slot-prob export supplied `slot_mismatch_prob`.
- The merged evidence was diagnostic-only.

This is not final model-consistent evidence. The predictions and slot probabilities came from separate runs, so the result must not be treated as a production guard evaluation or as evidence that final model behavior would match the merged diagnostic.

Stage138-A baseline:

- `n_total = 6200`
- `n_with_slot_prob = 6200`
- `support_pred_before = 758`
- `false_support_before = 14`
- `false_ne_before = 0`
- `accuracy_before = 0.997741935483871`
- `macro_f1_before = 0.9963904828740083`

## Global threshold result

The analyzer-selected threshold was `0.62`, but its tradeoff is too high:

- false SUPPORT decreases from `14` to `0`
- false NOT_ENTITLED increases from `0` to `503`
- SUPPORT recall drops to `0.3239247311827957`
- macro-F1 drops from `0.9963904828740083` to `0.8126764261018201`
- delta macro-F1 is approximately `-0.184`

Thresholds `0.30`, `0.40`, `0.50`, `0.55`, and `0.60` remove all false SUPPORT, but they also destroy all SUPPORT predictions. They change `758` SUPPORT predictions, leave `support_pred_after = 0`, create `744` false NOT_ENTITLED predictions, and reduce macro-F1 to about `0.641975`.

Threshold `0.64` creates harm without reducing false SUPPORT. It changes `139` SUPPORT predictions, creates `139` false NOT_ENTITLED predictions, and leaves false SUPPORT unchanged at `14`.

Thresholds `0.66+` do nothing, with `n_changed_total = 0`.

## Slot signal quality

The global ranking quality of `slot_mismatch_prob` is weak for this guard use:

- `slot_target_auc = 0.5331299512580994`
- `slot_target_auprc = 0.6766425559354456`
- `slot_target_positive_count = 2773`
- `slot_target_negative_count = 1550`

This supports the conclusion that the current slot signal is not reliable enough as a global SUPPORT-to-NOT_ENTITLED threshold.

## Group-level observation

All `14` false SUPPORT examples are in `intervention_type=location_swap`. For `location_swap`, thresholds at or below `0.62` remove those false SUPPORT cases without increasing false NOT_ENTITLED within that group.

However, the same global thresholds also convert many correct SUPPORT examples in `none`, `paraphrase`, and `polarity_flip`. This means a global threshold is unsafe, while a conditional location/slot-specific guard may be worth a separate export-only diagnostic analysis.

## Safety and leakage policy

Stage138-B is a static synthesis report. No code behavior changed.

Safety status:

- No final logits were modified.
- No final predictions were modified.
- Stage128 guard was not enabled.
- Stage15 was not used.
- No threshold was used for model selection.
- No external data was used for training.

## Stage139 recommendation

Stage139 should run export-only conditional guard diagnostics, focusing on location/slot-conditioned gating rather than global `slot_mismatch_prob` thresholding.

Stage139 should avoid:

- routing `slot_mismatch_prob` globally into final logits
- claiming final guard success from the cross-run merge
- using threshold `0.62` as a production rule
- repeating blind short fresh reruns
