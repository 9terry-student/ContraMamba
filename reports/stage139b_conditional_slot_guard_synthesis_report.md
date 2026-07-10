# Stage139-B Conditional Slot Guard Synthesis

## Summary decision

Decision: `STAGE139B_CONDITIONAL_LOCATION_GUARD_UPPER_BOUND_SAFE`.

Stage139-B records that the Stage139-A conditional location-aware diagnostic is safe as an upper-bound result. The best policy removes all 14 false SUPPORT cases without creating false NOT_ENTITLED cases, improving macro F1 from 0.9963904828740083 to 1.0.

This is an upper-bound diagnostic, not production evidence. It is cross-run, shadow-only, and depends on controlled metadata rather than deployable inference signals.

## Why global guard was rejected

Global `slot_mismatch_prob` thresholding remains rejected. Stage138-B showed that global thresholding could remove the 14 false SUPPORT examples only by causing large false-NE damage.

At threshold 0.62, false SUPPORT dropped from 14 to 0, but false NE increased from 0 to 503 and macro F1 fell from 0.9963904828740083 to 0.8126764261018201. Lower global thresholds were worse because they converted all 758 SUPPORT predictions to NOT_ENTITLED, while threshold 0.64 caused harm without reducing false SUPPORT.

## Stage139-A conditional upper-bound result

Stage139-A evaluated conditional and shadow-only policies on the Stage138-A cross-run merged diagnostic input. The input combined historical Stage123-B `core_only` predictions with Stage137-B `slot_mismatch_prob`, so it is diagnostic-only and not model-consistent final guard evidence.

The best observed policy was `oracle_location_swap_all`. It changed 14 predictions, all from SUPPORT to NOT_ENTITLED:

- false SUPPORT: 14 to 0
- false NE: 0 to 0
- support recall: 1.0 to 1.0
- support precision: 0.9815303430079155 to 1.0
- macro F1: 0.9963904828740083 to 1.0

The related `location_swap_and_slotprob_ge_t` policy was also safe for thresholds 0.30, 0.40, 0.50, 0.55, 0.60, and 0.62, producing the same 14 corrected SUPPORT-to-NOT_ENTITLED changes with no false-NE increase.

## Why the best policy is not deployable

The best policy is safe only because it uses controlled metadata `intervention_type=location_swap`.

That metadata identifies the diagnostic family directly and is not an inference-time signal. Therefore, `oracle_location_swap_all` is not deployable, even though it is useful as an upper-bound candidate. The result should not be routed into final logits, final predictions, checkpoint selection, or Stage128 behavior.

## Group-level interpretation

All 14 false SUPPORT examples were in `intervention_type=location_swap`. Within the location-swap group, the conditional guard changed exactly those 14 SUPPORT predictions to NOT_ENTITLED and created no false NE.

The representative pattern is a location mismatch such as claim location Dublin versus evidence location Cork. This suggests the remaining research question is not whether `slot_mismatch_prob` should be thresholded globally; it is whether a deployable location-mismatch condition can be inferred without `intervention_type`, `slot_mismatch_target`, gold labels, or diagnostic family metadata.

## Safety and leakage policy

Stage139-B is report-only. It does not modify training code, model code, export behavior, final logits, final predictions, checkpoint selection, Stage128 guard behavior, or evaluation behavior.

No Stage15 data is used. No external data is used for training. No threshold is used for model selection.

## Stage140 recommendation

Stage140 should design and evaluate deployable location-mismatch detection signals that approximate the Stage139 upper-bound condition.

Recommended directions:

- Use exported claim/evidence core text to derive location-mismatch features.
- Analyze whether existing frame/location scalars separate location-swap false SUPPORTs from correct SUPPORTs.
- Avoid `intervention_type`, `slot_mismatch_target`, gold labels, and diagnostic family metadata as inference signals.
- Keep evaluation shadow-only before any final-logit integration.

Stage140 should continue to avoid global `slot_mismatch_prob` guards and should not claim final model-consistent evidence from the Stage139-A cross-run merge.
