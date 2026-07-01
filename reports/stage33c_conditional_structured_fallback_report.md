# Stage33-C Conditional Structured Fallback Report

## Purpose

Stage33-C keeps structured Coverage Owner routing in shadow mode, but changes the shadow composer from a global replacement to a conditional fallback composer. High-precision structured rules can locally override the current final prediction; unresolved, residual, weak proxy, and blocked cases fall back to the current final prediction.

## Stage33-B Summary

Stage33-B confirmed that high-precision structured rules are useful on the Stage31 external diagnostic, but unsafe as a global replacement on dev. The dev failure mode was not high-precision contradiction or overclaim routing; it was collapse from routing residual and blocked proxy cases to `NOT_ENTITLED`.

## Conditional Fallback Policy

When `--stage33-structured-coverage-conditional-fallback` is enabled together with structured owner shadow mode:

- hard-core block cases fall back to the current final prediction by default
- high-precision `CONTRADICTION_REFUTE` routes shadow to `REFUTE`
- high-precision `OVERCLAIM_NE` routes shadow to `NOT_ENTITLED`
- allowed high-precision direct-support routes shadow to `SUPPORT`
- entailment-preserve plus positive polarity can shadow to `SUPPORT`
- all other structured cases fall back to the current final prediction

The fallback source flag accepts `current_final` and `h1_current`. In this implementation, `h1_current` is documented as equivalent to the already-computed current final prediction.

## New Flags

- `--stage33-structured-coverage-conditional-fallback`
- `--stage33-structured-coverage-fallback-source`

Stage33-B flags remain active:

- `--stage33-structured-coverage-direct-support-rules`
- `--stage33-structured-coverage-weak-rules-to-residual`
- `--stage33-structured-coverage-preserve-can-support`

## Exported Fields

- `stage33_conditional_fallback_enabled`
- `stage33_conditional_fallback_source`
- `stage33_conditional_action`
- `stage33_conditional_fallback_used`
- `stage33_conditional_override_applied`
- `stage33_conditional_override_type`
- `stage33_conditional_original_current_label`
- `stage33_conditional_shadow_label`

`stage32_shadow_label` remains the main evaluator shadow label and mirrors the conditional result when Stage33-C is enabled.

## Evaluator Diagnostics

The Stage32 shadow composer evaluator now reports:

- `stage33_conditional_action_counts`
- `stage33_conditional_fallback_used_count`
- `stage33_conditional_fallback_used_rate`
- `stage33_conditional_override_applied_count`
- `stage33_conditional_override_applied_rate`
- `stage33_conditional_override_type_counts`
- `stage33_conditional_original_current_label_counts`
- `stage33_conditional_shadow_label_counts`

Stage31-specific counters include:

- `stage33_conditional_support_recovered`
- `stage33_conditional_support_still_ne`
- `stage33_conditional_overclaim_to_support`
- `stage33_conditional_refute_to_support`
- `stage33_conditional_refute_recovered`

## Shadow-Only Guarantee

Stage33-C does not modify final logits, final predictions, H1 final decision logic, checkpoint selection, loss, calibration, threshold selection, training data, caps, or boosts.

## Remaining Risks

- Whole/part entailment remains unresolved until a targeted rule extension.
- Positive-polarity proxy recovery remains diagnostic and should be checked against dev safety.
- Conditional fallback safety depends on current final predictions being a reasonable fallback baseline.

## Next Step

If conditional fallback is promising in Kaggle validation, Stage33-D should add a focused whole/part extension.
