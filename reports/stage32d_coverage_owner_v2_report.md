# Stage32-D Coverage Owner v2 Report

## Purpose
Stage32-D introduces a shadow-only Coverage Owner v2 that can abstain to `UNRESOLVED_COVERAGE` instead of forcing low-confidence Coverage/Entailment predictions into overclaim routing.

This is diagnostic architecture plumbing. It does not change final logits, final predictions, H1 final decision behavior, losses, caps, entitlement, training data selection, or checkpoint selection.

## Stage32-C Unsafe Conclusion
Stage32-C showed that the Stage32-B shadow composer was unsafe:

| Split | Current Acc | Current Macro-F1 | Shadow Acc | Shadow Macro-F1 | Decision | Key Failure |
|---|---:|---:|---:|---:|---|---|
| Stage31 external | 0.3600 | 0.1818 | 0.3500 | 0.2064 | STAGE32_SHADOW_UNSAFE | zero SUPPORT recovery; over-routing through coverage overclaim |
| Controlled dev | 0.5611 | 0.3617 | 0.7111 | 0.2771 | STAGE32_SHADOW_UNSAFE | shadow prediction distribution collapsed to NOT_ENTITLED/REFUTE with SUPPORT=0 |

The failure indicates that the proxy Coverage/Entailment owner over-routes examples to `OVERCLAIM_NOT_ENTITLED`. It does not justify returning to a flat arbiter; it justifies a more cautious Coverage Owner.

## Coverage Owner v2 Logic
Coverage Owner v2 uses the existing Stage31 coverage-head probabilities as proxy inputs:

- `stage32_coverage_entails_support_prob`
- `stage32_coverage_overclaim_ne_prob`
- `stage32_coverage_contradicts_refute_prob`
- `stage32_coverage_confidence`
- `stage32_coverage_pred_label`

It computes:

- `top_prob`
- `second_prob`
- `margin = top_prob - second_prob`

When `--stage32-coverage-owner-v2-allow-abstain` is enabled:

| Condition | v2 Pred Label | v2 Route | v2 Reason |
|---|---|---|---|
| `top_prob < min_confidence` | `UNRESOLVED_COVERAGE` | `RESIDUAL` | `low_confidence_abstain` |
| `margin < min_margin` | `UNRESOLVED_COVERAGE` | `RESIDUAL` | `low_margin_abstain` |
| otherwise | original coverage label | mapped route | `confident_coverage_prediction` |

When abstain is disabled, v2 exports fields but follows v1 routing behavior.

## CLI Flags
| Flag | Default | Purpose |
|---|---:|---|
| `--stage32-coverage-owner-v2` | disabled | Enables v2 shadow routing and export fields. |
| `--stage32-coverage-owner-v2-min-confidence` | 0.50 | Minimum top probability before v2 accepts a route. |
| `--stage32-coverage-owner-v2-min-margin` | 0.05 | Minimum top-minus-second margin before v2 accepts a route. |
| `--stage32-coverage-owner-v2-allow-abstain` | disabled | Allows low-confidence/margin cases to route to residual unresolved. |

## Export Fields
When Stage32 owner-state export is enabled, Stage32-D adds:

- `stage32_coverage_v2_enabled`
- `stage32_coverage_v2_top_prob`
- `stage32_coverage_v2_second_prob`
- `stage32_coverage_v2_margin`
- `stage32_coverage_v2_min_confidence`
- `stage32_coverage_v2_min_margin`
- `stage32_coverage_v2_pred_label`
- `stage32_coverage_v2_route`
- `stage32_coverage_v2_reason`
- `stage32_coverage_v2_abstained`

## Shadow Composer v2 Routing
When v2 is enabled, the shadow composer uses `stage32_coverage_v2_route`:

| v2 Route | Shadow Label | Shadow Reason |
|---|---|---|
| `OVERCLAIM_NE` | `NOT_ENTITLED` | `coverage_v2_overclaim` |
| `CONTRADICTION_REFUTE` | `REFUTE` | `coverage_v2_contradiction` |
| `ENTAILMENT_PRESERVE` with SUPPORT polarity | `SUPPORT` | `coverage_v2_entails_support_with_positive_polarity` |
| `ENTAILMENT_PRESERVE` without SUPPORT polarity | `NOT_ENTITLED` | `coverage_v2_entailment_without_positive_polarity` |
| `RESIDUAL` | `NOT_ENTITLED` | `coverage_v2_unresolved_to_residual` |

Hard Core failure still takes priority and routes to `NOT_ENTITLED` with `hard_core_block`.

## Evaluator Updates
`scripts/evaluate_stage32_shadow_composer.py` now reports:

- `coverage_v2_pred_label_counts`
- `coverage_v2_route_counts`
- `coverage_v2_reason_counts`
- `coverage_v2_abstain_count`
- `coverage_v2_abstain_rate`

Stage31-specific v2 diagnostics include:

- `support_entailment_v2_entailment_preserve`
- `support_entailment_v2_unresolved`
- `overclaim_v2_overclaim_ne`
- `overclaim_v2_unresolved`
- `refute_v2_contradiction_refute`
- `refute_v2_unresolved`

## Shadow-Only Guarantee
Coverage Owner v2 affects only Stage32 owner-state export and shadow composer fields. It must not be used for training, calibration, threshold selection, checkpoint selection, final logits, or final predictions.

## Remaining Risks
- V2 may reduce blind overclaim routing without recovering SUPPORT.
- Thresholds may need a diagnostic sweep.
- The Stage31 probe remains diagnostic and must not be used for training or selection.
- The owner is still proxy-backed by the Stage31 coverage head.

## Next Step
Evaluate Coverage Owner v2 with confidence/margin sweeps before considering any Stage32 composer application.
