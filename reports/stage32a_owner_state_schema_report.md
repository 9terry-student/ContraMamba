# Stage32-A Owner-State Schema Report

## Purpose
Stage32-A introduces the owner-hierarchy output schema in shadow mode. Owner states are diagnostic proxies computed from existing model outputs. They do not replace the current v7/H1 final decision path.

## CLI Flags
| Flag | Default | Purpose |
|---|---:|---|
| `--stage32-use-owner-state-schema` | disabled | Enables the Stage32-A owner-state schema contract. |
| `--stage32-owner-state-export` | disabled | Adds flattened Stage32 owner-state fields to prediction exports. |
| `--stage32-owner-state-shadow-mode` | disabled | Enables export-time assertions that Stage32 owner-state construction leaves final logits and predictions unchanged. |

## Owner Fields and Proxy Sources
| Owner | Fields | Proxy Source |
|---|---|---|
| `hard_core` | `prob`, `pass`, `block_reason`, `source_fields` | `frame_prob`; pass is `frame_prob >= 0.5`; block reason is `low_frame_proxy` when false. |
| `coverage_entailment` | `entails_support_prob`, `overclaim_ne_prob`, `contradicts_refute_prob`, `pred_label`, `pred_id`, `confidence`, `input_mode`, `source_fields` | Stage31 coverage-entailment diagnostic head outputs when available; otherwise `UNAVAILABLE` with null probabilities. |
| `residual_adjudication` | `residual_prob`, `ambiguous_prob`, `underspecified_prob`, `pred_label`, `source_fields` | No stable residual owner exists yet; exported as nulls with `UNIMPLEMENTED_PROXY`. |
| `ani_diagnostic` | `novelty_prob`, `ambiguity_prob`, `ignorance_prob`, `pred_label`, `source_fields` | No ANI owner exists yet; exported as nulls with `UNIMPLEMENTED_PROXY`. |
| `polarity` | `support_energy`, `refute_energy`, `support_prob`, `refute_prob`, `pred_label`, `source_fields` | Existing `positive_energy` and `negative_energy`; support/refute probabilities are a two-way softmax over refute/support energies. |
| `composer_shadow` | `would_block_support`, `would_route_ne`, `would_route_refute`, `shadow_label`, `shadow_reason`, `note` | Deterministic diagnostic-only rules over owner proxies. |

## Shadow Composer Rules
1. If `hard_core.pass` is false, shadow label is `NOT_ENTITLED` with reason `hard_core_block`.
2. If coverage predicts `OVERCLAIM_NOT_ENTITLED`, shadow label is `NOT_ENTITLED` with reason `coverage_overclaim`.
3. If coverage predicts `CONTRADICTS_REFUTE`, shadow label is `REFUTE` with reason `coverage_contradiction`.
4. If coverage predicts `ENTAILS_SUPPORT` and polarity support is stronger, shadow label is `SUPPORT` with reason `coverage_entails_support_with_positive_polarity`.
5. Otherwise, shadow label is `NOT_ENTITLED` with reason `residual_or_unresolved`.

## Flattened Export Fields
Prediction exports include the following fields only when `--stage32-owner-state-export` is enabled:

| Prefix | Fields |
|---|---|
| `stage32_hard_core_*` | `prob`, `pass`, `block_reason` |
| `stage32_coverage_*` | `entails_support_prob`, `overclaim_ne_prob`, `contradicts_refute_prob`, `pred_label`, `pred_id`, `confidence`, `input_mode` |
| `stage32_residual_*` | `prob`, `pred_label` |
| `stage32_ani_*` | `novelty_prob`, `ambiguity_prob`, `ignorance_prob`, `pred_label` |
| `stage32_polarity_*` | `support_energy`, `refute_energy`, `support_prob`, `refute_prob`, `pred_label` |
| `stage32_shadow_*` | `label`, `reason`, `would_block_support`, `would_route_ne`, `would_route_refute` |

Nested owner-state objects are intentionally not exported to avoid breaking downstream prediction consumers.

## Shadow-Mode Guarantees
Stage32-A owner states do not modify:

- final logits
- final predictions
- H1 composer behavior
- entitlement logic
- caps
- losses
- training data selection
- checkpoint selection

When `--stage32-owner-state-shadow-mode` and `--stage32-owner-state-export` are both enabled, prediction export snapshots final logits and predictions and raises if owner-state construction mutates them.

## Known Limitations
- Hard Core is only a `frame_prob` proxy.
- Coverage/Entailment is only available if the Stage31 diagnostic head is enabled.
- Residual Adjudication and ANI are schema placeholders, not learned owners.
- Polarity uses existing energy proxies and is not a new polarity owner.
- The shadow label is diagnostic only and must not be used for training, calibration, threshold selection, checkpoint selection, or final prediction.

## Next Step
Stage32-B should introduce explicit Hard Core and Coverage/Entailment owner interfaces while preserving shadow-mode diagnostics until a rule-structured composer is validated.
