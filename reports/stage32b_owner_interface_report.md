# Stage32-B Owner Interface Report

## Purpose
Stage32-B splits Stage32-A owner-state construction into explicit shadow-mode owner interfaces. This is architecture plumbing only: it does not replace the H1 final decision, does not change final logits or predictions, and does not affect loss or checkpoint selection.

## Stage32-A Validation Summary
| Check | Result |
|---|---|
| shadow off dev rows | 720 |
| shadow on dev rows | 720 |
| id overlap | 720 |
| off prediction distribution | NOT_ENTITLED 370, REFUTE 350 |
| on prediction distribution | NOT_ENTITLED 370, REFUTE 350 |
| prediction mismatches | 0 |
| stage32 export columns | present when shadow/export is enabled |
| final predictions changed by Stage32 shadow export | no |

## CLI
Stage32-B keeps the Stage32-A flags:

- `--stage32-use-owner-state-schema`
- `--stage32-owner-state-export`
- `--stage32-owner-state-shadow-mode`

Stage32-B adds:

- `--stage32-use-owner-interfaces`

All flags default to disabled. The implementation builds Stage32 owner states through explicit owner interface functions and remains shadow/export-only.

## Owner Interfaces
| Interface | Responsibility | Proxy Source Fields | Notes |
|---|---|---|---|
| `build_stage32_hard_core_owner_state` | Determine hard-core validity proxy | `frame_prob`; annotates `v7_location_boundary_prob`, `v7_temporal_prob`, `temporal_mismatch_fused_prob` when present | Uses `frame_prob >= 0.5` for the proxy pass flag. |
| `build_stage32_coverage_entailment_owner_state` | Represent directional Coverage/Entailment owner | Stage31 coverage head outputs: entails/overclaim/contradicts probabilities, pred id/label, confidence, input mode | Falls back to `UNAVAILABLE` when the Stage31 diagnostic head is absent. |
| `build_stage32_residual_adjudication_owner_state` | Reserve residual unresolved owner interface | none | Currently `UNIMPLEMENTED_PROXY`. |
| `build_stage32_ani_diagnostic_state` | Reserve ANI novelty/ambiguity/ignorance readout | none | Currently `UNIMPLEMENTED_PROXY`. |
| `build_stage32_polarity_owner_state` | Represent support/refute polarity proxy | `positive_energy`, `negative_energy` | Computes two-way support/refute probabilities; does not change polarity loss. |
| `build_stage32_shadow_composer_state` | Produce diagnostic-only shadow route | all owner states | Adds priority trace; never applied to logits or predictions. |

## Shadow Composer Priority Rules
1. Hard Core fail -> `NOT_ENTITLED`, reason `hard_core_block`.
2. Coverage predicts `OVERCLAIM_NOT_ENTITLED` -> `NOT_ENTITLED`, reason `coverage_overclaim`.
3. Coverage predicts `CONTRADICTS_REFUTE` -> `REFUTE`, reason `coverage_contradiction`.
4. Coverage predicts `ENTAILS_SUPPORT` and polarity predicts `SUPPORT` -> `SUPPORT`, reason `coverage_entails_support_with_positive_polarity`.
5. Otherwise -> `NOT_ENTITLED`, reason `residual_or_unresolved`.

`priority_trace` examples use strings such as `hard_core:pass`, `coverage:OVERCLAIM_NOT_ENTITLED`, and `route:NOT_ENTITLED`.

## Export Additions
Stage32-B preserves the Stage32-A flattened fields and adds:

- `stage32_shadow_priority_trace`
- `stage32_hard_core_notes`
- `stage32_coverage_notes`
- `stage32_residual_notes`
- `stage32_ani_notes`
- `stage32_polarity_notes`

Nested owner objects are still not exported to avoid breaking downstream consumers.

## Metadata Fix
Coverage-entailment loss metadata now reports the active class list according to `--v7-coverage-entailment-num-classes`:

- 3-class mode: `ENTAILS_SUPPORT`, `OVERCLAIM_NOT_ENTITLED`, `CONTRADICTS_REFUTE`
- 4-class mode: also includes `OTHER_RESIDUAL`

This is reporting-only and does not change training behavior.

## Guarantees
Stage32-B owner interfaces do not modify:

- final logits
- final predictions
- H1 final decision behavior
- entitlement
- caps
- training data selection
- checkpoint selection
- loss computation

When Stage32 shadow mode and export are enabled together, export-time assertions verify that final logits and predictions are unchanged after owner-state construction.

## Known Limitations
- Owner interfaces are still proxy-backed.
- Residual Adjudication and ANI are placeholders.
- Coverage/Entailment depends on the Stage31 diagnostic head when available.
- Shadow routing is diagnostic only and must not be used for training, calibration, checkpoint selection, or final prediction.

## Next Step
Stage32-C should evaluate a shadow rule-structured composer using these explicit owner interfaces, still without replacing the final decision path.
