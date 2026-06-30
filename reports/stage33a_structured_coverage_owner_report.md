# Stage33-A Structured Coverage Owner v0 Report

## Purpose
Stage33-A introduces a deterministic structured directional Coverage Owner v0 in shadow mode. It is motivated by Stage32-D2: Coverage Owner v2 threshold sweeps stopped blind overclaim dominance only by collapsing coverage into unresolved/NOT_ENTITLED, with no safe SUPPORT recovery.

This implementation is diagnostic-only. It does not change final logits, final predictions, H1 final decision behavior, losses, caps, entitlement, training data selection, or checkpoint selection.

## Stage32-D2 Lock
Stage32-D2 concluded:

- decision: `STAGE32_D2_NO_SAFE_SUPPORT_RECOVERY`
- Stage31 external v2 route counts: `RESIDUAL = 200`
- support-entailment SUPPORT recovery: `0`
- v2 unresolved counts: support=80, overclaim=80, refute=40
- threshold tuning moved failure between overclaim collapse and unresolved collapse

The neural coverage head cannot serve as the Coverage Owner by itself.

## CLI Flags
| Flag | Default | Purpose |
|---|---:|---|
| `--stage33-use-structured-coverage-owner` | disabled | Compute deterministic structured coverage owner fields. |
| `--stage33-structured-coverage-owner-export` | disabled | Export structured coverage fields in prediction rows. |
| `--stage33-structured-coverage-owner-shadow-mode` | disabled | Let structured coverage route control only exported shadow labels. |
| `--stage33-structured-coverage-preserve-can-support` | disabled | In shadow mode only, allow entailment-preserve route to recover SUPPORT without SUPPORT polarity proxy. |

## Structured Labels and Routes
| Label | Route |
|---|---|
| `STRUCT_ENTAILMENT_PRESERVE` | `ENTAILMENT_PRESERVE` |
| `STRUCT_OVERCLAIM_NE` | `OVERCLAIM_NE` |
| `STRUCT_CONTRADICTION_REFUTE` | `CONTRADICTION_REFUTE` |
| `STRUCT_UNRESOLVED` | `RESIDUAL` |

## Implemented Rules
| Rule | Label | Route | Reason | Confidence |
|---|---|---|---|---:|
| evidence all/every/each/any, claim some/at least one | `STRUCT_ENTAILMENT_PRESERVE` | `ENTAILMENT_PRESERVE` | `quantifier_all_to_some` | 1.00 |
| evidence some/at least one, claim all/every/each/any | `STRUCT_OVERCLAIM_NE` | `OVERCLAIM_NE` | `quantifier_some_to_all` | 1.00 |
| evidence negative/no/none/never, claim existential | `STRUCT_CONTRADICTION_REFUTE` | `CONTRADICTION_REFUTE` | `none_to_some` | 1.00 |
| evidence existential, claim negative/no/none/never | `STRUCT_CONTRADICTION_REFUTE` | `CONTRADICTION_REFUTE` | `some_to_none` | 1.00 |
| evidence exclusive/only, claim drops exclusivity | `STRUCT_ENTAILMENT_PRESERVE` | `ENTAILMENT_PRESERVE` | `only_to_base` | 1.00 |
| evidence additive/also, claim exclusive/only | `STRUCT_OVERCLAIM_NE` | `OVERCLAIM_NE` | `also_to_only` | 1.00 |
| claim token set conservatively contained in richer evidence | `STRUCT_ENTAILMENT_PRESERVE` | `ENTAILMENT_PRESERVE` | `specific_to_general_proxy` | 0.75 |
| evidence token set conservatively contained in richer claim | `STRUCT_OVERCLAIM_NE` | `OVERCLAIM_NE` | `general_to_specific_proxy` | 0.75 |
| no rule fires | `STRUCT_UNRESOLVED` | `RESIDUAL` | `no_structured_rule_fired` | 0.00 |

## Export Fields
When Stage32 owner-state export and Stage33 structured owner export are enabled:

- `stage33_structured_coverage_enabled`
- `stage33_structured_coverage_label`
- `stage33_structured_coverage_route`
- `stage33_structured_coverage_reason`
- `stage33_structured_coverage_confidence`
- `stage33_structured_coverage_claim_cues`
- `stage33_structured_coverage_evidence_cues`
- `stage33_structured_coverage_rule_fired`
- `stage33_structured_coverage_priority_trace`

## Evaluator Updates
`scripts/evaluate_stage32_shadow_composer.py` reports:

- `stage33_structured_coverage_label_counts`
- `stage33_structured_coverage_route_counts`
- `stage33_structured_coverage_reason_counts`
- `stage33_structured_coverage_confidence_summary`

Stage31-specific diagnostics include support/overclaim/refute structured-route counts and structured shadow safety counters.

## Shadow-Only Guarantee
Structured coverage can affect only exported `stage32_shadow_label`, `stage32_shadow_reason`, and shadow priority traces when Stage33 shadow mode is enabled. It never changes the model's final logits or final predictions.

## Known Limitations
- Heuristic rules are brittle.
- No external parser or learned symbolic operator is used.
- Lexical containment is only a conservative proxy for specificity/generalization.
- Stage31 probe remains diagnostic-only and must not be used for training, calibration, threshold selection, or checkpoint selection.
- This is not a final architecture replacement.

## Next Step
Run leakage-safe shadow diagnostics to determine whether structured routes recover SUPPORT without overclaim/refute-to-SUPPORT safety errors.
