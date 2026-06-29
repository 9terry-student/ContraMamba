# Stage31-C3 Coverage/Entailment Diagnostic Head Report

## Purpose
Stage31-C3 adds a representation-access ablation for the Coverage/Entailment diagnostic head. The head remains diagnostic-only and is not allowed to modify final predictions, final logits, entitlement, caps, NE boosting, H1 composer logic, calibration, threshold selection, or checkpoint selection.

## Stage31-C2 Failure Summary
The corrected Stage31-C2 3-class hard-contrast auxiliary data is valid: 900 rows, 300 rows per class, train=720, dev=180, and no `OTHER_RESIDUAL`.

The Stage31-C2 diagnostic evaluation did not pass the safety bar:

| Metric | Value |
|---|---:|
| total_accuracy | 0.4500 |
| macro_f1 | 0.3535 |
| coverage_direction_alignment_accuracy | 0.3650 |
| support_entailment_recovered_by_head | 42 / 80 |
| overclaim_detected_by_head | 14 / 80 |
| refute_detected_by_head | 17 / 40 |
| overclaim_misread_as_entails_support | 32 |
| refute_misread_as_entails_support | 17 |

Interpretation: the current input `frame_pair_repr + predicate_pair_repr + sufficiency_repr` does not expose stable enough directional coverage/scope information for composer integration.

## Stage31-C3 Input Modes
The optional CLI flag `--v7-coverage-entailment-input-mode` controls the diagnostic head input:

| Mode | Representation |
|---|---|
| `current` | Existing C/C2 input: `frame_pair_repr`, `predicate_pair_repr`, `sufficiency_repr` |
| `raw_pair` | Existing `claim_frame_state`, `evidence_frame_state`, `abs(claim_frame_state - evidence_frame_state)`, `claim_frame_state * evidence_frame_state` |
| `hybrid` | `raw_pair` concatenated with `current` |

`raw_pair` uses claim/evidence sequence-level states already produced by the model's existing `FrameGate` forward path. It does not run the encoder twice, add a second backbone, or add external models.

## Head Architecture
The optional head is registered from `scripts/train_controlled_v6b_minimal.py` when `--v7-use-coverage-entailment-head` is enabled.

`--v7-coverage-entailment-num-classes` controls the output dimension. The default remains 3 for:

| Class | ID | Intended final label |
|---|---:|---|
| ENTAILS_SUPPORT | 0 | SUPPORT |
| OVERCLAIM_NOT_ENTITLED | 1 | NOT_ENTITLED |
| CONTRADICTS_REFUTE | 2 | REFUTE |

Legacy 4-class compatibility remains available only when explicitly requested.

## Exported Columns
Default 3-class exports:
- `coverage_entailment_input_mode`
- `coverage_entails_support_prob`
- `coverage_overclaim_ne_prob`
- `coverage_contradicts_refute_prob`
- `coverage_entailment_pred_id`
- `coverage_entailment_pred_label`
- `coverage_entailment_confidence`

Legacy 4-class mode additionally exports `coverage_other_residual_prob`.

## Evaluation Reporting
The Stage31 evaluator reports `coverage_entailment_input_mode` when present, plus:
- `coverage_direction_alignment_accuracy`
- `coverage_direction_alignment_by_group`
- `coverage_direction_confusion`
- `support_entailment_recovered_by_head`
- `overclaim_detected_by_head`
- `refute_detected_by_head`
- `refute_misread_as_entails_support`
- `overclaim_misread_as_entails_support`
- `support_misread_as_overclaim_ne`
- `support_misread_as_contradicts_refute`

## Decision Rule
Composer integration is allowed only if all criteria pass:

| Criterion | Required |
|---|---:|
| coverage_direction_alignment_accuracy | >= 0.65 |
| support_entailment_recovered_by_head | >= 60 |
| overclaim_detected_by_head | >= 60 |
| refute_detected_by_head | >= 25 |
| refute_misread_as_entails_support | <= 5 |
| overclaim_misread_as_entails_support | <= 10 |

Until those criteria pass, Stage31-D composer integration remains denied. Recommended next steps are better representation access or explicit symbolic/scope features.

## Leakage Policy
Do not use `data/stage31_coverage_entailment_probe.jsonl` for training, calibration, threshold selection, checkpoint selection, auxiliary loss construction, or model selection.
