# Stage31-C2 Coverage/Entailment Diagnostic Head Report

## Purpose
Stage31-C2 keeps the Coverage/Entailment owner diagnostic-only. The optional head is a readout signal for three hard minimal-contrast directions: entailment-preserving SUPPORT, overclaim NOT_ENTITLED, and contradiction REFUTE.

## Auxiliary Data
Stage31-C2 uses `data/stage31c_coverage_entailment_aux.jsonl`, separate from the Stage31 evaluation probe. It provides balanced hard minimal-contrast supervision:

| Class | ID | Intended final label |
|---|---:|---|
| ENTAILS_SUPPORT | 0 | SUPPORT |
| OVERCLAIM_NOT_ENTITLED | 1 | NOT_ENTITLED |
| CONTRADICTS_REFUTE | 2 | REFUTE |

Default aux counts are 900 rows total, 300 rows per class, with train=720 and dev=180. `OTHER_RESIDUAL` is not part of the default Stage31-C2 data.

## Head Architecture
The optional head is registered from `scripts/train_controlled_v6b_minimal.py` when `--v7-use-coverage-entailment-head` is enabled.

Input:
`concat([frame_pair_repr, predicate_pair_repr, sufficiency_repr])`

Output:
`--v7-coverage-entailment-num-classes` controls the output dimension. The default is 3. Legacy 4-class compatibility remains available only when explicitly requested.

Loss:
optional cross entropy over `coverage_direction_id`, enabled only with `--v7-use-coverage-entailment-loss` and `--v7-coverage-entailment-data`.

## Exported Columns
Default 3-class exports:
- `coverage_entails_support_prob`
- `coverage_overclaim_ne_prob`
- `coverage_contradicts_refute_prob`
- `coverage_entailment_pred_id`
- `coverage_entailment_pred_label`
- `coverage_entailment_confidence`

Legacy 4-class mode additionally exports `coverage_other_residual_prob`.

## Diagnostic-Only Constraints
Stage31-C2 does not modify `output["logits"]`, H1 final composer, entitlement, caps, NE boosting, calibration, threshold selection, or checkpoint selection. The Stage31 evaluation probe must not be used for this loss.

## Evaluation Alignment
For Stage31-C2 diagnostic exports, Stage31 probe alignment is interpreted as:
- SUPPORT entailment groups -> `ENTAILS_SUPPORT`
- NOT_ENTITLED overclaim groups -> `OVERCLAIM_NOT_ENTITLED`
- REFUTE groups -> `CONTRADICTS_REFUTE`

The evaluator supports both default 3-class exports and legacy 4-class exports, and reports safety counters for refute/overclaim/support direction misreads.

## Leakage Policy
Do not use `data/stage31_coverage_entailment_probe.jsonl` for training, calibration, threshold selection, checkpoint selection, or auxiliary loss construction.
