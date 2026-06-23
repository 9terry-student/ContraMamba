# Stage18-B1 Temporal Bias Calibration Notes

## Purpose

Stage18-B1 tests whether the finite temporal penalty identified in Stage18-A can be selected from a calibration split instead of manually chosen. This remains prediction-level calibration, not an end-to-end trained ContraMamba model.

## Method

The script reuses the Stage17 temporal comparator. For each example, it converts prediction probabilities to logits when true logits are unavailable:

```text
pseudo_logit = log(max(probability, eps))
```

For flagged temporal mismatches, a scalar alpha is applied:

```text
REFUTE       -= alpha
NOT_ENTITLED += alpha
SUPPORT      -= alpha
```

The adjusted probabilities are computed with softmax, and the adjusted prediction is the argmax.

## Split

Calibration and heldout splits are created by `stage15_source_id` when available. This keeps paired `temporal_mismatch` and `temporal_erased` variants from the same source in the same split.

## Objectives

Supported calibration objectives:

- `cross_entropy`
- `accuracy`
- `false_entitlement_guarded_accuracy`

The guarded objective chooses the smallest alpha that reduces calibration temporal false-entitlement by at least 90% while preserving `temporal_erased`, `surface_control`, and `sufficiency_control` predictions. If no alpha satisfies this, it falls back to best calibration accuracy and records that no valid guarded alpha was found.

## Caveat

Stage18-B1 should be described as a learned post-processing bias over existing predictions. It is not a trained temporal reasoning module and should not be presented as an architecture result.

