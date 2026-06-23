# Stage19-A Combined Temporal Bias + Frame-Supervised Diagnostic Notes

## Purpose

Stage19-A combines two partial diagnostic fixes:

1. frame-only slot supervision, expected to reduce location and role frame false-entitlement;
2. calibrated temporal bias, expected to reduce temporal mismatch false-entitlement.

The goal is to test whether these fixes are compatible when the temporal bias is applied to frame-supervised prediction files.

## Method

`scripts/apply_stage19_combined_temporal_frame_patch.py` loads Stage16-style prediction JSONs, infers the seed, selects an alpha from the Stage18 multiseed summary when available, and applies the Stage18 pseudo-logit temporal bias:

```text
REFUTE       -= alpha
NOT_ENTITLED += alpha
SUPPORT      -= alpha
```

The bias is applied only when the Stage17 temporal comparator flags a temporal mismatch.

## Analysis

`scripts/analyze_stage19_combined_temporal_frame_patch.py` reports false-entitlement before and after the temporal patch by seed and Stage15 probe group, then summarizes mean/std adjusted false-entitlement for:

- `temporal_mismatch`
- `frame_location_mismatch`
- `frame_role_mismatch`
- `predicate_mismatch`
- `temporal_erased`
- `surface_control`
- `sufficiency_control`

## Interpretation

If temporal mismatch falls near zero while frame-location and frame-role remain low, temporal and frame fixes are compatible. If predicate mismatch remains high, the next diagnostic target is a predicate guard.

## Caveat

Stage19-A is post-processing and analysis over existing predictions. It is not an end-to-end trained ContraMamba model and should not be described as an architecture result.

