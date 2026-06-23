# Stage19-B Frame-Model Temporal Bias Notes

## Purpose

Stage19-B recalibrates temporal alpha directly on Stage16 frame-only slot-supervised prediction files. This tests whether the Stage19-A seed2 failure was caused by alpha transfer mismatch rather than a temporal comparator failure.

## Method

For each frame-only prediction file, the script:

1. infers the seed from the filename;
2. merges predictions with the Stage15 slot-sensitivity probe;
3. computes Stage17 temporal mismatch flags;
4. splits by `stage15_source_id` when available;
5. grid-searches temporal alpha on the calibration split;
6. applies the selected alpha to all examples;
7. writes per-seed calibrated prediction JSON plus combined summary and alpha-grid CSVs.

The temporal bias is:

```text
REFUTE       -= alpha
NOT_ENTITLED += alpha
SUPPORT      -= alpha
```

and is applied only when the temporal comparator flags a mismatch.

## Compatibility criterion

Temporal + frame fixes are considered compatible when:

- temporal_mismatch adjusted false-entitled mean <= 5/100;
- frame_role adjusted false-entitled mean <= 5/20;
- frame_location adjusted false-entitled mean <= 10/20;
- temporal_erased, surface_control, and sufficiency_control changed_count == 0 in all seeds.

Predicate remains unresolved when predicate_mismatch adjusted false-entitled mean >= 30/100.

## Caveat

Stage19-B is target-model prediction-level calibration. It is not an end-to-end trained model and should not be presented as a ContraMamba architecture change.

