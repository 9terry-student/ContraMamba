# Stage18-B2 Temporal Bias Multiseed Notes

## Purpose

Stage18-B2 tests whether the Stage18-B1 temporal-bias calibration behavior is stable across multiple prediction seeds. It is a prediction-level calibration robustness analysis, not model training and not an architecture change.

## Workflow

`scripts/run_stage18_temporal_bias_multiseed.py` finds prediction files, infers seed ids from filenames, runs the same alpha-selection logic as Stage18-B1, and writes one calibrated prediction JSON per seed.

It also writes:

- `results/stage18_temporal_bias_multiseed_summary.csv`
- `results/stage18_temporal_bias_multiseed_alpha_grid.csv`

`scripts/analyze_stage18_temporal_bias_multiseed.py` aggregates the calibrated JSONs and writes:

- `results/stage18_temporal_bias_multiseed_group_metrics.csv`
- `results/stage18_temporal_bias_multiseed_examples.csv`
- `results/stage18_temporal_bias_multiseed_summary.md`

## Interpretation

The key quantities are:

- selected alpha per seed;
- calibration and heldout temporal false-entitled counts before and after calibration;
- whether `temporal_erased`, `surface_control`, and `sufficiency_control` predictions remain unchanged;
- aggregate mean/std selected alpha and heldout adjusted temporal false-entitled counts.

If selected alpha remains in a narrow range and heldout temporal false-entitlement drops sharply across seeds while controls are preserved, this supports the claim that a finite temporal decision bias is robust across prediction seeds.

## Caveat

Stage18-B2 remains prediction-level calibration. It should not be described as an end-to-end temporal reasoning module or a trained ContraMamba architecture.

