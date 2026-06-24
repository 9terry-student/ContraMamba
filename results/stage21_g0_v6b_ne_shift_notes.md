# Stage21-G0 v6B Unflagged NOT_ENTITLED Shift Sweep

## Setup
Stage21-G0 is an eval-only post-hoc calibration sweep for v6B.

The sweep subtracts a scalar shift from the final NOT_ENTITLED logit only for OOD records where both temporal and predicate OOD flags are zero. Flagged temporal/predicate mismatch records are left unchanged.

## Purpose
Stage21-F1 verified that temporal/predicate OOD gains are caused by the comparator flags. Stage21-G0 tests whether the remaining SUPPORT preservation failure can be rescued by relaxing the NOT_ENTITLED boundary on unflagged OOD records.

## Interpretation template
A successful shift should:
- reduce `surface_control` false-not-entitled rate;
- reduce `temporal_erased` false-not-entitled rate;
- preserve `temporal_mismatch` and `predicate_mismatch` false-entitled rate near zero;
- avoid large increases in `sufficiency_control`, `frame_location_mismatch`, and `frame_role_mismatch` false-entitled rates.

## Expected conclusion if seed1 pattern holds across seeds
A global unflagged NOT_ENTITLED shift is too blunt. It can restore SUPPORT controls, but it also pushes frame/sufficiency mismatch examples into entitled labels. This means the preservation failure is threshold-sensitive but requires a selective preservation gate rather than a global unflagged shift.
