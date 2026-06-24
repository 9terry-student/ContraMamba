# Stage21-F1 v6B OOD Comparator Ablation Notes

## Purpose
Stage21-F1 is a comparator-mechanism ablation to verify that v6B's Stage21-E3 OOD
improvements on `temporal_mismatch` and `predicate_mismatch` are caused by the
corresponding comparator flags rather than generic NOT_ENTITLED bias.

## Ablation modes
- **`current`**: both temporal and predicate OOD flags active (replicates Stage21-E3).
- **`no_flags`**: both temporal and predicate flags zeroed.
- **`temporal_only`**: temporal flags active, predicate flags zeroed.
- **`predicate_only`**: predicate flags active, temporal flags zeroed.

All four modes evaluate the same best-dev checkpoint with the same OOD probe data.
Only the flag tensors passed to the v6B comparator change.

## Main findings

### Flag-mechanism attribution
- **`current`** drives `temporal_mismatch` and `predicate_mismatch` false-entitled rates
  to zero (replicating Stage21-E3 v6B results).
- **`no_flags`** restores false-entitled errors on both `temporal_mismatch` and
  `predicate_mismatch`, confirming that the gains require active flags.
- **`predicate_only`** selectively fixes `predicate_mismatch` but not `temporal_mismatch`.
- **`temporal_only`** selectively fixes `temporal_mismatch` but not `predicate_mismatch`.

### SUPPORT preservation (surface_control, temporal_erased)
- `false_not_entitled_rate` on `surface_control` and `temporal_erased` remains high
  and is essentially unchanged across all four ablation modes.
- Over-rejection of SUPPORT examples is caused by the base sufficiency/entitlement
  decision boundary, not by the comparator flags.

## Interpretation
v6B's temporal and predicate OOD gains are mechanistically attributable to the
corresponding comparator flags: each flag independently and selectively guards its
target probe type, and removing the flags reverts the gains.

The SUPPORT preservation problem is orthogonal to the comparator mechanism and requires
a different intervention (e.g., recalibrating the entitlement boundary or adding
SUPPORT-preserving training signal).

## Conclusion
- v6B is a **verified targeted temporal/predicate guard**: the mechanism is confirmed.
- v6B is **not a complete selective OOD solution**: SUPPORT control preservation
  (surface_control, temporal_erased) remains weak regardless of flag configuration.

## Next stage
Stage21-G: explore boundary recalibration or explicit SUPPORT preservation signal to
reduce false-not-entitled rate on surface_control and temporal_erased while keeping
temporal/predicate guard gains.
