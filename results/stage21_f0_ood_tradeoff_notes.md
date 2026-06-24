# Stage21-F0 OOD Tradeoff Summary

## Source
Stage21-E3 best-dev checkpoints (3 seeds, mean across seeds).

## Key Findings

### v6B strengths — targeted guard signal
- **temporal_mismatch false-entitled rate**: v5=0.230 → v6B=0.000
  (reduced to zero across all 3 seeds)
- **predicate_mismatch false-entitled rate**: v5=0.203 → v6B=0.000
  (reduced to zero across all 3 seeds)
- **Overall OOD accuracy**: v5=0.559 → v6B=0.636 (+0.077)
- **Overall OOD macro-F1**: v5=0.279 → v6B=0.324 (+0.045)

### v6B failures — SUPPORT control over-rejection
- **surface_control false-not-entitled rate**: v5=0.797 → v6B=0.697
  (improvement of 0.100 but still severe)
- **temporal_erased false-not-entitled rate**: v5=0.830 → v6B=0.787
  (improvement of 0.043 but still severe)

### v6B regressions — frame mismatch
- **frame_location_mismatch false-entitled rate**: v5=0.250 → v6B=0.333 (-0.083, v6B worsens)
- **frame_role_mismatch false-entitled rate**: v5=0.200 → v6B=0.350 (-0.150, v6B worsens)

## Conclusion
v6B is a targeted temporal/predicate guard, not a complete selective OOD solution.
The temporal and predicate comparators eliminate false entitlement on their target probe
types but the model still severely over-rejects SUPPORT controls (surface_control,
temporal_erased) and regresses on frame mismatch detection.

## Next Stage
Stage21-F1: comparator ablation (current / no-flags / temporal-only / predicate-only)
with full Stage15 OOD evaluation to isolate which comparator drives each effect.
