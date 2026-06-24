# Stage21-E3 Best-Dev OOD Summary

## Setup
- Train data: `data/controlled_v5_seed_no_time_swap.jsonl`
- OOD data: `data/stage15_slot_sensitivity_probe.jsonl`
- Backbone: Mamba
- Training: balanced sampler, 3 epochs, best-dev checkpoint used for OOD evaluation
- Seeds: 1, 2, 3
- v6B flags:
  - train flag source: `controlled_heuristic`
  - OOD flag source: `stage15_probe_type`
  - predicate comparator enabled
  - temporal comparator enabled but no time_swap training data

## Main finding
v6B improves targeted temporal/predicate mismatch rejection over the v5 balanced baseline.

Across three seeds:
- temporal_mismatch false-entitled rate:
  - v5: 0.230
  - v6B: 0.000
- predicate_mismatch false-entitled rate:
  - v5: 0.203
  - v6B: 0.000
- overall OOD accuracy:
  - v5: 0.559
  - v6B: 0.636
- overall OOD macro-F1:
  - v5: 0.279
  - v6B: 0.324

## Limitation
v6B is not yet a satisfactory selective OOD solution.

Preservation/control probes still suffer severe over-rejection:
- surface_control false-not-entitled rate:
  - v5: 0.797
  - v6B: 0.697
- temporal_erased false-not-entitled rate:
  - v5: 0.830
  - v6B: 0.787

v6B also worsens frame mismatch behavior:
- frame_location false-entitled rate:
  - v5: 0.250
  - v6B: 0.333
- frame_role false-entitled rate:
  - v5: 0.200
  - v6B: 0.350

## Interpretation
Stage21-E3 supports v6B as a targeted temporal/predicate guard signal, but not as a complete selective OOD solution. The next stage should preserve the temporal/predicate guard while reducing over-rejection on SUPPORT controls and avoiding frame mismatch regressions.
