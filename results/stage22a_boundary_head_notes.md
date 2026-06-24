
## Stage22-A3 mini diagnostic result

Stage22-A3 added a separate `frame_violation_head` alongside the existing preservation/boundary head and ran a 3-seed mini sweep over six configs.

### Config sweep

| Config | frame_violation_head weight | boundary_head weight |
|---|---|---|
| no_heads | — | — |
| frame_head_w0 | 0.0 (probe only) | — |
| frame_w0p05 | 0.05 | — |
| frame_w0p2 | 0.2 | — |
| both_b0p02_f0p05 | 0.05 | 0.02 |
| both_b0p02_f0p2 | 0.2 | 0.02 |

### DEV target-check (in-domain frame intervention signal)

| Config | train/dev frame-target gap |
|---|---|
| frame_head_w0 | ~0.004 (flat — no supervision) |
| frame_w0p05 | ~0.221 |
| frame_w0p2 | ~0.329 |
| both_b0p02_f0p05 | ~0.376 |
| both_b0p02_f0p2 | ~0.345 |

The head learns the in-domain frame intervention target under supervision.

### OOD compact result (Stage15 frame mismatch probe)

| Config | frame_location frame_violation_prob | frame_role frame_violation_prob | surface frame_violation_prob | temporal_erased frame_violation_prob |
|---|---|---|---|---|
| frame_w0p05 | 0.394 | 0.395 | 0.421 | 0.457 |
| frame_w0p2 | 0.252 | 0.251 | 0.297 | 0.368 |

OOD ranking is inverted: surface controls and temporal_erased receive `frame_violation_prob` as high as or higher than frame_location/frame_role mismatch groups.

| Config | frame_location FE | frame_role FE | surface FNE | temporal_erased FNE |
|---|---|---|---|---|
| no_heads | 0.617 | 0.483 | 0.487 | 0.817 |
| frame_w0p05 | 0.517 | 0.417 | 0.527 | 0.810 |
| frame_w0p2 | 0.767 | 0.583 | 0.293 | 0.773 |
| both_b0p02_f0p05 | 0.717 | 0.683 | — | — |
| both_b0p02_f0p2 | 0.633 | 0.617 | — | — |

Both-head configs add preservation pressure but worsen frame false-entitled rates relative to `frame_w0p05`.

Boundary scores for both-head configs (surface / temporal_erased):

| Config | boundary surface | boundary temporal_erased |
|---|---|---|
| both_b0p02_f0p05 | 0.661 | 0.578 |
| both_b0p02_f0p2 | 0.561 | 0.436 |

### Interpretation

1. Stage22-A3 is valid as a diagnostic extension — the head can be trained and probed without breaking the main classifier.
2. The `frame_violation_head` successfully learns in-domain frame intervention targets (DEV gap rises with weight).
3. It fails to generalize to Stage15 OOD frame mismatch: surface/temporal controls score as high or higher than true frame mismatches.
4. The OOD ranking failure reflects a train/dev intervention target space that does not align with the Stage15 OOD frame mismatch structure — this is a supervision design problem, not a missing auxiliary head.
5. Both-head configs do not compound benefits; they raise frame false-entitled rates, so preservation + frame heads do not yet form a safe recovery gate.

### Rejected next step

Stage22-B positive recovery gate (gate SUPPORT predictions on high preservation + low frame_violation probability) is rejected. The `frame_violation_prob` signal is unsafe for OOD gating because it ranks surface controls and temporal_erased as likely frame violations.

### Recommended next step

Design-level intervention before any logit gate: align frame violation supervision with the OOD frame mismatch structure. Options include (a) explicit frame-location / frame-role contrastive supervision using Stage15-style OOD negatives at train time, or (b) a separate contrastive head trained on frame-role vs. surface-control pairs to force the representation to discriminate frame mismatch from surface change. Do not add a logit gate until the `frame_violation_prob` OOD ranking is correct.

---

## Stage22-A2 mini diagnostic result

Stage22-A2 tested whether the single preservation boundary head can provide a safe recovery signal.

Mini 3-seed sweep compared:
- no_boundary
- head_only_w0
- head_w0p02
- head_w0p05
- head_w0p2

Main finding:
The single boundary head is not a safe SUPPORT recovery signal.

Key observations:
- The no-boundary baseline already has high frame OOD false-entitled rates:
  - frame_location_mismatch FE = 0.6167
  - frame_role_mismatch FE = 0.4833
- Therefore the frame OOD failure is not caused by boundary-loss interference alone.
- With head_only_w0, boundary_prob is nearly flat across OOD groups, so the existing slot representation does not expose a clean preservation-vs-frame boundary without supervision.
- Increasing boundary loss raises boundary_prob for both surface controls and frame mismatches.
- head_w0p02 improves frame_location/frame_role FE somewhat, but worsens SUPPORT preservation:
  - surface_control FNE increases from 0.4867 to 0.5800
  - temporal_erased FNE increases from 0.8167 to 0.8267
- head_w0p2 slightly improves surface_control/temporal_erased FNE, but frame_location/frame_role remain unsafe:
  - frame_location_mismatch FE = 0.5833
  - frame_role_mismatch FE = 0.4833

Conclusion:
Stage22-B positive recovery gate is rejected for the single-head boundary design. The boundary probability does not separate preservation from frame violation safely. The next design should split preservation and frame violation into separate diagnostic heads, so that SUPPORT recovery can require high preservation evidence and low frame_violation evidence.

