
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

