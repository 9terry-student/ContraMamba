# Stage179-A mixed/insufficient closure

## Decision

`STAGE179A_FRAME_SEMANTICS_REPRESENTATION_CAUSE_MIXED_OR_INSUFFICIENT`

Stage179-A exactly recovered the canonical seed-174, epoch-20 clean-dev frame-head contract. The classifier input is `output["frame_pair_repr"]` with shape `[720, 128]`; the scalar linear readout is `frame_gate.frame_classifier`. The maximum absolute error when reconstructing the native logit as `w dot h + b` was `7.15255737e-07`.

The audit found zero normalized exact claim/evidence duplicate groups and therefore zero conflicting frame-label duplicate groups. Near duplicates remain review candidates only and do not establish annotation conflict.

None of the prioritized automated localization gates passed:

- semantic cross-channel interference did not meet its gate;
- the gold-conditioned centroid diagnostic was not meaningfully superior to the native frame head (hard-39 AUROC `0.05142857142857143`, one corrected native-head error, no introduced error, net `+1`);
- representation insensitivity did not meet its gate; and
- readout alignment failure did not meet its gate.

The evidence is consequently insufficient for a single automated causal localization. Stage179-A closes automatic label-error, pair-offset, threshold/calibration, pairwise-objective, representation-repair, and readout-retraining paths. It does not authorize a new loss, readout retraining, architecture change, relabeling, or additional training.

The only authorized next stage is `STAGE180_HARD_FRAME_CASE_MANUAL_TAXONOMY_AND_DATA_DESIGN_AUDIT`: a blinded manual taxonomy and data-design audit of the fixed hard-39 cohort with matched correct controls.

## Safety boundary

The closure is reporting-only. It uses the clean controlled internal data and the already-produced Stage176–179 evidence. It performs no model forward, training, fitting, threshold search, calibration, checkpoint selection or modification, external evaluation, external labeling, time-swap analysis, dataset modification, or automatic adjudication.
