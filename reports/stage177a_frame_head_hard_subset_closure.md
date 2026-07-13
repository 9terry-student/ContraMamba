# Stage177-A frame-head hard-subset closure

**Decision:** `STAGE177A_FRAME_PAIRWISE_SIGNAL_PRESENT_ABSOLUTE_DISCRIMINATION_WEAK`

Stage177-A is closed as a single-seed, internal clean-dev observational audit. The primary ranking and discrimination signal is the exact native `frame_logit` with identity normalization. Fixed-threshold confusion, Brier score, and ECE use the exact native `frame_prob`. No frame score was reconstructed from final classifier logits.

## Results

| View | Baseline | Treatment |
|---|---:|---:|
| Full-dev frame AUROC | 0.931242 | 0.932816 |
| Full-dev average precision | 0.941278 | 0.942292 |
| Full-dev balanced accuracy | 0.840278 | 0.826389 |
| Hard-39 frame AUROC | 0.414286 | 0.437143 |
| Same-pair compatible/incompatible ranking accuracy | 0.939352 | 0.944444 |

The frame head therefore has strong global clean-dev discrimination and approximately 94% same-pair ranking accuracy, but it fails to provide comparable absolute scores across pairs on the Stage176 hard-39 transition cohort. This is not a global frame-output collapse. The supported diagnosis is weak inter-pair absolute score comparability with preserved pair-relative distinction. Stage175 treatment does not destroy that relative ordering.

## Closure

A fixed threshold or post-hoc calibration is not supported as a separator of beneficial and harmful Stage176 transitions. The Stage174-C final-classifier SUPPORT-logit pairwise route remains closed and must not be reapplied. Hidden-state representation collapse, causal mechanism identification, a deployable threshold, broad frame-head uselessness, and statistically confirmed treatment superiority are not established.

Stage177-B may audit only the feasibility of a clean, frame-head-specific same-pair relative objective. It may not train, modify the trainer, search a weight or margin, tune a threshold, select a checkpoint, or use external or `time_swap` data. Stage177-C implementation and training remain unauthorized until the Stage177-B gate passes.

The machine-readable closure is in `reports/stage177a_frame_head_hard_subset_closure.json`.
