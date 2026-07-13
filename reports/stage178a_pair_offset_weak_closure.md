# Stage178-A pair-offset weak-path closure

## Decision

`STAGE178A_PAIR_OFFSET_EXPLANATION_WEAK_PATH_CLOSED`

The canonical baseline retained strong same-pair frame ranking (`0.939352`). Cross-pair ranking improved from `0.931105` on raw logits to `0.945920` under leave-one-row-out pair centering, a delta of `0.014815`, but none of the 60 dev pairs was offset-misaligned at the native zero threshold.

## Centering attribution

On the fixed Stage176-A hard-39 cohort, leave-one-row-out centering corrected one error, introduced none, and produced a net change of `+1`. Across the full dev split it changed 18 wrong rows to correct and 23 correct rows to wrong, for a net change of `-5`. Pair centering therefore has collateral damage and is not an authorized inference mechanism.

## Association and stability

Surface/construction association with pair centers was observed, but it did not satisfy the offset-explanation gate. Pair-center structure was stable across the baseline and Stage177-C pilot. The pilot primarily changed compatible–incompatible gap rather than center offset and did not establish pair-offset removal.

These findings do not support pair offset as the primary explanation of the hard-frame absolute-comparability failure. They are observational and do not establish a causal nuisance mechanism.

## Closed paths

Pair-centering deployment, pair-offset loss, score-normalization or global-threshold sweeps, calibration, a Stage177-C weight sweep, multi-seed expansion, and external evaluation remain closed.

## Authorized next stage

Only `STAGE179_FRAME_LABEL_SEMANTICS_AND_INPUT_REPRESENTATION_AUDIT` is authorized: an evaluation-only audit of native frame-label semantics, the exact frame-head input representation, and exact readout decomposition. Training, relabeling, architecture changes, fitted probes, and inference-time centering remain unauthorized.
