# Gen5 Temporal Mechanism Recovery Correction Authority

SOURCE_EXECUTION_HEAD=d9b790b62f8db9f875cfa13d7ba155ac75137a2b
SOURCE_TEMPORAL_MECHANISM_AUTHORITY_COMMIT=adad44c6fb304500c700b9ea59d727cefd71c2d9
SOURCE_RECOVERY_ARCHIVE_SHA256=3af11ca1f0fe97182199fc80cb8549031d365f5ddd2828b15d1bc344bf42e784
SOURCE_WORKER0_SHA256=3f49147ff5d8a1bff7b6e5d3d607e9342ea1f328f00abba41d3d300f92eff3e9
SOURCE_WORKER1_SHA256=f8ac8f0dcb20e1da1b1e93696137f9a329ff469fd3d32c2f77e8f61a12910e91
SOURCE_T20_DIAGNOSTIC_SHA256=e64ca120d7ae1ea55ee3145b600d0cad4c48566fb36ec80c76f5396cce300afc

STATUS=READY_FOR_RECOVERY_CORRECTION_IMPLEMENTATION_AND_PARTIAL_EVIDENCE_FREEZE

CURRENT_USER_AUTHORITY=EXPLICIT_SUPERSEDING_CORRECTION_AUTHORIZATION
GPU_REEXECUTION_ALLOWED=NO_FOR_THIS_RECOVERY
TRAINING_ALLOWED=NO
OPTIMIZER_CONSTRUCTION_ALLOWED=NO
OPTIMIZER_STEP_ALLOWED=NO
PARAMETER_UPDATE_ALLOWED=NO
BACKWARD_ALLOWED=NO
NEW_SEEDS_ALLOWED=NO
DATA_SPLIT_LABEL_CHANGE_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
HISTORICAL_FROZEN_ARTIFACT_OVERWRITE_ALLOWED=NO
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## Purpose

This correction preserves the completed eight-hour temporal-mechanism worker
computation, corrects two implementation defects discovered only in the parent
post-worker merge/authentication path, and freezes the recoverable evidence
without rerunning the scientific workload.

Historical frozen artifacts are not rewritten. This document supersedes only
the defective temporal-mechanism runtime expectations/accumulation semantics
described below.

## Defect 1: t=1 A-axis aggregate hardcoded to 72

`temporal_mechanism_prediction_factor_disagreement()` sums all nine unordered
same-training-RNG / different-A pairs in the frozen 3x3 factorial grid.

The frozen t=1 behavioral predictions and the recovered worker predictions
both produce exactly:

- A-axis disagreement sum: `144`
- R-axis disagreement sum: `0`
- frozen prediction mismatch count: `0`

The parent runtime nevertheless required A-axis sum `72`.

Correction:

- freeze the runtime expectation to `144`;
- retain R-axis expectation `0`;
- add a regression test that derives `144/0` directly from the frozen
  behavioral cell-metrics artifact.

This is an aggregate-definition implementation defect, not a scientific
difference.

## Defect 2: task-visible stage-local statistics did not apply valid-token masking

The no-grad geometry path explicitly indexes only
`attention_mask.bool()` valid tokens.

The task-visible stage-local accumulator, however, flattened the entire padded
sequence tensor for stage residual statistics, the local two-margin Jacobian
row-space projection, and signed-permutation controls.

Correction:

- require `attention_mask` in the temporal task-visible stage accumulator;
- zero padded coordinates before residual statistics, Jacobian row-space
  projection, and signed-permutation controls;
- construct visible/complement intervention residuals only in the valid-token
  coordinate subspace;
- keep the real source/target forward values and existing replay
  authentication;
- do not change rows, seeds, times, stages, controls, thresholds, or model
  semantics.

The recovered defective local-projector values are not retroactively corrected.

## Recovery evidence boundary

Valid recovered evidence includes:

- all t=0..20 behavioral logits/predictions/margins;
- exact frozen behavioral authentication;
- t=1 four-state micro-decomposition;
- B-update-only versus full-t1 comparison;
- permanent prediction reconvergence t=17..20;
- critical-time valid-token geometry;
- finite-intervention/replay quantities that authenticated at t=20.

Not promoted to a fully authenticated claim:

- t=1/t17 earliest task-visible-stage gate based on the defective local
  projector-energy accumulator.

No GPU rerun is authorized or required for this recovery freeze.

## Authorized repository delta

Implementation correction:

- `scripts/audit_reason_router_gen5_ainit_temporal_birth.py`
- `tests/test_reason_router_gen5_ainit_temporal_birth.py`

Recovery documentation:

- this correction authority;
- the partial-recovery validated evidence report.

Recovered executed artifacts may be imported only under:

`reports/reason_router_gen5_ainit_temporal_mechanism_recovery_runs/gen5-ainit-temporal-mechanism-d9b790b-r1-partial-recovery/`

Do not overwrite the original frozen endpoint or temporal artifacts.
