# Gen5 Stage E Learned-B-Plane Positive Control Execution Authority

## Authority identities

DESIGN_IMPLEMENTATION_AUTHORITY_COMMIT=c9ef6c55448a3458b46f5b76b5c88244cc7b726e

IMPLEMENTATION_FREEZE_COMMIT=b8fa20dc3fb058412610375c255d8c25e93e14bf

STAGE_E_EVIDENCE_FREEZE_COMMIT=382a4961ee701ddf38b39ef9fa75aa932ce34f39

SCIENTIFIC_EXECUTION_ALLOWED=YES_STAGE_E_BFREE_THREE_CELL_POSITIVE_CONTROL

CUDA_PREFLIGHT_ALLOWED=YES

TRAINING_ALLOWED=YES_EXACT_STAGE_E_BFREE_THREE_CELL

EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV

BACKWARD_ALLOWED=YES_TRAINING_ONLY

OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS

CONFIRMATORY_9601_9900_ALLOWED=NO

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP

## Scientific purpose

This execution is a diagnostic positive control for the completed Stage E
R22/C22 fixed-plane result.

It asks only:

Can the frozen Stage E QMA parameterization relearn useful correction when Q is
fixed to the seed-matched output plane of the already successful unrestricted
Phase3A P0 solution?

The execution must not be interpreted as an independent confirmatory treatment,
because the BFREE plane was derived from the same task objective.

## Exact implementation

Execution must use the implementation frozen at:

`b8fa20dc3fb058412610375c255d8c25e93e14bf`

Exact frozen implementation paths:

1. `src/contramamba/gen5_stage_e_learned_b_plane_positive_control.py`
2. `scripts/train_reason_router_gen5_stage_e_learned_b_plane_positive_control.py`
3. `tests/test_reason_router_gen5_stage_e_learned_b_plane_positive_control.py`

No modification of those files is authorized during execution.

## Exact three-cell matrix

Single arm:

`E-BFREE`

Exactly:

- seed6201 / E-BFREE
- seed6202 / E-BFREE
- seed6203 / E-BFREE

Pressure:

`P0`

Exactly three scientific training cells.

## Frozen learned-plane sources

Only these seed-matched Phase3A P0 final corrections may define Q.

### seed6201

Source SHA256:

`157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf`

Frozen Q SHA256:

`1cfc7e1b55b68b0b71404f75c4920788331c9fcccb14ed25b40a316973721705`

### seed6202

Source SHA256:

`1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214`

Frozen Q SHA256:

`44b81288f73f605bc12fbab90a51cdb87f421f5cbd6fec7b648dd6621d77ff57`

### seed6203

Source SHA256:

`c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770`

Frozen Q SHA256:

`28dd4b583b8019688ca656cc063924e75bd4b2fee9c35fa6680b5570d0953330`

The implementation-authenticated deterministic float64 thin-QR and
positive-diagonal sign convention is authoritative.

Because exact CPU QR bytes were observed to depend on multithreaded
linear-algebra execution state, every static verification, CUDA preflight,
single-cell runtime, and matrix runtime for this execution must inherit:

`OMP_NUM_THREADS=1`

`MKL_NUM_THREADS=1`

`OPENBLAS_NUM_THREADS=1`

`NUMEXPR_NUM_THREADS=1`

Under that exact single-thread CPU linear-algebra contract, three fresh
Kaggle processes produced byte-identical seed-specific Q hashes:

- seed6201: `1cfc7e1b55b68b0b71404f75c4920788331c9fcccb14ed25b40a316973721705`
- seed6202: `44b81288f73f605bc12fbab90a51cdb87f421f5cbd6fec7b648dd6621d77ff57`
- seed6203: `28dd4b583b8019688ca656cc063924e75bd4b2fee9c35fa6680b5570d0953330`

The earlier local-development Q hashes and multithreaded Kaggle Q hashes are
not execution identities and must not be used.

No alternative basis derivation is allowed.

## Parameterization

For each seed-specific frozen Q:

`B_eff = Q M`

with:

- Q frozen `[24576,2]`;
- M trainable `[2,2]`;
- A trainable `[2,768]`;
- correction operator `Q M A`;
- no bias.

Initialization:

- M exactly zero;
- A exactly seed-matched to the Stage E initialization convention.

Exactly two trainable tensors:

- `A_theta.weight`
- `M_theta.weight`

Expected trainable parameter count:

`1540`

## Parent model

Frozen parent checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Parent parameters must remain frozen.

Parent fingerprint must be identical before and after every cell.

## Data contract

Use the same frozen Phase3A population as Stage E.

Training:

- 3360 rows
- 480 source pairs

Dev evaluation:

- 840 rows
- 120 source pairs

Split seed:

`16384`

Pressure:

`P0`

Confirmatory population `9601..9900` is forbidden.

## Optimization contract

Exactly:

- AdamW
- learning rate `0.001`
- weight decay `0.0001`
- 20 optimizer steps per cell
- gradient clip norm `5.0`
- no scheduler
- final three-way cross entropy only
- no early stopping
- no checkpoint selection
- final fixed step only

Total scientific optimizer steps:

`60`

## Frozen unrestricted reference

Do not retrain unrestricted P0.

Use:

- seed6201 free gain: `0.4984860420227051`
- seed6202 free gain: `0.5020102858543396`
- seed6203 free gain: `0.4994615912437439`

## Frozen Stage E comparison

Already-frozen Stage E recoveries:

### seed6201
- R22: `0.001975318561968087`
- C22: `0.0017156096081790623`

### seed6202
- R22: `0.002009419003162425`
- C22: `0.0020609486561624537`

### seed6203
- R22: `0.0024332976314431222`
- C22: `0.0018053421563742791`

Frozen Stage E means:

- R22: `0.00213934506552454`
- C22: `0.00186063347357193`

These values are references only and must not be recomputed by rerunning
Stage E.

## Primary quantity

For each seed:

`gain_BFREE = CE_ZERO - CE_BFREE`

`recovery_BFREE = gain_BFREE / gain_FREE_P0_seed`

Report seedwise values and descriptive mean.

No scientific p-values.

No post-hoc success threshold.

## CUDA preflight

One bounded CUDA preflight is authorized before the three-cell matrix.

Use one seed-matched E-BFREE cell.

Purpose:

- authenticate runtime and kernels;
- authenticate seed-specific BFREE plane loading;
- instantiate parent and positive-control wrapper;
- execute forward and backward plumbing;
- validate finite A/M gradients;
- validate no parent gradients;
- validate fixed-plane geometry.

Preflight restrictions:

- optimizer constructed: false;
- optimizer step count: 0;
- training executed: false;
- task evaluation executed: false;
- confirmatory data loaded: false.

Preflight is runtime validation only and is not scientific evidence.

## GPU topology

Full matrix must use:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

Required:

- exactly two Tesla T4 devices;
- independent single-GPU workers;
- fresh parent model per cell;
- no DDP;
- no shared model;
- no shared optimizer.

## Required artifact fields

Each cell must preserve:

- seed;
- arm;
- exact source correction SHA;
- exact Q SHA;
- train/dev identities;
- training losses;
- step-0 loss;
- post-step20 matched-RNG loss;
- gradient norms;
- optimizer-step count;
- final dev CE;
- final dev accuracy;
- gain;
- seed-matched unrestricted gain;
- recovery;
- fixed-plane geometry;
- A hash;
- M hash;
- parent fingerprint before/after;
- execution/runtime provenance.

`scientific_conclusion` must remain null in execution artifacts.

## Prohibited execution

Not authorized:

- R22 rerun;
- C22 rerun;
- unrestricted free-B rerun;
- random-plane controls;
- new learned-plane search;
- QR convention changes;
- seed expansion;
- pressure expansion;
- layer sweep;
- rank sweep;
- learning-rate sweep;
- optimizer sweep;
- step-count sweep;
- token search;
- new dataset;
- early stopping;
- checkpoint selection;
- confirmatory evaluation;
- scientific p-values;
- implementation edits during execution.

If execution exposes an implementation defect, stop.

Do not patch the frozen implementation in place.

## Interpretation boundary

Successful execution alone is not a scientific conclusion.

Keep separate:

1. runtime/code correctness;
2. execution success;
3. artifact/provenance validity;
4. scientific interpretation.

Scientific interpretation is allowed only after successful collect/import and
artifact validation.

## Prospective interpretation

If BFREE recovery is qualitatively far larger than the already-frozen
approximately 0.2 percent R22/C22 recovery across the seed-matched cells, the
fixed-plane QMA parameterization is demonstrated capable of useful relearning
inside a known-successful output plane.

This would strengthen the bounded interpretation that Stage E failure is
plane-orientation-specific rather than generic fixed-plane incapacity.

If BFREE recovery is also negligible, Stage E must not be used to establish
an orientation-specific failure.

If results are materially heterogeneous across seeds, preserve that
heterogeneity rather than averaging it away.
