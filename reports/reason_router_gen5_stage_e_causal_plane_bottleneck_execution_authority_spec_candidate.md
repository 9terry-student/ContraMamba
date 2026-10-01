# Gen5 Stage E — Causal-Plane Bottleneck Execution Authority

## Authority

DESIGN_FREEZE_COMMIT=c2f0c92da999d6508b712eb82f20f33cfcac4624

IMPLEMENTATION_AUTHORITY_COMMIT=33c2c10b8dd14ad4fea826802f3b9163664d2148

IMPLEMENTATION_FREEZE_COMMIT=88625179cd63d4e61e8719045b9a58b611f9825e

SCIENTIFIC_EXECUTION_ALLOWED=YES_STAGE_E_FIXED_PLANE_SIX_CELL_MATRIX

CUDA_PREFLIGHT_ALLOWED=YES

TRAINING_ALLOWED=YES_EXACT_STAGE_E_SIX_CELL

EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV

BACKWARD_ALLOWED=YES_TRAINING_ONLY

OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS

CONFIRMATORY_9601_9900_ALLOWED=NO

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP

## Scientific objective

Execute the prospectively frozen Stage E comparison:

- R22-constrained rank-2 correction;
- C22-constrained rank-2 matched control.

The question is whether task optimization can recover useful correction
capacity when the WRITE22 output plane is fixed to the native-causal R22
subspace, and whether that recovery differs from an otherwise matched C22
control plane.

No scientific criterion may be changed after execution begins.

## Exact implementation

Execution must use the three implementation files frozen at:

`88625179cd63d4e61e8719045b9a58b611f9825e`

No modification of those files is authorized during this execution phase.

Authorized frozen implementation paths:

1. `src/contramamba/gen5_stage_e_causal_plane_bottleneck.py`
2. `scripts/train_reason_router_gen5_stage_e_causal_plane_bottleneck.py`
3. `tests/test_reason_router_gen5_stage_e_causal_plane_bottleneck.py`

## Exact parameterization

For selected frozen plane Q:

`B_eff = Q M`

with:

- Q shape `[24576, 2]`, frozen;
- M shape `[2, 2]`, trainable;
- A shape `[2, 768]`, trainable;
- realized correction operator `Q M A`;
- no bias;
- M exactly zero initialized;
- A seed-deterministic and matched between R22/C22 arms.

Exactly two trainable tensors:

- `A_theta.weight`
- `M_theta.weight`

Expected trainable parameter count:

`1540`

## Frozen plane identities

R22 SHA256:

`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22 SHA256:

`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

No plane fitting, interpolation, rotation, search, or replacement is allowed.

## Exact six-cell matrix

Pressure:

`P0`

Cells:

- seed6201 / E-R22
- seed6201 / E-C22
- seed6202 / E-R22
- seed6202 / E-C22
- seed6203 / E-R22
- seed6203 / E-C22

Exactly six scientific training cells.

No unrestricted free-B training arm is authorized.

No PR or PC arm is authorized.

## Frozen data and optimization contract

Training population:

- frozen Phase3A train population;
- 3360 rows;
- 480 source pairs.

Evaluation population:

- frozen Phase3A dev population;
- 840 rows;
- 120 source pairs.

Split seed:

`16384`

Training:

- AdamW;
- learning rate `0.001`;
- weight decay `0.0001`;
- exactly `20` optimizer steps per cell;
- gradient clip norm `5.0`;
- no scheduler;
- final three-way cross entropy only;
- no early stopping;
- no checkpoint selection;
- final fixed step only;
- no hyperparameter tuning.

Parent model remains frozen.

Parent checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

## Frozen unrestricted reference

The Stage E matrix must reuse, not retrain, the existing unrestricted P0
reference.

Frozen P0 ZERO dev CE:

`1.3409655094146729`

Seed-matched unrestricted P0 gain:

- seed6201: `0.4984860420227051`
- seed6202: `0.5020102858543396`
- seed6203: `0.4994615912437439`

No new free-B baseline is authorized.

## Primary quantities

For each Stage E cell:

`gain_Q = CE_ZERO - CE_Q`

and:

`recovery_Q = gain_Q / gain_FREE_P0_seed`

For each seed:

`delta_RC = recovery_R22 - recovery_C22`

Report seedwise values and descriptive means only.

No scientific p-values are authorized.

No post-hoc success threshold may be created after seeing the results.

## Required output validation

Each cell must record and authenticate:

- seed;
- arm;
- selected frozen basis;
- train/dev identities;
- step-0 objective;
- final fixed-step objective;
- gradient norms;
- optimizer-step count;
- final dev CE;
- final dev accuracy;
- gain vs ZERO;
- frozen seed-matched unrestricted gain;
- recovery fraction;
- effective output/operator rank;
- fixed-plane residual;
- cross-plane leakage;
- A tensor SHA256;
- M tensor SHA256;
- parent fingerprint before and after;
- runtime and execution provenance.

Scientific conclusion fields must remain null in execution artifacts.

## Fixed-plane firewall

For every trained cell the realized effective output matrix must remain inside
the selected frozen plane.

The implementation must validate:

`||(I - Q Q^T) B_eff||`

and fail closed if the authorized numerical tolerance is exceeded.

R22/C22 cross-plane leakage must remain consistent with the frozen
orthogonality contract.

## CUDA preflight

Before full six-cell training, one bounded CUDA preflight may be executed.

Purpose:

- authenticate runtime/model/checkpoint/kernel plumbing;
- instantiate exactly one Stage E arm;
- execute one training-population forward;
- execute one backward;
- validate finite A/M gradients;
- validate no parent gradient;
- validate fixed-plane geometry.

CUDA preflight must satisfy:

- optimizer constructed: false;
- optimizer step count: 0;
- training executed: false;
- task evaluation executed: false;
- confirmatory data loaded: false.

The preflight result is implementation/runtime validation, not scientific
evidence.

## GPU execution topology

Full Stage E matrix must use:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

Required environment:

- exactly two visible Tesla T4 devices;
- each worker independently loads a fresh parent model;
- no DDP;
- no shared model;
- no shared optimizer;
- no shared scientific state between workers.

Worker queues are fixed by the frozen implementation.

## Confirmatory firewall

The population `xg1_fact_9601..9900` is forbidden.

It must not be loaded, evaluated, inspected, or used for model selection.

## Prohibited execution

Not authorized:

- unrestricted free-B retraining;
- PR/PC Stage E training;
- seed expansion;
- layer sweep;
- plane sweep;
- rank sweep;
- token search;
- learning-rate sweep;
- optimizer sweep;
- training-step sweep;
- basis modification;
- checkpoint selection;
- early stopping;
- confirmatory evaluation;
- scientific p-values;
- implementation edits during execution.

If frozen execution fails because of an implementation defect, stop.

Do not patch the frozen implementation in-place.

A correction implementation freeze and new execution authority would be
required.

## Interpretation boundary

Successful execution does not itself establish a scientific claim.

Keep separate:

1. code/runtime correctness;
2. execution success;
3. artifact/provenance validity;
4. scientific interpretation.

Scientific interpretation is allowed only after successful collection,
local import, provenance validation, and artifact validation.

## Prospective interpretation map

No outcome category is selected in advance.

The frozen design's interpretation map remains authoritative:

- stronger R22 recoverability than C22:
  evidence for task-usable R22 capacity not privileged by free optimization;
- similar substantial recovery for both:
  evidence for broad fixed-plane functional substitutability;
- low recovery for both:
  evidence that freely chosen output-plane orientation is important;
- C22 exceeding R22:
  no evidence for special R22 task-objective utility under this constrained
  correction.

These are bounded interpretations only.
