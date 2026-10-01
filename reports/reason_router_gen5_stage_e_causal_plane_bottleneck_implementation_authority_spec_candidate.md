# Gen5 Stage E — Causal-Plane Bottleneck Implementation Authority

## Authority

DESIGN_FREEZE_COMMIT=c2f0c92da999d6508b712eb82f20f33cfcac4624

IMPLEMENTATION_ALLOWED=YES_BOUNDED_STAGE_E_FIXED_PLANE

SCIENTIFIC_EXECUTION_ALLOWED=NO

CUDA_EXECUTION_ALLOWED=NO

TRAINING_ALLOWED=NO

EVALUATION_ALLOWED=NO

BACKWARD_ALLOWED=NO

OPTIMIZER_STEP_ALLOWED=NO

CONFIRMATORY_9601_9900_ALLOWED=NO

## Goal

Implement, but do not scientifically execute, the frozen Stage E comparison
between a rank-2 correction constrained to R22 and an otherwise identical
rank-2 correction constrained to C22.

The implementation must preserve the prospective scientific design frozen at:

`c2f0c92da999d6508b712eb82f20f33cfcac4624`

No scientific criterion may be changed during implementation.

## Authorized implementation paths

Exactly these new paths are authorized:

1. `src/contramamba/gen5_stage_e_causal_plane_bottleneck.py`
2. `scripts/train_reason_router_gen5_stage_e_causal_plane_bottleneck.py`
3. `tests/test_reason_router_gen5_stage_e_causal_plane_bottleneck.py`

No existing source, script, test, data, report, checkpoint, basis, or artifact
file may be modified by this implementation phase.

## Fixed-plane parameterization

The Stage E correction must implement:

`B_eff = Q M`

with:

- `Q` frozen shape `[24576, 2]`;
- `M` trainable shape `[2, 2]`;
- `A` trainable shape `[2, 768]`;
- correction operator `Q M A`;
- no bias;
- `M` exactly zero initialized;
- A initialized deterministically on CPU from the scientific seed;
- A initialization identical between E-R22 and E-C22 for a matched seed.

Authorized arms:

`E-R22`
- `Q = frozen R22`

`E-C22`
- `Q = frozen C22`

Frozen basis SHA256 identities:

R22:

`a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214`

C22:

`c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4`

No learned, rotated, interpolated, fitted, or searched output basis is allowed.

## Required trainable surface

Exactly two trainable tensors are allowed:

- `A_theta.weight`, shape `[2, 768]`
- `M_theta.weight`, shape `[2, 2]`

Expected trainable parameter count:

`1540`

The parent model must remain frozen.

The fixed R22/C22 bases must remain buffers, not parameters.

## Required WRITE22 semantics

The implementation must target the same layer-22 correction surface used by
the frozen Gen5 Phase2/Phase3A correction mechanism.

The implementation must preserve:

- target layer 22;
- hidden size 768;
- intermediate size 1536;
- state size 16;
- state-write width 24576;
- rank 2;
- active-token masking semantics;
- frozen native recurrence coefficients;
- separate additive correction recurrence;
- no duplicate native output bias.

The existing frozen Phase2 implementation may be imported or used as a
reference, but it must not be edited.

## Frozen Stage E matrix

The implementation may expose future execution for exactly:

- seed6201 / E-R22
- seed6201 / E-C22
- seed6202 / E-R22
- seed6202 / E-C22
- seed6203 / E-R22
- seed6203 / E-C22

Pressure is P0 only.

No PR/PC implementation matrix is authorized.

## Frozen training contract encoded by the runner

The future runner must encode, but not execute in this implementation phase:

- Phase3A frozen train population: 3360 rows;
- Phase3A frozen dev population: 840 rows;
- Phase3A split seed: 16384;
- seeds: 6201, 6202, 6203;
- P0 only;
- AdamW;
- learning rate 0.001;
- weight decay 0.0001;
- 20 optimizer steps;
- gradient clip norm 5.0;
- no scheduler;
- final 3-way cross entropy only;
- frozen parent;
- final fixed step only;
- no early stopping;
- no checkpoint selection;
- no hyperparameter tuning.

Parent checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

The implementation must reuse the frozen Phase3A data/split/tokenizer identities
rather than reconstructing a new population.

## Frozen unrestricted reference

The implementation must encode the already frozen P0 unrestricted reference
rather than provide a new unrestricted training arm.

ZERO dev CE:

`1.3409655094146729`

Unrestricted P0 gain by seed:

- 6201: `0.4984860420227051`
- 6202: `0.5020102858543396`
- 6203: `0.4994615912437439`

No free-B retraining mode is authorized.

## Required future outputs

For each future Stage E cell, the implementation must be able to record:

- seed;
- arm;
- exact basis identity;
- train/dev identity hashes;
- step-0 objective;
- final fixed-step objective;
- gradient norms;
- optimizer-step count;
- final dev CE;
- final dev accuracy;
- `gain_Q = CE_ZERO - CE_Q`;
- frozen seed-matched unrestricted gain;
- `recovery_Q = gain_Q / gain_FREE_P0`;
- effective operator rank;
- fixed-plane residual;
- A tensor SHA256;
- M tensor SHA256;
- parent fingerprint before/after;
- execution/provenance identities.

Scientific conclusion fields must remain null during execution.

No p-values are authorized.

## Required fixed-plane validation

The implementation and tests must directly verify that the realized output
operator lies in the selected frozen plane.

For effective output matrix:

`B_eff = Q M`

the implementation must validate the residual outside Q:

`||(I - Q Q^T) B_eff||`

to a numerically justified tolerance.

The test suite must also verify:

- R22/C22 exact frozen geometry;
- matched-seed A initialization equality;
- exact zero M initialization;
- E-R22 output lies in R22;
- E-C22 output lies in C22;
- cross-plane leakage is consistent with frozen R22/C22 orthogonality;
- only A and M are trainable;
- expected trainable numel = 1540;
- parent fingerprint is unchanged by wrapper installation;
- active-mask behavior;
- forward/operator algebra;
- unsupported arm rejection.

## Runner modes

The runner may implement:

`--static-verify-only`

This is the only runner mode authorized to execute during the current
implementation phase.

It must perform CPU/read-only authentication and must report:

- checkpoint loaded: false;
- model instantiated: false;
- CUDA executed: false;
- backward executed: false;
- optimizer constructed: false;
- optimizer step count: 0;
- training executed: false;
- task evaluation executed: false;
- confirmatory data loaded: false.

The runner may also contain future CUDA preflight/training modes needed for
the frozen six-cell experiment, but their presence in source code does not
authorize their execution.

## Validation allowed now

Allowed:

- Python syntax/compile checks;
- CPU unit tests;
- static repository/data/basis authentication;
- `git diff --check`;
- `cm ship`.

Forbidden now:

- model checkpoint loading;
- model instantiation;
- CUDA execution;
- forward passes through the scientific model;
- backward;
- optimizer construction or step;
- training;
- task evaluation;
- confirmatory-data access.

## Stop conditions

Stop implementation validation if any of the following would be required:

- modification of an existing frozen implementation file;
- alteration of Stage E arms;
- alteration of seeds;
- alteration of P0-only scope;
- alteration of train/dev population;
- alteration of optimizer or 20-step budget;
- alteration of R22/C22 basis identities;
- introduction of a basis/plane/rank/layer/token/hyperparameter sweep;
- introduction of a newly trained unrestricted baseline;
- access to confirmatory 9601..9900;
- scientific execution before a later explicit execution authority.

## Commit boundary

Implementation files must be independently reviewed with CPU/static validation
and frozen in a separate implementation commit.

A later explicit Stage E execution authority is required before CUDA,
backward, optimizer construction, training, or task evaluation.
