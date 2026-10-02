# Gen5 Stage E AFIX-MONLY SCALEMATCH Design + Implementation Authority

## Status

IMPLEMENTATION_AUTHORITY_CANDIDATE

This document authorizes only the bounded implementation and static
verification of one Gen5 Stage E diagnostic:

`E-BFREE-AFIX-MONLY-SCALEMATCH`

It does not authorize CUDA execution, model forward execution, backward,
optimizer construction, training, task evaluation, Kaggle scientific
execution, confirmatory-data access, or scientific interpretation of new
runtime results.

## Authority basis

Static scale/mode evidence freeze:

`8abab776801eb106ddd97c39883a99f83257b833`

Validated AFIX-MONLY evidence freeze:

`d3d0f86fca9111ab19944f020c1efbb3d6b37d0a`

Frozen AFIX-MONLY implementation:

`b1ab97600d47e0f86ca3a427befffb777a10829c`

The static evidence establishes that the previous absolute unrestricted-B
versus fixed-plane-QMA comparison was confounded by parameterization-dependent
Adam step scale.

Under the existing 20-step, AdamW, lr=0.001 contract:

- unrestricted B has `24576 * 2 = 49152` trainable coordinates;
- AFIX-MONLY M has `2 * 2 = 4` trainable coordinates;
- the diagnostic coordinate-normalized Frobenius movement scales differ by
  `sqrt(49152 / 4) = sqrt(12288)`.

The next experiment must remove only that scale confound.

## Scientific question

Does matching the nominal Adam coordinate-normalized movement budget of the
4-parameter M core to the original unrestricted-B parameterization materially
restore:

1. task recovery;
2. core norm;
3. the second source singular mode;
4. source-core/operator fidelity?

This is a single diagnostic control, not a learning-rate sweep.

## Experimental arm

Exact arm name:

`E-BFREE-AFIX-MONLY-SCALEMATCH`

Seeds:

`6201,6202,6203`

Pressure:

`P0`

Output plane:

same exact seed-matched learned-B plane `Q_B` used by AFIX-MONLY.

Read-side matrix:

same exact seed-matched unrestricted `A_free`, frozen for the entire run.

Core:

`M in R^(2x2)`, exact zero initialization.

Trainable tensor names:

`M_theta.weight`

Trainable tensor count:

`1`

Trainable parameter count:

`4`

Parent model:

frozen.

## Single allowed scientific delta

Original AFIX-MONLY learning rate:

`0.001`

Unrestricted-B coordinate count:

`49152`

M coordinate count:

`4`

Scale factor:

`sqrt(49152 / 4) = sqrt(12288)`

Exact SCALEMATCH learning rate:

`0.11085125168440814`

Equivalent definition:

`SCALEMATCH_LEARNING_RATE = 0.001 * sqrt(49152 / 4)`

This value is determined only from:

- the frozen original learning rate;
- the unrestricted-B parameterization dimension;
- the M parameterization dimension.

It must not be fitted to:

- observed `R_B` norms;
- AFIX-MONLY recovery;
- dev loss;
- any task outcome.

## Everything else remains frozen

The implementation must preserve exactly:

- seeds `6201,6202,6203`;
- pressure `P0`;
- train split;
- dev split;
- split seed;
- tokenizer/model/runtime identities;
- parent checkpoint;
- learned-B source checkpoints;
- canonical QR/source-plane construction;
- exact `A_free` initialization;
- A frozen throughout;
- exact zero M initialization;
- only M trainable;
- AdamW optimizer family;
- weight decay `0.0001`;
- gradient clipping norm `5.0`;
- no scheduler;
- exactly 20 optimizer steps per cell;
- exactly 60 optimizer steps across three cells;
- final fixed-step checkpointing;
- final 3-way cross entropy only;
- no early stopping;
- no checkpoint selection;
- two independent single-GPU workers;
- no DDP;
- worker assignment `(6201,6203)` and `(6202)`;
- confirmatory IDs `9601..9900` forbidden;
- no scientific p-values.

No rank, layer, pressure, optimizer, horizon, initialization, plane, seed, or
additional learning-rate sweep is authorized.

## Implementation scope

Create exactly these three new files:

1. `src/contramamba/gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py`
2. `scripts/train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py`
3. `tests/test_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py`

Do not modify any existing file.

The new implementation should reuse the already-frozen AFIX-MONLY primitives
where possible rather than reimplementing source authentication, correction
construction, A-free freezing, M-only trainability, or parent-model logic.

## Required implementation invariants

The implementation must enforce:

`ARM=E-BFREE-AFIX-MONLY-SCALEMATCH`

`TRAINING_SEEDS=(6201,6202,6203)`

`PRESSURE=P0`

`UNRESTRICTED_B_TRAINABLE_NUMEL=49152`

`M_TRAINABLE_NUMEL=4`

`SCALEMATCH_FACTOR=sqrt(12288)`

`LEARNING_RATE=0.11085125168440814`

`WEIGHT_DECAY=0.0001`

`GRADIENT_CLIP_NORM=5.0`

`TOTAL_OPTIMIZER_STEPS=20`

`A_REQUIRES_GRAD=False`

`M_REQUIRES_GRAD=True`

`TRAINABLE_TENSOR_NAMES=M_theta.weight`

`TRAINABLE_TENSOR_COUNT=1`

`TRAINABLE_NUMEL=4`

`M_INIT_EXACT_ZERO=True`

`STEP0_CORRECTION_EXACT_ZERO=True`

The runner must verify A byte identity against the frozen same-seed `A_free`
identity before training and after every optimizer step whenever runtime
execution is later authorized.

A gradient must remain `None`.

No parent parameter gradient or mutation is allowed.

## Static verification only under this authority

Before a separate execution authority exists, the only permitted runner mode
is static verification.

Static verification must:

- authenticate repository HEAD and allowed implementation scope;
- authenticate all three frozen source correction checkpoints;
- verify source A/B identities;
- verify rank-2 source B;
- verify numerical QR reconstruction within `1e-12`;
- verify exact source-operator representability within `1e-12`;
- verify A exact copy and frozen state;
- verify M exact zero initialization;
- verify only M is trainable;
- verify exactly 4 trainable parameters;
- verify SCALEMATCH LR is derived from dimensions and original lr;
- verify all non-LR training constants equal AFIX-MONLY;
- verify exact worker assignment;
- verify confirmatory data are not loaded;
- load no parent checkpoint;
- instantiate no parent model;
- execute no model forward;
- execute no CUDA;
- execute no backward;
- construct no optimizer;
- execute no optimizer step;
- execute no training;
- execute no task evaluation;
- compute no scientific p-values.

Expected static terminal marker:

`GEN5_STAGE_E_BFREE_AFIX_MONLY_SCALEMATCH_STATIC_VERIFY_PASS`

## Required tests

The dedicated test file must cover at minimum:

1. exact arm identity;
2. exact seed set;
3. exact scale-factor derivation;
4. exact SCALEMATCH LR;
5. proof that SCALEMATCH LR depends on dimensions and original AFIX lr, not
   target `R_B` values;
6. exact A-free copy and `requires_grad=False`;
7. exact M-zero initialization and `requires_grad=True`;
8. only `M_theta.weight` trainable;
9. exactly 4 trainable parameters;
10. source checkpoint authentication for all seeds;
11. exact three-cell/two-worker assignment;
12. static mode rejection of runtime authority arguments;
13. runtime modes blocked without a future execution authority;
14. existing frozen AFIX-MONLY implementation files are not modified by this
    implementation scope.

## Validation command

After creating the three files, run only:

```powershell
python -m pytest tests/test_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py -q
python scripts/train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_scalematch_control.py --static-verify-only --expected-head <CURRENT_HEAD> --allow-opening-worktree
```

The second command may authenticate only the three authorized new
implementation paths as opening-worktree changes.

## Training / evaluation authority

`CUDA_ALLOWED=NO`

`PARENT_MODEL_INSTANTIATION_ALLOWED=NO`

`MODEL_FORWARD_ALLOWED=NO`

`BACKWARD_ALLOWED=NO`

`OPTIMIZER_ALLOWED=NO`

`TRAINING_ALLOWED=NO`

`TASK_EVALUATION_ALLOWED=NO`

`KAGGLE_SCIENTIFIC_EXECUTION_ALLOWED=NO`

`CONFIRMATORY_9601_9900_ALLOWED=NO`

A separate implementation freeze and separate execution authority are required
before any runtime execution.

## Stop conditions

Stop implementation immediately if:

- any existing file must be modified;
- any scientific delta other than the LR is needed;
- the scale factor cannot be derived exactly from frozen dimensions;
- static source identities fail;
- A is trainable;
- any tensor other than M is trainable;
- trainable numel differs from 4;
- static verification attempts model/CUDA/backward/optimizer/training/eval;
- confirmatory data are accessed.

## Required implementation report

After implementation/static validation, report:

- exact HEAD;
- exact three generated file paths;
- SHA256 for all three files;
- pytest result;
- static verifier terminal marker;
- derived scale factor;
- exact SCALEMATCH LR;
- trainable names/count/numel;
- A/M requires-grad state;
- step-zero correction state;
- source identities for all three seeds;
- confirmation that no CUDA/model/backward/optimizer/training/evaluation or
  confirmatory access occurred.

No scientific conclusion may be drawn from implementation validation.
