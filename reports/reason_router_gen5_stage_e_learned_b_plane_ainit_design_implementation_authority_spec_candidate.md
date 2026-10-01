# Gen5 Stage E Learned-B-Plane A-Initialization Diagnostic
# Design and Implementation Authority

## Status

DESIGN_FROZEN=YES

IMPLEMENTATION_ALLOWED=YES_BOUNDED_AINIT_DIAGNOSTIC

SCIENTIFIC_EXECUTION_ALLOWED=NO

CUDA_ALLOWED=NO

TRAINING_ALLOWED=NO

EVALUATION_ALLOWED=NO

BACKWARD_ALLOWED=NO

OPTIMIZER_STEP_ALLOWED=NO

CONFIRMATORY_9601_9900_ALLOWED=NO

## Parent evidence

BFREE_EVIDENCE_FREEZE_COMMIT=a17006164d6fc73de64cb138b12b6a7975750b4e

BFREE_IMPLEMENTATION_FREEZE_COMMIT=b8fa20dc3fb058412610375c255d8c25e93e14bf

BFREE_EXECUTION_COMMIT=467148d074f7cbd91a5e107a520df21ed4fe508c

PHASE3A_SOURCE_EXECUTION_COMMIT=d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e

## Scientific motivation

The validated E-BFREE positive control fixed the layer-22 correction output
plane to the seed-matched unrestricted Phase3A P0 final-B span.

Observed mean recovery relative to the frozen unrestricted P0 gain:

- E-R22: 0.00213934506552454
- E-C22: 0.00186063347357193
- E-BFREE: 0.07142508872336398

E-BFREE therefore recovered approximately 33.39 times the E-R22 recovery and
38.39 times the E-C22 recovery.

The direction was consistent for seeds 6201, 6202, and 6203.

However, E-BFREE recovered only approximately 7.14 percent of the unrestricted
seed-matched P0 gain.

Therefore output-plane orientation materially affects short-horizon
optimization accessibility, but the learned-B output span alone does not
explain the unrestricted solution.

The principal unresolved ambiguity is whether the remaining gap is caused
substantially by relearning the input/read-side A geometry from the Stage E
restart initialization.

## Exact scientific question

If the output plane remains exactly the seed-matched unrestricted final-B
span, but A is initialized from the same seed's unrestricted final
A_theta.weight, does the identical 20-step QMA optimization contract recover
materially more task benefit than E-BFREE?

This experiment changes exactly one scientific factor relative to E-BFREE:

A initialization.

No output-plane change, optimizer change, data change, training-horizon change,
rank change, layer change, or pressure change is allowed.

## Diagnostic arm

Single arm:

`E-BFREE-AINIT`

Exactly three cells:

- seed6201 / E-BFREE-AINIT
- seed6202 / E-BFREE-AINIT
- seed6203 / E-BFREE-AINIT

No E-BFREE rerun.

No R22/C22 rerun.

No unrestricted rerun.

No PR/PC.

No additional seed.

## Exact source checkpoints

Use only the existing seed-matched unrestricted Phase3A P0 final corrections.

### seed6201

Path:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed6201/P0/final_correction.pt`

SHA256:

`157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf`

### seed6202

Path:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed6202/P0/final_correction.pt`

SHA256:

`1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214`

### seed6203

Path:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed6203/P0/final_correction.pt`

SHA256:

`c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770`

Each source must authenticate:

- schema `GEN5_PHASE3A_FINAL_CORRECTION_V1`;
- matching seed;
- arm `G5-C0`;
- pressure `P0`;
- exact parent checkpoint SHA256;
- exact source-file SHA256;
- state_dict contains exactly the required A/B tensors;
- `A_theta.weight` shape `[2, 768]`;
- `B_theta.weight` shape `[24576, 2]`;
- both tensors finite;
- B effective rank exactly 2;
- source tensor hashes authenticated when present in the payload.

No PR or PC correction may be used.

## Output-plane derivation

The output plane is unchanged from E-BFREE.

For each seed:

`B_free = B_theta.weight`

Use the existing frozen learned-B-plane QR derivation:

`B_free = Q_B R_B`

with float64 thin QR and positive-diagonal sign canonicalization.

No SVD alternative, random rotation, plane search, interpolation, or
post-hoc basis transformation is allowed.

The later execution contract must retain the previously established
single-thread CPU linear-algebra environment for exact Q identity:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

The implementation-stage static verification may report platform-local Q
bytes, but local-development Q hashes must not be promoted to execution
identity.

## A initialization

Let:

`A_free = source state_dict["A_theta.weight"]`

For E-BFREE-AINIT:

- initialize `A_theta.weight` exactly from seed-matched `A_free`;
- initialize `M_theta.weight` exactly to zero;
- keep both A and M trainable during later authorized training;
- Q_B remains frozen;
- no bias.

Parameterization:

`correction(x) = Q_B M A x`

Shapes:

- Q_B `[24576, 2]`, frozen;
- M `[2, 2]`, trainable;
- A `[2, 768]`, trainable.

Exactly two trainable tensors:

- `A_theta.weight`
- `M_theta.weight`

Expected trainable parameter count:

`1540`

## Step-zero firewall

Because M is exactly zero initialized:

`Q_B M A_free x = 0`

for every input x at step zero.

Static verification must establish:

- M contains exactly zero nonzero elements;
- deterministic finite test input produces exactly zero correction output;
- no source B amplitude or unrestricted correction output is installed at
  step zero.

A_free initialization therefore supplies read-side geometry only.

It does not initialize the learned correction itself.

## Exact representability check

The implementation must establish before any scientific execution that the
QMA parameterization can exactly represent the seed-matched unrestricted
final correction operator.

For each seed, define:

`B_free = Q_B R_B`

and source:

`A_free`.

The exact unrestricted source operator is:

`O_free = B_free A_free`

The corresponding QMA representation is:

`O_qma = Q_B R_B A_free`

Require the relative operator residual to be within frozen numerical
tolerance.

Do not materialize a `[24576, 768]` matrix merely to perform this check.

Use the factorized Frobenius identity.

Let:

`D = B_free - Q_B R_B`

and:

`G_A = A_free A_free^T`

Then:

`||D A_free||_F^2 = tr((D^T D) G_A)`

and:

`||B_free A_free||_F^2 = tr((B_free^T B_free) G_A)`

Compute:

`representability_relative = sqrt(residual_sq / denominator_sq)`

in float64.

Require:

- denominator positive and finite;
- residual finite;
- `representability_relative <= 1e-12`.

Also record:

- A_free tensor SHA256;
- B_free tensor SHA256;
- Q tensor SHA256;
- R tensor SHA256;
- A_free effective row rank;
- B_free rank;
- QR reconstruction relative error;
- factorized operator representability relative error.

No task forward, CUDA, backward, optimizer, training, or evaluation is allowed
for this static representability check.

## Frozen later training contract

If a later execution authority is issued, it must preserve exactly the BFREE
training contract:

- Phase3A train population;
- Phase3A dev population;
- train rows 3360;
- dev rows 840;
- split seed 16384;
- pressure P0;
- same parent checkpoint;
- AdamW;
- learning rate 0.001;
- weight decay 0.0001;
- exactly 20 optimizer steps;
- gradient clip norm 5.0;
- no scheduler;
- final 3-way cross entropy only;
- no early stopping;
- final fixed step only.

Parent checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Confirmatory population 9601..9900 remains forbidden.

Because AdamW uses decoupled weight decay, A_free may begin changing after the
first optimizer step even though the task-loss gradient to A is zero while
M remains exactly zero at initialization.

This behavior is part of the matched optimizer contract and must not be
special-cased.

## Primary metrics

Use the existing frozen ZERO dev CE and seed-matched unrestricted P0 gains.

For each seed:

`gain_AINIT = CE_ZERO - CE_AINIT`

`recovery_AINIT = gain_AINIT / gain_FREE_P0`

Primary descriptive comparison:

`recovery_AINIT - recovery_BFREE`

Use the already frozen E-BFREE seed-matched recovery values:

- seed6201: 0.0690419274517625
- seed6202: 0.0737218360466545
- seed6203: 0.0715115026716749

Frozen E-BFREE mean:

`0.07142508872336398`

No p-value.

No post-hoc success threshold.

No rerunning BFREE.

## Interpretation contract

If E-BFREE-AINIT is consistently and materially higher than E-BFREE across
the matched seeds, then relearning A/read-side geometry is implicated as a
major contributor to the remaining fixed-plane optimization bottleneck.

If E-BFREE-AINIT remains near E-BFREE, then A restart initialization is not a
sufficient explanation and the remaining ambiguity shifts toward QMA
scale/gauge conditioning or short-horizon optimization dynamics.

If outcomes are materially seed-dependent, preserve seed-dependent ambiguity.

A low AINIT result must not be interpreted as evidence that A geometry is
irrelevant in all optimization regimes.

A high AINIT result must not be interpreted as independent confirmation,
because A_free and Q_B were selected by the same prior task optimization.

## Explicitly forbidden expansion

This authority does not permit:

- random-plane controls;
- additional seeds;
- rank sweep;
- layer sweep;
- learning-rate sweep;
- optimizer sweep;
- weight-decay sweep;
- training-step sweep;
- alternative QR/SVD basis construction;
- freezing A during scientific execution;
- nonzero M initialization;
- source final-B amplitude initialization;
- confirmatory-set access;
- scientific training or evaluation.

## Authorized implementation scope

Create exactly these three new files:

1. `src/contramamba/gen5_stage_e_learned_b_plane_ainit_control.py`
2. `scripts/train_reason_router_gen5_stage_e_learned_b_plane_ainit_control.py`
3. `tests/test_reason_router_gen5_stage_e_learned_b_plane_ainit_control.py`

Do not modify the frozen E-BFREE implementation.

Do not modify Stage E, Phase3A, Phase2, dataset, tokenizer, model, or prior
evidence files.

The new implementation should import and reuse frozen BFREE/Stage-E
primitives wherever possible rather than duplicate them.

## Required static validation

Before implementation freeze, require:

- exact authority authentication;
- exact three-file implementation scope;
- source checkpoint SHA/schema/seed/arm/pressure authentication;
- A_free and B_free tensor authentication;
- Q/R geometry authentication;
- A_free initialization identity;
- M exact-zero initialization;
- exact zero correction output at initialization;
- trainable tensor names exactly A and M;
- trainable parameter count exactly 1540;
- factorized exact-operator representability check PASS for all three seeds;
- no parent checkpoint load;
- no parent model construction;
- no model forward;
- no CUDA;
- no backward;
- no optimizer construction;
- no optimizer step;
- no task training;
- no task evaluation;
- confirmatory population not loaded.

Static verification must print seed-specific A_free SHA256 and
representability residuals.

## Stop conditions

Stop implementation work if:

- any source checkpoint identity differs;
- A or B source tensor schema differs;
- B rank is not exactly 2;
- factorized representability residual exceeds tolerance;
- step-zero correction is nonzero;
- implementation requires modification of frozen BFREE files;
- implementation scope expands beyond the exact three authorized files;
- any scientific execution is required to complete static validation.

A later implementation-freeze commit and a separate execution authority are
required before CUDA preflight, backward, optimizer steps, training, or task
evaluation.
