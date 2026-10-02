# Gen5 Stage E Learned-B-Plane A-Fixed M-Only Diagnostic
# Design and Implementation Authority

## Status

DESIGN_FROZEN=YES

IMPLEMENTATION_ALLOWED=YES_BOUNDED_AFIX_MONLY_DIAGNOSTIC

SCIENTIFIC_EXECUTION_ALLOWED=NO

CUDA_ALLOWED=NO

TRAINING_ALLOWED=NO

EVALUATION_ALLOWED=NO

BACKWARD_ALLOWED=NO

OPTIMIZER_STEP_ALLOWED=NO

CONFIRMATORY_9601_9900_ALLOWED=NO

## Parent evidence

STATIC_DECOMPOSITION_FREEZE_COMMIT=5ffc5ea29a148dedac97a2f093458e550fa38b03

AINIT_EVIDENCE_FREEZE_COMMIT=473ed74a70c02484c93746a022c4122632d55416

AINIT_EXECUTION_COMMIT=51bfc4d1d2e1d3fae970e7c09a75cf3aa72cf488

AINIT_IMPLEMENTATION_FREEZE_COMMIT=69f92f54f6a143940b7687c5e4b8e1931c5fae8d

AINIT_DESIGN_IMPLEMENTATION_AUTHORITY_COMMIT=c24bc199b6b8fb94002c0563535ff7b5284794b2

BFREE_EVIDENCE_FREEZE_COMMIT=a17006164d6fc73de64cb138b12b6a7975750b4e

PHASE3A_SOURCE_EXECUTION_COMMIT=d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e

## Scientific basis

Validated E-BFREE mean recovery relative to the frozen unrestricted
seed-matched P0 gain:

`0.07142508872336398`

Validated E-BFREE-AINIT mean recovery:

`0.133576230985308`

Mean AINIT minus BFREE recovery delta:

`0.0621511422619444`

The recovery increase was positive for all three matched seeds.

Therefore initialization from the seed-matched unrestricted final
`A_theta.weight` materially improved short-horizon optimization accessibility.

However, approximately 86.64 percent of the unrestricted gain remained
unrecovered.

The frozen read-only static decomposition then compared the final E-BFREE and
E-BFREE-AINIT corrections against the unrestricted seed-matched source
factorization

`B_free = Q_B R_B`.

For E-BFREE-AINIT, the mean values were:

- A row-space affinity to A_free: `0.93180364918`
- effective operator relative error: `0.98117067811`
- effective operator norm ratio: `0.0228588324149`
- M-only relative error with exact A_free restored: `0.988275293096`
- gauge-aligned M relative error: `0.986030986272`

The static decomposition supports:

`RESIDUAL_FIXED_PLANE_RECOVERY_FAILURE_LOCALIZES_PRIMARILY_TO_CORE_M_ACQUISITION_RATHER_THAN_FINAL_A_ROWSPACE_MISMATCH`

but does not distinguish whether the poor M-core acquisition arises from:

1. joint A/M optimization interference;
2. scale or conditioning of the zero-initialized M parameterization;
3. the fixed 20-step horizon.

The smallest discriminating next experiment is therefore to remove joint A
optimization while preserving every other frozen factor.

## Exact scientific question

With the output plane fixed to the same seed-matched unrestricted final-B span,
and A installed exactly from the same seed's unrestricted final
`A_theta.weight` but frozen for the entire run, can training only the 2x2 M
core from exact zero recover materially more task benefit than
E-BFREE-AINIT under the identical 20-step contract?

This experiment changes exactly one scientific factor relative to
E-BFREE-AINIT:

`A trainability`

No output-plane change, A initialization change, M initialization change,
optimizer-family change, learning-rate change, weight-decay change, data
change, training-horizon change, rank change, layer change, or pressure change
is allowed.

## Diagnostic arm

Single arm:

`E-BFREE-AFIX-MONLY`

Exactly three cells:

- seed6201 / E-BFREE-AFIX-MONLY
- seed6202 / E-BFREE-AFIX-MONLY
- seed6203 / E-BFREE-AFIX-MONLY

No E-BFREE rerun.

No E-BFREE-AINIT rerun.

No R22/C22 rerun.

No unrestricted rerun.

No PR/PC.

No additional seed.

## Source checkpoints and output plane

Use only the same seed-matched unrestricted Phase3A P0 final corrections and
the same authenticated source identities already frozen by E-BFREE-AINIT.

### seed6201

Source file SHA256:

`157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf`

A_free tensor SHA256:

`23796c2fbdaeee6e76d58bf613b85fdf52d25a47c86d98bc9376422478e79469`

Execution Q_B SHA256:

`1cfc7e1b55b68b0b71404f75c4920788331c9fcccb14ed25b40a316973721705`

### seed6202

Source file SHA256:

`1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214`

A_free tensor SHA256:

`46472cf48fd5973fc2635c99ebc3d0b77798141c359ae5798b1ec9019e1f24f9`

Execution Q_B SHA256:

`44b81288f73f605bc12fbab90a51cdb87f421f5cbd6fec7b648dd6621d77ff57`

### seed6203

Source file SHA256:

`c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770`

A_free tensor SHA256:

`5f47d66270a2ee2900741395e54dd274d4cb12e8359cbc223ea9fa1be07911c8`

Execution Q_B SHA256:

`28dd4b583b8019688ca656cc063924e75bd4b2fee9c35fa6680b5570d0953330`

For each seed:

`B_free = Q_B R_B`

must use the existing deterministic float64 thin-QR procedure with
positive-diagonal sign canonicalization.

No alternative QR, SVD, random rotation, interpolation, or post-hoc plane
selection is allowed.

A later execution authority must retain:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

for exact execution Q identity.

## Parameterization and initialization

Use:

`correction(x) = Q_B M A_free x`

with:

- Q_B `[24576, 2]`, frozen;
- A `[2, 768]`, copied exactly from seed-matched A_free and frozen;
- M `[2, 2]`, trainable;
- no bias.

Initialization:

- A = exact seed-matched A_free;
- M = exact zero.

The initial correction output must therefore be exactly zero for every input.

A must remain byte-identical to its installed source value for the entire
later scientific execution.

## Trainable parameter contract

Exactly one trainable tensor:

`M_theta.weight`

Shape:

`[2, 2]`

Expected trainable parameter count:

`4`

`A_theta.weight` must have `requires_grad=False`.

Q_B must remain frozen.

The parent model must remain frozen.

A later optimizer may receive only the four M parameters.

No optimizer parameter group may contain A or any parent parameter.

## Exact representability

The static implementation must preserve the existing exact-representability
check.

With:

`B_free = Q_B R_B`

the parameterization can exactly represent the unrestricted source operator at:

`M = R_B`

and:

`A = A_free`.

Require the existing factorized float64 representability residual:

`<= 1e-12`

without materializing the full `[24576, 768]` operator.

Static verification must record:

- source file SHA256;
- A_free tensor SHA256;
- B_free tensor SHA256;
- Q tensor SHA256;
- R tensor SHA256;
- A_free effective row rank;
- B_free rank;
- QR reconstruction relative error;
- factorized operator representability relative error.

## Frozen later training contract

If and only if a separate later execution authority is issued, preserve:

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
- final fixed step only;
- parent checkpoint SHA256
  `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`.

The only optimizer parameter is `M_theta.weight`.

Because A is frozen, AdamW weight decay must not modify A.

Confirmatory IDs 9601..9900 remain forbidden.

## Frozen comparison evidence

Do not rerun comparison arms.

Use the already frozen E-BFREE-AINIT recovery values:

- seed6201: `0.135931570756102`
- seed6202: `0.124898380318522`
- seed6203: `0.139898741881301`

Mean:

`0.133576230985308`

Secondary context may retain the already frozen E-BFREE values:

- seed6201: `0.0690419274517625`
- seed6202: `0.0737218360466545`
- seed6203: `0.0715115026716749`

Mean:

`0.07142508872336398`

## Primary metric

For each seed:

`gain_AFIX_MONLY = CE_ZERO - CE_AFIX_MONLY`

`recovery_AFIX_MONLY = gain_AFIX_MONLY / gain_FREE_P0`

Primary descriptive comparison:

`recovery_AFIX_MONLY - recovery_AINIT`

No p-value.

No post-hoc success threshold.

No comparison-arm rerun.

## Interpretation contract

If E-BFREE-AFIX-MONLY is consistently and materially higher than
E-BFREE-AINIT across the matched seeds, joint A drift or joint A/M coupling is
implicated as a major remaining short-horizon optimization bottleneck.

If E-BFREE-AFIX-MONLY remains near E-BFREE-AINIT, freezing A does not resolve
the residual gap and the remaining ambiguity shifts strongly toward M-core
scale/conditioning or the fixed 20-step horizon.

If outcomes are materially seed-dependent, preserve seed-dependent ambiguity.

A high AFIX-MONLY result must not be interpreted as independent confirmation:
Q_B and A_free are both inherited from the same prior unrestricted task
optimization.

A low AFIX-MONLY result must not be interpreted as evidence that the learned-B
plane is intrinsically unusable.

## Explicitly forbidden expansion

This authority does not permit:

- scientific execution;
- CUDA;
- backward;
- optimizer steps;
- task training;
- task evaluation;
- A training;
- nonzero M initialization;
- source R_B initialization of M;
- random-plane controls;
- additional seeds;
- rank sweep;
- layer sweep;
- learning-rate sweep;
- optimizer sweep;
- weight-decay sweep;
- training-step or horizon sweep;
- alternative QR/SVD construction;
- confirmatory-set access;
- scientific p-values.

## Authorized implementation scope

Create exactly these three new files:

1. `src/contramamba/gen5_stage_e_learned_b_plane_afix_monly_control.py`
2. `scripts/train_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control.py`
3. `tests/test_reason_router_gen5_stage_e_learned_b_plane_afix_monly_control.py`

Do not modify frozen E-BFREE or E-BFREE-AINIT implementation files.

Do not modify Stage E, Phase3A, Phase2, dataset, tokenizer, model, prior
evidence, or prior authority files.

The new implementation should reuse frozen E-BFREE-AINIT source-loading,
representability, Q derivation, layer-installation, and runtime primitives
wherever possible.

## Required static implementation behavior

The primitive implementation must:

- construct the same QMA correction used by AINIT;
- copy exact A_free;
- set `A_theta.weight.requires_grad=False`;
- zero `M_theta.weight` exactly;
- leave only `M_theta.weight` trainable;
- expose a trainable-parameter audit returning exactly one tensor and 4
  parameters;
- retain exact-zero step-zero correction firewall;
- retain exact source and representability authentication;
- preserve parent parameter fingerprint during installation.

The runner implementation must support static verification without loading the
parent checkpoint or constructing the parent model.

Runtime modes may be implemented structurally for later authority, but must be
hard-gated by a separate execution-authority file/commit that does not yet
exist.

## Required static validation

Before implementation freeze, require:

- exact implementation-authority authentication;
- exact three-new-file implementation scope;
- all inherited source checkpoint identities authenticated;
- all A_free identities authenticated;
- Q/R geometry authentication;
- factorized representability `<= 1e-12` for all three seeds;
- A copied exactly from A_free;
- A `requires_grad=False`;
- M exact zero;
- M `requires_grad=True`;
- exact zero correction output at initialization;
- trainable tensor names exactly `["M_theta.weight"]`;
- trainable tensor count exactly `1`;
- trainable parameter count exactly `4`;
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

Static verification must print seed-specific:

- source SHA256;
- A_free SHA256;
- Q SHA256;
- R SHA256;
- A row rank;
- B rank;
- QR reconstruction relative error;
- representability relative error;
- A frozen identity;
- M exact-zero identity.

## Stop conditions

Stop implementation work if:

- any inherited source identity differs;
- A or B source schema differs;
- B rank is not exactly 2;
- representability exceeds tolerance;
- A is trainable;
- M is not the only trainable tensor;
- trainable parameter count is not exactly 4;
- step-zero correction is nonzero;
- parent fingerprint changes on installation;
- implementation requires modification of frozen BFREE/AINIT files;
- scope expands beyond the exact three new files;
- any scientific execution is required to complete static validation.

A later implementation-freeze commit and a separate execution authority are
required before CUDA preflight, backward, optimizer construction/steps,
training, or task evaluation.
