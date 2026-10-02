# Gen5 A-init Residual Nullness and Task-Sensitive Subspace Audit Authority

SOURCE_RESIDUAL_LOCALIZATION_EVIDENCE_FREEZE_COMMIT=3c0a3d8a67e9910f91de2354ba29a5c4b3b28942
SOURCE_RESIDUAL_LOCALIZATION_EXECUTION_COMMIT=69473ac76629ef1e7c78b45c8c802a89e1e9695b
SOURCE_QUOTIENT_EVIDENCE_FREEZE_COMMIT=0e6191fd54e23388abcce1abd9e01a453d2dc73c
SOURCE_FUNCTIONAL_EQUIVALENCE_EVIDENCE_COMMIT=5f079a66f7b0eb0caea30a8d5bc9a0fe757cc449

STATUS=READY_FOR_AINIT_RESIDUAL_NULLNESS_AND_TASK_SENSITIVE_SUBSPACE_AUDIT

TRAINING_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
PARAMETER_GRADIENT_UPDATE_ALLOWED=NO
ANALYSIS_AUTOGRAD_ALLOWED=YES_DOWNSTREAM_LAYER22_BOUNDARY_ONLY
BACKWARD_METHOD_ALLOWED=TORCH_AUTOGRAD_GRAD_ONLY
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
CUDA_EVALUATION_ALLOWED=YES_FROZEN_DEV_ONLY
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## Scientific question

The frozen residual-localization evidence established that same-training-RNG /
different-A-init pairs retain a substantial layer-22 correction-output
normalized residual:

`0.155009066845`

while the final centered-delta-logit residual is only:

`0.0128792116035`

This geometric decay alone does not determine whether the layer-22 residual is
task-relevant information.

The present audit asks two prior questions:

1. Is the actual A-init residual locally aligned with the kernel or low-gain
   directions of the frozen downstream task map?
2. Does the downstream task-sensitive geometry concentrate into a
   lower-dimensional hidden-direction subspace in which A-init solutions become
   substantially more similar?

The audit precedes any precursor search or latent intervention.

## Frozen population and grid

Use only:

DEV_ROWS=840
SPLIT_SEED=16384
ARM=G5-C0
PRESSURE=P0

Use the exact frozen 3x3 grid:

A_INIT_SEED in {6201,6202,6203}
TRAINING_RNG_SEED in {6201,6202,6203}

No new scientific seed, training population, confirmatory population,
checkpoint, rank, optimizer, or architecture is authorized.

## Layer-22 intervention boundary

For each frozen cell, run the exact frozen Phase3A P0 dev computation.

At the output of `Phase2Layer22MixerWrapper.forward`, replace the returned
tensor by an equal-valued detached leaf requiring gradient.

This preserves the exact forward value while disconnecting all computation at
and before the layer-22 mixer output.

All analysis gradients therefore measure only the frozen downstream map from
the layer-22 mixer-output boundary to final task logits.

No parent or correction parameter gradient may be requested or accumulated.

## Task output coordinates

Use two independent centered-logit margin coordinates:

`m_refute = logit_refute - logit_not_entitled`

`m_support = logit_support - logit_not_entitled`

These span the two nontrivial degrees of freedom of three-class logits after
removing common-mode logit shifts.

For every dev chunk and cell, call `torch.autograd.grad` on the summed margin
coordinates with respect to the detached layer-22 boundary tensor.

Do not call `.backward()`.

## Actual residual directions

For each pair of frozen cells evaluated on the same dev examples, define the
layer-22 boundary residual:

`d_ij = h_i - h_j`

Because all cells share the same frozen upstream computation before the
layer-22 correction, this is the exact cell-to-cell boundary difference under
the tested contract.

Primary groups:

1. same A-init, different training RNG;
2. same training RNG, different A-init.

## Endpoint-local Jacobian audit

At each endpoint cell, let:

`J = [grad(m_refute); grad(m_support)]`

for each example, flattened across sequence and hidden dimensions.

For every pair residual `d`, report at both endpoints and symmetrically averaged:

### Directional downstream gain

`G(d) = ||J d||_2 / ||d||_2`

### Local task-row-space energy fraction

Let `P_J` be the Euclidean orthogonal projector onto the row space of `J`.

Report:

`E_task(d) = ||P_J d||_2^2 / ||d||_2^2`

This is the exact local fraction of residual energy lying in the
two-coordinate downstream task-sensitive row space at that endpoint.

Report raw per-example distributions and pair/group summaries.

Do not call a direction exactly null unless the observed `J d` is numerically
zero under the recorded computation. Otherwise use the continuous terms
`low-gain`, `task-insensitive`, or `task-sensitive` as warranted by comparisons.

## Prospectively defined orientation controls

A high-dimensional latent map to two centered-logit coordinates generically
has a large kernel. Therefore small task sensitivity of an arbitrary direction
is not itself evidence of selective null alignment.

For each actual residual `d`, construct norm-preserving orientation controls by
applying fixed signed permutations of the hidden dimension independently of
all final-logit outcomes.

Control transformations:

- preserve sequence position;
- preserve each token vector norm exactly;
- use the same transform for all examples for a given control index;
- use deterministic analysis-only RNG seeds recorded in provenance;
- do not use 9601-9900 or any training/evaluation seed namespace.

Use exactly eight signed-permutation controls.

For each control, compute the same `G(d)` and `E_task(d)` without additional
model forwards.

Primary null-alignment contrasts:

- actual A-init residual versus its signed-permutation controls;
- actual A-init residual versus same-A/different-RNG natural residuals.

Report ratios and raw values descriptively. No scientific p-value is
authorized.

## Global hidden-direction task-sensitive subspace

The endpoint-local Jacobian lives in sequence-by-hidden space and answers the
exact local nullness question.

Separately, construct a 768x768 hidden-direction sensitivity covariance by
summing, over all frozen dev examples, valid token positions, cells, and both
centered-logit margin gradients:

`C_task = sum g_t g_t^T`

where `g_t` is the 768-dimensional gradient at a valid token position.

Eigendecompose `C_task`.

Report:

- eigenvalue spectrum;
- participation-ratio effective dimension;
- cumulative sensitivity energy at k in
  `{1,2,4,8,16,32,64,128,256}`.

This is a descriptive task-sensitive hidden-direction geometry, not a universal
intrinsic dimension.

## A-init residual energy in task-sensitive dimensions

For each primary cell pair, accumulate the valid-token hidden-direction
residual covariance:

`C_d = sum d_t d_t^T`

Without using final-logit outcomes to choose k, report for the prospectively
fixed k values:

`Q_k(d) = tr(V_k^T C_d V_k) / tr(C_d)`

where `V_k` contains the top-k eigenvectors of `C_task`.

Report grouped means/ranges for:

1. same A-init, different RNG;
2. same RNG, different A-init;
3. the signed-permutation controls.

This directly tests whether high-dimensional A-init differences collapse,
concentrate, or remain diffuse when viewed in the task-sensitive
hidden-direction subspace.

## Functional authentication

For every frozen cell, final logits from this audit must authenticate against
the already frozen functional-fingerprint evidence within a predeclared
float32 evaluation tolerance.

The audit must not redefine the functional endpoint.

## Prospective interpretation cases

### Selective null / low-gain alignment

If actual different-A residuals have substantially lower directional gain and
task-row-space energy than their norm-preserving controls, support the bounded
interpretation that A-init-dependent solution variation is preferentially
aligned with downstream-insensitive directions.

### Low-dimensional task-visible core

If `C_task` is strongly concentrated and different-A residual energy is small
inside the leading task-sensitive dimensions while final task behavior remains
shared, support a bounded decomposition in which ambient A-init variation is
large but the task-visible coordinate is comparatively stable.

### Generic high-dimensional kernel only

If actual residuals are no less task-sensitive than norm-preserving controls,
then the near-equivalence may be explained by the generic large kernel of the
high-dimensional-to-two-dimensional downstream map rather than selective
A-init null alignment.

### No low-dimensional concentration

If task sensitivity is not concentrated in a small hidden-direction subspace,
do not force a low-dimensional latent-core interpretation.

## Next-step boundary

Only after selective null/low-gain alignment or a stable task-visible subspace
is established may a subsequent stage authorize:

- causal intervention along task-visible versus task-null components;
- upstream localization of the task-visible component;
- precursor analysis across earlier layers, tokens, or recurrent time.

No precursor fishing is authorized in this audit.

## Required artifacts

Write only under:

`reports/reason_router_gen5_ainit_residual_nullness_runs/<run-name>/`

Required files:

- `ainit_residual_nullness_summary.json`
- `task_sensitive_subspace.pt`
- `run_provenance.json`

Do not persist full per-example hidden states or full per-example gradients.

## Runtime constraints

Use the same frozen Mamba snapshot, parent checkpoint, tokenizer/runtime, and
validated two-T4 Kaggle environment as the source evidence.

The audit may use GPU 0 only.

It must:

- use `model.eval()`;
- perform frozen-dev evaluation only;
- use autograd only on detached layer-22 boundary tensors;
- request no parameter gradients;
- construct no optimizer;
- perform no training;
- mutate no checkpoint;
- load no confirmatory data;
- preserve parent and correction checkpoint identities.

## Stop conditions

Stop if:

- source evidence identity mismatches;
- the frozen 3x3 grid is incomplete;
- parent or correction checkpoint identity mismatches;
- frozen dev encoding or row order mismatches;
- layer-22 boundary semantics differ from the frozen wrapper output;
- a parameter gradient is requested or accumulated;
- functional logits fail authentication against frozen fingerprints;
- control generation depends on observed final-logit outcomes;
- training, optimizer construction, or checkpoint mutation would occur;
- confirmatory data would be accessed;
- an output collision exists.

## Result boundary

This audit can characterize local downstream sensitivity, null/low-gain
alignment, and task-sensitive hidden-direction concentration only under the
frozen Phase3A P0 dev contract.

It cannot establish a global nonlinear null manifold, universal latent
controllability, a universal intrinsic task dimension, a precursor mechanism,
or behavior outside the tested contract.
