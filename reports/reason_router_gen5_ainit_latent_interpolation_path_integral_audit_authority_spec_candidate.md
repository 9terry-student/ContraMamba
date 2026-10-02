# Gen5 A-init Latent Interpolation and Jacobian Path-Integral Audit Authority

SOURCE_TASK_SENSITIVITY_EVIDENCE_FREEZE_COMMIT=ed196d5003f279dbfe1dc9a631e84e22dc049ac7
SOURCE_TASK_SENSITIVITY_EXECUTION_COMMIT=b8b1a5e95c7932df2c0319e766d10b10f19b3081
SOURCE_RESIDUAL_LOCALIZATION_EVIDENCE_FREEZE_COMMIT=3c0a3d8a67e9910f91de2354ba29a5c4b3b28942
SOURCE_FUNCTIONAL_EQUIVALENCE_EVIDENCE_COMMIT=5f079a66f7b0eb0caea30a8d5bc9a0fe757cc449

STATUS=READY_FOR_AINIT_LATENT_INTERPOLATION_PATH_INTEGRAL_AUDIT

TRAINING_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
PARAMETER_GRADIENT_UPDATE_ALLOWED=NO
ANALYSIS_AUTOGRAD_ALLOWED=YES_DOWNSTREAM_LAYER22_BOUNDARY_ONLY
BACKWARD_METHOD_ALLOWED=TORCH_AUTOGRAD_GRAD_ONLY
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
LATENT_INTERVENTION_ALLOWED=YES_LAYER22_CHORD_INTERPOLATION_ONLY
CUDA_EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_P0_DEV_ONLY
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## Scientific question

The frozen task-sensitivity evidence established simultaneously that:

1. the downstream hidden-direction sensitivity geometry is strongly
   low-dimensional under the frozen Phase3A P0 dev contract;
2. same-training-RNG / different-A-init layer-22 residuals are more aligned
   with downstream-sensitive directions than norm-preserving orientation
   controls;
3. the different-A endpoints nevertheless remain nearly functionally
   equivalent at final centered logits.

Therefore the simple explanation

`different A-init -> generic flat downstream null displacement -> same output`

is not supported.

This audit asks whether the two functionally near-equivalent endpoints are
connected by a non-null latent chord whose downstream effect changes along the
path, producing nonlinear excursion and/or cancellation.

## Frozen population and grid

Use only:

DEV_ROWS=840
SPLIT_SEED=16384
ARM=G5-C0
PRESSURE=P0

Use the exact frozen 3x3 A-init x training-RNG grid:

A_INIT_SEED in {6201,6202,6203}
TRAINING_RNG_SEED in {6201,6202,6203}

Primary pairs:

all 9 unordered pairs with the same training RNG and different A-init.

Natural controls:

all 9 unordered pairs with the same A-init and different training RNG.

No new scientific seed, checkpoint, training population, confirmatory
population, rank, optimizer, architecture, or learned parameter is authorized.

## Layer-22 boundary semantics

For each frozen cell and dev chunk, capture the exact output tensor returned by:

`Phase2Layer22MixerWrapper.forward`

For a pair of endpoints `h0` and `h1`, define:

`d = h1 - h0`

and the straight latent chord:

`h(alpha) = h0 + alpha * d`

The intervention replaces only the layer-22 wrapper output with `h(alpha)`.

All computation before the wrapper remains frozen and common.

All computation after the wrapper remains the exact frozen downstream model.

The interpolation is an analysis path in ambient layer-22 latent space. It is
not assumed to lie on the trained-data manifold or on an exact equivalence
manifold.

## Prospective interpolation grid

Use exactly:

`alpha = {0, 1/6, 2/6, 3/6, 4/6, 5/6, 1}`

for every primary and control pair.

No alpha location may be chosen after inspecting outcomes.

Endpoint alpha values must authenticate the already frozen cell outputs.

## Task output coordinates

Use the same two independent centered-logit margin coordinates as the frozen
task-sensitivity audit:

`m_refute = logit_refute - logit_not_entitled`

`m_support = logit_support - logit_not_entitled`

Let:

`m(alpha) = [m_refute(alpha), m_support(alpha)]`

For each alpha, detach the substituted layer-22 boundary as the only analysis
leaf and compute with `torch.autograd.grad`:

`D(alpha) = J(h(alpha)) d`

where `J` is the downstream Jacobian of the two margin coordinates.

Do not call `.backward()`.

Do not request or accumulate parameter gradients.

## Primary path endpoints

For each dev example and pair report:

### Endpoint chord size

`E = ||m(1) - m(0)||_2`

### Nonlinear deviation from endpoint interpolation

Define the linear endpoint interpolation:

`L(alpha) = (1-alpha)m(0) + alpha*m(1)`

and:

`N(alpha) = ||m(alpha) - L(alpha)||_2`

Report maximum and mean interior `N(alpha)`.

### Functional excursion

Report:

`X(alpha) = min(||m(alpha)-m(0)||_2, ||m(alpha)-m(1)||_2)`

and maximum interior excursion.

Also report excursion normalized by:

`max(E, 1e-8)`

as a descriptive ratio only.

Large normalized ratios may arise when endpoint differences are extremely
small, so always retain the raw margin-space excursion.

## Jacobian path-integral endpoints

Numerically integrate the sampled directional derivative with the composite
trapezoid rule:

`I = integral_0^1 D(alpha) d(alpha)`

Compare `I` with the exact endpoint margin difference:

`Delta = m(1)-m(0)`

Report the quadrature residual:

`||I - Delta||_2`

This residual is an interpolation-grid adequacy diagnostic. It is not a
scientific endpoint.

## Pathwise cancellation

Define sampled directional total variation:

`TV = integral_0^1 ||D(alpha)||_2 d(alpha)`

using the same trapezoid grid.

Define:

`C = 1 - ||I||_2 / max(TV, 1e-12)`

clamped descriptively to `[0,1]`.

`C` near zero indicates little vector cancellation along the sampled chord.

Large `C` indicates that local downstream effects along the chord point in
different directions and substantially cancel in the path integral.

## Derivative reversal

For each example and each of the two margin coordinates, define:

`tau = max(1e-10, 1e-4 * max_alpha |D_coordinate(alpha)|)`

A coordinate has a sampled sign reversal only if some alpha has derivative
greater than `tau` and another alpha has derivative less than `-tau`.

Report:

- fraction of examples with a reversal in either coordinate;
- fraction with reversals in both coordinates;
- per-coordinate reversal fractions.

## Functional authentication

Endpoint alpha values must authenticate the already frozen functional
fingerprint evidence within the same predeclared float32 tolerance used by the
task-sensitivity audit.

The run must fail closed if endpoint logits do not reproduce the frozen
functional endpoint.

## Primary comparisons

Report all path metrics separately for:

1. same training RNG / different A-init;
2. same A-init / different training RNG.

The primary scientific contrast is whether different-A paths show stronger
nonlinearity, excursion, or pathwise cancellation than the natural fixed-A
training-RNG controls.

No scientific p-value is authorized.

## Prospective interpretation cases

### Flat or near-flat corridor

If primary paths have small raw nonlinear deviation, small excursion, low
cancellation, and stable directional derivatives, then the endpoint
task-sensitivity enrichment does not imply a strongly curved path.

Do not claim curved equivalence geometry.

### Nonlinear return path

If different-A endpoints are near-equivalent but their interpolation paths
show substantial interior margin excursion and/or strong derivative
cancellation/reversal, support the bounded interpretation that endpoint
functional equivalence coexists with a non-null chord through nonlinear
downstream geometry.

### Generic nonlinearity

If same-A/different-RNG controls show comparable path behavior, do not attribute
the effect specifically to A initialization.

### A-init-specific nonlinear equivalence geometry

If different-A paths show materially stronger excursion/cancellation than
same-A controls while endpoints remain near-equivalent, support a bounded
A-init-specific nonlinear-equivalence interpretation under the frozen dev
contract.

## Required artifacts

Write only under:

`reports/reason_router_gen5_ainit_latent_interpolation_runs/<run-name>/`

Required files:

- `ainit_latent_interpolation_summary.json`
- `pathwise_metrics.pt`
- `run_provenance.json`

Do not persist full hidden-state trajectories or full Jacobian tensors.

## Runtime constraints

Use the same frozen Mamba snapshot, parent checkpoint, tokenizer/runtime, and
validated two-T4 Kaggle environment as the source evidence.

The audit may use GPU 0 only.

It must:

- use `model.eval()`;
- perform frozen-dev evaluation only;
- interpolate only at the layer-22 wrapper output;
- use autograd only on the detached substituted boundary;
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
- endpoint functional authentication fails;
- an interpolation alpha outside the frozen grid would be used;
- a parameter gradient is requested or accumulated;
- training, optimizer construction, or checkpoint mutation would occur;
- confirmatory data would be accessed;
- an output collision exists.

## Result boundary

This audit can characterize the downstream response along the straight
layer-22 chord between frozen endpoints.

It cannot establish:

- that the straight chord is a natural model trajectory;
- a global equivalence manifold;
- global manifold curvature;
- universal latent controllability;
- a precursor mechanism;
- behavior outside the frozen Phase3A P0 dev distribution.

Only after the path geometry is resolved may a later stage authorize targeted
task-sensitive latent intervention or upstream precursor localization.
