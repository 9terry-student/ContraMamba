# Gen5 A-init Task-Visible vs Complement Causal Intervention Audit Authority

SOURCE_PROJECTION_CLUSTERING_EVIDENCE_FREEZE_COMMIT=a82ca54f331274c7037368f2c458bcd5dedbeae9
SOURCE_FORWARD_JACOBIAN_RECOVERY_EVIDENCE_FREEZE_COMMIT=a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e
SOURCE_PROJECTION_CLUSTERING_EXECUTION_COMMIT=cc2cd80990f4ef8ae8e2a0350ebaafe3195b93b2

STATUS=READY_FOR_GEN5_AINIT_VISIBLE_COMPLEMENT_CAUSAL_INTERVENTION_AUDIT

TRAINING_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
PARAMETER_GRADIENT_UPDATE_ALLOWED=NO
ANALYSIS_AUTOGRAD_ALLOWED=NO
BACKWARD_METHOD_ALLOWED=NO
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
CUDA_EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_P0_DEV_ONLY

## Scientific question

The frozen projection-clustering evidence established that A-init explains
nearly all ambient layer-22 endpoint variation in the 3x3 grid, while most of
that A-init-induced squared-distance energy lies outside the leading recovered
task-sensitive hidden coordinates.

The remaining question is causal:

> Does the A-init-specific complement component actually have little downstream
> functional effect when transplanted between real frozen endpoints, while the
> smaller task-visible component carries most of the measurable endpoint
> functional difference?

This audit is the decisive test of whether the observed endpoint decomposition
is merely descriptive geometry or behaves like a downstream gauge-like /
non-identifiable representation degree of freedom.

No optimization trajectory claim is authorized.

## Frozen inputs

Use only:

- source evidence freeze
  `a82ca54f331274c7037368f2c458bcd5dedbeae9`;
- the same frozen Phase3A P0 dev population, 840 rows, split seed 16384;
- the same exact 3x3 A-init x training-RNG grid with seeds
  `{6201,6202,6203}`;
- the exact nine frozen correction checkpoints authenticated by the source
  projection-clustering evidence;
- the exact recovered task-sensitive basis from the frozen true-forward-
  Jacobian recovery evidence;
- the same parent checkpoint, tokenizer/runtime snapshot, and functional
  fingerprint.

No new scientific seed is authorized.

## Primary pair class

Primary causal pairs are the exactly nine:

- same training RNG;
- different A-init.

The exactly nine same-A-init / different-training-RNG pairs are required as a
descriptive control class.

The remaining 18 different-A / different-RNG pairs may be reported as context
but must not drive the primary conclusion.

## Intervention boundary

Intervene only at the output of `Phase2Layer22MixerWrapper.forward`.

For a real frozen endpoint pair `(i,j)` with aligned valid-token layer-22
states:

`d = h_j - h_i`

For each prospectively frozen leading task-sensitive basis `V_k`, define:

`P_k = V_k V_k^T`

`d_visible = P_k d`

`d_complement = d - d_visible`

and the two bounded hybrid states:

`h_visible = h_i + d_visible`

`h_complement = h_i + d_complement`

The four corners are therefore:

- `h_i` : real endpoint i;
- `h_j` : real endpoint j;
- `h_visible` : only the top-k visible part moved from i toward j;
- `h_complement` : only the orthogonal-complement part moved from i toward j.

These are affine hybrids defined by the exact real endpoint chord. Do not add
arbitrary random state perturbations in this stage.

For symmetry, repeat the same decomposition with j as the source and i as the
target, or equivalently authenticate that the reverse intervention produces
the same two affine hybrid corners within numerical precision.

## Prospectively fixed k values

Use exactly:

`k = {1,2,4,8,16,32,64,128,256}`

Do not select k from intervention outcomes.

For interpretation, `k=8` is the primary compact task-visible reference because
the already frozen recovered spectrum assigns approximately 89.2% cumulative
task-sensitivity energy to the leading eight directions while the source
projection-clustering evidence assigns only approximately 7.93% of the
same-RNG/different-A residual energy to those directions.

All k values remain required to establish robustness rather than relying on
the primary reference alone.

## Downstream functional readout

From each real or hybrid layer-22 state, run only the frozen downstream
computation to the final logits.

Use centered three-class logits and the two frozen task margins:

- `margin_refute = logit_refute - logit_not_entitled`;
- `margin_support = logit_support - logit_not_entitled`.

No gradient is required.

For each pair and k, define endpoint functional difference:

`Delta_full = F(h_j) - F(h_i)`

visible-only intervention effect:

`Delta_visible = F(h_visible) - F(h_i)`

complement-only intervention effect:

`Delta_complement = F(h_complement) - F(h_i)`

and nonlinear interaction residual:

`Delta_interaction = Delta_full - Delta_visible - Delta_complement`

where `F` is evaluated separately for centered logits and for the two-margin
vector.

## Required aggregate metrics

For each k and pair class report aggregate squared effect energy over the full
frozen dev population:

- `E_full = sum ||Delta_full||^2`;
- `E_visible = sum ||Delta_visible||^2`;
- `E_complement = sum ||Delta_complement||^2`;
- `E_interaction = sum ||Delta_interaction||^2`.

Report stable aggregate ratios:

- `R_visible = E_visible / E_full`;
- `R_complement = E_complement / E_full`;
- `R_interaction = E_interaction / E_full`.

Do not use mean per-example ratios when the endpoint denominator can be tiny.

Also report:

- absolute centered-logit RMS effect;
- absolute two-margin RMS effect;
- prediction disagreement count versus the source endpoint;
- prediction disagreement count versus the target endpoint;
- final 3-way CE change relative to the source endpoint;
- endpoint-to-hybrid and hybrid-to-target symmetry diagnostics.

## Required additivity / path diagnostics

For each k, report both affine two-step orders:

1. visible-first:
   `h_i -> h_visible -> h_j`;
2. complement-first:
   `h_i -> h_complement -> h_j`.

Measure downstream effect energy on both legs.

This distinguishes:

- a genuinely low-effect complement;
- a visible component that carries the endpoint functional difference;
- strong nonlinear interaction between visible and complement coordinates.

No path-integral Jacobian calculation is authorized in this stage.

## Functional authentication

Before interpreting interventions:

1. authenticate all nine real frozen endpoint logits against the frozen
   functional fingerprint under the established float32 tolerance;
2. authenticate the recovered basis identity and source artifact hashes;
3. authenticate all nine correction-checkpoint SHA256 identities;
4. authenticate valid-token count and frozen dev row order.

Stop on any identity mismatch.

## Primary interpretation gates

### Gauge-like / downstream-non-identifiable support

Strong support requires, across the primary same-RNG/different-A pairs and a
stable range of k values including k=8:

- complement-only intervention output effect is small relative to the real
  endpoint functional difference;
- visible-only intervention accounts for most of the measurable endpoint
  centered-logit / margin difference;
- nonlinear interaction is not large enough to explain away that separation;
- endpoint predictions remain largely invariant under complement-only swaps.

This would support the bounded statement:

`A_INIT_SELECTS_DIFFERENT_LAYER22_REPRESENTATIVES_PRIMARILY_ALONG_DOWNSTREAM_LOW_GAIN_OR_EFFECTIVELY_NULL_DIRECTIONS_WHILE_A_SMALL_VISIBLE_COMPONENT_CARRIES_THE_MEASURABLE_FUNCTIONAL_DIFFERENCE`

Do not call this an exact mathematical gauge symmetry unless the intervention
evidence is correspondingly exact.

### Complement is causally important

If complement-only intervention produces output effects comparable to
`Delta_full`, changes many predictions, or dominates margin changes, the simple
null/complement interpretation is falsified.

The likely alternatives would be:

- the recovered linear task-sensitive basis is incomplete for finite
  interventions;
- strong nonlinear downstream coupling makes nominally low-gain directions
  causally important;
- endpoint equivalence depends on coupled visible/complement coordinates.

### Strong nonlinear interaction

If both isolated interventions are small but the interaction residual is
large, do not claim independent visible and null components.

The correct result would be a coupled nonlinear equivalence geometry.

## Required artifacts

Write only under:

`reports/reason_router_gen5_ainit_visible_complement_intervention_runs/<run-name>/`

Required files:

- `ainit_visible_complement_intervention_summary.json`
- `visible_complement_intervention_metrics.pt`
- `run_provenance.json`

The metrics artifact may contain compact per-example functional-effect
aggregates required for verification, but must not contain full hidden-state
dumps.

## Runtime constraints

Use the exact validated two-T4 Kaggle environment and frozen runtime identities
from the source execution.

- model eval mode only;
- GPU 0 only within the validated 2xT4 environment;
- frozen-dev execution only;
- no autograd;
- no backward;
- no optimizer;
- no training;
- no parameter update;
- no checkpoint mutation;
- no confirmatory population;
- no new scientific seed;
- no outcome-conditioned k selection.

## Scientific boundary

This audit can establish whether the layer-22 endpoint complement is causally
low-effect under bounded endpoint-to-endpoint affine interventions.

It cannot establish:

- the actual optimization trajectory;
- where during training the A-init-specific component first arose;
- a universal gauge symmetry;
- a global nonlinear manifold of equivalent states;
- OOD invariance;
- safety of arbitrary complement perturbations.

If this audit supports the gauge-like interpretation, the next meaningful
A-init question is upstream precursor localization: identify the earliest
layer where A-init-specific complement separation emerges while task-visible
function remains stable.

If this audit fails, precursor localization should be deferred until the
finite-intervention coupling mechanism is understood.
