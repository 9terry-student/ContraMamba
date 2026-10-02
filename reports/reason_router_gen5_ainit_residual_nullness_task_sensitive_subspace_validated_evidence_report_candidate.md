# Gen5 A-init Residual Nullness and Task-Sensitive Subspace Validated Evidence Report Candidate

## Status

VALIDATED_AINIT_RESIDUAL_NULLNESS_AND_TASK_SENSITIVE_SUBSPACE_EVIDENCE_CANDIDATE

## Evidence identity

Nullness / task-sensitive-subspace authority commit:

`b8b1a5e95c7932df2c0319e766d10b10f19b3081`

Source residual-propagation evidence freeze:

`3c0a3d8a67e9910f91de2354ba29a5c4b3b28942`

Source residual-localization execution:

`69473ac76629ef1e7c78b45c8c802a89e1e9695b`

Source task-reachable quotient evidence freeze:

`0e6191fd54e23388abcce1abd9e01a453d2dc73c`

Source functional-equivalence evidence freeze:

`5f079a66f7b0eb0caea30a8d5bc9a0fe757cc449`

Run:

`gen5-ainit-residual-nullness-b8b1a5e-r1`

Pinned command SHA256:

`9420b16246a974d1a03bf077d051a9791180c70e2bc83b9a38896350b9b67fa7`

Imported ZIP SHA256:

`7e711d2a26f12d79a148805cc2262cd7d074156439ba754564dfe3793ef71890`

Run log SHA256:

`ecb481fecb292d70eaf0dc8728811119e795a044d00db09d1bd2ed6e67b30b87`

Run meta SHA256:

`cc294bde6022d1ced9885b3df4115cf24e2829f0d45a403474072cf7e0e07ad3`

Collector status:

`PASS`

Import status:

`PASS`

Validated imported files:

`3`

## Execution boundary

Frozen Phase3A P0 dev contract:

- arm: `G5-C0`
- pressure: `P0`
- dev rows: `840`
- split seed: `16384`
- frozen 3x3 A-init x training-RNG grid

Analysis boundary:

`OUTPUT_OF_PHASE2_LAYER22_MIXER_WRAPPER`

Task output coordinates:

- `refute - not_entitled`
- `support - not_entitled`

Analysis gradients were obtained only with `torch.autograd.grad` from a detached
layer-22 boundary leaf.

No `.backward()` call, optimizer construction, parameter-gradient accumulation,
training, checkpoint mutation, or confirmatory 9601-9900 access occurred.

Functional authentication against the frozen functional-fingerprint artifact
passed with maximum absolute error:

`1.43051147461e-06`

## Task-sensitive hidden-direction spectrum

The 768-dimensional hidden-direction downstream-sensitivity covariance was
strongly concentrated.

Participation-ratio effective dimension:

`3.05342209203`

Cumulative sensitivity energy:

- top 1: `0.538847923826`
- top 2: `0.711126713964`
- top 4: `0.810986652685`
- top 8: `0.892193205233`
- top 16: `0.937394356294`
- top 32: `0.963580052658`
- top 64: `0.980762752268`

This is descriptive evidence of a strongly low-dimensional downstream-sensitive
hidden-direction geometry under the frozen dev contract.

It does not establish a universal intrinsic dimension.

## Same training RNG, different A-init: local task sensitivity

Mean actual directional downstream gain:

`0.000152554560786`

Mean signed-permutation-control directional gain:

`0.0000172633763458`

Actual / control directional-gain ratio:

`9.82203214456`

Mean actual local task-row-space residual-energy fraction:

`0.00415859952355`

Mean control task-row-space residual-energy fraction:

`0.0000814704305204`

Actual / control task-row-space-energy ratio:

`57.1938263949`

The absolute row-space energy fraction is small, as expected for a very
high-dimensional layer-22 boundary mapped to two centered-logit coordinates.

However, actual A-init residuals are far more aligned with the local
task-sensitive row space than norm-preserving signed-permutation controls.

Therefore the different-A residual is not preferentially aligned with a generic
downstream null direction.

## Same training RNG, different A-init: projection into global task-sensitive hidden directions

Fraction of actual residual energy captured by the top-k task-sensitive hidden
directions, compared with signed-permutation controls:

### k = 1

Actual:

`0.0460206877364`

Control:

`0.00109887453231`

Actual / control:

`46.0684302505`

### k = 2

Actual:

`0.0542434745167`

Control:

`0.00227799191647`

Actual / control:

`24.8278863611`

### k = 4

Actual:

`0.0638893185295`

Control:

`0.00457903513175`

Actual / control:

`14.2826366702`

### k = 8

Actual:

`0.0792970771805`

Control:

`0.00974133145543`

Actual / control:

`8.20839126673`

### k = 16

Actual:

`0.0932692040846`

Control:

`0.0200494849267`

Actual / control:

`4.66127190039`

### k = 32

Actual:

`0.131998949128`

Control:

`0.0428095052971`

Actual / control:

`3.08727910735`

### k = 64

Actual:

`0.203966361211`

Control:

`0.086435849137`

Actual / control:

`2.35914873915`

Thus different-A residual energy is selectively enriched, not depleted, in the
directions to which the downstream task map is most sensitive.

## Same A-init, different training RNG control comparison

Actual / signed-permutation-control directional-gain ratio:

`3.36068797722`

Actual / signed-permutation-control local task-row-space-energy ratio:

`8.57639813575`

Fixed-A residuals are also more task-aligned than purely orientation-randomized
controls, but the enrichment is substantially weaker than for changing A
initialization.

## Primary validated result

The simple flat-null hypothesis is not supported.

The data do not support the interpretation:

`A-init -> arbitrary large null-space displacement -> same task-visible state`

Instead, under the frozen contract:

1. downstream task sensitivity is highly concentrated in a small number of
   hidden directions;
2. different-A residuals are selectively enriched in those task-sensitive
   directions relative to norm-preserving orientation controls;
3. nevertheless, final centered-logit behavior remains nearly equivalent across
   A-init cells.

Therefore the large A-init-dependent latent difference cannot be explained
merely as variation that avoids the downstream task-sensitive geometry.

## Geometric implication

Let the frozen downstream centered-logit map from the layer-22 boundary be
`F(h)`.

The frozen functional evidence establishes approximately:

`F(h_i) ~= F(h_j)`

for different-A solutions.

The present audit establishes that the chord:

`d = h_i - h_j`

is not preferentially aligned with the local kernel of `F` at the endpoints.

Therefore endpoint functional equivalence does not imply that the connecting
chord is a local null direction.

A natural next hypothesis is nonlinear / curved equivalence geometry:

`F(h_j) - F(h_i) = integral_0^1 J(h_i + alpha d) d d(alpha)`

may remain small because directional downstream effects vary and cancel along
the path, even though `J(h_i)d` and `J(h_j)d` are non-negligible.

This is a hypothesis for the next audit, not an established result.

## Supported bounded conclusions

`GEN5_DOWNSTREAM_TASK_SENSITIVITY_AT_THE_LAYER22_BOUNDARY_IS_STRONGLY_LOW_DIMENSIONAL_UNDER_THE_FROZEN_PHASE3A_P0_DEV_CONTRACT`

`GEN5_DIFFERENT_A_INIT_LAYER22_RESIDUALS_ARE_MORE_TASK_SENSITIVE_THAN_NORM_PRESERVING_ORIENTATION_CONTROLS`

`GEN5_DIFFERENT_A_INIT_RESIDUALS_ARE_SELECTIVELY_ENRICHED_IN_LEADING_TASK_SENSITIVE_HIDDEN_DIRECTIONS`

`GEN5_SIMPLE_GENERIC_NULL_SPACE_VARIATION_DOES_NOT_EXPLAIN_THE_OBSERVED_A_INIT_FUNCTIONAL_EQUIVALENCE`

`GEN5_ENDPOINT_FUNCTIONAL_EQUIVALENCE_COEXISTS_WITH_A_NON_NULL_ALIGNED_INTER_SOLUTION_CHORD_UNDER_THE_TESTED_LOCAL_SENSITIVITY_MEASURE`

## What is not established

The present evidence does not establish:

- a global nonlinear equivalence manifold;
- that pathwise Jacobian effects cancel between different-A endpoints;
- the shape or curvature of any equivalence set;
- that the top task-sensitive directions are causally controllable;
- that a stable low-dimensional task-visible coordinate is identical across
  different-A solutions;
- that a task-sensitive component can be manipulated independently of the
  surrounding representation;
- any upstream precursor;
- behavior outside the frozen Phase3A P0 dev distribution;
- confirmatory-seed behavior;
- a universal Mamba mechanism.

## Next scientific action

Perform a frozen latent interpolation / Jacobian path-integral audit between
same-training-RNG / different-A layer-22 boundary endpoints.

For each primary pair and each dev example, prospectively evaluate a fixed
interpolation grid:

`h(alpha) = (1-alpha) h_i + alpha h_j`

with alpha values chosen before execution and shared across all pairs.

At each interpolation point measure:

1. centered logits;
2. deviation from linear interpolation of endpoint centered logits;
3. directional derivative `J(h(alpha)) d`;
4. cumulative numerical path integral of `J(h(alpha)) d`;
5. sign / directional reversals of the two centered-logit margin derivatives;
6. maximum functional excursion away from both endpoints.

Primary question:

> Are functionally near-equivalent A-init endpoints connected by a chord along
> which the downstream task response leaves the endpoint neighborhood and
> returns, consistent with curved nonlinear equivalence geometry rather than a
> flat null direction?

The interpolation audit should include natural same-A/different-RNG controls and
must use no new training or confirmatory population.

Only after this path geometry is resolved should the research proceed to causal
latent intervention or precursor localization.
