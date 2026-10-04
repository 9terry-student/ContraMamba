# Gen5 A-init Task-Visible Projection Clustering / Null-Complement Validated Evidence Report

## Evidence identity

- execution authority commit:
  `cc2cd80990f4ef8ae8e2a0350ebaafe3195b93b2`
- source forward-Jacobian recovery evidence freeze:
  `a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e`
- run:
  `gen5-ainit-projection-clustering-cc2cd80-r4`
- run command SHA256:
  `96abb90efe3739de6ee3a49f2193cb3e590589524422819e4be32a194dead04b`
- imported handoff ZIP SHA256:
  `56d8de98d4d4150f718d1b56b3a36e43499025dcdcb869463e0089cb73f6aa23`
- run log SHA256:
  `9920640c177c516165774a8e3c2047a2511b598113927b19a90a431f00aec936`
- run meta SHA256:
  `329ea5f8320a1a9f5296b8a6487adcb9fb0dacc8669b561a8dc54bb37aa0c711`

Imported run-artifact identities:

- `ainit_projection_clustering_summary.json`
  SHA256 `f8e5935b9abd581f9ac6b9353d9ec2f44e4b6f932e194d1bd8846ef829fbe909`
- `projection_clustering_metrics.pt`
  SHA256 `04ea243e13812f694bf78392156defdbc74ccf8e1e1890bcb481450b5ab93aab`
- recovered task-sensitive subspace source
  SHA256 `a8ec07113cf50f302874f797060da194366f87ccee2b6178358396a8960d2197`
- recovery summary source
  SHA256 `74549167ba204a78bc509c1c9b7eec3ce18c129f71b4dbd5698efe2d6845af57`
- frozen functional fingerprint source
  SHA256 `5cf5b69014b632e32aaaad5d38d33e4cf84036d789bc3fe0f7dbcb67f04c0e20`
- parent checkpoint
  SHA256 `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

## Execution and provenance validity

The run completed successfully on the exact authorized commit and frozen
Phase3A P0 dev contract.

Observed execution identity and guards:

- grid cells: 9
- dev rows: 840
- valid tokens: 60,094
- functional-authentication max absolute error:
  `1.43051147461e-06`
- recovered-Q authentication max absolute delta:
  `3.8945629377e-07`
- training executed: false
- optimizer constructed: false
- analysis autograd executed: false
- `.backward()` called: false
- parameter gradients accumulated: false
- checkpoint mutation: false
- confirmatory 9601-9900 loaded: false

The run therefore measures frozen trained-endpoint geometry only. It does not
measure an optimization trajectory.

## Ambient endpoint separation is overwhelmingly A-init controlled

For the same-training-RNG / different-A-init pair class:

- mean full-space squared distance:
  `114.688049255`

For the same-A-init / different-training-RNG pair class:

- mean full-space squared distance:
  `0.268697519737`

Their ratio is:

`426.829579102x`

Thus changing A-init while holding training RNG fixed produces an ambient
layer-22 endpoint separation more than two orders of magnitude larger than
changing training RNG while holding A-init fixed.

The balanced 3x3 factorial endpoint decomposition is consistent with that
pairwise result:

- A-init main-effect fraction:
  `0.997660841249` = `99.7660841249%`
- training-RNG main-effect fraction:
  `0.00155051942365` = `0.155051942365%`
- A x RNG interaction fraction:
  `0.000788703438988` = `0.0788703438988%`
- full-space A/R main-effect energy ratio:
  `643.436532321x`

Under this frozen 3x3 contract, ambient endpoint geometry is therefore
dominated by A-init rather than training RNG.

## Projection into the recovered task-sensitive basis

The same-training-RNG / different-A-init residual is strongly compressed in the
leading recovered task-sensitive coordinates.

Prospectively frozen top-k results:

- k=1: projected residual energy `4.602%`, complement `95.398%`, A-main-effect retention `4.829%`, projected A/R energy ratio `4249.31x`, complement A/R ratio `616.88x`
- k=2: projected residual energy `5.424%`, complement `94.576%`, A-main-effect retention `5.709%`, projected A/R energy ratio `1608.34x`, complement A/R ratio `620.88x`
- k=4: projected residual energy `6.389%`, complement `93.611%`, A-main-effect retention `6.801%`, projected A/R energy ratio `1399.40x`, complement A/R ratio `619.03x`
- k=8: projected residual energy `7.930%`, complement `92.070%`, A-main-effect retention `8.364%`, projected A/R energy ratio `1076.73x`, complement A/R ratio `620.64x`
- k=16: projected residual energy `9.327%`, complement `90.673%`, A-main-effect retention `9.815%`, projected A/R energy ratio `967.79x`, complement A/R ratio `620.79x`
- k=32: projected residual energy `13.200%`, complement `86.800%`, A-main-effect retention `13.661%`, projected A/R energy ratio `896.04x`, complement A/R ratio `615.96x`
- k=64: projected residual energy `20.397%`, complement `79.603%`, A-main-effect retention `20.949%`, projected A/R energy ratio `822.99x`, complement A/R ratio `608.27x`
- k=128: projected residual energy `30.497%`, complement `69.503%`, A-main-effect retention `31.331%`, projected A/R energy ratio `762.81x`, complement A/R ratio `600.56x`
- k=256: projected residual energy `48.837%`, complement `51.163%`, A-main-effect retention `50.092%`, projected A/R energy ratio `707.34x`, complement A/R ratio `589.94x`

These values establish two simultaneous facts.

First, most of the A-init-induced ambient separation lies outside the leading
task-sensitive coordinates. For example:

- top-2 retains only about `5.42%` of residual squared energy;
- top-8 retains about `7.93%`;
- top-16 retains about `9.33%`;
- top-64 retains about `20.40%`;
- even top-256 retains only about `48.84%`.

Equivalently, the corresponding complements retain approximately:

- top-2 complement: `94.58%`;
- top-8 complement: `92.07%`;
- top-16 complement: `90.67%`;
- top-64 complement: `79.60%`;
- top-256 complement: `51.16%`.

Second, the task-visible component is not random or negligible in structure.
Within every reported leading task-sensitive subspace, the A-init factorial
main effect remains dramatically larger than the training-RNG main effect.

Therefore projection suppresses A-init separation strongly in absolute
energy, but it does not erase A-init identity from task-visible coordinates.

## Relation to the prior local-Jacobian result

The prior recovered true-forward-Jacobian audit established that the exact
local two-margin Jacobian row space captured only approximately `0.416%` of
same-training-RNG / different-A-init residual squared energy.

This projection-clustering audit asks a broader question using a global
task-sensitive hidden basis accumulated over the frozen dev population.

Accordingly, the present top-k retained-energy fractions are larger than
`0.416%` and must not be conflated with the local two-margin row-space result.

The two findings are compatible:

1. immediate local visibility to the exact two centered-margin rows is very
   small;
2. a broader globally recovered task-sensitive hidden subspace captures a
   larger, still minority, portion of the A-init separation.

## Scientific interpretation

The validated endpoint evidence supports the bounded statement:

`GEN5_A_INIT_LAYER22_ENDPOINT_GEOMETRY_IS_DOMINATED_BY_A_INIT_SPECIFIC_NULL_OR_LOW_GAIN_VARIATION_WITH_A_SMALLER_BUT_STRONGLY_STRUCTURED_TASK_VISIBLE_A_INIT_COMPONENT`

A useful descriptive decomposition is:

`h_(A,R) = q + v_A + n_A + epsilon_R + epsilon_(A,R)`

where:

- `q` denotes a shared task-relevant reference component;
- `v_A` denotes the smaller A-init-specific component that survives projection
  into the recovered task-sensitive basis;
- `n_A` denotes the dominant A-init-specific ambient component lying in the
  tested task-sensitive complement / low-gain region;
- `epsilon_R` denotes the much smaller training-RNG main effect under this
  frozen grid;
- `epsilon_(A,R)` denotes the small A x RNG interaction.

This decomposition is descriptive and local to the tested layer-22 boundary
and frozen Phase3A P0 dev contract.

## What the result does and does not establish

The result supports:

- A-init is the dominant source of trained layer-22 endpoint variation in this
  3x3 causal grid;
- most A-init-induced squared-distance energy lies outside the leading
  recovered task-sensitive coordinates;
- the surviving task-visible A-init component is highly structured rather than
  disappearing completely;
- nearly functionally equivalent final behavior can coexist with very large
  A-init-specific ambient hidden-state separation.

The result does not establish:

- that all A-init solutions collapse to one identical task-visible latent point;
- that the entire A-init residual is mathematically null;
- a universal null manifold;
- a universal intrinsic task dimension;
- that different A-init conditions followed different optimization
  trajectories during training;
- causal invariance to arbitrary interventions in the complement;
- behavior outside the frozen Phase3A P0 dev population.

In particular, the original intuition that A-init may select different
representatives largely along task-null / low-gain directions is strongly
supported at the endpoint level, but the stronger statement that task-visible
coordinates are A-init invariant is falsified by the retained structured
A-init main effect.

## Consequence for the next experiment

The sharp next scientific question is causal rather than descriptive:

> Can the A-init-specific complement component be exchanged or removed at the
> layer-22 boundary with little final-output effect, while the task-visible
> component accounts for the measurable functional difference?

The next stage should therefore be a bounded frozen-dev visible-vs-complement
intervention audit.

It should remain:

- no training;
- no optimizer;
- no parameter update;
- no confirmatory 9601-9900 population;
- frozen 3x3 endpoint checkpoints only;
- layer-22 intervention only;
- prospectively fixed top-k values;
- explicit functional-authentication and provenance guards.

Upstream precursor localization or training-trajectory measurement should not
begin before this causal visible-vs-complement intervention is resolved.
