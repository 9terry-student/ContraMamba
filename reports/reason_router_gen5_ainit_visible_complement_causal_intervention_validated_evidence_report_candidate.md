# Gen5 A-init Visible-vs-Complement Causal Intervention Validated Evidence Report

## Evidence identity

- execution authority commit:
  `4a973b0725b8fe9eebc8c07950457e58869e10b3`
- source projection-clustering evidence freeze:
  `a82ca54f331274c7037368f2c458bcd5dedbeae9`
- source forward-Jacobian recovery evidence freeze:
  `a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e`
- run:
  `gen5-ainit-visible-complement-intervention-4a973b0-r1`
- run command SHA256:
  `26f1259ab098b18451e2b06ea058975cc6520bc48c3b33ddc0872b2878b13cf1`
- imported handoff ZIP SHA256:
  `8b39aac06d01f7b294e76ff158738798349141aa88d53a685d4124c434917e62`
- run log SHA256:
  `9e97971715a44f93142c19d803c975d9cace9a7244669381a2433746e54d939a`
- run meta SHA256:
  `988410c5ceff5ad333ebce5c1531dedbe26c25127c7d157219f9190fe441eda9`

Imported artifact identities:

- `ainit_visible_complement_intervention_summary.json`
  SHA256 `5f47cb6170e9c33ddb5f17911f1035122511baeb079f647ffddc44e702f85a3c`
- `visible_complement_intervention_metrics.pt`
  SHA256 `7ae9a1e1d18b4da5447afd066e16a117936aa85fbac022ab9a4eee6496a98e12`

## Execution and provenance validity

The run completed successfully on the exact authorized commit and frozen
Phase3A P0 dev contract.

Observed execution identity:

- result:
  `PASS_GEN5_AINIT_VISIBLE_COMPLEMENT_CAUSAL_INTERVENTION_AUDIT`
- dev rows: `840`
- valid tokens: `60,094`
- executed pairs: `18`
- primary same-training-RNG / different-A-init pairs: `9`
- control same-A-init / different-training-RNG pairs: `9`
- functional-authentication max absolute error:
  `1.43051147460938e-06`
- source projection-Q authentication max absolute delta:
  `3.98642385074366e-07`

Execution guardrails:

- analysis autograd executed: false
- `.backward()` called: false
- parameter gradients accumulated: false
- optimizer constructed: false
- training executed: false
- checkpoint mutation: false
- confirmatory 9601-9900 loaded: false

Both primary and control real-endpoint pair classes have zero prediction
disagreement between their frozen endpoints.

## Primary causal intervention result

For each same-training-RNG / different-A-init endpoint pair and each prospectively
fixed `k`, the actual endpoint chord was decomposed at layer 22 into:

`d = d_visible + d_complement`

with:

`d_visible = P_k d`

`d_complement = (I - P_k) d`

The downstream centered-logit and two-margin effects were then measured after
applying the visible-only or complement-only component.

The aggregate effect-energy ratios relative to the real endpoint functional
difference are:

| k | hidden Q_k | centered R_visible | centered R_complement | centered R_interaction | margin R_visible | margin R_complement | margin R_interaction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.046021 | 0.706552 | 0.036866 | 0.001918 | 0.708908 | 0.036308 | 0.001879 |
| 2 | 0.054243 | 0.835127 | 0.012282 | 0.002043 | 0.835977 | 0.012160 | 0.002011 |
| 4 | 0.063889 | 1.048402 | 0.001436 | 0.000158 | 1.049422 | 0.001469 | 0.000160 |
| 8 | 0.079297 | 1.012775 | 0.001069 | 0.000284 | 1.014452 | 0.001109 | 0.000295 |
| 16 | 0.093269 | 0.998622 | 0.000806 | 0.000289 | 1.000524 | 0.000843 | 0.000312 |
| 32 | 0.131999 | 1.017742 | 0.000602 | 0.000197 | 1.018563 | 0.000625 | 0.000208 |
| 64 | 0.203966 | 1.013302 | 0.000237 | 0.000066 | 1.013546 | 0.000242 | 0.000068 |
| 128 | 0.304966 | 1.010841 | 0.000106 | 0.000027 | 1.010954 | 0.000108 | 0.000027 |
| 256 | 0.488375 | 1.002749 | 0.000015 | 0.000003 | 1.002710 | 0.000015 | 0.000003 |

The decisive pattern is stable from `k=4` onward.

At the primary compact reference `k=8`:

- hidden residual energy retained in the visible projection:
  `7.9297%`
- centered-logit visible-only effect ratio:
  `1.012775`
- centered-logit complement-only effect ratio:
  `0.00106868`
- centered-logit nonlinear interaction ratio:
  `0.00028408`
- two-margin visible-only effect ratio:
  `1.014452`
- two-margin complement-only effect ratio:
  `0.00110852`
- two-margin nonlinear interaction ratio:
  `0.00029534`

Thus a component containing only about `7.93%` of the layer-22 A-init residual
squared energy reproduces essentially the entire measurable endpoint functional
difference, while the complement containing about `92.07%` of the hidden
residual energy contributes only about `0.1%` of the endpoint output-effect
energy.

The slight `R_visible > 1` overshoot is compatible with the small but nonzero
complement and nonlinear interaction terms and should not be interpreted as a
violation of the decomposition.

## Robustness across k

The result is not an isolated `k=8` effect.

For centered logits:

- `k=4`: complement-only ratio `0.001436`
- `k=8`: `0.001069`
- `k=16`: `0.000806`
- `k=32`: `0.000602`
- `k=64`: `0.000237`
- `k=128`: `0.000106`
- `k=256`: `0.0000146`

Over the same range, visible-only intervention remains near the full endpoint
functional effect, and the nonlinear interaction remains very small.

The two-margin readout shows the same qualitative and quantitative pattern.

The weakest low-k projections behave as expected:

- `k=1` leaves a larger complement functional contribution
  (`~3.7%` of centered-logit effect energy);
- `k=2` reduces that to `~1.2%`;
- by `k=4`, complement-only output effect is already approximately `0.14%`.

This monotone strengthening is consistent with the recovered task-sensitive
basis capturing the finite downstream-relevant component progressively rather
than the result depending on a single hand-picked dimension.

## Prediction and CE diagnostics

Across all reported `k` values for the primary same-RNG/different-A pairs:

- visible-only prediction disagreement versus the source endpoint: `0`
- complement-only prediction disagreement versus the source endpoint: `0`

At `k=8`:

- visible-only mean CE change from source:
  `-0.000667803`
- complement-only mean CE change from source:
  `-0.0000175385`

At larger `k`, complement-only CE changes approach numerical zero.

The maximum reverse-corner state symmetry error is only on the order of
`1.5e-05` to `3.1e-05`, consistent with the intended affine intervention
construction.

## Control-class behavior

The same-A-init / different-training-RNG control class shows the same broad
downstream decomposition structure, with visible projections explaining nearly
all measurable endpoint functional differences and complement-only effects
becoming very small as `k` grows.

This does not weaken the A-init result.

The prior frozen 3x3 endpoint analysis established that A-init is the dominant
source of ambient layer-22 variation:

- same-RNG / different-A full-space distance is hundreds of times larger than
  same-A / different-RNG;
- the balanced factorial A-init main effect accounts for approximately
  `99.77%` of endpoint variation.

The intervention result therefore identifies where that dominant A-init
variation is functionally expressed downstream.

## Scientific interpretation

The causal intervention evidence strongly supports the bounded statement:

`A_INIT_SELECTS_DIFFERENT_LAYER22_REPRESENTATIVES_PRIMARILY_ALONG_DOWNSTREAM_EFFECTIVELY_NULL_OR_VERY_LOW_GAIN_DIRECTIONS_WHILE_A_SMALL_TASK_VISIBLE_COMPONENT_CARRIES_ESSENTIALLY_ALL_MEASURABLE_ENDPOINT_FUNCTIONAL_DIFFERENCE`

This is substantially stronger than the prior descriptive endpoint result.

The evidence now supports all of the following at the frozen layer-22 boundary:

1. A-init causes very large changes in hidden representation.
2. Most of that hidden squared-distance energy lies outside the recovered
   leading task-sensitive coordinates.
3. Replacing only that large complement component has almost no downstream
   functional effect.
4. Replacing only the much smaller task-visible component reproduces
   essentially the full endpoint functional difference.
5. The residual nonlinear interaction is very small over the tested endpoint
   chords.
6. Predictions remain invariant under both intervention classes on the frozen
   dev population.

This is strong evidence for downstream representational non-identifiability:
multiple substantially different layer-22 hidden representatives implement
nearly the same downstream behavior.

The term `gauge-like` is appropriate as a bounded descriptive analogy for this
layer-22 frozen-dev phenomenon.

The evidence is not sufficient to claim an exact mathematical gauge symmetry.

## What the result does not establish

The present audit does not establish:

- a universal gauge group or exact symmetry;
- a globally flat null manifold;
- invariance under arbitrary complement perturbations;
- equivalence outside endpoint-to-endpoint affine chords;
- invariance outside the frozen Phase3A P0 dev population;
- the optimization trajectory that produced the representatives;
- the layer at which A-init-specific representational separation first appears;
- out-of-distribution equivalence.

The complement is therefore best described as `effectively null / very low
gain under the tested bounded endpoint interventions`, not mathematically
null in every context.

## Scientific consequence

The endpoint-level A-init mechanism is now sufficiently resolved to justify
moving upstream.

The next meaningful A-init question is:

> At which earliest network layer does the A-init-specific representational
> separation emerge, and does that separation already lie primarily in a
> downstream-low-gain component?

The next stage should therefore be an upstream precursor-localization audit.

It should remain frozen-dev and no-training, and should measure layerwise:

- same-RNG / different-A endpoint separation;
- same-A / different-RNG control separation;
- downstream-visible versus complement decomposition;
- final-output functional stability;
- first layer at which large A-init-specific ambient separation appears.

No training-trajectory claim should be made from such an endpoint layerwise
audit.
