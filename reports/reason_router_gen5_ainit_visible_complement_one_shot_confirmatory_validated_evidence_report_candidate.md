# Gen5 A-init Visible-vs-Complement One-Shot Confirmatory Validated Evidence Report

## Evidence identity

- execution authority commit:
  `9cced33e32c4a5f84a1b5c6d3fa38ea85d45360f`
- source dev causal evidence freeze:
  `5694f962855bd2ab4f4035feb15cf1f4bfb3f784`
- source forward-Jacobian recovery evidence freeze:
  `a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e`
- run:
  `gen5-ainit-visible-complement-confirmatory-9cced33-r2`
- executed command SHA256:
  `fb0d8c298e15fe89705324f6d1f2461f2e869c7527e69706330185a3b93699a3`
- imported handoff ZIP SHA256:
  `f04de99faaaf62a4d781825c3f062cbd02b091ccfb962aca5d26c9375cca4e99`
- run log SHA256:
  `772ccd742174c9f1f92ad5c21f0a76d7696828eac8536a81d0118e1213908362`
- run meta SHA256:
  `3d19405a5040b8b4954e593ba20b4404125d2615813963cf872c92d5e1180f7b`

Imported artifact identities:

- `ainit_visible_complement_confirmatory_summary.json`
  SHA256 `f2e1267758cab99dc5eb00035afcf17414ed9870183c92db8c30149defeef81f`
- `visible_complement_confirmatory_metrics.pt`
  SHA256 `5727bc53da1b8363b8641c14ecd8c6485fa72ef46ae2fe88ca6c0f387bb98e56`

## Execution validity

The scientifically valid confirmatory execution completed successfully on the
exact authorized commit with the previously frozen one-shot contract.

Observed execution:

- confirmatory rows: `1800`
- source pairs: `300`
- primary same-training-RNG / different-A-init checkpoint pairs: `9`
- control same-A-init / different-training-RNG checkpoint pairs: `9`
- valid tokens: `128300`
- GPU0: primary pair class
- GPU1: control pair class
- both GPUs used for scientific execution: true
- pair split across GPUs: false
- cross-GPU scientific tensor reduction: false

No training, optimizer construction, parameter gradients, autograd, backward,
checkpoint mutation, label loading, CE computation, or p-value computation
occurred.

The previously unopened `xg1_fact_9601..xg1_fact_9900` population is now
scientifically consumed by this one-shot hypothesis test and must not be reused
for threshold tuning, basis fitting, primary-k selection, or rescue.

## Preregistered result

All preregistered strong-replication Gates A through G passed.

Scientific outcome:

`STRONG_REPLICATION`

This classification was determined from thresholds frozen before the
confirmatory outputs were observed.

## Ambient A-init separation

On the unseen confirmatory population:

`D_full_same_RNG_diff_A / D_full_same_A_diff_RNG = 424.368825354`

This passes the preregistered Gate A threshold of `>= 20`.

The large A-init dominance observed on development data therefore persists on
the unseen confirmatory population.

## Frozen k=8 decomposition

The prospectively frozen compact reference remained `k=8`.

Primary same-RNG/different-A hidden residual energy:

`Q_8 = 0.0850977686487`

Thus only about `8.51%` of A-init residual squared energy lies in the frozen
top-8 task-sensitive subspace, while about `91.49%` remains in its complement.

This passes preregistered Gate B (`Q_8 <= 0.20`).

### Centered three-class logits

At `k=8`:

- `R_visible = 0.987246730356`
- `R_complement = 0.000437253352847`
- `R_interaction = 0.000162674193961`

The visible-only intervention therefore reproduces about `98.72%` of the real
endpoint functional-effect energy.

The complement-only intervention contributes about `0.0437%` of the endpoint
functional-effect energy despite containing about `91.49%` of the hidden
residual squared energy.

The visible/complement effect concentration is approximately:

`0.987246730356 / 0.000437253352847 ~= 2258`

### Two-margin readout

At `k=8`:

- `R_visible = 0.988493915749`
- `R_complement = 0.000422822503292`
- `R_interaction = 0.000155027674465`

The two-margin readout independently reproduces the same causal decomposition.

## Robust compact-subspace replication

The preregistered robustness band was `k={4,8,16}`.

Centered-logit ratios:

- `k=4`
  - `R_visible = 1.03387454741`
  - `R_complement = 0.000962392253824`
  - `R_interaction = 0.000132502561841`
- `k=8`
  - `R_visible = 0.987246730356`
  - `R_complement = 0.000437253352847`
  - `R_interaction = 0.000162674193961`
- `k=16`
  - `R_visible = 0.972675277067`
  - `R_complement = 0.000473237604858`
  - `R_interaction = 0.0000640139952757`

All values lie comfortably inside the prospectively frozen Gate G bounds.

The confirmatory success is therefore not attributable to post hoc selection
of a single favorable dimensionality.

## Prediction diagnostics

- primary real-endpoint prediction disagreement: `0`
- control real-endpoint prediction disagreement: `0`
- k=8 visible-only prediction disagreement versus source: `0`
- k=8 complement-only prediction disagreement versus source: `0`

The confirmatory claim remains about continuous downstream logit/margin effects,
not about rescuing or changing class predictions.

## Scientific interpretation

The frozen dev-derived layer-22 task-sensitive basis generalizes to a
previously unseen population from the same synthetic generator family.

The validated bounded conclusion is:

`A_INIT_SELECTS_SUBSTANTIALLY_DIFFERENT_LAYER22_REPRESENTATIVES_WHOSE_DOMINANT_RESIDUAL_ENERGY_IS_DOWNSTREAM_EFFECTIVELY_NULL_OR_VERY_LOW_GAIN_WHILE_A_SMALL_FROZEN_TASK_VISIBLE_COMPONENT_CARRIES_ESSENTIALLY_ALL_MEASURABLE_FUNCTIONAL_DIFFERENCE_ON_UNSEEN_EXAMPLES`

The result now supports more than a development-set geometric observation.

It establishes that:

1. A-init remains the dominant source of ambient layer-22 endpoint variation
   on the unseen confirmatory population.
2. Most A-init residual energy remains outside the frozen compact
   task-sensitive subspace.
3. The frozen visible component alone reproduces essentially the entire
   measurable endpoint output effect.
4. The much larger complement component remains nearly functionally silent.
5. Nonlinear visible/complement interaction remains negligible.
6. The pattern is stable across the preregistered compact k band.
7. The result is not created by refitting the basis or choosing k on the
   confirmatory population.

This is strong evidence for bounded downstream representational
non-identifiability at the layer-22 boundary.

`Gauge-like` remains an appropriate descriptive analogy under this bounded
intervention contract.

## What remains unestablished

This evidence does not establish:

- an exact mathematical gauge symmetry;
- a global gauge group;
- a globally flat null manifold;
- invariance to arbitrary perturbations;
- natural-domain generalization beyond the frozen synthetic generator family;
- Transformer inferiority or Mamba superiority;
- a claim about the full optimization trajectory;
- a nontrivial upstream network-layer precursor before layer 22.

In particular, an upstream layer-0..21 precursor search would be structurally
tautological in the current Gen5 setup because the parent Mamba is frozen and
the A-init-dependent trainable tensors belong to the layer-22 correction.

## Consequence for the A-init research axis

The layer-22 endpoint A-init question is now sufficiently resolved and should
be considered closed as a descriptive-and-causal endpoint phenomenon.

The next scientifically meaningful question is internal mechanism localization
within the layer-22 correction pathway:

> At which internal transformation does the very large A-init-specific hidden
> residual become strongly downstream-low-gain, and which transformation
> preserves the small task-visible component?

The next stage should therefore localize the decomposition across the internal
state-write and readout chain rather than across frozen upstream network layers.

Candidate internal boundaries include:

1. correction write coordinates;
2. recurrently propagated correction state;
3. C-projected correction contribution;
4. gated mixer contribution;
5. layer-22 mixer output;
6. downstream final logits/margins.

The next audit should remain no-training, frozen-confirmatory-independent, and
should use already consumed populations only for mechanism analysis unless a
new prospective holdout is separately frozen.

No further use of `xg1_fact_9601..xg1_fact_9900` is authorized for model
selection or threshold tuning.
