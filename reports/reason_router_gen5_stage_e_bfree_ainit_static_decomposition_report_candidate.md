# Gen5 Stage E BFREE/AINIT Static Optimization Decomposition Report

## Status

VALIDATED_STATIC_ANALYSIS

This report records the read-only static decomposition performed after the
validated E-BFREE-AINIT evidence freeze.

It is not a new execution authority and authorizes no CUDA, training,
evaluation, backward pass, or optimizer step.

## Evidence identity

Evidence freeze commit:

`473ed74a70c02484c93746a022c4122632d55416`

Compared arms:

- `E-BFREE`
- `E-BFREE-AINIT`

Seeds:

- 6201
- 6202
- 6203

The analysis used only already-frozen Phase3A source corrections and the
already-frozen E-BFREE / E-BFREE-AINIT final correction checkpoints.

CUDA was not used.

No task evaluation was executed.

Confirmatory IDs 9601--9900 were not loaded.

No scientific p-values were computed.

## Decomposition

For each seed, the unrestricted Phase3A source factorization was written as

`B_free = Q_B R_B`

with deterministic canonical thin QR.

Because `Q_B` is orthonormal, the full operator discrepancy

`||Q_B M A - B_free A_free||_F`

is exactly the low-dimensional core discrepancy

`||M A - R_B A_free||_F`.

Therefore the analysis did not materialize the full `24576 x 768` operator.

The analysis measured:

- row-space geometry of `A_final` versus `A_free`;
- effective operator error/cosine/norm ratio;
- `M A_free` versus `R_B A_free` to isolate the learned core with ideal source A;
- `R_B A_final` versus `R_B A_free` to isolate the final A effect;
- direct and gauge-aligned `M` versus `R_B`.

## Results

### E-BFREE means

- recovery: `0.0714250887234`
- A row-space affinity to `A_free`: `0.960375217034`
- effective operator relative error: `0.987793413123`
- effective operator cosine: `0.886116170407`
- effective operator norm ratio: `0.0138131429696`
- M-only relative error at exact `A_free`: `0.988579644897`
- A-only relative error at exact `R_B`: `0.28083265917`
- gauge-aligned M relative error: `0.988853730621`

### E-BFREE-AINIT means

- recovery: `0.133576230985`
- A row-space affinity to `A_free`: `0.93180364918`
- effective operator relative error: `0.98117067811`
- effective operator cosine: `0.827225610208`
- effective operator norm ratio: `0.0228588324149`
- M-only relative error at exact `A_free`: `0.988275293096`
- A-only relative error at exact `R_B`: `0.84629156723`
- gauge-aligned M relative error: `0.986030986272`

### Paired direction across seeds

For every seed:

- E-BFREE-AINIT recovery exceeded E-BFREE recovery;
- E-BFREE-AINIT A row-space affinity was lower than E-BFREE A row-space affinity;
- effective operator relative error improved only slightly;
- M-only relative error remained approximately `0.988`.

The seedwise E-BFREE-AINIT recovery deltas were:

- seed6201: `+0.0668896433043`
- seed6202: `+0.0511765442719`
- seed6203: `+0.0683872392096`

The seedwise A-affinity deltas were:

- seed6201: `-0.028790832589`
- seed6202: `-0.0251639239458`
- seed6203: `-0.0317599470267`

The seedwise effective-operator relative-error deltas were:

- seed6201: `-0.00699943387305`
- seed6202: `-0.00591989149815`
- seed6203: `-0.00694887966943`

The seedwise M-only relative-error deltas were:

- seed6201: `-0.000564367083202`
- seed6202: `+0.000116402654994`
- seed6203: `-0.000465090973664`

## Interpretation

The remaining E-BFREE-AINIT gap is not well explained by loss of the source
A row-space alone.

E-BFREE-AINIT improved task recovery in every seed even though its final A
row-space was farther from `A_free` than E-BFREE.

More importantly, when final M was combined with the exact source `A_free`,
the mean relative operator error remained approximately `0.9883`.

Thus even a counterfactual restoration of the source read-side A geometry
would leave the learned M core extremely far from the exact source target
`R_B A_free`.

The same conclusion survives 2x2 gauge alignment: the mean gauge-aligned
M error remained approximately `0.9860`.

The dominant residual bottleneck therefore localizes to failure to acquire the
source-equivalent 2x2 core transformation under the matched zero-M,
20-step QMA optimization contract.

The supported conclusion is:

`RESIDUAL_FIXED_PLANE_RECOVERY_FAILURE_LOCALIZES_PRIMARILY_TO_CORE_M_ACQUISITION_RATHER_THAN_FINAL_A_ROWSPACE_MISMATCH`

This does not yet distinguish:

1. joint A/M optimization interference;
2. scale or conditioning of the zero-initialized M parameterization;
3. the fixed 20-step horizon.

No inferential claim is made from the three-seed descriptive comparison.

## Stage E status

Stage E remains open.

The next bounded experiment should isolate joint-A interference from core-M
optimization.

### Required next diagnostic: E-BFREE-AFIX-MONLY

For each seed:

- use the same frozen seed-matched learned-B output plane `Q_B`;
- initialize A exactly from the same seed's unrestricted `A_free`;
- freeze A for the entire run;
- initialize M exactly to zero;
- train only the `2 x 2` M matrix;
- preserve P0, the same train/dev split, optimizer family, learning rate,
  weight decay, gradient clipping, fixed 20-step horizon, parent model,
  final-only checkpointing, and seed set 6201/6202/6203;
- preserve the exact two-independent-T4-worker topology if/when scientific
  execution is later authorized;
- keep confirmatory IDs 9601--9900 forbidden;
- compute no scientific p-values.

Interpretation:

- if E-BFREE-AFIX-MONLY materially exceeds E-BFREE-AINIT, then joint A drift
  or A/M coupling is a major remaining bottleneck;
- if it remains near E-BFREE-AINIT, the residual ambiguity shifts strongly
  toward M-core scale/conditioning or the 20-step horizon;
- mixed seed behavior preserves seed-dependent ambiguity.

No learning-rate, optimizer, horizon, rank, random-plane, layer, or broader
parameter sweep is warranted before this diagnostic.

This report authorizes no implementation or execution.
