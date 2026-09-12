# K0-RVG Layer-23 Hidden In-Projection Transfer Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

Runner commit:

`8e9b5fd6c356889b866f3eae4890351f522471be`

Runner SHA256:

`65810137bebb768c12bb11fd279a630d6a6a1f1a6116b3b12f00a0c286ea8d64`

Parent evidence freeze:

`5305afe5369f4952aa50e63c0fb381c30f577cd4`

No tokenizer execution, logits, task heads, training, intervention, PCA, or
learned probe was executed. Raw vectors were not persisted.

## Exact boundary

At layer 23:

`H_t = W_H X_t`

where:

- `X_t` is the width-768 mixer input;
- `W_H` is the first 1536 rows of the bias-free in-projection;
- `H_t` is the current-token width-1536 pre-convolution hidden branch.

Therefore:

`delta_H_t = W_H delta_X_t`

and the exact scalar magnitude factorization is:

`||delta_H|| = ||delta_X|| * hidden_inproj_transfer`

with:

`hidden_inproj_transfer = ||delta_H|| / ||delta_X||`

The transfer term is observationally interpretable as direction-conditioned
transfer through the fixed hidden-branch projection. It is not a causal
intervention.

## Artifact authentication

Full local metrics:

`hidden_inproj_transfer_metrics.jsonl`

SHA256:

`54eb59d5d7a4e1e2561e9c12d284f71344a5cebf5f75ea8b8caf8070aa6bdbcf`

summary.json SHA256:

`d9bd8ea89ec9c7e561164a1cd6f4803d69943748c2249d769b748576b7ca953f`

execution_manifest.json SHA256:

`662933d9ae42e5994926e67f0cd20b528d0ef7b2d988374cd8e7e1684b676de4`

Trajectory rows:

`5376`

Common DDSSSSS cohort:

`330`

Scientific forwards:

`1344`

Parent delta-H reproduction:

`PASS`

Maximum hidden in-projection reconstruction relative residual:

`5.651929189193336e-07`

Maximum magnitude-factorization absolute error:

`3.552713678800501e-15`

Maximum magnitude-factorization relative error:

`2.0178486471843738e-16`

At k=-1, delta-X, delta-H, and transfer were all exactly zero for 672/672
pair-role trajectories.

## Common-cohort medians

corr:

- k1: DX 3.862948, DH 9.502844, transfer 2.406057
- k2: DX 4.737126, DH 6.951394, transfer 1.439837
- k3: DX 1.129992, DH 2.837561, transfer 2.770709

ctrl:

- k1: DX 2.415712, DH 8.073882, transfer 3.336379
- k2: DX 0.953187, DH 2.827695, transfer 3.009877
- k3: DX 0.891142, DH 2.399005, transfer 2.750858

## Corr k1 -> k2

Counts:

- delta-X up: 193/330
- delta-X down: 137/330
- in-projection transfer up: 39/330
- in-projection transfer down: 291/330
- delta-H up: 42/330
- delta-H down: 288/330

Absolute-log contribution dominance:

- delta-X: 139/330
- in-projection transfer: 191/330

Joint direction counts:

- delta-X up + transfer down: 193/330
- delta-X down + transfer up: 39/330

Median log ratios:

- delta-X: +0.157391
- in-projection transfer: -0.481306
- delta-H: -0.340143

Thus corr current-H contraction from k1 to k2 occurs despite a median increase
in upstream delta-X magnitude.

The dominant changing factor is more often reduced direction-conditioned
transfer through the fixed hidden-branch in-projection. This transfer decrease
overcomes the upstream delta-X increase in a large fraction of items.

This is an observational/algebraic projection effect, not evidence that the
projection causally determines downstream model behavior.

## Corr k2 -> k3

Counts:

- delta-X up: 2/330
- delta-X down: 328/330
- in-projection transfer up: 300/330
- in-projection transfer down: 30/330
- delta-H up: 0/330
- delta-H down: 330/330

Absolute-log contribution dominance:

- delta-X: 320/330
- in-projection transfer: 10/330

Joint direction counts:

- delta-X down + transfer up: 300/330

Median log ratios:

- delta-X: -1.275437
- in-projection transfer: +0.364306
- delta-H: -0.836166

Therefore corr k2-to-k3 current-H collapse is primarily already present in
the upstream layer-input delta-X magnitude.

The in-projection transfer usually moves in the opposite direction and
partially offsets that upstream collapse.

## Corr versus ctrl at k2

Counts:

- corr delta-X > ctrl: 330/330
- corr transfer > ctrl: 14/330
- corr delta-H > ctrl: 330/330

Absolute-log contribution dominance:

- delta-X: 329/330
- in-projection transfer: 1/330

Direction structure:

- delta-X favors corr while transfer favors ctrl: 316/330
- both delta-X and transfer favor corr: 14/330

Median paired log corr/ctrl ratios:

- delta-X: +1.484877
- in-projection transfer: -0.623545
- delta-H: +0.844772

Thus the corr-vs-ctrl k2 current-H separation is already substantially larger
at the layer-23 input delta-X boundary.

The fixed hidden-branch in-projection does not create this role separation.
Instead, for 316/330 common-cohort items it transfers the corr delta-X
direction less strongly than the ctrl delta-X direction and therefore
attenuates the upstream role separation.

The surviving delta-H separation is the net result of a much larger upstream
delta-X advantage and an opposing in-projection transfer effect.

## Validated scientific conclusion

The layer-23 hidden-branch in-projection has two distinct roles across the
observed trajectory.

First, for corr k1 -> k2, the current-H contraction is not inherited directly
from upstream delta-X magnitude. Delta-X commonly increases while
direction-conditioned hidden-branch transfer falls. Transfer is the larger
absolute log factor in 191/330 items. This localizes much of the temporal
k1-to-k2 H contraction to projection geometry at this boundary.

Second, for corr k2 -> k3, the situation reverses. Delta-X collapses in
328/330 items and is the dominant log factor in 320/330, while projection
transfer usually increases and partially compensates the collapse.

Third, the corr-vs-ctrl k2 role separation is overwhelmingly upstream of the
hidden in-projection. Corr delta-X exceeds ctrl in 330/330 items and delta-X
is the dominant log contribution in 329/330. The hidden projection usually
opposes rather than generates this separation.

Therefore temporal evolution and role-level separation must not be collapsed
into one mechanism:

- temporal corr k1 -> k2: projection-transfer contraction is important and
  often dominant;
- temporal corr k2 -> k3: upstream delta-X contraction is dominant;
- corr-vs-ctrl k2: upstream delta-X role separation is dominant and projection
  transfer attenuates it.

All statements are observational/algebraic.

## Relation to the prior convolution result

The four-tap convolution audit established that the layer-23 convolution
output is dominated by the current-token Q0 tap, and that current-token
delta-H magnitude is the dominant factor for corr-vs-ctrl k2 and for the
corr k2-to-k3 Q0 collapse.

The present audit moves one boundary upstream.

For corr-vs-ctrl k2, the large current-H magnitude separation is already
present even more strongly as layer-23 input delta-X separation.

For corr k2-to-k3, current-H collapse is likewise primarily inherited from
upstream delta-X collapse.

For corr k1-to-k2, however, the hidden projection substantially reshapes the
trajectory: delta-X does not exhibit the same contraction, while
direction-conditioned transfer does.

## Next scientific boundary

The next primary boundary is the exact producer of layer-23 mixer input X.

This choice is motivated by two validated observations:

1. corr-vs-ctrl k2 delta-X separation dominates the role-level current-H
   separation;
2. corr k2-to-k3 delta-X collapse dominates the temporal current-H collapse.

Before any new scientific forward, the layer-23 block source path should be
statically authenticated to determine exactly whether X corresponds to a
normalized residual stream, a residual update, or another block-internal
boundary in this Transformers Mamba implementation.

The corr k1-to-k2 projection-transfer effect remains a validated secondary
mechanism and may later justify a dedicated W_H directional-geometry audit,
but it is not the next primary boundary because it does not explain the
dominant role-level k2 separation.

No new scientific execution is authorized by this report before this evidence
is frozen.