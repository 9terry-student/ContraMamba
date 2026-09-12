# K0-RVG Divergence Magnitude Trajectory

## Status

VALIDATED SCIENTIFIC EVIDENCE

Execution commit:

`32b2a65b631b8b7031649d21a66b6cb223b1521c`

Parent transition-localization freeze:

`c96b58e72f7deceb02f734c5e6c2aa5ccd23ac6a`

The experiment reused frozen token IDs and the validated equal-length
divergence-aligned CPU slow-path protocol. No tokenizer, logits, task heads,
training, geometry analysis, or causal intervention was executed.

## Scope

Population:

- 336 frozen items
- corr and ctrl roles
- 672 role-level matched/swapped comparisons

Observation:

- layer 23
- S_post
- k=-1 through k=+6
- equal-length prefixes through k=+6

Primary metric:

`||S_post(matched) - S_post(swapped)||_2`

Supporting metrics included RMS magnitude, state-scale-normalized L2, and
pair-specific magnitude normalized to k=0.

Total model forwards:

`1344`

Raw state vectors were not persisted.

## Artifact authentication

execution_manifest.json SHA256:

`b8250f1c4b0ca38e2826d0c3b72ab35a95daf8c9d5834ebf0b7b724378e7a8a0`

summary.json SHA256:

`aeb49f30d45ef0a9f8510514104bda9a495bf72e3896b4d7a672e9e36fe01a6d`

Full trajectory_metrics.jsonl remains in the local validated run artifact.
Its SHA256 is:

`20e6c6cb7f7a53b101a40e7662761ecd4f27cf89b0cfc8193e990eb60f223bc1`

All manifest-declared artifact hashes, trajectory medians, stepwise counts,
and boundary conditions were independently recomputed and validated.

## Boundary integrity

At k=-1:

- exact S_post identity: 672/672
- divergence magnitude: 0

At k=0:

- exact S_post difference: 672/672
- normalized magnitude: 1.0 for all 672 pairs

## Full-cohort trajectory

Median k-relative magnitude, normalized to each pair's k=0 magnitude:

corr:

- k=0: 1.000
- k=1: 0.470
- k=2: 0.874
- k=3: 0.136
- k=4: 0.196
- k=5: 0.221
- k=6: 0.203

ctrl:

- k=0: 1.000
- k=1: 0.469
- k=2: 0.235
- k=3: 0.111
- k=4: 0.171
- k=5: 0.184
- k=6: 0.200

Both roles retain approximately 20% of their initial divergence magnitude
at k=+6, but their intermediate trajectories are not the same.

## Token-conditioned decomposition

330/336 items share the same token-equality signature in both roles:

`DDSSSSS`

Thus matched/swapped token IDs differ at k=0 and k=1, and are identical from
k=2 through k=6.

The remaining 6 items have signature:

`DDDDDDD`

The corr k=2 rebound cannot be explained by renewed current-token-value
difference. Among the common 330-item cohort:

- corr k=1 -> k=2 increases: 313/330
- ctrl k=1 -> k=2 increases: 35/330
- corr k=1 -> k=2 UP while ctrl DOWN: 278/330

At k=2:

- corr magnitude > ctrl magnitude: 330/330
- ctrl magnitude > corr magnitude: 0/330
- median corr/ctrl normalized-magnitude ratio: 3.8488838104
- median corr-minus-ctrl normalized magnitude: 0.6271621643

## Transient localization

Post-divergence peak coordinate among k=1 through k=6:

corr:

- k=1: 17/330
- k=2: 313/330

ctrl:

- k=1: 278/330
- k=2: 34/330
- k=4: 10/330
- k=5: 5/330
- k=6: 3/330

The corr rebound is immediately followed by a cohort-wide collapse:

- corr k=3/k=2 median magnitude ratio: 0.1389042497
- corr k=2 -> k=3 decreases: 330/330

For ctrl:

- ctrl k=3/k=2 median magnitude ratio: 0.4543864447
- ctrl k=2 -> k=3 decreases: 289/330
- ctrl k=2 -> k=3 increases: 41/330

## Scientific interpretation

The divergence-aligned layer-23 trajectory is not adequately described as
simple monotonic forgetting.

The common 330-item paired cohort exhibits a strong role-associated transient:
corr shows a sharp k=2 rebound followed by an immediate k=3 collapse, whereas
ctrl usually continues to attenuate through k=2 and k=3.

Because current token IDs are already identical at k=2 for this cohort, the
corr rebound is not attributable to renewed token-value contrast at that
coordinate.

This does not yet establish the mechanism of the rebound. Identical token IDs
do not imply identical upstream hidden/context representations. The current
evidence therefore establishes a propagated role-associated state-magnitude
transient, not autonomous recurrence amplification or a causal decision
mechanism.

## Next scientific boundary

Magnitude trajectory evidence is complete.

The next minimal native-Mamba question is whether the corr k=2 rebound and
k=3 collapse are primarily associated with the recurrence carry component
or the write component in:

`S_post = G * S_prev + W`

That decomposition requires a new scientific recurrence capture.