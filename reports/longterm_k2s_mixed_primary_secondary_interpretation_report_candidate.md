# K2S Mixed-Primary Secondary Interpretation Report Candidate

**Status:** read-only secondary scientific interpretation.

**Authority status:** NOT execution authority.

**Primary-verdict status:** frozen and unchanged.

## Source identity

K2S scientific/result archive commit:

`6069213234793286e658948a2e6f3b4f1105543d`

K2S implementation/runtime commit:

`d6d901cb3f1ab636d4db8b6cbea0bdf1e0581631`

Frozen primary verdict:

`MIXED_PRIMARY_PAIR_SPECIFICITY_RESULT`

This report uses only the already-produced K2S artifacts.

No model execution, new native-state capture, item selection, threshold selection, layer selection, window selection, metric selection, or significance testing was performed.

## Primary verdict remains unchanged

The frozen primary endpoints at layer 23 were:

- R: 84 positive, 56 negative, 10 zero; sign effect +0.20; Holm-adjusted p = 0.022154955980559526.
- D: 93 positive, 47 negative, 10 zero; sign effect +0.32857142857142857; Holm-adjusted p = 0.00025125894680797265.
- P: 20 positive, 120 negative, 10 zero; sign effect -0.7142857142857143; Holm-adjusted p = 4.256501810949398e-18.

Therefore the primary verdict remains mixed.

Secondary analyses below are explanatory only.

## Depth structure

Layerwise pair-specificity is strongly non-monotonic.

At layer 22:

- R sign effect = 0.08571428571428572
- D sign effect = 0.17142857142857143
- P sign effect = 0.2571428571428571

At the frozen primary layer 23:

- R sign effect = 0.2
- D sign effect = 0.32857142857142857
- P sign effect = -0.7142857142857143

The most notable terminal-layer descriptive transition is therefore P:

`+0.2571428571428571 -> -0.7142857142857143`

from layer 22 to layer 23.

R and D remain positive at layer 23.

The layerwise observations do not support a simple monotonic claim that pair-specificity merely increases with depth.

They also do not authorize replacing the frozen primary layer with a post-hoc favorable layer.

## Event-relative temporal structure

The first post-prefix position k=1 is exactly zero for both speed and turning pair-specificity in all 150 blocks.

This is consistent with the frozen token geometry, in which the correction-versus-control branches first diverge only at p+2 or p+3.

Therefore k=1 functions as an internal temporal negative-control observation.

At layer 23, speed pair-specificity is descriptively:

- positive at k=2;
- negative at k=3 and k=4;
- positive again at k=5;
- strongly positive at k=6 and k=7;
- weakly positive at k=8.

Turning pair-specificity is descriptively:

- non-positive or near-neutral through much of k=2..4;
- positive at k=5;
- negative at k=6;
- strongly positive at k=7 and k=8.

The observed response is therefore temporally structured and delayed rather than a simple immediate scalar separation at evidence onset.

These observations do not authorize a best-time or reduced-window confirmatory claim.

## Path-efficiency decomposition

Layer-23 pair-specificity decomposes as:

- path length:
  positive=84,
  negative=56,
  zero=10,
  sign effect=0.2

- displacement:
  positive=36,
  negative=104,
  zero=10,
  sign effect=-0.4857142857142857

- efficiency:
  positive=20,
  negative=120,
  zero=10,
  sign effect=-0.7142857142857143

Mean speed is path length divided by the fixed W=8 window, so the path-length direction necessarily mirrors R.

The key secondary observation is that matched assignment increases correction-versus-control local movement contrast, while reciprocal swapped assignment produces the larger correction-versus-control net-displacement contrast.

The strong negative efficiency specificity is therefore not explained by matched trajectories simply moving less.

Instead, local movement magnitude and net endpoint displacement separate.

## Within-block dissociation

P is negative in 120 of 150 blocks.

Among those P-negative blocks:

- 97 / 120 have R positive or D positive;
- 50 / 120 have both R positive and D positive.

The single most common primary sign pattern is:

`R+ D+ P- = 50` blocks.

Therefore the mixed result is not primarily explained by one subset of blocks producing positive R/D and a disjoint subset producing negative P.

Local positive pair-specificity and negative net-path specificity frequently coexist within the same reciprocal blocks.

## Shared exact-zero blocks

Exactly 10 blocks are zero simultaneously for R, D, and P.

Their block indices are:

`[0, 35, 76, 91, 98, 106, 109, 122, 125, 143]`

Thus the exact-zero observations are a shared block-level phenomenon, not three unrelated endpoint-specific zero sets.

No block is removed or downweighted because of this observation.

## Scientific interpretation

The strongest defensible secondary interpretation is:

**Matched semantic prefix-continuation assignment is associated with stronger local movement and directional correction-versus-control dynamics, while reciprocal semantic mismatch is associated with stronger net endpoint-displacement and trajectory-efficiency contrast at the terminal primary layer.**

This constitutes a local-versus-net trajectory dissociation.

It does not establish two causal mechanisms.

It does not establish an epistemic-state ontology.

It does not establish a confident-error precursor.

It does not convert the K2S primary verdict into a global positive result.

## Research boundary

K3 recurrent-state intervention remains unauthorized.

The current evidence instead motivates, at most, a future separately preregistered confirmatory study of local-versus-net trajectory dissociation.

Such a successor must not:

- choose layer 23 because it maximizes a secondary effect;
- choose k=7 or any other time point as a post-hoc primary target;
- discard P;
- redefine efficiency;
- treat R/D alone as the completed K2S success criterion;
- reuse this dataset as though the new dissociation hypothesis were prospectively frozen.

A future confirmatory design should use independent prospective evidence or a genuinely held-out population and freeze its temporal/geometric summaries before observing its native-state outcome.

## Final secondary interpretation state

`K2S_PRIMARY_VERDICT = MIXED_PRIMARY_PAIR_SPECIFICITY_RESULT`

`K2S_SECONDARY_LOCAL_VS_NET_DISSOCIATION_OBSERVED = YES`

`K2S_PRIMARY_PROMOTION_CHANGED = NO`

`K3_CAUSAL_INTERVENTION_AUTHORIZED = NO`

`NEW_SCIENTIFIC_EXECUTION_AUTHORIZED_BY_THIS_REPORT = NO`
