# Gen4-K XG2-Basis Projector-Contrast Mechanism Localization

## Status

Static mechanistic localization of the frozen XG2-basis cross-family fresh-index holdout result.

This analysis used only:

- the frozen XG2 and XG4 Phase-1 `alignment_delta_h.pt` evidence;
- the committed XG2/XG4 source-pair 601..900 signed directional-Jacobian observations;
- deterministic linear algebra on the resulting frozen subspaces and observed signed `J` coordinates.

No new model forward, training, backward pass, task-head evaluation, logits read, new p-value, subgroup test, tail test, epsilon sweep, k sweep, rescue analysis, or response-guided model execution was performed.

## Evidence identity

Result commit:

`6252ffca422b15227863b069b3efc4ad5580dfba`

Frozen holdout runner blob:

`2d35e5ed936fd37f4ecfc063e060290304c6bf10`

Frozen family-subspace runner blob:

`03f3bf1482913bce30bfdb665ab223a67e6e4159`

XG2 observation items SHA256:

`41677dd4e5594f9a3eb3f33477c014b44368644fe2c1fe9038fd647b61119f6e`

XG4 observation items SHA256:

`739fdc74ae64d4a30bfc1d772c3be4516c762d58313be45397633104207a8c7a`

Frozen Phase-1 plan SHA256:

- XG2: `b2cfaeaa02eaf013f2837339296c6f3252e9c26bac414d161ced4afcba6b819c`
- XG4: `792487f6ef7d1cd122f0fe6fb34594ea92747a3f52fc232382a5ed154264ec6f`

All analysis inputs were read from exact Git-object bytes to avoid Windows working-tree line-ending transformations.

## Frozen geometry

Let the frozen orthonormal five-dimensional bases be `B2` and `B4`, with projectors:

`P2 = B2 B2^T`

`P4 = B4 B4^T`

The two subspaces have union rank `10` in ambient dimension `395`.

Principal angles:

| Principal pair | cos(theta) | theta (deg) | abs eigenvalue of P2-P4 |
|---|---:|---:|---:|
| P1 | 0.49195939975654601 | 60.5305512095 | 0.87061814189182785 |
| P2 | 0.31960691239131289 | 71.3608457153 | 0.94755022112376275 |
| P3 | 0.16115855024319592 | 80.7258509922 | 0.98692852916688512 |
| P4 | 0.054942378933413208 | 86.8504476394 | 0.99848952673382474 |
| P5 | 0.016251944499344834 | 89.0687911755 | 0.99986792842854511 |

Chordal distance squared:

`4.6265725015235093`

Thus the frozen XG2 and XG4 top-5 subspaces are strongly separated rather than substantially overlapping.

The positive contrast eigenvalues are distinct, with minimum positive eigengap:

`0.0013784016947203659`

so the five individual positive contrast-mode identities are numerically non-degenerate under the frozen geometry.

## Phase-1 variance check

Normalized Phase-1 top-5 second-moment mass:

- XG2: `0.76002691588376836`
- XG4: `0.77697325589099264`

Therefore XG2 does not have greater total frozen top-5 Phase-1 mass than XG4.

The observed XG2-basis response advantage cannot be explained simply by a larger total Phase-1 top-5 variance mass.

Likewise, P3 is not the principal plane with the largest projector-contrast eigenvalue: P4 and P5 have larger `sin(theta)` values.

Therefore P3 response dominance is not explained merely by maximal geometric separation between the two subspaces.

## Exact Q decomposition

For each item:

`Q = (1/5) g^T (P2 - P4) g`

under the stored directional-Jacobian coordinates.

The projector contrast decomposes into five orthogonal principal planes. Each non-degenerate plane has one positive and one negative contrast eigenmode with eigenvalues:

`+sin(theta_i)` and `-sin(theta_i)`.

The reconstructed principal-plane decomposition reproduced the stored item-level `Q` to numerical precision.

Maximum absolute Q reconstruction residual:

- XG2: `3.705769144237564e-22`
- XG4: `6.3527471044072525e-22`

No new response observation was needed for this identity.

## XG2 source-family localization

Mean observed Q:

`1.3987206143704023e-07`

Total positive contrast gain:

`1.7447429013884855e-07`

Total negative contrast penalty:

`-3.4602228701808441e-08`

Effective positive contrast-mode count:

`3.03414211634`

Effective positive-net principal-plane count:

`2.79684479982`

Dominant positive mode:

`P3+`

P3+ share of positive gain:

`0.491967699447`

Dominant positive-net plane:

`P3`

P3 share of total positive-net principal-plane contribution:

`0.527178977905`

Per-plane mean contributions:

| Plane | Positive gain | Negative penalty | Net | Positive-net share |
|---|---:|---:|---:|---:|
| P1 | 2.0000565576529136e-08 | -4.079978123371159e-09 | 1.5920587453157979e-08 | 0.113114322975 |
| P2 | 2.9655242304908906e-08 | -8.9439540574989332e-09 | 2.0711288247409971e-08 | 0.147151815530 |
| P3 | 8.583571513231499e-08 | -1.1636455476405792e-08 | 7.4199259655909202e-08 | 0.527178977905 |
| P4 | 1.8015989907775935e-09 | -2.6772964092091862e-09 | -8.7569741843159263e-10 | 0 |
| P5 | 3.7181168134317933e-08 | -7.2645446353233729e-09 | 2.9916623498994552e-08 | 0.212554883589 |

## XG4 source-family localization

Mean observed Q:

`3.7773950385492771e-07`

Total positive contrast gain:

`5.1803013850641158e-07`

Total negative contrast penalty:

`-1.4029063465148413e-07`

Effective positive contrast-mode count:

`3.11248585893`

Effective positive-net principal-plane count:

`2.45912271184`

Dominant positive mode:

`P3+`

P3+ share of positive gain:

`0.461139188294`

Dominant positive-net plane:

`P3`

P3 share of total positive-net principal-plane contribution:

`0.555490065950`

Per-plane mean contributions:

| Plane | Positive gain | Negative penalty | Net | Positive-net share |
|---|---:|---:|---:|---:|
| P1 | 4.4500064363403997e-08 | -2.5830041452326291e-08 | 1.8670022911077699e-08 | 0.046731773735 |
| P2 | 1.671928843445138e-09 | -2.3446961223178255e-08 | -2.1775032379733117e-08 | 0 |
| P3 | 2.3888399758288564e-07 | -1.6957641501928693e-08 | 2.2192635608095694e-07 | 0.555490065950 |
| P4 | 1.2047741121262137e-07 | -4.4215359621867849e-09 | 1.1605587525043461e-07 | 0.290492246776 |
| P5 | 1.1249673650405537e-07 | -6.9634454511864107e-08 | 4.2862281992191257e-08 | 0.107285913539 |

## Cross-family agreement

Positive-gain profile cosine similarity:

`0.87871544002347313`

Net principal-plane profile cosine similarity:

`0.81288191995762282`

Both source families have:

- the same dominant positive contrast mode: `P3+`;
- the same dominant positive-net principal plane: `P3`.

The secondary realization differs.

XG2 has positive net contributions from P1, P2, P3, and P5, with P4 approximately neutral-to-negative.

XG4 has positive net contributions from P1, P3, P4, and P5, while P2 is negative.

Therefore the aggregate XG2-basis dominance is not generated by an identical five-plane mixture in the two source families.

## Signed-orientation diagnostic

The synthesized contrast-mode `J` coordinates were reconstructed as linear combinations of the already observed signed finite-difference `J` values.

They were not directly probed with new model forwards.

Under the deterministic ambient sign convention, P3+ has the same positive mean orientation in both source families.

The reconstructed P3+ mean signed coordinates are:

- XG2: `0.00064019604041729613`
- XG4: `0.0010943328180143467`

Both have descriptive reconstructed sign coherence `1`.

P1+ and P5+ also have matching mean signs across the two families, whereas P2+ and P4+ do not.

Because these are synthesized linearized coordinates rather than directly measured finite-difference responses along the contrast modes, this analysis does not establish a universal signed causal direction.

## Mechanistic localization

The observed XG2-basis cross-family sensitivity effect is best localized as:

**a shared dominant P3 projector-contrast plane embedded in a broader distributed, family-specific secondary geometry.**

The evidence does not support a pure single-mode account because the effective positive-net plane count is materially greater than one in both source families.

The evidence also does not support a completely family-specific internal realization because the same P3+ contrast mode is the largest positive mode and the same P3 plane is the largest positive-net plane in both families.

P3 dominance is not explained by:

1. larger total XG2 Phase-1 top-5 variance mass; or
2. P3 having the largest possible projector separation.

Instead, the frozen response gradients are preferentially aligned with the P3 positive contrast geometry in both source families.

This identifies P3 as the strongest shared mechanistic candidate produced by the current static localization.

## Scientific disposition

No further 601..900 post-hoc localization is required before transport.

For the next prospective transport stage, the mechanism target should be frozen from the current geometry before any new-generator response is inspected.

The appropriate primary mechanistic object is the frozen P3 principal contrast plane derived deterministically from the exact frozen XG2 and XG4 bases.

The unsigned positive-versus-negative P3 contrast should remain the primary transport object.

The deterministic P3+ signed orientation may be frozen as a prospective diagnostic, but it must not be treated as already validated by direct finite-difference observation.

External generator/data transport remains a new prospective experiment and is not established by this localization result.
