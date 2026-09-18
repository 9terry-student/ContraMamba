# Gen4 PP3-Anchored Interaction Direction Localization Result

## Scope

This report freezes the limited static interaction-localization
follow-up requested by the completed PP3-anchored plane-additivity
Stage 3 result.

Repository head before this report:

`2b942fcd290d863eb790bcc38742bef380cf59b0`

Population:

- family: XG1
- source pairs: `xg1_fact_2401..2700`
- N: 300
- epsilon: 0.025

No new model execution was performed.

- new scientific model forwards: 0
- training: none
- backward passes: none
- task-head evaluation: none
- primary inference: none
- multiplicity correction: none

This is a descriptive static decomposition of already-frozen raw
scientific evidence.

## Frozen inputs

Stage 3 PP3-anchored interaction raw item SHA256:

`bc04b53d4c6d30cc5214bd249e3c5a9fa6f688010b93b9ef65f8b4e9f662c3cb`

Frozen native XG1 raw item SHA256:

`9d5dabbef82a8fcfaccc4e610bbb4d91f1e7ee9ea2f2627a92f999c88034ac49`

The Stage 3 interaction for partner k is:

`I_3k = A_3k - A_3 - A_k`

The raw direction probes permit an exact directional decomposition of
the same endpoint.

For an XG2 basis direction d:

`q_d = +J_d^2 / 5`

For an XG4 basis direction d:

`q_d = -J_d^2 / 5`

The direction-level interaction is:

`I_3k,d = q_3,d + q_k,d - q_3k,d - q_0,d`

and the ten direction-level terms reconstruct the frozen scalar
interaction.

## Numerical reconstruction

The static direction decomposition reproduced the frozen scalar
interaction values with:

- maximum absolute reconstruction error:
  `2.3491929396505986e-22`
- maximum reconstruction error divided by numerical scale:
  `4.675668572938755e-15`

These discrepancies are floating-point summation-path effects and do
not alter the stored scientific quantities.

## Family localization

Fraction of direction-level interaction L2 energy associated with each
family:

| Partner | XG2 fraction | XG4 fraction |
|---|---:|---:|
| P1 | 0.8912 | 0.1088 |
| P2 | 0.8891 | 0.1109 |
| P4 | 0.8728 | 0.1272 |
| P5 | 0.9070 | 0.0930 |

Thus, for every tested PP3 partner, the interaction energy is strongly
concentrated in the XG2 basis directions.

This is an energy localization statement. It does not imply that the
signed XG2 contribution alone determines each partner's population
mean because substantial within-family and cross-family cancellation
is present.

## Direction localization

The dominant directions are consistently concentrated toward the
higher-index XG2 basis directions.

Combined L2 share of `xg2_2`, `xg2_3`, and `xg2_4`:

| Partner | Combined L2 share |
|---|---:|
| P1 | 0.7173 |
| P2 | 0.7589 |
| P4 | 0.7081 |
| P5 | 0.7878 |

The two largest directions are:

- P1: `xg2_4`, then `xg2_3`
- P2: `xg2_3`, then `xg2_4`
- P4: `xg2_4`, then `xg2_3`
- P5: `xg2_3`, then `xg2_4`

`xg4_4` is the strongest recurring secondary XG4 direction.

The interaction is therefore not diffusely distributed across all ten
basis directions.

## Partner-specific signed structure

Although the dominant directional support is shared, the signed mean
direction profiles differ substantially across partners.

Cosine similarities of the ten-dimensional mean interaction signatures:

| Pair | Cosine |
|---|---:|
| P1 : P2 | +0.4659 |
| P1 : P4 | +0.2894 |
| P1 : P5 | -0.1978 |
| P2 : P4 | +0.6673 |
| P2 : P5 | -0.5667 |
| P4 : P5 | -0.7138 |

P2 and P4 have the most similar signed profiles.

P5 has an oppositely oriented mean directional profile relative to P2
and especially P4.

Therefore the previously observed cross-partner item-level correlation
does not reduce the four PP3 interactions to a single common signed
interaction mechanism.

## Interpretation

The limited localization follow-up supports the following descriptive
structure:

1. PP3-anchored interaction energy is predominantly localized to the
   XG2 family rather than XG4.
2. Within XG2, the interaction is concentrated principally in
   `xg2_2`, `xg2_3`, and `xg2_4`, with `xg2_3` and `xg2_4` dominating.
3. `xg4_4` provides a recurring but secondary contribution.
4. The dominant directional support is shared across partners.
5. The signed combination of those directions is partner-specific.
6. P5 is particularly distinct from P2/P4 in signed direction space.

The frozen descriptive conclusion is:

`PP3_ANCHORED_INTERACTION_ENERGY_IS_CONCENTRATED_IN_XG2_HIGH_INDEX_DIRECTIONS_WITH_PARTNER_SPECIFIC_SIGNED_PROFILES_ON_XG1_2401_2700`

This is not a claim of a uniquely identified biological or mechanistic
cause. It is a localization of the measured causal-intervention
interaction endpoint in the frozen response basis.

No post-hoc threshold, p-value, equivalence margin, or direction
selection criterion is introduced.

## Stage boundary

The limited interaction-localization follow-up is complete.

The evidence does not justify expanding to a 32-cell/full-factorial
interaction experiment before independent checkpoint replication.

The next research step is the already-prepared independent seed181
checkpoint replication. Its endpoint, homolog-plane matching rule,
control-plane rule, population, and success criteria remain frozen
independently of this localization result.
