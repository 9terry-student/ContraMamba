# Gen4-K XG2/XG4 Family-Subspace Sensitivity Confirmatory Result

## Status

Confirmatory result candidate based exclusively on frozen imported observation artifacts.

No new model execution, training, backward pass, task-head evaluation, logits read, subgroup analysis, tail analysis, basis-wise significance test, dimension search, epsilon search, or response-guided adaptation was performed for this analysis.

## Frozen evidence identity

Artifact freeze commit:

`d7a2f56268c7671dbbc7c5f07be85c6b2d1a2ae6`

Frozen artifact roots:

- `reports/reason_router_gen4_family_subspace_sensitivity_2ad38ed_r2/xg2/`
- `reports/reason_router_gen4_family_subspace_sensitivity_2ad38ed_r2/xg4/`

Observation population:

- XG2: 300 frozen source pairs
- XG4: 300 frozen source pairs
- pair ids: 301..600 within each family
- subspace dimension: `k = 5`
- finite-difference epsilon: `0.025`
- scientific forwards: 12,000 per family
- total scientific forwards: 24,000
- new baseline forwards: 0

Both imported artifact sets passed the frozen runner's artifact validation before confirmatory analysis.

For every pair:

`D = E_own - E_cross`

where:

`E_own = mean_j J(v_own,j)^2`

and:

`E_cross = mean_j J(v_cross,j)^2`

over the five frozen ordered basis directions.

## Pre-specified confirmatory rule

For each family independently:

`H1: mean(D) > 0`

using a one-sample Student t-test on the 300 paired `D` values.

Multiplicity family:

- XG2
- XG4

Holm correction was applied at `alpha = 0.05`.

The frozen positive outcome label:

`FAMILY_SPECIFIC_SUBSPACE_SENSITIVITY_REPLICATED`

requires both:

1. positive mean `D` with Holm rejection for XG2; and
2. positive mean `D` with Holm rejection for XG4.

Otherwise the frozen outcome is:

`FAMILY_SPECIFIC_SUBSPACE_SENSITIVITY_NOT_ESTABLISHED`

The word `REPLICATED` above is the pre-frozen result label. This study was prospectively defined for previously unobserved subspace-sensitivity outcomes but is not an independent replication study.

## Confirmatory results

| Family | N | Mean D | Median D | SD D | t | df | Raw one-sided p | Holm p | Holm reject |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| XG2 | 300 | 1.4050168357562629e-07 | 1.3088981262241912e-07 | 6.5922790875540311e-08 | 36.91531430478986 | 299 | 1.1069210751042617e-113 | 2.2138421502085234e-113 | TRUE |
| XG4 | 300 | -3.7956923311125934e-07 | -3.8499704190413078e-07 | 6.3269063583292152e-08 | -103.91068865325654 | 299 | 1 | 1 | FALSE |

Therefore the pre-specified joint confirmatory outcome is:

`FAMILY_SPECIFIC_SUBSPACE_SENSITIVITY_NOT_ESTABLISHED`

## Pre-authorized descriptive statistics

### XG2

`E_own`:

- mean: `1.7380713365860543e-07`
- median: `1.7040727632395762e-07`
- SD: `7.068696909424963e-08`

`E_cross`:

- mean: `3.3305450082979156e-08`
- median: `2.5088878676126171e-08`
- SD: `2.1510779236084463e-08`

`D`:

- mean: `1.4050168357562629e-07`
- median: `1.3088981262241912e-07`
- SD: `6.5922790875540311e-08`

Mean absolute directional derivative:

- own basis: `0.00033684955631213765`
- cross basis: `0.00014652556419330942`

### XG4

`E_own`:

- mean: `1.5367349760622704e-07`
- median: `1.5295526148358009e-07`
- SD: `3.930239661580275e-08`

`E_cross`:

- mean: `5.3324273071748638e-07`
- median: `5.2704021509957918e-07`
- SD: `8.0417025839173166e-08`

`D`:

- mean: `-3.7956923311125934e-07`
- median: `-3.8499704190413078e-07`
- SD: `6.3269063583292152e-08`

Mean absolute directional derivative:

- own basis: `0.00033430695616713633`
- cross basis: `0.00065513599086659032`

## Interpretation

The pre-specified symmetric family-specific concentration hypothesis is not established.

For XG2, the frozen XG2 basis has greater local squared directional sensitivity than the frozen XG4 basis, with positive mean `D` and Holm rejection under the pre-specified one-sided test.

For XG4, the observed descriptive direction is the reverse: the frozen XG2 cross basis has substantially larger `E_cross` than the XG4 own basis, producing a negative mean and median `D`. Under the pre-specified greater-than-zero test, XG4 therefore does not satisfy the required family-specific own-subspace condition.

The joint failure is consequently not a near-zero or low-power pattern. It is an observed family asymmetry in the own-versus-cross endpoint.

This result does not establish:

- that the XG2 basis is a universal or generator-independent sensitivity subspace;
- that the XG4 reverse direction is independently statistically significant, because no opposite-direction confirmatory test was pre-specified or executed;
- that every XG2 basis direction is causal;
- a common signed adverse direction;
- transport to XG3 or another generator;
- superiority of `k = 5` over another dimension;
- robustness to alternative epsilon values, bases, layers, offsets, checkpoints, subgroups, or tails.

No rescue analysis is authorized from this confirmatory result.

## Scientific disposition

The tested hypothesis:

> each generator family's local first-order response sensitivity is preferentially concentrated in its own frozen five-dimensional Phase-1 geometry rather than the other family's frozen geometry

is not supported jointly across XG2 and XG4.

The observed XG2/XG4 asymmetry may motivate a distinct future hypothesis about cross-family geometry, but that would constitute a new prospective research question and must not be treated as a positive reinterpretation of this failed confirmatory test.

XG3 remains outside this experiment because the frozen current anchor semantics provide zero complete eligible source pairs.
