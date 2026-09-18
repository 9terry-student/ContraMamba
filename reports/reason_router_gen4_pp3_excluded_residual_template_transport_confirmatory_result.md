# Gen4 PP3-Excluded Residual Template Transport — Confirmatory Result

## Status

`PP3_EXCLUDED_XG2_LIKE_RESIDUAL_TEMPLATE_TRANSPORT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

This report records the single frozen confirmatory inference defined before observing fresh XG1 `1201..1500` outcomes.

No rescue analysis, subgroup analysis, alternate metric, alternate tail, plane selection, epsilon sweep, layer sweep, token-position sweep, checkpoint sweep, or additional p-value was executed.

## Frozen design

Prospective design:

`reports/reason_router_gen4_pp3_excluded_residual_template_transport_design.md`

Design commit:

`d3cc008fad221862e6fe9718b67b6ba0c87d0368`

Fresh confirmatory population:

`xg1_fact_1201..xg1_fact_1500`

Required pair count:

`N = 300`

Residual plane order:

`[P1, P2, P4, P5]`

Primary endpoint:

`D_TEMPLATE_i = C_XG2_i - C_XG4_i`

Hypotheses:

- `H0: mean(D_TEMPLATE) <= 0`
- `H1: mean(D_TEMPLATE) > 0`

Frozen test:

- one-sample Student t-test
- one-sided alternative: greater
- `N = 300`
- `df = 299`
- `alpha = 0.05`
- exactly one confirmatory p-value
- no multiplicity correction

## Raw execution provenance

Execution HEAD:

`64d1b3abddad79e7320eb8717356addbc7e28c54`

Run:

`g4k-pp3-residual-template-xg1-1201-1500-64d1b3a-retry1`

Imported raw artifact directory:

`reports/reason_router_gen4_pp3_excluded_residual_template_transport_64d1b3a_retry1/`

Representative checkpoint SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Execution contract validated:

- GPU count: `2`
- GPU 0: `xg1_fact_1201..xg1_fact_1350`, `6000` scientific model forwards
- GPU 1: `xg1_fact_1351..xg1_fact_1500`, `6000` scientific model forwards
- total scientific model forwards: `12000`
- baseline model forwards: `0`
- training executed: `False`
- backward executed: `False`
- task heads executed: `False`
- logits read: `False`
- primary inference executed in raw artifact: `False`
- raw artifact scientific conclusion: `None`

The imported artifact passed checksum, schema, pair-order, endpoint-identity, Q-reconstruction, residual-norm, shard, checkpoint, and execution-boundary validation before confirmatory inference.

## Confirmatory inference

Observed statistic:

`t = 29.357724311692319`

One-sided confirmatory p-value:

`p = 2.2607093023100906e-90`

Exactly one confirmatory p-value was computed.

No additional p-values were computed.

## Frozen positive-label gates

Gate 1:

`mean(C_XG2) = 0.86608386493388712 > 0`

Result: `PASS`

Gate 2:

`mean(D_TEMPLATE) = 0.5853125793356333 > 0`

Result: `PASS`

Gate 3:

`p = 2.2607093023100906e-90 < 0.05`

Result: `PASS`

All three frozen positive-label gates passed.

Therefore the frozen protocol assigns:

`PP3_EXCLUDED_XG2_LIKE_RESIDUAL_TEMPLATE_TRANSPORT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

## Allowed descriptive results

`C_XG2`:

- mean: `0.86608386493388712`
- sample SD: `0.11622169218083186`

`C_XG4`:

- mean: `0.28077128559825382`
- sample SD: `0.2604699022409554`

`D_TEMPLATE`:

- mean: `0.5853125793356333`
- sample SD: `0.3453234709049785`

Fraction of items with `D_TEMPLATE > 0`:

`0.81333333333333335`

Mean residual net vector in `[P1, P2, P4, P5]` order:

`[2.8292869137243096e-08, 3.4347213212917032e-08, 2.8659583820718861e-09, 4.8317782684423612e-08]`

Residual effective absolute-net plane count:

`2.9969338136290076`

Maximum absolute Q reconstruction residual:

`6.3527471044072525e-22`

## Interpretation

The fresh XG1 `1201..1500` holdout supports prospective transport of an XG2-like PP3-excluded secondary residual geometry under the frozen test.

The result establishes that the fresh XG1 PP3-excluded local susceptibility residual is, on average, more cosine-aligned with the frozen XG2 residual template than with the frozen XG4 residual template, with all three pre-specified positive-label gates satisfied.

This result does not establish:

- causality of the residual template;
- necessity or sufficiency of P1, P2, P4, or P5;
- that XG1 and XG2 are globally mechanistically identical;
- universal family-conditioned residual geometry;
- behavioral or downstream-task effects;
- architecture-wide universality.

No individual residual plane is promoted by this result.

## Inferential boundary

Confirmatory p-value count:

`1`

Multiplicity correction:

`False`

Additional p-values:

`0`

Rescue analysis:

`False`

Outcome-dependent filtering:

`False`

Scientific conclusion:

`PP3_EXCLUDED_XG2_LIKE_RESIDUAL_TEMPLATE_TRANSPORT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`
