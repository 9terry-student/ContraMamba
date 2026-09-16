# Gen4-K XG2/XG4 Fresh Response Confirmatory H1/H2 Result

## Frozen evidence

Phase-2 artifact freeze:

`0ec7179fcce06765cffb104fd7519cf4ac64d1b7`

Phase-2 execution HEAD:

`bac46b860dfd472069be0e918a86ce56d1f0ec0e`

Frozen artifacts:

- `reports/reason_router_gen4_xg2_xg4_fresh_response_restartable_phase2_5bddad1_r1/xg2/`
- `reports/reason_router_gen4_xg2_xg4_fresh_response_restartable_phase2_5bddad1_r1/xg4/`

The confirmatory analysis was read-only.

- model forwards during analysis: 0
- additional scientific execution: 0
- family pooling: none
- additional p-values: none
- threshold scans: none

## Frozen tests

For each family independently:

H1:

`mean(R_ALIGN | LARGE) < 0`

One-sample Student t-test, one-sided less-than-zero.

H2:

`mean(R_ALIGN | LARGE) < mean(R_ALIGN | SMALL)`

Independent two-sample Welch t-test, one-sided LARGE < SMALL.

Multiplicity:

Holm correction over exactly H1 and H2 within each family.

Alpha:

`0.05`

## XG2

Group sizes:

- LARGE = 92
- SMALL = 208

Observed response:

- mean R_ALIGN LARGE = `-2.7252865002142935e-05`
- mean R_ALIGN SMALL = `7.630933112903642e-07`
- SD R_ALIGN LARGE = `0.00017108405543401277`
- SD R_ALIGN SMALL = `1.5513218909252825e-05`

H1:

- t = `-1.527905668898224`
- raw p = `0.065002264495718318`
- Holm-adjusted p = `0.12036446343727453`
- direction PASS
- Holm rejection = false

H2:

- t = `-1.5678394562507083`
- raw p = `0.060182231718637264`
- Holm-adjusted p = `0.12036446343727453`
- direction PASS
- Holm rejection = false

Frozen family result:

`XG2_CROSS_GENERATOR_ADVERSE_REGIME_NOT_ESTABLISHED`

The observed mean directions are consistent with the prospective adverse-regime
direction, but the frozen confirmatory rejection criteria are not satisfied.

## XG4

Group sizes:

- LARGE = 57
- SMALL = 243

Observed response:

- mean R_ALIGN LARGE = `-5.4470894031657115e-06`
- mean R_ALIGN SMALL = `-6.9938478707027219e-07`
- SD R_ALIGN LARGE = `3.5907976874623064e-05`
- SD R_ALIGN SMALL = `3.8874565987304408e-05`

H1:

- t = `-1.1452782007633111`
- raw p = `0.12848248735124052`
- Holm-adjusted p = `0.25696497470248103`
- direction PASS
- Holm rejection = false

H2:

- t = `-0.8840716768048783`
- raw p = `0.18951366869879699`
- Holm-adjusted p = `0.25696497470248103`
- direction PASS
- Holm rejection = false

Frozen family result:

`XG4_CROSS_GENERATOR_ADVERSE_REGIME_NOT_ESTABLISHED`

The observed mean directions are consistent with the prospective adverse-regime
direction, but the frozen confirmatory rejection criteria are not satisfied.

## Joint selected-family statement

The prospective conjunction:

`SELECTED_ELIGIBLE_FAMILY_RESPONSE_REPLICATION_SUPPORTED`

is not assigned because neither XG2 nor XG4 independently satisfies both frozen
confirmatory rejection requirements.

No pooled response test or pooled p-value is used.

## Scientific interpretation boundary

These results do not establish cross-generator replication of the adverse
LARGE-regime causal response under the frozen XG2/XG4 prospective design.

They also do not establish a zero effect or prove the absence of the proposed
mechanism.

The precise result is that both prospective families exhibited the predicted
mean direction, but neither family satisfied the pre-specified Holm-corrected
confirmatory rejection criteria.
