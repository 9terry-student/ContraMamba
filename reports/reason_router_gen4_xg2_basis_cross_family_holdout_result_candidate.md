# Gen4-K XG2-Basis Cross-Family Fresh-Index Holdout Result

## Status

Prospective result candidate based exclusively on the imported frozen observation artifacts from source-pair indices 601..900.

No additional model execution, training, backward pass, task-head evaluation, logits read, subgroup analysis, tail analysis, basis-wise significance test, epsilon sweep, k sweep, post-hoc thresholding, or rescue analysis was performed for the confirmatory inference.

## Evidence identity

Execution HEAD:

`c8fcb97e9471cb32d3456bb0feb01d52c362943c`

Execution run:

`g4k-xg2-basis-cross-family-holdout-c8fcb97-retry1`

Authorized runner:

`scripts/reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda.py`

Runner Git blob:

`2d35e5ed936fd37f4ecfc063e060290304c6bf10`

Imported artifact root:

`reports/reason_router_gen4_xg2_basis_cross_family_holdout_c8fcb97_retry1/`

Collector/import provenance:

- command SHA256: `50f3638827730c4591294af18be5464a69b0b2221f76f6ad733eb83defcfa7b7`
- run log SHA256: `be5b192e7db9b7f4218c8dc6cd78449f374456f7609d087f651e017c761e818b`
- run meta SHA256: `e97f9a58faa43e8bd39b68f5065804b59c68abf2909ad35404ce05347f018340`
- handoff ZIP SHA256: `0d16cd17d5e396711011849d17c9db54c44ac90ff430d7eb2f6fa96e52451dc0`
- collector files: `8`
- execution exit code: `0`
- import validation: `8`
- imported files copied: `8`
- import result: `PASS`

Observation artifact identities:

### XG2

- source pairs: `xg2_fact_601..xg2_fact_900`
- N: `300`
- items SHA256: `41677dd4e5594f9a3eb3f33477c014b44368644fe2c1fe9038fd647b61119f6e`
- summary SHA256: `5e38a8c9511181ae3fd07e7747258985f22b418e42db61af32d6566adcfcd030`
- scientific forwards: `12000`
- new baseline forwards: `0`

### XG4

- source pairs: `xg4_fact_601..xg4_fact_900`
- N: `300`
- items SHA256: `739fdc74ae64d4a30bfc1d772c3be4516c762d58313be45397633104207a8c7a`
- summary SHA256: `c75a3c4450db40104848ae070bfd15e835a4b4689f6dc61231737413f464e4d4`
- scientific forwards: `12000`
- new baseline forwards: `0`

Total new scientific forwards: `24000`.

Both imported artifact sets passed the frozen runner's `validate_artifact()` before inference.

## Frozen endpoint

The frozen XG2 and XG4 five-dimensional bases were reconstructed from the previously frozen Phase-1 evidence.

For every fresh source pair:

`E_XG2 = (1/5) * sum_j J(v_XG2,j)^2`

`E_XG4 = (1/5) * sum_j J(v_XG4,j)^2`

and the pre-specified primary endpoint was:

`Q = E_XG2 - E_XG4`

with:

- subspace dimension: `k = 5`
- finite-difference epsilon: `0.025`
- model forwards per direction: `4`
- model forwards per pair: `40`

Negative `Q` values were retained without clipping or replacement.

The scientific hypothesis was evaluated separately within the XG2 and XG4 source families:

`H1: mean(Q) > 0`

## Pre-specified inference

For each family independently:

- one-sample Student t-test;
- one-sided alternative `mean(Q) > 0`;
- N = `300`.

Multiplicity family consisted of exactly two tests:

- XG2 source family;
- XG4 source family.

Holm correction was applied with familywise `alpha = 0.05`.

The pre-specified positive result label:

`XG2_BASIS_CROSS_FAMILY_SENSITIVITY_SUPPORTED_ON_FRESH_INDEX_HOLDOUT`

required:

1. positive mean `Q` in XG2 and Holm rejection; and
2. positive mean `Q` in XG4 and Holm rejection.

Otherwise the frozen result label was:

`XG2_BASIS_CROSS_FAMILY_SENSITIVITY_NOT_ESTABLISHED`

## Confirmatory results

| Source family | N | Mean Q | SD Q | t | df | Raw one-sided p | Holm adjusted p | Holm reject |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| XG2 | 300 | 1.3987206143704026e-07 | 6.4833768577290609e-08 | 37.367181066998441 | 299 | 5.5562279414683582e-115 | 5.5562279414683582e-115 | TRUE |
| XG4 | 300 | 3.7773950385492776e-07 | 6.1895400181421826e-08 | 105.70478755850664 | 299 | 3.6505385540722314e-239 | 7.3010771081444628e-239 | TRUE |

Holm step-down order was XG4 first and XG2 second:

- XG4 threshold: `0.025`
- XG2 threshold: `0.05`

Both family means were positive and both family hypotheses were rejected after the frozen Holm correction.

Therefore the pre-specified joint result is:

`XG2_BASIS_CROSS_FAMILY_SENSITIVITY_SUPPORTED_ON_FRESH_INDEX_HOLDOUT`

## Interpretation

On the prospectively held-out same-generator source-pair indices 601..900, the frozen XG2 five-dimensional Phase-1 basis has greater mean local squared directional sensitivity than the frozen XG4 five-dimensional basis in both tested source families.

For XG2 source pairs, this reproduces the directional pattern observed in the earlier 301..600 family-subspace experiment: the XG2 basis produces larger local squared sensitivity than the XG4 basis.

For XG4 source pairs, the fresh holdout prospectively confirms the distinct cross-family hypothesis motivated after the earlier symmetric family-specific test failed: the XG2 basis also produces larger local squared sensitivity than the XG4 basis.

The result is therefore evidence for an XG2-basis cross-family sensitivity effect under the frozen checkpoint, layer/intervention semantics, tokenizer, anchor construction, k=5 basis, epsilon=0.025, and the same XG2/XG4 generator families.

## Scope limitations

This result does not establish:

- generator-independent or external replication;
- transport to XG3 or to a new generator family;
- transport to independently authored or natural data;
- that the XG2 basis is globally optimal;
- superiority of `k = 5` over another dimension;
- robustness to alternative epsilon values;
- robustness to alternative checkpoints, layers, offsets, or intervention definitions;
- that every individual XG2 basis direction is independently significant;
- a universal signed causal direction;
- downstream task-head or logit effects.

The holdout is fresh-index within the same frozen synthetic generator semantics. It is stronger evidence than reusing indices 301..600, but it is not an independent generator or external-data replication.

No subgroup, tail, opposite-direction, per-basis, epsilon-sweep, k-sweep, or rescue inference was executed.

## Scientific disposition

The prospectively frozen hypothesis:

> the frozen XG2 top-5 Phase-1 subspace has greater local squared first-order sensitivity than the frozen XG4 top-5 subspace on fresh source-pair indices, separately within both XG2 and XG4 source families

is supported on the pre-specified fresh-index holdout.

Final frozen label:

`XG2_BASIS_CROSS_FAMILY_SENSITIVITY_SUPPORTED_ON_FRESH_INDEX_HOLDOUT`
