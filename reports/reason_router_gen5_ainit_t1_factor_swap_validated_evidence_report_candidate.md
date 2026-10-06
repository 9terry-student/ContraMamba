# Gen5 M7 t1 A0/B1 Factor-Swap — Validated Evidence Report

## Status

`PASS_VALIDATED_M7_FACTOR_SWAP_EVIDENCE`

Execution commit: `c95194f1169ae75db242bdb751bd98649486c95f`
Authority commit: `e05948ffacab4210158fbd87eeac7bb25b733556`
Run: `gen5-m7-t1-factor-swap-c95194f-r1`

This report is a static validation and bounded interpretation of the imported
M7 artifacts. It performs no model forward, CUDA work, training, backward,
optimizer construction, parameter update, or new-seed evaluation.

## 1. Provenance and execution validity

- Internal artifact manifest: PASS; `40` manifested run files authenticated by byte size and SHA256.
- Execution HEAD and implementation freeze: `c95194f1169ae75db242bdb751bd98649486c95f`.
- Phase-A trajectory SHA256: `0f7cd4248faa92223597e0816597b59e426f08829dadd366f9603f56a8de809e`.
- Frozen temporal behavioral-coordinate SHA256: `41518c0dbac345b972bf61920fe98681541f6393d6770494acb0df1cce149101`.
- Conceptual states: 27 = 9 frozen matched references + 18 new cross-A interventions.
- Dev rows: 840; frozen vulnerable subset: 120 rows.
- Row batch size: 32.
- Worker partition: `[0,420)` and `[420,840)`.
- Planned new downstream cross-hybrid calls: 252 per worker, 504 total.
- Full 840-row matched-anchor recomputation: 0.
- Matched-anchor authentication prediction mismatch: 0.
- Matched-anchor authentication max logit error: `5.60283660889e-06` <= `5e-5`.
- Frozen B_UPDATE_ONLY vs FULL_T1 prediction mismatch: 0.
- Frozen B_UPDATE_ONLY vs FULL_T1 max logit error: `1.43051147461e-06` <= `5e-5`.
- Training/backward/optimizer/optimizer-step/parameter-grad/confirmatory-seed flags: all false.
- Both workers: 14/14 durable chunks authenticated; worker results and provenance SHA links PASS.

## 2. Finite factor-effect decomposition

Two-margin mean pairwise L2 effect:

| Factor axis | Mean pair effect |
|---|---:|
| recipient A0 | 0.0428052854608 |
| donor A / B1-history | 0.0185676429055 |
| donor RNG | 0.000154306412049 |

Ratios:

- recipient-A / donor-A = `2.30536992114`
- donor-A / donor-R = `120.329691157`
- recipient-A / donor-R = `277.404450614`

Thus, within this frozen finite intervention, the recipient-A axis is the
largest of the three measured factor effects, donor-A/B1-history is second,
and donor-R is much smaller than either A-linked axis. This statement is
descriptive of the preregistered 27-state finite grid, not a global
training-dynamics claim.

## 3. Cross-A hybrid affinity to matched references

The preregistered affinity is

`S = (d_donor - d_recipient) / (d_donor + d_recipient + eps)`,

so positive values mean the cross-A hybrid is closer to the matched recipient
reference and negative values mean it is closer to the matched donor reference.

### All 840 dev rows x 18 cross-A hybrids

Two-margin coordinates:

- observations: 15120
- mean affinity: `0.300482146398`
- median affinity: `0.43731331113`
- positive / negative / zero: `10171 / 4949 / 0`
- positive fraction: `0.672685185185`
- mean distance to recipient / donor: `0.0193864457537 / 0.0447792156849`
- hybrid-level mean-affinity signs: `{'positive': 12, 'negative': 6, 'zero': 0}`
- primary descriptive class: `RECIPIENT_CLOSER_ON_AVERAGE`

Centered-logit coordinates:

- mean affinity: `0.299794179153`
- median affinity: `0.449019728721`
- positive / negative / zero: `10131 / 4989 / 0`
- positive fraction: `0.67003968254`
- mean distance to recipient / donor: `0.0162491481193 / 0.0375694728981`
- hybrid-level mean-affinity signs: `{'positive': 12, 'negative': 6, 'zero': 0}`

Prediction relationships over the 15,120 cross-hybrid/row observations:

- recipient-only agreement: `218` (1.4417989418%)
- donor-only agreement: `70` (0.462962962963%)
- agrees with both references: `14787` (97.7976190476%)
- agrees with neither reference: `45` (0.297619047619%)

Two-margin recipient-to-donor segment diagnostics:

- outside-segment observations: `9156` / `15120`
- degenerate recipient/donor reference pairs: `0` / `15120`

## 4. Frozen vulnerable-120 subset

Across 2,160 cross-hybrid/vulnerable-row observations:

Two-margin coordinates:

- mean affinity: `0.298874165068`
- median affinity: `0.508074518461`
- positive / negative / zero: `1416 / 744 / 0`
- positive fraction: `0.655555555556`
- mean distance to recipient / donor: `0.0149057083581 / 0.0317869301654`
- hybrid-level mean-affinity signs: `{'positive': 12, 'negative': 6, 'zero': 0}`

Centered-logit coordinates:

- mean affinity: `0.299065852107`
- median affinity: `0.516509344868`
- positive / negative / zero: `1415 / 745 / 0`
- positive fraction: `0.655092592593`
- mean distance to recipient / donor: `0.0141036238743 / 0.0300476300069`
- hybrid-level mean-affinity signs: `{'positive': 12, 'negative': 6, 'zero': 0}`

Prediction relationships:

- recipient-only: `0` (0%)
- donor-only: `0` (0%)
- both: `2160` (100%)
- neither: `0` (0%)

Two-margin outside-segment observations:
`1186 / 2160`.

## 5. What M7 establishes

The validated finite intervention directly separates recipient `A0`, donor
`B1` history, and donor training-RNG effects at t1 while holding the frozen
downstream model and dev inputs fixed.

The factor-effect magnitudes establish the ordering

`recipient-A > donor-A/B1-history >> donor-R`

for mean two-margin pairwise effect on this frozen grid.

Reference-affinity and prediction diagnostics above determine whether the
cross-A hybrid outputs are predominantly recipient-like, donor-like, or mixed.
Off-segment / neither-reference observations are retained as compatibility
diagnostics; they are not silently reclassified as recipient or donor effects.

## 6. Claim boundary

This evidence does not establish global training-dynamics causality,
uniqueness of the latent factorization, gauge equivalence, unseen-seed
generalization, or a universal Mamba mechanism.

No projector-recovery claim is introduced here. The earlier
earliest-internal-task-visible-stage claim remains blocked.

M8 representational-equivalence / task-relevant quotient interpretation may
proceed only after this M7 evidence is frozen.
