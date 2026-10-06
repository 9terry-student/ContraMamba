# Gen5 A-init Behavioral Onset — Validated Evidence Freeze

## Status

`VALIDATED_STATIC_FREEZE`

This report freezes the successful behavioral-onset run `gen5-ainit-behavioral-onset-1633d67-r1` and a
read-only static alignment against the already frozen Phase A operator trajectory.
It is an evidence report, not a new execution or implementation authority.

## Provenance

- source execution HEAD: `1633d67146a7f049c6ecf28b333f53fdf79ad00e`
- source ZIP SHA256: `dffbdf75b1d9d485cee30309a443553aea6995d5f109e7c4720c8c994bf62afd`
- source manifest SHA256: `20fa51056fbbe0ce0bff695a59055d440b6070776d0378ef24c723cd3a8f0ce3`
- Phase A summary SHA256: `37cab4866c729ad03262df562762ac8b6b91e4e24e2498223483848871ad1df6`
- behavioral run status: `PASS_BEHAVIORAL_ONSET_SCAN`
- endpoint authentication: `PASS`
- training executed by behavioral scan: `false`
- optimizer constructed by behavioral scan: `false`
- backward executed by behavioral scan: `false`
- confirmatory seeds 9601..9900 loaded: `false`

All eight manifest-listed payloads were byte-count and SHA256 authenticated before
being copied into the repository.

## Frozen behavioral result

- t0: all 9 cells have identical predictions.
- first same-training-RNG / different-A prediction disagreement: **t=1**.
- first same-A / different-training-RNG prediction disagreement: **t=2**.
- first newly entered decisive-wrong state since t0: **t=4**.
- peak A-init behavioral disagreement: **t=10, count=868**.
- peak accuracy spread: **t=11, spread=0.186904761904762**.
- permanent prediction reconvergence across all 9 cells: **t=17**.
- t20: all 9 cells again have identical predictions and accuracy.
- t20 per cell: 240 wrong predictions, but 0 decisive-wrong predictions.

## Shared transient vulnerability set

Each of the 9 `(A,R)` cells has exactly 120 `new_decisive_wrong_since_t0` event
rows. The row identities are exactly the same in all 9 cells:

- intersection count: **120**
- union count: **120**
- pairwise Jaccard min/mean/max: **1 / 1 / 1**
- re-entry rows: **0 in every cell**
- persistent-at-t20 rows: **0 in every cell**

A-init changes onset timing much more strongly than training RNG:

- A6201: onset min/median/max = **6 / 9 / 12**
- A6202: onset min/median/max = **4 / 7 / 11**
- A6203: onset min/median/max = **6 / 8 / 11**

Thus the current executed evidence supports the bounded description that the
vulnerability **identity** is shared, while A-init strongly shifts its transient
timing.

## Geometry × behavior alignment

The already frozen Phase A operator geometry is nonzero from t=1 and continues
to separate strongly by A-init. In contrast, behavioral A-init disagreement
peaks at t=10 and permanently returns to zero at t=17.

The descriptive Pearson correlation between
`log(operator A mean-squared distance)` and A-init behavioral disagreement over
t=1..20 is **-0.100529131314**.

Therefore a monotone explanation of the form “larger representation/operator
distance implies larger behavioral divergence” is not supported by this
trajectory. Behavior reconverges while operator separation continues to grow.

## Interpretation boundary

`t=4` is **not** “hallucination birth.” It is the first step at which a row that
was not decisive-wrong at t0 newly enters a SUPPORT/REFUTE wrong state. The
correct terms here are `transient decisive-error entry` and `behavioral
excursion`.

This freeze also does not establish why the 120 rows are vulnerable or prove a
causal factor. Those questions remain for the next mechanism-closure stages:
first-update origin decomposition, excursion/reconvergence decision geometry,
and bounded factor-swap interventions.
