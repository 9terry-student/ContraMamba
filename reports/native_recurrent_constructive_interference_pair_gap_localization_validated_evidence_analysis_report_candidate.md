# Native Recurrent Constructive-Interference Pair-Gap Localization — Validated Evidence Analysis Report Candidate

Status: `VALIDATED_EVIDENCE_ANALYSIS_READY_FOR_FREEZE`

Execution commit:

`57ea0b5bc88d494279dbad14b22fe9e398436fbb`

Validated run:

`gen5-recurrent-pair-gap-localization-57ea0b5-r2`

Artifact root:

`reports/native_recurrent_constructive_interference_pair_gap_localization_runs/gen5-recurrent-pair-gap-localization-57ea0b5-r2`

## 1. Scope

This report interprets the validated recurrence-only pair-gap interference audit.

It does not introduce training, checkpoint fitting, projector refitting, downstream transport, confirmatory seed loading, VitaminC execution, threshold tuning, or a new causal intervention.

The scientific question is:

> Which source-token pair separations account for the PRIMARY_A-specific loss of complement-side constructive interference relative to matched CONTROL_R?

The exact additive diagnostic is:

`H_c(d) = C_c(d) / S_c`

with:

`Q_c = 1 + sum_d H_c(d)`.

The aggregate log endpoint remains:

`L_interference = log(Q_complement / Q_visible)`.

Gap terms are additive for `H`, not for `L_interference`.

## 2. Provenance and execution validity

The imported run passed the full frozen validation chain:

- execution HEAD: `57ea0b5bc88d494279dbad14b22fe9e398436fbb`;
- exact Python runtime: `3.12.13`;
- NumPy: `2.0.2`;
- PyTorch: `2.10.0+cu128`;
- Transformers: `5.0.0`;
- two independent Tesla T4 workers, no DDP;
- 840 dev rows;
- 60,094 valid tokens;
- 36 ordered orientations;
- 243 source-gradient forwards;
- 486 pair-gap group decompositions;
- recurrence-only;
- downstream stage transport calls: 0;
- final head transport calls: 0;
- training executed: false;
- parameter gradients accumulated: false.

Artifact validation:

- validated kernel replay max relative error:
  `4.401570390602549e-07`;
- pair-gap `C` reconstruction max absolute error:
  `0.04993460502009839`;
- pair-gap `Q` reconstruction max absolute error:
  `3.7932139633767292e-06`;
- raw reconstruction max absolute error:
  `7.450580596923828e-09`.

Imported artifact SHA256 identities:

- `pair_gap_localization_summary.json`:
  `8c30db83f4cfb17089778a43199d61b043b15cc67340c1639470cdd556d40cab`;
- `orientation_pair_gap_metrics.jsonl`:
  `ede01dc8ba696293005b36558c622ba621943c1b8942edfd02b63489ab8d0949`;
- `shard_manifest.json`:
  `8e69d080bb6f8d0eb97688e53f4843654f2a895d66caef9adb0dda22fde8832c`;
- `run_provenance.json`:
  `7e81ff74037dee483f04c7ca9234f9f2c73f14ec9443d61d2310dedfe16b5ad0`.

The failed earlier run identity
`gen5-recurrent-pair-gap-localization-4967c08-r1`
is not reused as scientific evidence.

## 3. Aggregate replay

The validated aggregate endpoint replayed the prior recurrent-kernel evidence.

Source-matched PRIMARY_A minus CONTROL_R:

- `Δ log Q_visible`:
  median `-0.00539111120273`,
  negative `6/9`,
  positive `3/9`,
  range `[-0.0160051963625, 0.0106885906286]`;
- `Δ log Q_complement`:
  median `-1.04353376191`,
  negative `9/9`,
  range `[-1.34700978917, -0.78276775795]`;
- `Δ L_interference`:
  median `-1.03744402508`,
  negative `9/9`,
  range `[-1.3499055141, -0.788123585285]`.

Therefore the factor-specific interference differential remains overwhelmingly complement-side.

The visible component remains an effective control: its source-matched log-Q shift is near zero in median and mixed in sign.

## 4. Fixed-band pair-gap structure

Source-matched PRIMARY_A minus CONTROL_R median `ΔH`:

Visible:

- gap 1: `-0.00207208361334`;
- gaps 2–4: `-0.0145868929604`;
- gaps 5–8: `-0.0435285594587`;
- gaps 9+: `-0.0306161040103`;
- total: `-0.115338291377`.

Complement:

- gap 1: `-0.494831603369`;
- gaps 2–4: `-1.32525162835`;
- gaps 5–8: `-1.29162511924`;
- gaps 9+: `-3.29216284767`;
- total: `-6.4021552872`.

Every complement band is negative in `9/9` matched source cells.

The large magnitude of the `9+` band must not be interpreted as evidence that very long gaps dominate. That band aggregates many more individual gap indices than the shorter fixed bands.

## 5. Full-gap localization

There are 127 resolved source-token gaps.

For the complement component:

- 113/127 gaps are negative in all `9/9` matched source cells;
- 8/127 gaps are positive in all `9/9`;
- 6/127 gaps are mixed-sign across matched source cells.

The largest absolute median deficits occur at the shortest gaps and decay smoothly with separation:

- `d=1`: median `-0.494831603369`;
- `d=2`: `-0.479856866741`;
- `d=4`: `-0.403763707395`;
- `d=8`: `-0.276312358008`;
- `d=16`: `-0.146892151047`;
- `d=32`: `-0.0396122975481`;
- `d=64`: `-0.0012797827901`;
- `d=96`: approximately `-2.55e-10`;
- `d=127`: approximately `-3.23e-10`.

Thus the complement-side deficit is strongest at short separations and exhibits a smooth decay rather than a narrow isolated peak.

## 6. Cumulative concentration

Median cumulative complement `ΔH`:

- `d <= 1`: `-0.494831603369`;
- `d <= 4`: `-1.81837206669`;
- `d <= 8`: `-3.10999243953`;
- `d <= 16`: `-4.69812203818`;
- `d <= 32`: `-5.95507559955`;
- `d <= 64`: `-6.39841127382`;
- `d <= 127`: `-6.4021552872`.

Median fraction of each source-matched signed total accumulated by each prefix:

- `d <= 1`: `0.0803069979556`;
- `d <= 4`: `0.29318525855`;
- `d <= 8`: `0.494731375147`;
- `d <= 16`: `0.741905667197`;
- `d <= 32`: `0.931961790768`;
- `d <= 64`: `0.999457808102`.

Therefore approximately half of the signed complement deficit is already accumulated by gap 8, roughly three quarters by gap 16, over 93% by gap 32, and effectively all of it by gap 64.

## 7. Short-to-long decomposition

Complement source-matched `ΔH` block medians:

- gaps 1–8: `-3.10999243953`, negative `9/9`;
- gaps 9–16: `-1.58812959866`, negative `9/9`;
- gaps 17–32: `-1.25695356137`, negative `9/9`;
- gaps 33–64: `-0.443335674272`, negative `9/9`;
- gaps 65–127: `-0.00374401337463`, negative `9/9`.

This provides the cleanest scale interpretation:

- gaps 1–32 contain the dominant effect;
- gaps 33–64 form a smaller decaying tail;
- gaps 65+ are negligible in aggregate magnitude.

## 8. Scientific conclusion

The original strong localization hypothesis — that a narrow bounded source-token separation range would dominate the PRIMARY_A-specific interference differential — is not supported.

The supported result is:

> PRIMARY_A exhibits a broad but strongly distance-decaying loss of complement-side constructive cross-token coherence. The deficit is maximal at the shortest source-token separations, remains substantial through short-to-mid gaps, and is practically saturated by about 32 tokens, with only a small tail through 64 tokens and negligible aggregate mass beyond 64.

This is therefore better described as a **short-to-mid-range distributed coherence deficit** than as a single-gap or narrow-band localization.

The visible-side control remains qualitatively distinct:

- aggregate `Δ log Q_visible` is near zero;
- sign is mixed across matched source cells;
- full-gap visible profiles show extensive mixed-sign structure rather than the complement-side coherent negative pattern.

The result strengthens the prior conclusion that the PRIMARY_A-specific `L_interference` shift is driven by loss of complement constructive amplification rather than by a matched visible-side change.

## 9. Falsification outcome

The pair-gap analysis does not falsify the aggregate complement-side interference mechanism.

It does falsify or narrow the stronger temporal-localization version of the hypothesis:

- no isolated narrow gap regime uniquely carries the effect;
- the deficit is distributed over many consecutive short-to-mid gaps;
- the profile decays smoothly with temporal separation;
- the effect is almost entirely accumulated by `d <= 64`, with the dominant mass already present by `d <= 32`.

## 10. Claim boundary

This evidence supports a statement about source-token temporal separation within the frozen recurrence-only population.

It does not establish:

- semantic identity of the interacting tokens;
- lexical or syntactic causes;
- a unique native state coordinate;
- a causal effect of any particular token pair;
- universality beyond the frozen population;
- an additive attribution of `L_interference` to individual gaps;
- a new training rule or architecture.

No p-values or post-hoc mechanism thresholds are introduced.

## 11. Recommended next scientific question

The next useful question is no longer “which single pair gap causes the deficit?”

The evidence instead motivates a bounded follow-up on **what distinguishes the interacting token pairs inside the dominant `d <= 32` regime**.

A future analysis may examine token-relative roles or pre-existing frozen token classes inside that window, but only after a separate design freezes:

- the exact token-level quantity to inspect;
- how to avoid post-hoc semantic fishing;
- how visible-side controls are retained;
- how the analysis avoids treating correlation within the pair-gap profile as token-pair causality.

No new training or confirmatory-seed execution is authorized by this report.
