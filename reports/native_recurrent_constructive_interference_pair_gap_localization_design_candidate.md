# Native Recurrent Constructive-Interference Pair-Gap Localization — Scientific Design Candidate

Status: `SCIENTIFIC_DESIGN_READY_FOR_FREEZE`

Base validated evidence commit:

`d64a4f6f60419de529eb239516b82ae62e613b17`

Source validated run:

`gen5-recurrent-kernel-decomposition-09577cf-r1`

## 1. Scientific target

The validated recurrent-kernel decomposition established that the
`raw_write -> recurrent_state` selective-retention effect is jointly produced by
self-propagation and cross-token interference, with the larger
PRIMARY_A-versus-CONTROL_R differential carried by `L_interference`.

The next question is narrower:

> Which temporal separation between distinct raw-write tokens produces the
> PRIMARY_A-specific loss of complement-side constructive interference?

This is not a new kernel-gain sweep and not a downstream transport study.

No training, checkpoint fitting, projector refit, downstream basis search,
mechanism threshold tuning, confirmatory 9601/9900 loading, or VitaminC
execution is allowed.

## 2. Corrected interpretation of the validated interference term

For component `c` in `{VISIBLE, COMPLEMENT}`, the validated audit defines:

`S_c =` propagated self-energy,

`E_c =` full recurrent energy,

`C_c = E_c - S_c`,

`Q_c = E_c / S_c = 1 + C_c / S_c`.

The validated interference term is:

`L_interference = log(Q_complement / Q_visible)`.

The imported 36-orientation artifact shows:

- `C_visible > 0` in `36/36` orientations;
- `C_complement > 0` in `36/36` orientations;
- `Q_visible > 1` in `36/36` orientations;
- `Q_complement > 1` in `36/36` orientations.

Therefore the observed negative `L_interference` must not be described as
evidence that the complement is undergoing net destructive cancellation.

The supported interpretation is instead:

> Both visible and complement components receive net constructive cross-token
> interference, but the visible component receives much stronger constructive
> amplification. The PRIMARY_A-specific differential is dominated by a
> reduction of complement-side constructive amplification relative to matched
> CONTROL_R.

Orientation-level medians:

- PRIMARY_A:
  - `Q_visible ≈ 21.1975`
  - `Q_complement ≈ 3.70067`
- CONTROL_R:
  - `Q_visible ≈ 21.3580`
  - `Q_complement ≈ 10.1463`

In log space:

- PRIMARY_A median `log Q_visible ≈ 3.05388`;
- CONTROL_R median `log Q_visible ≈ 3.06143`;
- PRIMARY_A median `log Q_complement ≈ 1.30851`;
- CONTROL_R median `log Q_complement ≈ 2.31711`.

Across the nine source-matched cells:

- `Δ log Q_complement = PRIMARY_A - CONTROL_R` is negative in `9/9`;
- `Δ log Q_visible` has mixed sign (`3/9` positive, `6/9` negative);
- median `Δ log Q_complement ≈ -1.04353`;
- median `Δ log Q_visible ≈ -0.00539`;
- median `ΔL_interference ≈ -1.03744`.

Thus the next analysis should target the temporal structure of the
**complement-side constructive-interference deficit**, while continuing to
measure the visible side as a mandatory control.

## 3. Existing self-propagation context

The existing artifact already contains lag-resolved **self-energy** diagnostics.
These do not localize cross-token interference, but they constrain the next
hypothesis.

Across ordered orientations:

PRIMARY_A:

- visible mean self-propagation lag median: approximately `12.87`;
- complement mean self-propagation lag median: approximately `1.13`;
- visible lag-0 self-energy fraction median: approximately `0.0545`;
- complement lag-0 self-energy fraction median: approximately `0.7814`.

CONTROL_R:

- visible mean self-propagation lag median: approximately `12.77`;
- complement mean self-propagation lag median: approximately `3.01`;
- visible lag-0 self-energy fraction median: approximately `0.0542`;
- complement lag-0 self-energy fraction median: approximately `0.5475`.

This already shows that complement self-propagation is much shorter-lived than
visible self-propagation, especially under PRIMARY_A.

However, self-energy lag summaries cannot identify which *pairs of distinct
tokens* generate `C_c`. A separate exact cross-token decomposition is required.

## 4. Exact pair-gap interference identity

For a fixed component `c`, example, and recurrent coordinate, write:

`z_(t,tau)^c = K_(t,tau) * delta_w_tau^c`

for the propagated contribution of source token `tau` to recurrent target time
`t`.

Then:

`E_c = sum_t || sum_(tau <= t) z_(t,tau)^c ||^2`

and:

`S_c = sum_t sum_(tau <= t) ||z_(t,tau)^c||^2`.

Therefore the exact cross-token term is:

`C_c = 2 * sum_t sum_(tau < sigma <= t)
             <z_(t,tau)^c, z_(t,sigma)^c>`.

Define source-token pair gap:

`d = sigma - tau`, with `d >= 1`.

Define the exact gap-resolved interference:

`C_c(d) =
  2 * sum_t sum_(tau < sigma <= t, sigma - tau = d)
      <z_(t,tau)^c, z_(t,sigma)^c>`.

Then exactly:

`C_c = sum_(d >= 1) C_c(d)`.

Normalize by propagated self-energy:

`H_c(d) = C_c(d) / S_c`.

Then:

`Q_c = 1 + sum_(d >= 1) H_c(d)`.

This is the primary next-stage identity.

`H_c(d)` may be positive or negative even though the aggregate `C_c` is
positive. Therefore preserve signed values and never clamp pair-gap
interference to zero.

## 5. Primary scientific quantities

For every ordered orientation and both components report:

- exact `S_c`;
- exact `C_c`;
- exact `Q_c`;
- full signed vector `C_c(d)` for `d = 1..L-1`;
- full signed vector `H_c(d) = C_c(d)/S_c`;
- reconstruction error:
  `abs(C_c - sum_d C_c(d))`;
- reconstruction error:
  `abs(Q_c - (1 + sum_d H_c(d)))`.

For compact descriptive summaries, use the already-established temporal bands:

- gap `1`;
- gaps `2..4`;
- gaps `5..8`;
- gaps `9+`.

These bands are descriptive aggregations of the exact full vector, not fitted
bins and not mechanism thresholds.

For each band report signed:

`H_c(B) = sum_(d in B) H_c(d)`.

Do not take absolute values before aggregation.

## 6. Factor-specific comparison

The primary factor-specific question is whether the validated
PRIMARY_A-specific interference differential is concentrated on the complement
side and, if so, at which pair gaps.

For each source cell, compare the mean of its two PRIMARY_A orientations with
the mean of its two matched CONTROL_R orientations.

Mandatory quantities:

- `Δ log Q_visible`;
- `Δ log Q_complement`;
- `ΔL_interference`;
- `ΔH_visible(B)` for each fixed pair-gap band;
- `ΔH_complement(B)` for each fixed pair-gap band;
- the full source-matched `ΔH_c(d)` vectors.

Primary interpretation target:

> Does the `9/9` negative `Δ log Q_complement` arise predominantly from a
> bounded region of pair-gap space, or is the deficit broadly distributed over
> temporal separations?

The visible-side profile is a mandatory negative/control comparison because the
validated source-matched `Δ log Q_visible` is near zero in median and mixed in
sign.

## 7. Non-additivity boundary for the log statistic

The gap terms are exactly additive for:

`C_c / S_c = sum_d H_c(d)`.

They are **not** additive components of:

`L_interference =
 log[(1 + sum_d H_complement(d)) /
     (1 + sum_d H_visible(d))]`.

Therefore:

- do not assign percentages of `L_interference` to individual gap bands by
  naively normalizing log changes;
- do not introduce Shapley attribution, arbitrary ordering, or post-hoc
  decomposition unless separately authorized;
- use the exact additive `H_c(d)` profiles to localize cross-token interaction
  structure;
- retain `L_interference` only as the authenticated aggregate endpoint.

This prevents a nonlinear log-ratio from being misrepresented as an additive
lag decomposition.

## 8. Optional cumulative diagnostic

A cumulative profile may be reported descriptively:

`Q_c(<=D) = 1 + sum_(1 <= d <= D) H_c(d)`.

When both component cumulative `Q` values are positive, also report:

`L_interference(<=D) =
 log[Q_complement(<=D) / Q_visible(<=D)]`.

This is a descriptive prefix trajectory only.

It must not be interpreted as an additive contribution or a causal percentage
for gap `D`.

If any cumulative `Q_c(<=D) <= 0`, omit the corresponding log value and report
the signed cumulative `Q` without transformation.

## 9. Exact computational form

For each recurrent coordinate, let `P_t` be the prefix sum of native
`log A_t`.

For a pair `tau < sigma`, its total contribution over valid recurrent target
times is:

`2 * delta_w_tau * delta_w_sigma
   * exp(P_sigma - P_tau)
   * T_sigma`

where:

`T_sigma =
 sum_(t >= sigma, valid t) exp(2 * (P_t - P_sigma))`.

`T_sigma` can be computed by a backward stable recurrence using only
non-positive native log-decay differences.

The remaining all-gap pair correlation can be computed on GPU by an
overflow-safe cross-correlation using the same structural principle as the
validated dyadic FFT path:

- source time is always earlier than target/source-partner time;
- exponent differences are therefore non-positive under the frozen native
  stable recurrence;
- use batched GPU FFT/dyadic rectangles for large-span cases;
- no CPU-only pair loop as the scientific execution path;
- no duplicate pair accounting;
- each ordered token pair `tau < sigma` must contribute to exactly one pair-gap
  entry.

The exact implementation may choose an algebraically equivalent stable GPU
form, but it must authenticate against brute force on small tensors.

## 10. Efficiency boundary

The next audit should preserve the existing efficient execution structure.

Required:

- recurrence-only;
- same 840 examples / 60,094 valid tokens;
- same 36 ordered orientations;
- same two independent single-GPU workers;
- no DDP;
- same source-local two-margin projector;
- no downstream projector refit;
- no downstream readout/gate/out-projection/head transport;
- no additional model forwards solely for pair-gap accounting.

Within a batch/source context:

- compute recurrence coefficients once;
- compute the backward survival factor needed for pair interactions once and
  reuse it across visible/complement target groups when algebraically valid;
- reuse the existing source-gradient projector calculation exactly as before;
- perform pair-gap decomposition as deterministic GPU arithmetic after the
  frozen raw-write components are constructed.

A new run must not duplicate the prior downstream transport work.

## 11. Mandatory validation

Before any scientific execution, implementation validation must include:

1. exact small-tensor brute-force `C(d)` comparison;
2. exact identity `sum_d C(d) = C`;
3. exact identity `1 + sum_d H(d) = Q`;
4. mixed positive/negative pair-interference synthetic case;
5. non-causal padding-tail invariance;
6. large active-prefix dynamic-range regression beyond float64 centered-exp
   range;
7. no duplicate token-pair accounting;
8. exact lag-index direction;
9. frozen 36-orientation population and worker shards;
10. no downstream transport calls;
11. no training/optimizer/backward/checkpoint mutation;
12. replay of aggregate `S`, `E`, `C`, `Q`, and `L_interference` against the
    validated `d64a4f6` artifacts.

Tolerance widening after seeing scientific output is forbidden.

## 12. Falsification / decision logic

The working hypothesis is:

> The dominant A-init-specific `L_interference` differential is a
> complement-side loss of constructive cross-token coherence, and this loss may
> be temporally localized by source-token pair separation.

This hypothesis is falsified or narrowed if any of the following occurs:

- pair-gap terms fail exact reconstruction of aggregate `C` or `Q`;
- the previously observed `9/9` negative source-matched
  `Δ log Q_complement` does not replay;
- visible-side `Δ log Q_visible` becomes comparably large under the exact same
  frozen population, indicating the static readout was misleading;
- the complement deficit is broadly distributed across all gaps rather than
  concentrated in a bounded temporal regime;
- the dominant signed pair-gap terms cancel strongly across bands, making a
  one-band localization unsupported.

No threshold is required to distinguish these outcomes. Report the continuous
profiles and sign structure.

## 13. Claim boundary

This audit may establish the temporal pair-separation structure of the
aggregate constructive-interference differential.

It does not establish:

- semantic identity of the interacting tokens;
- a unique lexical or syntactic cause;
- a unique native state dimension;
- universality beyond the frozen population;
- causality of a particular token pair without a separately designed
  intervention;
- a new architecture or training rule.

If pair-gap localization succeeds, semantic/token-class inspection is a later
question and must be separately authorized.

## 14. Next action

Freeze this scientific design before implementation.

After freeze, one bounded implementation may add a recurrence-only
pair-gap-interference audit and its tests.

Training/Evaluation allowed: `NO` for the design and implementation phases.

Kaggle scientific execution allowed: `NO` until a separately frozen
implementation commit passes local validation and a fresh execution gate.
