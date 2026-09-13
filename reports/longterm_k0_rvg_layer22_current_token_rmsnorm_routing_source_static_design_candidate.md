# ContraMamba K0-RVG Layer-22 Current-Token RMSNorm Routing-Source Decomposition
## Static Design Candidate

## 1. Status

**Phase:** new K0 scientific static design.

**Parent evidence freeze commit:**

`8b494a72d48528c3bdb8985a1907766fded040e0`

**Parent validated report:**

`reports/longterm_k0_rvg_layer22_current_token_inproj_strong_routing_validated_evidence_analysis_report_candidate.md`

**Parent report SHA256:**

`391f671d22f7b369f109288b649f61c667e83696c920684147445f7d3dc7e8ce`

**Parent corrected implementation commit / runtime HEAD:**

`cd602f036b03e36e171837d9532541a530799954`

**Parent runner SHA256:**

`8d6c45636197ee6cc1d9e6f8c423a1f8fe57fc654ae450a4b3eb6b8657dfaf5b`

This document opens one new scientific question upstream of the now-closed fixed layer-22 hidden in-projection routing decomposition.

It does not authorize training, evaluation, intervention, learned geometry, post-hoc channel search, or K1.

A bounded static/runtime preflight after implementation will not require a separate authority document.

---

## 2. Frozen parent result

The immediately preceding validated stage established at layer 22, current token, common-330, `k=2`:

- the fixed bias-free hidden in-projection is:
  `H_t = W_H X_t`;
- all `1536` hidden-output channels are preregistered;
- the downstream lag-0 strong/weak partition remains:
  `240 / 1296 / 0`;
- corr strong-partition transfer exceeds ctrl for:
  `330/330`;
- corr weak-partition transfer exceeds ctrl for only:
  `76/330`;
- median enrichments are:
  - `G_H = +0.025515793449323046`;
  - `G_S = +0.1968593236956881`;
  - `G_W = -0.03423699371158165`;
- mean transfer-squared role differences are:
  - all:
    `+1.1589093253784961`;
  - strong:
    `+2.3029343955594896`;
  - weak:
    `-1.144025070180993`;
- `P_S,corr > P_S,ctrl` for:
  `328/330`;
- exact all/strong/weak channel contribution identities close at numerical precision.

The parent conclusion is therefore already frozen:

**within fixed `W_H`, the role-dependent source is the direction of current-token `ΔX_t`, which preferentially aligns corr with the fixed strong-output rows and relatively away from weak-output rows.**

There is no remaining norm-routing term to discover inside fixed `W_H`.

This stage must not repeat that decomposition.

---

## 3. Exact upstream architecture boundary

The authenticated block boundary is:

`R22 -> RMSNorm22 -> X22 -> Mixer22`.

The downstream residual update is separately:

`R23 = R22 + Y22`.

Therefore the current-token mixer input studied by the parent stage is the output of the layer-22 pre-mixer RMSNorm.

The runtime RMSNorm implementation is:

`variance = mean(R²)`

`s = rsqrt(variance + eps)`

`X = gamma ⊙ (s R)`.

The frozen runtime has:

- hidden width:
  `768`;
- learned RMSNorm weight:
  `gamma ∈ R^768`;
- `gamma` is fixed across matched/swapped and corr/ctrl branches;
- epsilon:
  `1e-5`;
- residual stream evaluated in float32;
- RMSNorm output observed in float32.

This stage analyzes this exact boundary only.

---

## 4. Scientific question

The bounded question is:

> Why does the layer-22 current-token mixer input `ΔX_t` arrive with the corr-specific direction that the fixed `W_H` map routes toward the downstream strong-kernel output rows and relatively away from weak-output rows?

More specifically:

> Is the validated strong/weak routing redistribution primarily associated with the raw residual-stream difference `ΔR22`, with branch-specific RMS scaling contrast `Δs22`, with their vector interaction, or with a genuinely mixed combination?

The numerical float32 reconstruction bridge is retained explicitly and is not silently folded into a scientific term.

---

## 5. Population and fixed scope

Scientific population:

- source layer:
  `22`;
- boundary:
  layer-22 pre-mixer RMSNorm;
- current token only;
- relative coordinate:
  `k=2`;
- common DDSSSSS cohort:
  `330`;
- roles:
  corr and ctrl;
- same matched/swapped definitions as the frozen K0 lineage;
- same checkpoint and runtime lineage;
- same equal-length prefix execution protocol;
- same fixed `W_H`;
- same downstream strong/weak partition.

The runner may execute the existing frozen `672` pair-role plan for exact parent comparability.

Expected full forward count:

`1344`.

Scientific summaries remain restricted to common-330 `k=2`.

No other layer, coordinate, lag, token window, seed, or channel subset may be searched.

---

## 6. Branch notation

For one item and one role, let matched/swapped current-token layer-22 residual vectors be:

`R_m, R_s ∈ R^768`.

Let observed runtime RMSNorm outputs be:

`X_m, X_s ∈ R^768`.

Let branch RMS scalars be:

`s_m = rsqrt(mean(R_m²) + eps)`

`s_s = rsqrt(mean(R_s²) + eps)`.

Let the fixed learned RMSNorm weight be:

`gamma ∈ R^768`.

Define:

`ΔR = R_m - R_s`

`R_bar = (R_m + R_s)/2`

`Δs = s_m - s_s`

`s_bar = (s_m + s_s)/2`.

The observed mixer-input difference is:

`x_obs = X_m - X_s`.

---

## 7. Exact scientific RMSNorm decomposition

Using captured float32 branch operands converted to float64, define:

`Q_R = gamma ⊙ (s_bar ΔR)`.

Define:

`Q_s = gamma ⊙ (R_bar Δs)`.

The symmetric bilinear identity gives:

`Q_R + Q_s
 = gamma ⊙ (s_m R_m - s_s R_s)`.

Interpretation:

- `Q_R`:
  residual-difference transport under the mean branch RMS scale;
- `Q_s`:
  branch-specific RMS-scale contrast acting on the mean residual vector.

These are the two scientific RMSNorm factors.

No additional fitted factor is introduced.

---

## 8. Explicit numerical branch bridge

Runtime `X_m` and `X_s` are float32 outputs and must not be conflated with a float64 operand replay.

Define float64 operand-replay branch values:

`X_alg,m = gamma ⊙ (s_m R_m)`

`X_alg,s = gamma ⊙ (s_s R_s)`.

Define branch numerical errors:

`epsilon_m = X_m - X_alg,m`

`epsilon_s = X_s - X_alg,s`.

Define:

`Q_eps = epsilon_m - epsilon_s`.

Then the observed branch difference satisfies:

`x_obs = Q_R + Q_s + Q_eps`.

`Q_eps` is a numerical execution bridge.

It is **not** a third scientific mechanism.

The implementation must persist only scalar summaries of this bridge, not raw vectors.

---

## 9. Reconstruction semantics

The frozen RMSNorm branch reconstruction tolerance remains:

`1e-6`.

It applies branchwise to:

`X_b` versus `gamma ⊙ (s_b R_b)`,

for:

`b ∈ {m,s}`.

The runner must not reuse this branch-relative tolerance as a gate on:

`Q_R + Q_s` versus `x_obs`.

The difference-relative quantity:

`||Q_R + Q_s - x_obs|| / ||x_obs||`

is cancellation-sensitive and may be recorded as a diagnostic only.

Instead require the exact error identity:

`(Q_R + Q_s) - x_obs = -Q_eps`.

Equivalently:

`x_obs - (Q_R + Q_s) = Q_eps`.

The maximum absolute identity residual must be gated at:

`5e-12`.

This rule is fixed before runtime execution.

---

## 10. Exact input-space energy decomposition

Let:

`D_X = ||x_obs||²`.

For common-330 `k=2`, require:

`D_X > 0`.

Define normalized input-space terms:

`I_R = ||Q_R||² / D_X`

`I_s = ||Q_s||² / D_X`

`I_Rs = 2 <Q_R, Q_s> / D_X`.

Define the complete numerical-bridge term:

`I_eps = (
    ||Q_eps||²
    + 2<Q_R,Q_eps>
    + 2<Q_s,Q_eps>
) / D_X`.

Then exactly:

`1 = I_R + I_s + I_Rs + I_eps`.

This identity describes how the observed `ΔX_t` vector energy is assembled.

`I_Rs` and `I_eps` may be negative.

No positive-share interpretation may be imposed on signed cross terms.

---

## 11. Propagation through the already-frozen hidden in-projection

The fixed parent operator is:

`W_H ∈ R^(1536×768)`.

Define:

`H_R = W_H Q_R`

`H_s = W_H Q_s`

`H_eps = W_H Q_eps`.

Then:

`H_obs_alg = W_H x_obs`

and:

`H_obs_alg = H_R + H_s + H_eps`

up to float64 numerical closure.

This stage does not reopen row gains or row-alignment geometry inside `W_H`.

It uses the already-frozen `W_H` only as a fixed readout of the upstream RMSNorm factors.

---

## 12. Channelwise exact routing-source decomposition

For hidden output channel `j`, let:

`h_R,j = H_R[j]`

`h_s,j = H_s[j]`

`h_eps,j = H_eps[j]`.

Normalize by the **observed full input difference norm**:

`D_X = ||x_obs||²`.

Define scientific residual term:

`e_R,j = h_R,j² / D_X`.

Define scientific RMS-scale term:

`e_s,j = h_s,j² / D_X`.

Define scientific residual×scale interaction:

`e_Rs,j = 2 h_R,j h_s,j / D_X`.

Define the complete numerical bridge contribution:

`e_eps,j = (
    h_eps,j²
    + 2 h_R,j h_eps,j
    + 2 h_s,j h_eps,j
) / D_X`.

Then:

`e_total,j
 = e_R,j + e_s,j + e_Rs,j + e_eps,j`.

And:

`e_total,j = (W_H x_obs)_j² / ||x_obs||²`.

This gives an exact decomposition of the parent normalized routing energy at every output channel.

---

## 13. Paired corr/ctrl decomposition of the parent `D_j`

For aligned corr/ctrl items define parent routing difference:

`D_total,j = e_total,corr,j - e_total,ctrl,j`.

Define component contrasts:

`D_R,j = e_R,corr,j - e_R,ctrl,j`

`D_s,j = e_s,corr,j - e_s,ctrl,j`

`D_Rs,j = e_Rs,corr,j - e_Rs,ctrl,j`

`D_eps,j = e_eps,corr,j - e_eps,ctrl,j`.

Require exactly:

`D_total,j
 = D_R,j + D_s,j + D_Rs,j + D_eps,j`.

The parent stage already established that:

`D_total,j`

is equivalently the fixed-row routing contrast:

`r_j²(A_corr,j - A_ctrl,j)`.

The new stage therefore decomposes the previously frozen `D_j` **one boundary upstream**, into RMSNorm source terms.

---

## 14. All/strong/weak partition closures

Let:

- `A` be all 1536 channels;
- `S` be the frozen strong-kernel partition, 240 channels;
- `W` be the frozen weak-kernel partition, 1296 channels.

For each partition `P ∈ {A,S,W}` define:

`Delta_R,P = Σ_(j∈P) D_R,j`

`Delta_s,P = Σ_(j∈P) D_s,j`

`Delta_Rs,P = Σ_(j∈P) D_Rs,j`

`Delta_eps,P = Σ_(j∈P) D_eps,j`.

Require:

`Delta_total,P
 = Delta_R,P
 + Delta_s,P
 + Delta_Rs,P
 + Delta_eps,P`.

The total partition contrasts must reproduce the frozen parent quantities:

- all:
  `T_H,corr² - T_H,ctrl²`;
- strong:
  `T_S,corr² - T_S,ctrl²`;
- weak:
  `T_W,corr² - T_W,ctrl²`.

This is the primary scientific bridge.

---

## 15. Parent reproduction gates

Before scientific interpretation is accepted, the runner must reproduce the frozen parent current-token routing evidence.

### 15.1 Parent artifact identities

Authenticate exactly the evidence frozen at:

`8b494a72d48528c3bdb8985a1907766fded040e0`.

Required parent output hashes:

- item metrics:
  `f3a34918c02b1715af5ec57221d9d3a34b23afb71849c3962f1aedbe458b14c8`;
- channel summary:
  `2cf91c7611cd57cd8bc1deda9269cb7e36b6008af1baab523e8a6b4818752cf2`;
- cumulative profile:
  `dac78490055ffbd0f723bdf07570305d1d51ff48e4550f9f980769355f4772b5`;
- summary:
  `c61f5faabe7a47fc71a516c9035bf739fc7c47f5707041558ab9f7d6f98cc47a`;
- manifest:
  `91d91b4d52966242e3c2b7d2532d91e3070991eaae8f2dec4f5d5293fe4a83df`.

### 15.2 Parent scalar metrics

For every common-330 item reproduce from observed `x_obs` and fixed `W_H`:

- `delta_x_current_l2_corr`;
- `delta_x_current_l2_ctrl`;
- `T_H,corr`;
- `T_H,ctrl`;
- `T_S,corr`;
- `T_S,ctrl`;
- `T_W,corr`;
- `T_W,ctrl`;
- `P_S,corr`;
- `P_S,ctrl`;
- `P_W,corr`;
- `P_W,ctrl`.

Use frozen parent scalar tolerances:

- relative:
  `1e-13`;
- absolute:
  `1e-13`.

### 15.3 Parent population-level ordering

Reproduce:

- `P_S,corr > P_S,ctrl`:
  `328/330`;
- `T_H,corr > T_H,ctrl`:
  `243/330`;
- `T_S,corr > T_S,ctrl`:
  `330/330`;
- `T_W,corr > T_W,ctrl`:
  `76/330`.

### 15.4 Parent aggregate routing contrasts

Reproduce:

- mean all:
  `+1.1589093253784961`;
- mean strong:
  `+2.3029343955594896`;
- mean weak:
  `-1.144025070180993`.

No scientific claim from this stage is accepted if these bridges fail.

---

## 16. RMSNorm instrumentation precedent

A frozen earlier K0 runner already authenticated the generic RMSNorm boundary:

`R -> RMSNorm -> X -> mixer`

and the symmetric identity:

`ΔX = Q_R + Q_s`.

That precedent is:

`scripts/longterm_k0_rvg_rmsnorm_residual_factorization_audit.py`.

Frozen runner SHA256:

`a4c5a1aea466df2e70b933050714c564ccf2d7db85f8bb9d36dc8c5e06937c4c`.

Its evidence lineage is frozen under:

`b29e05dd384cddddf8754a9e9925ff9753f49bdb`.

That older stage targeted layer 23.

Its **implementation semantics** may be reused as an instrumentation precedent.

Its layer-23 scientific result must not be imported as layer-22 evidence.

The new runner must resolve and capture layer 22 explicitly.

---

## 17. Runtime capture policy

At layer 22, register read-only hooks on the pre-mixer RMSNorm:

- forward pre-hook:
  capture current branch residual `R`;
- forward hook:
  capture observed normalized output `X`.

The already-authenticated parent capture must independently observe the exact same `X` at `mixer.in_proj` input.

At the target token, require exact float32 equality between:

- RMSNorm forward-hook `X`;
- parent mixer-input `X`.

This is a blocking boundary-identity gate.

Capture only the exercised target positions.

Compute scientific algebra in float64 after read-only float32 capture.

Do not alter model forward semantics.

---

## 18. Fixed operator authentication

Authenticate:

- layer-22 RMSNorm module identity;
- `gamma.shape == (768,)`;
- `gamma.dtype == float32`;
- `variance_epsilon == 1e-5`;
- layer-22 mixer identity;
- fixed `W_H.shape == (1536,768)`;
- `W_H` bias-free hidden branch;
- downstream lag-0 strong/weak partition:
  `240 / 1296 / 0`.

Authenticate frozen Mamba source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`.

No new model operator may be introduced.

---

## 19. Item-level persisted metrics

Persist one scalar row per common-330 aligned item.

Required fields include:

### Parent bridges
- parent `T_H`, `T_S`, `T_W`, `P_S`, `P_W` for corr/ctrl;
- parent paired ordering flags.

### RMS branch diagnostics
For corr and ctrl separately:
- `||ΔR||`;
- `||x_obs||`;
- `|Δs|`;
- `||Q_R||`;
- `||Q_s||`;
- `||Q_eps||`;
- branch RMS reconstruction residual;
- difference-relative reconstruction diagnostic.

### Input-space exact terms
For corr and ctrl:
- `I_R`;
- `I_s`;
- `I_Rs`;
- `I_eps`;
- input-energy closure residual.

### Routing-source partition terms
For each of all/strong/weak:
- `Delta_R,P`;
- `Delta_s,P`;
- `Delta_Rs,P`;
- `Delta_eps,P`;
- `Delta_total,P`;
- closure residual.

### Exact identities
- maximum channelwise component closure residual;
- parent all/strong/weak reproduction residuals;
- numerical bridge identity residual.

Raw vectors are not persisted.

---

## 20. Channel aggregate output

Persist one row for each of all `1536` hidden output channels.

Required fields:

- channel index;
- frozen downstream partition;
- frozen lag-0 kernel magnitude rank;
- fixed `W_H` row gain squared;
- mean `D_total,j`;
- mean `D_R,j`;
- mean `D_s,j`;
- mean `D_Rs,j`;
- mean `D_eps,j`;
- median values for the same component contrasts;
- positive/negative/zero paired counts for each component contrast where defined.

Do not rank or select channels by the new component values.

All 1536 channels remain preregistered.

---

## 21. Fixed kernel-rank cumulative profile

Use the already-frozen descending lag-0 squared-kernel order.

For ranks `1..1536`, persist cumulative means for:

- total parent `D`;
- residual-source `D_R`;
- RMS-scale-source `D_s`;
- residual×scale interaction `D_Rs`;
- numerical bridge `D_eps`.

Do not reorder by any new result.

No optimal cutoff may be selected.

---

## 22. Threshold-free component attribution

For each partition:

`P ∈ {all,strong,weak}`

let aggregate mean signed component nets be:

`M_R,P`

`M_s,P`

`M_Rs,P`

`M_eps,P`.

Require:

`M_total,P = M_R,P + M_s,P + M_Rs,P + M_eps,P`.

Define total absolute component mass:

`A_P = |M_R,P| + |M_s,P| + |M_Rs,P| + |M_eps,P|`.

When `A_P > 0`, report absolute component shares:

`F_R,P = |M_R,P| / A_P`

`F_s,P = |M_s,P| / A_P`

`F_Rs,P = |M_Rs,P| / A_P`

`F_eps,P = |M_eps,P| / A_P`.

Also report signed net ratios to `M_total,P` when the denominator is nonzero.

Ratios may exceed one in magnitude under cancellation.

Do not misinterpret such ratios as probability-like shares.

No arbitrary dominance threshold is preregistered.

The largest absolute signed aggregate component may be identified descriptively.

---

## 23. Scientific outcome classes

The stage is designed to distinguish the following without post-hoc redefinition.

### Outcome A: residual-vector transport localized

The residual-source component `D_R` carries the main signed strong-positive / weak-negative routing pattern, while RMS-scale and interaction terms are materially smaller.

Interpretation:

the corr-specific `ΔX_t` direction is already mainly present in the raw residual difference `ΔR22`, with RMSNorm mostly transporting/reweighting it.

### Outcome B: RMS-scale contrast localized

`D_s` carries the main role-routing contrast.

Interpretation:

branch-specific reciprocal-RMS scaling materially creates the strong/weak routing phenotype from the mean residual vector.

### Outcome C: residual×scale interaction localized

`D_Rs` is the largest signed source.

Interpretation:

the phenotype is not attributable to either factor alone; it is produced primarily by their vector interaction after fixed gamma weighting and `W_H`.

### Outcome D: mixed

No single scientific component clearly carries the observed strong-positive / weak-negative routing pattern.

Report the exact mixture and cancellation.

### Numerical-bridge condition

`D_eps` is always reported separately.

If the numerical bridge is comparable to or larger than the scientific components, do not promote a residual/RMS scientific source claim.

Do not change tolerances after observing this outcome.

---

## 24. Falsification conditions

The proposed upstream localization is falsified or blocked if any of the following occurs:

1. layer-22 RMSNorm output does not exactly equal the parent mixer-input `X` in float32;
2. branch RMS reconstruction exceeds `1e-6`;
3. the symmetric `Q_R + Q_s` algebraic identity fails its float64 closure;
4. the explicit numerical-bridge identity fails `5e-12`;
5. the observed parent `ΔX` metrics cannot be reproduced;
6. the frozen `W_H` strong/weak parent metrics cannot be reproduced;
7. all/strong/weak routing-source component sums do not close;
8. parent paired counts do not reproduce;
9. the strong/weak partition changes;
10. any raw per-item vector is persisted;
11. any learned geometry, probe, PCA/SVD, channel search, intervention, training, tokenizer, logits, or task-head path is invoked.

A scientifically unexpected component pattern is not a validation failure.

It is a valid result.

---

## 25. Tolerances fixed before runtime

Use:

- branch RMS reconstruction relative tolerance:
  `1e-6`;
- parent scalar relative tolerance:
  `1e-13`;
- parent scalar absolute tolerance:
  `1e-13`;
- float64 vector/component max-absolute closure tolerance:
  `5e-12`;
- float64 squared-energy relative closure tolerance:
  `1e-12`;
- kernel RMS relative tolerance:
  `1e-13`;
- kernel RMS absolute tolerance:
  `1e-13`.

Difference-relative observed `Q_R+Q_s` versus `x_obs` is diagnostic-only.

No tolerance may be weakened after preflight or scientific execution.

---

## 26. Full execution plan

The full authorized population, after implementation freeze and bounded preflight PASS, is:

- frozen `672` pair-role plan;
- matched and swapped branch forward per pair-role;
- expected forwards:
  `1344`;
- scientific summaries:
  common-330, current-token, `k=2` only.

GPU is not required.

CPU execution is sufficient.

No Kaggle execution is needed unless local runtime becomes technically impossible.

---

## 27. Artifact policy

A later authorized full execution may persist:

1. `330` item scalar rows;
2. `1536` channel aggregate rows;
3. `1536` fixed-kernel-rank cumulative rows;
4. summary JSON;
5. execution manifest.

Do not persist:

- `R_m`, `R_s`;
- `X_m`, `X_s`;
- `Q_R`;
- `Q_s`;
- `Q_eps`;
- `H_R`;
- `H_s`;
- `H_eps`;
- per-item channel vectors;
- token IDs beyond already-frozen identifiers;
- logits;
- task-head outputs.

Temporary in-memory vectors must be released after aggregation.

---

## 28. Provenance requirements

The execution manifest must record:

- runtime Git HEAD;
- branch;
- this static-design freeze commit and SHA256;
- parent evidence freeze:
  `8b494a72d48528c3bdb8985a1907766fded040e0`;
- parent runner SHA:
  `8d6c45636197ee6cc1d9e6f8c423a1f8fe57fc654ae450a4b3eb6b8657dfaf5b`;
- exact parent artifact hashes;
- RMSNorm precedent runner SHA:
  `a4c5a1aea466df2e70b933050714c564ccf2d7db85f8bb9d36dc8c5e06937c4c`;
- Mamba source SHA;
- handoff ZIP SHA;
- checkpoint SHA;
- encoder canonical digest;
- encoder raw concat digest;
- forward count;
- layer/coordinate/cohort;
- all tolerances;
- raw-vector persistence flags;
- training/intervention/search flags;
- output file SHA256 hashes.

Any provenance mismatch blocks interpretation.

---

## 29. Interpretation boundary

Passing this stage may support a statement of the form:

> the corr-specific layer-22 current-token strong/weak `W_H` routing pattern is algebraically localized to a particular combination of raw residual-stream difference, branch-specific RMS scaling, and their interaction at the layer-22 pre-mixer RMSNorm boundary.

It may not support:

- causal necessity;
- causal sufficiency;
- semantic interpretation of individual channels;
- generalization to another layer/seed/task;
- learned subspace claims;
- K1 claims.

This remains observational/algebraic localization.

---

## 30. What this stage intentionally does not ask

Do not ask:

- which top-k channels matter;
- whether another layer shows a larger effect;
- whether a learned projection separates corr/ctrl;
- whether an intervention improves performance;
- whether `gamma` should be tuned;
- whether a different epsilon changes the effect;
- whether an alternative RMSNorm implementation performs better.

The architecture and frozen model are read-only.

---

## 31. Next-step rule after validated evidence

If the residual-source term carries the routing pattern, the next unresolved K0 boundary moves upstream into construction of the layer-22 raw residual `R22`.

If the RMS-scale term or residual×scale interaction materially carries the pattern, the next K0 question remains at the normalization boundary and asks why the matched/swapped branch residual magnitudes generate that scale contrast.

Do not choose that next branch before validated evidence exists.

Do not transition to K1 automatically.

---

## 32. Static-design conclusion

The correct next K0 experiment is:

**an exact layer-22 current-token RMSNorm routing-source decomposition that splits the observed mixer-input role difference into raw-residual transport, branch-specific RMS-scale contrast, residual×scale interaction, and an explicitly separated numerical bridge, then propagates those terms through the already-frozen `W_H` and strong/weak channel partition to decompose the previously validated routing contrast without new channel selection or learned geometry.**

This is the narrowest upstream experiment that directly answers the unresolved parent question.
