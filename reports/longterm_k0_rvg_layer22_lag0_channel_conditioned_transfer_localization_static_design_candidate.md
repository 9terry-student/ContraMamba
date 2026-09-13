# ContraMamba K0-RVG Layer-22 Lag-0 Channel-Conditioned Transfer Localization
## Static Design Candidate

## 1. Phase and authority boundary

**Phase:** scientific static design only.

This stage follows the validated layer-22 four-tap causal-convolution evidence freeze and narrows the remaining K0 question to the already-frozen dominant lag-0 transfer.

This document may authorize implementation after it is frozen.

It does **not** itself authorize scientific model execution.

Bounded implementation/runtime preflights do not require a separate authority document once this design and its implementation are frozen; full scientific execution still requires an explicit controller decision after preflight.

No K1 transition is authorized.

---

## 2. Repository boundary

Repository:

`9terry-student/ContraMamba`

Branch:

`longterm-k-series-native-state-kinematics`

Frozen parent evidence commit:

`6b0d282c22d0b1c8f8387d0853205c3d4226a392`

Expected unrelated local files to remain untouched/untracked:

- `scripts/longterm_k1_native_state_kinematics.py`
- `tests/test_longterm_k1_native_state_kinematics.py`

---

## 3. Frozen parent identities

Parent implementation:

`3dd3791cab718421fd30ef119d44d3d2a4defde9`

Parent runner:

`scripts/longterm_k0_rvg_layer22_four_tap_convolution_decomposition_audit.py`

Runner SHA256:

`bc48b1bcb222dbf75828ad61fb61c0d766a5024e2456d19c8ccee6ccdfcbe088`

Parent run directory:

`reports/longterm_k0_rvg_layer22_four_tap_convolution_decomposition_3dd3791_v1`

Metrics SHA256:

`65d845ecc50021a7dfb1d3b3cd4fd3431843c9d7c8318268f09df0945b4fcf76`

Summary SHA256:

`620413f7f33a372a082a5ab0e1e938552dc6fa330836768b7e5b3e996f1a9872`

Execution manifest SHA256:

`77e2a06147637f7a233843ed2df2e2ad2f558071785c2ea06bd8c7726fda2e07`

Validated-evidence report:

`reports/longterm_k0_rvg_layer22_four_tap_convolution_decomposition_validated_evidence_analysis_report_candidate.md`

Validated-evidence report SHA256:

`d63df7810ba30647fd6b245080e891aac3d9f84ea743f840012cc88fc2979cea`

Parent runtime HEAD:

`a4c0a48411788cb5bb3ee32fa92e97f91d87de2d`

---

## 4. Frozen scientific frontier

At common-330 `k=2`, the validated parent evidence established:

- lag 0 is the dominant contribution in corr for `330/330` items;
- lag-0 incoming hidden-difference enrichment:
  `E_H0 = +0.41294006625397023`;
- lag-0 contribution enrichment:
  `E_Q0 = +0.5675913225723508`;
- additional lag-0 fixed-tap transfer enrichment:
  `G_TAP0 = +0.15367770768001138`;
- corr lag-0 normalized transfer median:
  `1.409556698171893`;
- ctrl lag-0 normalized transfer median:
  `1.210189909405911`.

Therefore the remaining question is not which lag dominates.

That is frozen.

The remaining question is why the fixed lag-0 depthwise kernel produces a larger **norm transfer** for corr than ctrl.

---

## 5. Algebraic refinement: sign cannot explain lag-0 norm transfer

For layer-22 lag 0 and channel `j`, define:

`h_j = ΔH_t,j`

and the authenticated lag-0 kernel:

`k_j = K[j, lag0]`.

The lag-0 contribution is:

`q_j = k_j h_j`.

The tap-transfer magnitude is:

`t = ||q||₂ / ||h||₂`.

Therefore:

`t² = Σ_j k_j² h_j² / Σ_j h_j²`.

Define normalized hidden-difference energy:

`p_j = h_j² / Σ_r h_r²`.

Then:

`t² = Σ_j p_j k_j²`.

Thus lag-0 norm transfer depends only on:

1. fixed squared kernel magnitude `k_j²`; and
2. where hidden-difference energy `p_j` is placed across channels.

The signs of `k_j` and `h_j` do **not** affect `t`.

Therefore this stage must not invoke "signed alignment" as an explanation for lag-0 transfer magnitude.

The exact scientific object is **channelwise energy exposure to the fixed `k²` profile**.

---

## 6. Kernel-normalized exposure identity

Let:

`d = 1536`

and:

`μ_k2 = (1/d) Σ_j k_j²`.

The authenticated parent lag-0 kernel RMS is:

`sqrt(μ_k2) = 0.24383223809052498`.

Define the squared normalized transfer:

`N² = t² / μ_k2`.

Then:

`N² = Σ_j p_j (k_j² / μ_k2)`.

Define centered kernel strength:

`a_j = (k_j² / μ_k2) - 1`.

Since `Σ_j p_j = 1`:

`N² - 1 = Σ_j p_j a_j`.

This identity exactly measures whether hidden-difference energy is concentrated on channels whose squared lag-0 kernel strength is above or below the uniform-channel kernel mean.

---

## 7. Exact corr-vs-ctrl channel contribution identity

For an aligned common-330 item, define:

`p_corr,j`

and:

`p_ctrl,j`.

Define the paired energy redistribution:

`Δp_j = p_corr,j - p_ctrl,j`.

Because both role energy distributions sum to one:

`Σ_j Δp_j = 0`.

Define each channel's exact contribution to the squared normalized-transfer difference:

`c_j = a_j Δp_j`.

Then:

`Σ_j c_j = N_corr² - N_ctrl²`.

This is the primary decomposition.

It has no fitted weights, no learned geometry, no selected channel subset, and no residual term.

A positive `c_j` occurs when the corr-vs-ctrl energy redistribution at channel `j` moves normalized transfer upward relative to the kernel-mean baseline.

A negative `c_j` moves it downward.

---

## 8. Structural above-mean / below-mean partition

The only binary channel partition permitted in the primary analysis is determined by the algebraic baseline itself:

- **strong-kernel channels:** `k_j² > μ_k2`;
- **weak-kernel channels:** `k_j² < μ_k2`;
- exact-equality channels, if any: `k_j² = μ_k2`.

This is not a tuned threshold.

It is the zero point of:

`a_j = k_j²/μ_k2 - 1`.

For each role/item define:

`P_strong = Σ_(j: k_j² > μ_k2) p_j`

`P_weak = Σ_(j: k_j² < μ_k2) p_j`.

Required paired diagnostics include corr-vs-ctrl differences in these energy masses.

This partition is supportive; the full exact 1536-channel decomposition remains primary.

---

## 9. Full-channel reporting policy

All `1536` channels are preregistered.

No channel may be selected after viewing results and then presented as if it were the prespecified target.

The evidence artifact must contain a full channel summary table with, for every channel index:

- channel index;
- `k_j`;
- `k_j²`;
- `k_j² / μ_k2`;
- centered strength `a_j`;
- kernel-magnitude rank determined only from the frozen kernel;
- common-330 mean `p_corr,j`;
- common-330 mean `p_ctrl,j`;
- common-330 mean `Δp_j`;
- common-330 mean `c_j`;
- median `c_j`;
- counts `c_j > 0`, `< 0`, `= 0`.

Raw per-item `h_j`, `p_j`, or `c_j` vectors must not be persisted.

Only aggregate channel summaries may be persisted.

---

## 10. Threshold-free concentration diagnostics

To determine whether the role-difference contribution profile is diffuse or concentrated without choosing top-k channels, define the aggregate mean channel contribution:

`\bar c_j = mean_i(c_i,j)`.

Define absolute contribution mass:

`M_abs = Σ_j |\bar c_j|`.

Define net contribution:

`M_net = Σ_j \bar c_j`.

Define cancellation ratio when `M_abs > 0`:

`R_cancel = M_net / M_abs`.

Define threshold-free effective channel count:

`N_eff = (Σ_j |\bar c_j|)² / Σ_j \bar c_j²`.

Interpretation:

- small `N_eff`: contribution profile is concentrated;
- large `N_eff`: contribution profile is diffuse;
- `R_cancel` near 1: little signed cancellation;
- smaller positive values: substantial positive/negative channel cancellation.

No arbitrary top-k cutoff is permitted as a primary metric.

---

## 11. Kernel-rank cumulative profile

To visualize/diagnose concentration without data-dependent channel selection, channels must also be ordered once by the frozen quantity:

`k_j²`

from strongest to weakest.

Let this fixed order be `π(1),...,π(d)`.

Define cumulative mean paired contribution:

`C_m = Σ_(r=1)^m \bar c_(π(r))`

for every `m = 1..d`.

Also define cumulative mean role energy:

`P_role,m = Σ_(r=1)^m mean_i(p_role,i,π(r))`.

The artifact may persist these complete `1536`-point cumulative curves.

Scientific interpretation must use the whole preregistered curve rather than choosing an optimal post-hoc cutoff.

---

## 12. Primary common-330 k2 itemwise metrics

For each aligned common-330 item at `k=2`, record scalar values only:

- `delta_h_lag0_l2_corr`;
- `delta_h_lag0_l2_ctrl`;
- `q0_l2_corr`;
- `q0_l2_ctrl`;
- `t0_corr`;
- `t0_ctrl`;
- `n0_corr`;
- `n0_ctrl`;
- `n0_sq_corr`;
- `n0_sq_ctrl`;
- `delta_n0_sq = n0_sq_corr - n0_sq_ctrl`;
- `strong_energy_mass_corr`;
- `strong_energy_mass_ctrl`;
- `delta_strong_energy_mass`;
- `weak_energy_mass_corr`;
- `weak_energy_mass_ctrl`;
- `delta_weak_energy_mass`;
- `channel_contribution_sum = Σ_j c_j`;
- exact identity residual:
  `channel_contribution_sum - delta_n0_sq`.

No epsilon is allowed for zero denominators.

If `||ΔH_t||₂ = 0`, role-specific energy distributions and transfer quantities are undefined.

Undefined counts must be reported explicitly.

---

## 13. Parent reproduction requirements

The new audit must reproduce the frozen parent common-330 k2 lag-0 quantities before scientific interpretation:

- `delta_h_lag0_l2`;
- `q0_l2`;
- `q0_tap_transfer`;
- `q0_transfer_over_weight_rms`.

It must also reproduce the authenticated lag-0 kernel RMS:

`0.24383223809052498`.

The runner must authenticate the parent artifact identities and parent runner identity before execution.

Numerical tolerances must be fixed in implementation before runtime preflight and must not be tuned after viewing results.

---

## 14. Runtime capture boundary

The implementation should reuse the frozen parent layer-22 `H_RF` capture path.

For the target token, only the lag-0/current-token hidden vector is scientifically required:

`ΔH_t`.

The exact layer-22 lag-0 kernel must be read from the authenticated runtime model.

The scientific calculation should be performed in float64 after read-only float32 capture, consistent with the preceding K0 audits.

No model parameter or forward semantic may be changed.

---

## 15. Execution population

The scientific target is fixed to:

- source layer: `22`;
- causal lag: `0`;
- relative coordinate: `k=2`;
- common DDSSSSS cohort: `330`;
- both roles: corr and ctrl;
- same frozen matched/swapped definitions.

To preserve numerical comparability with the parent execution, the model forward should retain the frozen equal-length prefix protocol through `k+6`.

The implementation may execute the full frozen 672 pair-role plan if needed for exact parent reproduction, but scientific channel statistics must be restricted to the preregistered common-330 `k=2` target.

No additional layer, lag, coordinate, cohort, or channel search is authorized.

---

## 16. Required role-level summaries

For common-330 k2 report:

- medians of `t0`, `n0`, and `n0²` for corr and ctrl;
- corr > ctrl / corr < ctrl / equal counts;
- median paired `delta_n0_sq`;
- median and mean strong-kernel energy mass by role;
- paired strong-kernel energy-mass counts;
- median and mean weak-kernel energy mass by role;
- identity residual maxima;
- undefined counts.

The already-frozen `G_TAP0` may be reproduced as a bridge, but this stage must not replace the exact squared-transfer decomposition with a log-only analysis.

---

## 17. Required channel-level summaries

For all 1536 channels report the full preregistered table described above.

Also report global aggregate values:

- `M_abs`;
- `M_net`;
- `R_cancel`;
- `N_eff`;
- number of channels with positive mean contribution;
- number with negative mean contribution;
- number exactly zero;
- total mean contribution from strong-kernel channels;
- total mean contribution from weak-kernel channels.

No top-k list is a primary scientific result.

A small convenience list of highest absolute contributors may be printed only if clearly labeled descriptive and only alongside the complete channel table and threshold-free concentration metrics.

---

## 18. Exact algebraic gates

For every defined common-330 item require:

### 18.1 Energy normalization

`Σ_j p_role,j = 1`

within fixed tolerance.

### 18.2 Centered-kernel identity

`N_role² - 1 = Σ_j p_role,j a_j`.

### 18.3 Paired channel-contribution closure

`N_corr² - N_ctrl² = Σ_j c_j`.

### 18.4 Strong/weak energy closure

`P_strong + P_weak + P_equal = 1`.

### 18.5 Parent transfer bridge

Directly reconstructed:

`t² = Σ_j p_j k_j²`

must agree with the frozen parent lag-0 tap-transfer value.

All tolerances must be fixed before runtime scientific execution.

---

## 19. Interpretation rules

### 19.1 Strong-kernel energy redistribution explanation

If corr systematically allocates more hidden-difference energy to channels with `k_j² > μ_k2`, the exact channel contribution sum is positive, and this closes the observed transfer difference, the permitted conclusion is:

**corr's lag-0 transfer advantage is explained by preferential hidden-difference energy exposure to stronger fixed lag-0 kernel channels.**

This is an algebraic localization, not a causal claim.

### 19.2 Diffuse versus concentrated channel organization

Use `N_eff`, the full channel table, and the fixed kernel-rank cumulative profile.

If `N_eff` is large and the cumulative profile changes gradually, describe the effect as diffuse.

If `N_eff` is small and the cumulative profile is sharply concentrated, describe it as concentrated.

Do not invent a threshold separating these regimes.

### 19.3 Mixed positive/negative redistribution

If strong positive channel contributions coexist with substantial negative contributions, report the signed cancellation structure using `R_cancel`.

Do not discard negative channels.

### 19.4 No additional transfer mechanism

Because the lag-0 tap is depthwise and fixed, and because:

`t² = Σ p_j k_j²`

is exact, once parent transfer and channel-exposure identities close, there is no residual norm-transfer mechanism inside the lag-0 tap to discover.

Any remaining upstream question concerns how layer-22 `ΔH_t` acquired that channel-energy distribution, not another hidden operation inside the tap.

---

## 20. Claims explicitly prohibited

This audit cannot establish:

- that any channel is causally necessary;
- that any channel is sufficient;
- that ablating a channel would alter task behavior;
- that high-|k| channels encode a semantic concept;
- that kernel sign matters for lag-0 norm transfer;
- that the corr/control distinction originates in the convolution;
- that a channel subset generalizes to another layer/window;
- that the result establishes K1.

No intervention is authorized.

---

## 21. Artifact policy

A later authorized execution must persist only:

1. item-level scalar metrics for the common-330 k2 target;
2. aggregate per-channel summary table;
3. fixed kernel-rank cumulative summary;
4. summary JSON;
5. execution manifest.

It must not persist raw hidden-state vectors or per-item channel vectors.

The manifest must record:

- authority commit/hash;
- parent evidence commit/hashes;
- runner hash;
- runtime HEAD;
- handoff/checkpoint identities;
- lag-0 kernel RMS;
- scientific target layer/lag/k/cohort;
- output hashes;
- no-tokenizer/no-logits/no-training/no-intervention/no-search flags.

Atomic `.partial` output behavior is required.

---

## 22. Stop conditions

Stop if:

- branch is incorrect;
- the parent evidence freeze is not an ancestor;
- any parent artifact/hash mismatches;
- parent runner identity mismatches;
- layer 22 or lag 0 cannot be authenticated exactly;
- lag-0 kernel RMS differs from the frozen parent value beyond the fixed bridge tolerance;
- parent lag-0 metrics cannot be reproduced;
- raw vectors would need to be persisted;
- a learned transform is proposed;
- a top-k or threshold must be tuned after seeing results;
- unrelated tracked worktree changes exist;
- either unrelated K1 file changes state.

---

## 23. Static-design success criterion

This stage is complete when implementation can proceed without result-dependent scientific choices and the following are fixed:

1. exact lag-0 norm-transfer identity;
2. sign-independence of the norm-transfer mechanism;
3. normalized channel-energy distribution `p_j`;
4. centered fixed-kernel strength `a_j`;
5. exact channel contribution `c_j`;
6. all-1536-channel preregistration;
7. structural strong/weak kernel partition;
8. threshold-free concentration metrics;
9. fixed kernel-rank cumulative profile;
10. parent reproduction gates;
11. common-330 k2 population boundary;
12. no raw-vector persistence;
13. no channel/layer/lag/window search;
14. observational/algebraic interpretation boundary.

Completion of this design does not itself authorize model forward execution.
