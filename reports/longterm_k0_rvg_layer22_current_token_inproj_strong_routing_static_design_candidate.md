# ContraMamba K0-RVG Layer-22 Current-Token In-Projection Strong-Routing Decomposition
## Static Design Candidate

## 1. Phase and authority boundary

**Phase:** scientific static design only.

This stage follows the validated layer-22 lag-0 channel-conditioned transfer evidence freeze.

It narrows the remaining K0 question to the exact bias-free current-token hidden in-projection:

`ΔH_t = W_H ΔX_t`.

This document may authorize implementation after it is frozen.

It does **not** itself authorize scientific execution.

Bounded static/runtime preflights do not require a separate authority document once this design and its implementation are frozen.

No K1 transition is authorized.

---

## 2. Repository boundary

Repository:

`9terry-student/ContraMamba`

Branch:

`longterm-k-series-native-state-kinematics`

Frozen parent evidence commit:

`bfa626261aba575ab7316877bc69ca6e5df38157`

Expected unrelated local files to remain untouched/untracked:

- `scripts/longterm_k1_native_state_kinematics.py`
- `tests/test_longterm_k1_native_state_kinematics.py`

---

## 3. Frozen parent evidence

Parent scientific stage:

**Layer-22 lag-0 channel-conditioned transfer localization.**

Parent implementation/runtime commit:

`4c19d02d94600e47f39c15bebd839f5a4820a473`

Parent runner:

`scripts/longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_audit.py`

Parent runner SHA256:

`dd1e8ffb0516c4b8392389578815fa5c5c2f05ba826eb37055025b6eda6fae0c`

Parent run directory:

`reports/longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_4c19d02_v1`

Parent artifact SHA256 values:

- item metrics:
  `81396fafb131b1efa5e62877d29ddd9c5b860e718fe94adb1f0c5ddbb8b92f81`
- channel summary:
  `459af9290b110304160b04148ff46b511aca50ca8bae7849aa80dabe5461b451`
- kernel-rank cumulative:
  `ba838da6f12286a6dd7699275d5257a04366015c262f1a070b28f5ef5ace4c50`
- summary:
  `ca33ff9a00d2527ce480cbefd4244f7830bc6dbd32298fd6356030cc0ecfc3a3`
- execution manifest:
  `5b304a571dfc17af26c21faa185bd7b9b9a74e3432ae1b80e085fa44b8e8a94c`

Validated-evidence report:

`reports/longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_validated_evidence_analysis_report_candidate.md`

Report SHA256:

`39d6365ad628e27b96654989d91c71e9594b0cc624ef4290ae7e61d1792faa35`

---

## 4. Additional frozen upstream bridge

The existing layer-22 U-path source-localization stage already authenticated the exact bias-free boundary:

`H_t = W_H X_t`

and reconstructed it with maximum relative residual:

`3.9620320852752387e-07`

against tolerance:

`1e-6`.

The authenticated hidden-branch in-projection has:

- input width:
  `768`;
- output width:
  `1536`;
- no bias;
- hidden-half weight:
  `W_H ∈ R^(1536×768)`.

The same stage records current-token diagnostics:

- `delta_x_current_l2`;
- `delta_h_current_l2`.

The present stage must use the same authenticated boundary and must not introduce a different definition of `X_t` or `H_t`.

---

## 5. Frozen scientific frontier

The parent lag-0 channel-transfer stage established at common-330 `k=2`:

- strong lag-0 kernel channels:
  `240`;
- weak lag-0 kernel channels:
  `1296`;
- equal-to-mean channels:
  `0`;
- corr mean strong-channel `ΔH_t` energy mass:
  `0.3063219149358646`;
- ctrl mean strong-channel `ΔH_t` energy mass:
  `0.22028264559893468`;
- corr strong-energy mass > ctrl:
  `328/330`;
- corr normalized lag-0 transfer squared > ctrl:
  `323/330`.

The exact lag-0 fixed-tap norm-transfer mechanism is already closed.

Therefore this stage must not re-explain the convolution tap.

The unresolved question is upstream:

> How does the fixed bias-free current-token hidden in-projection map the corr and ctrl `ΔX_t` directions into the different `ΔH_t` channel-energy distributions observed at the lag-0 convolution input?

---

## 6. Important distinction from the prior U-path result

The prior U-path source-localization report found that the **full four-position receptive-field** hidden in-projection transfer:

`||ΔH_RF|| / ||ΔX_RF||`

was relatively corr-attenuating.

That result used the complete four-lag receptive-field norm.

The present scientific target is different and narrower:

- current token only;
- lag-0 `H_t`;
- current-token `X_t`;
- the exact downstream strong/weak H-channel partition frozen by the lag-0 kernel.

Therefore the present stage must measure a current-token in-projection transfer directly.

It must not substitute the older full-RF transfer statistic for the current-token statistic.

---

## 7. Exact current-token boundary

For each aligned role/item at common-330 `k=2`, define:

`x = ΔX_t ∈ R^768`

and:

`h = ΔH_t ∈ R^1536`.

The authenticated bias-free identity is:

`h = W_H x`.

For output channel `j`, let the corresponding in-projection row be:

`w_j ∈ R^768`.

Then:

`h_j = w_j^T x`.

This stage is entirely observational/algebraic.

No parameter is changed.

---

## 8. Frozen downstream strong/weak channel partition

Reuse exactly the parent lag-0 kernel partition.

Let:

`k_j`

be the frozen layer-22 lag-0 convolution kernel weight for H channel `j`.

Let:

`μ_k2 = mean_j(k_j²)`.

Then:

- strong channels:
  `S = {j : k_j² > μ_k2}`;
- weak channels:
  `W = {j : k_j² < μ_k2}`;
- equal channels:
  `E = {j : k_j² = μ_k2}`.

Frozen counts:

- `|S| = 240`;
- `|W| = 1296`;
- `|E| = 0`.

No channel threshold may be tuned in this stage.

---

## 9. Current-token total and partitioned routing

Define the current-token total hidden transfer:

`T_H = ||h||₂ / ||x||₂`.

Define strong-output transfer:

`T_S = ||h_S||₂ / ||x||₂`

where `h_S` retains only H channels in `S`.

Define weak-output transfer:

`T_W = ||h_W||₂ / ||x||₂`.

If equal channels exist in runtime despite the frozen expectation, also define:

`T_E = ||h_E||₂ / ||x||₂`

and treat that as a blocker unless the parent partition is reproduced exactly.

Because the output channel groups are disjoint coordinates:

`T_H² = T_S² + T_W² + T_E²`.

With the frozen `|E|=0`:

`T_H² = T_S² + T_W²`.

---

## 10. Strong-channel H energy mass identity

Define:

`P_S = ||h_S||₂² / ||h||₂²`.

Then:

`P_S = T_S² / T_H²`.

Similarly:

`P_W = T_W² / T_H²`.

With no equal channels:

`P_S + P_W = 1`.

The parent lag-0 stage already measured `P_S` from `ΔH_t`.

The present stage must reproduce parent `P_S` itemwise before any new interpretation is accepted.

---

## 11. Row-gain × directional-alignment factorization

For every in-projection output row `w_j`, define the fixed row gain:

`r_j = ||w_j||₂`.

For nonzero `x` and nonzero row `w_j`, define:

`A_j(x) = cos²(w_j, x)`

so that:

`A_j(x) = (w_j^T x)² / (||w_j||₂² ||x||₂²)`.

Then the channel's squared current-token transfer is exactly:

`e_j(x) = h_j² / ||x||₂²`.

And:

`e_j(x) = r_j² A_j(x)`.

This is the central factorization.

The row gain `r_j²` is fixed and role-independent.

The role-specific term is the squared directional alignment `A_j(x)`.

Therefore corr-vs-ctrl transfer differences cannot be caused by a role-dependent change in the in-projection weights.

They can only arise because corr and ctrl `ΔX_t` directions expose the fixed in-projection row geometry differently.

---

## 12. Exact role-difference channel contribution

For an aligned item define:

`A_corr,j`

and:

`A_ctrl,j`.

Define:

`ΔA_j = A_corr,j - A_ctrl,j`.

Define channel contribution to the current-token transfer-squared difference:

`D_j = r_j² ΔA_j`.

Then:

`D_j = e_corr,j - e_ctrl,j`.

And exactly:

`Σ_j D_j = T_H,corr² - T_H,ctrl²`.

For the downstream strong partition:

`Σ_(j∈S) D_j = T_S,corr² - T_S,ctrl²`.

For the weak partition:

`Σ_(j∈W) D_j = T_W,corr² - T_W,ctrl²`.

This decomposition is exact.

It has no learned basis, fitted weight, or residual explanatory term.

---

## 13. Structural row-gain diagnostics

The fixed in-projection row geometry may be structurally coupled to the downstream lag-0 kernel partition.

Before examining role-dependent alignment, report the frozen operator-only quantities:

For all 1536 rows:

- channel index;
- downstream lag-0 `k_j`;
- downstream lag-0 `k_j²`;
- strong/weak partition;
- in-projection row norm `r_j`;
- in-projection row norm squared `r_j²`.

Required group summaries:

- strong-group mean and median `r_j²`;
- weak-group mean and median `r_j²`;
- strong/weak mean row-gain ratio;
- strong/weak median row-gain ratio.

These are descriptive properties of the fixed operator.

They are not role-specific evidence.

---

## 14. Role-specific alignment diagnostics

For every common-330 aligned item, the runner may hold in memory:

- `A_corr,j`;
- `A_ctrl,j`;
- `ΔA_j`;
- `D_j`.

Raw per-item vectors must not be persisted.

Persist only aggregate per-channel summaries.

For every output channel report:

- channel index;
- strong/weak partition;
- `r_j²`;
- mean `A_corr,j`;
- mean `A_ctrl,j`;
- mean `ΔA_j`;
- mean `D_j`;
- median `D_j`;
- counts `D_j > 0`, `< 0`, `= 0`.

All 1536 channels are preregistered.

---

## 15. Primary item-level scalar metrics

For each common-330 item persist only scalar metrics:

- `delta_x_current_l2_corr`;
- `delta_x_current_l2_ctrl`;
- `delta_h_current_l2_corr`;
- `delta_h_current_l2_ctrl`;
- `t_h_corr`;
- `t_h_ctrl`;
- `t_s_corr`;
- `t_s_ctrl`;
- `t_w_corr`;
- `t_w_ctrl`;
- `p_s_corr`;
- `p_s_ctrl`;
- `p_w_corr`;
- `p_w_ctrl`;
- `delta_t_h_sq`;
- `delta_t_s_sq`;
- `delta_t_w_sq`;
- `sum_d_all`;
- `sum_d_strong`;
- `sum_d_weak`;
- exact closure residuals.

No epsilon is allowed for zero denominators.

Undefined cases must be counted explicitly.

---

## 16. Role enrichment metrics

For defined aligned items define:

`G_H = log(T_H,corr / T_H,ctrl)`.

Define:

`G_S = log(T_S,corr / T_S,ctrl)`.

Define:

`G_W = log(T_W,corr / T_W,ctrl)`.

The strong-energy-mass role enrichment satisfies:

`log(P_S,corr / P_S,ctrl) = 2(G_S - G_H)`.

The weak-energy-mass role enrichment satisfies:

`log(P_W,corr / P_W,ctrl) = 2(G_W - G_H)`.

These identities must be enforced itemwise.

This makes explicit whether corr's strong-channel energy advantage is associated with:

- selective preservation/amplification of strong-output routing;
- selective suppression of weak-output routing;
- or both.

---

## 17. Current-token versus full-RF distinction

The summary must separately label:

- current-token transfer:
  `T_H`;
- historical full-RF transfer from the older stage:
  `T_XH_RF`.

The new runner may reproduce the frozen full-RF statistic as a bridge if convenient, but it must not mix the two quantities.

Scientific conclusions from this stage must refer to current-token routing only unless explicitly comparing the two.

---

## 18. Threshold-free concentration diagnostics

Define the aggregate mean channel contribution:

`\bar D_j = mean_i(D_i,j)`.

Define:

`M_abs = Σ_j |\bar D_j|`.

Define:

`M_net = Σ_j \bar D_j`.

When `M_abs > 0`, define:

`R_cancel = M_net / M_abs`.

Define effective channel count:

`N_eff = (Σ_j |\bar D_j|)² / Σ_j \bar D_j²`.

Compute these for:

1. all 1536 channels;
2. strong channels only;
3. weak channels only.

Do not convert `N_eff` into an exact number of mechanistically active channels.

No arbitrary top-k threshold is a primary result.

---

## 19. Fixed downstream-kernel-rank cumulative profile

Use the already-frozen descending downstream lag-0 `k_j²` order from the parent stage.

Do not reorder channels by the new `D_j` values.

For rank `m=1..1536`, persist cumulative:

- mean `D_j`;
- mean corr alignment energy;
- mean ctrl alignment energy;
- fixed in-projection row-gain mass.

This allows the routing difference to be viewed against the downstream kernel-strength ordering without post-hoc channel selection.

---

## 20. Optional fixed row-gain-rank profile

A second cumulative profile may be persisted using a preregistered fixed ordering by:

`r_j²`

from largest to smallest.

This order is operator-only and role-independent.

If implemented, the full 1536-point curve must be reported.

No optimal cutoff may be selected after viewing results.

---

## 21. Parent reproduction gates

Before scientific metrics are accepted, the runner must reproduce:

### 21.1 Current-token boundary

From the frozen U-path capture:

- `delta_x_current_l2`;
- `delta_h_current_l2`.

### 21.2 Parent strong-energy exposure

From the frozen lag-0 channel-transfer evidence:

- `p_s_corr`;
- `p_s_ctrl`;
- strong-energy paired ordering;
- strong/weak channel counts;
- lag-0 kernel RMS;
- exact channel partition.

### 21.3 Bias-free in-projection reconstruction

For every exercised target:

`W_H ΔX_t`

must reproduce observed:

`ΔH_t`

within the frozen in-projection tolerance:

`1e-6`.

Implementation may set a tighter additional float64 algebraic closure tolerance, but it must not weaken the existing runtime reconstruction gate.

---

## 22. Exact algebraic gates

For every defined item/role require:

### 22.1 H reconstruction

`h ≈ W_H x`

within the frozen runtime tolerance.

### 22.2 Channel transfer identity

For every channel:

`e_j = r_j² A_j`

within fixed implementation tolerance.

### 22.3 Total transfer closure

`T_H² = Σ_j e_j`.

### 22.4 Strong/weak transfer closure

`T_H² = T_S² + T_W²`.

### 22.5 Strong energy identity

`P_S = T_S² / T_H²`.

### 22.6 Paired all-channel difference closure

`T_H,corr² - T_H,ctrl² = Σ_j D_j`.

### 22.7 Paired strong-channel difference closure

`T_S,corr² - T_S,ctrl² = Σ_(j∈S) D_j`.

### 22.8 Paired weak-channel difference closure

`T_W,corr² - T_W,ctrl² = Σ_(j∈W) D_j`.

### 22.9 Enrichment identities

`log(P_S,corr/P_S,ctrl) = 2(G_S-G_H)`.

`log(P_W,corr/P_W,ctrl) = 2(G_W-G_H)`.

All tolerances must be fixed before scientific runtime execution.

---

## 23. Execution population

The scientific population is fixed to:

- layer:
  `22`;
- current token only;
- relative coordinate:
  `k=2`;
- common DDSSSSS cohort:
  `330`;
- both corr and ctrl roles;
- same frozen matched/swapped definitions;
- same checkpoint/runtime lineage.

The runner may execute the frozen 672 pair-role plan for exact parent comparability.

Scientific summaries must remain restricted to common-330 `k=2`.

No other layer, lag, token window, or cohort may be searched.

---

## 24. Runtime capture policy

Reuse the frozen layer-22 U-path capture.

At the current target token extract only:

- `X_RF32[0,:]`;
- `H_RF32[0,:]`.

The index `0` is already authenticated by the parent capture as the current-token / lag-0 position.

Compute scientific algebra in float64 after read-only float32 capture.

Do not alter model forward semantics.

---

## 25. Artifact policy

A later authorized execution may persist:

1. common-330 item scalar metrics;
2. all-1536 channel aggregate summary;
3. downstream-kernel-rank cumulative profile;
4. optional in-projection-row-gain-rank cumulative profile;
5. summary JSON;
6. execution manifest.

It must not persist:

- raw `ΔX_t` vectors;
- raw `ΔH_t` vectors;
- per-item `A_j` vectors;
- per-item `D_j` vectors;
- model activations beyond aggregate/scalar evidence.

Atomic `.partial` output behavior is required.

---

## 26. Interpretation rules

### 26.1 Selective strong routing

If:

- parent `P_S` is reproduced;
- corr has larger `P_S`;
- `G_S > G_H` systematically;
- strong-channel `ΣD_j` is positive;

then the permitted conclusion is:

**the fixed bias-free current-token in-projection routes the corr `ΔX_t` direction relatively more strongly into the downstream-strong H-channel partition than it routes the ctrl direction.**

This is direction-conditioned transfer through a fixed operator.

It is not a causal intervention claim.

### 26.2 Weak-channel suppression

If `G_W < G_H` and weak-channel contribution is negative, then part of the strong-energy-mass increase is due to relatively weaker corr routing into downstream-weak channels.

### 26.3 Mixed mechanism

If both strong positive routing and weak negative routing occur, report both.

Do not collapse them into a single vague "reorientation" statement.

### 26.4 Fixed row-gain structure

If strong channels have larger fixed in-projection row gains than weak channels, report this as operator structure that magnifies directional alignment on the strong partition.

Do not call fixed row gain a role-specific cause.

Role-specific differences arise from `ΔX_t` direction relative to the fixed rows.

### 26.5 Concentration

Use the full channel tables, `N_eff`, cancellation ratios, and fixed-rank cumulative profiles.

Do not promote a post-hoc top-k set as the mechanism.

---

## 27. Falsification outcomes

This stage remains informative if the expected qualitative picture fails.

Possible outcomes include:

### A. Strong-routing selective

`G_S` substantially exceeds `G_H`, while weak routing does not.

Interpretation:

corr input direction is selectively routed toward downstream-strong H channels.

### B. Weak-suppression selective

`G_W` is substantially below `G_H`, while strong routing changes little.

Interpretation:

strong energy mass rises mainly because weak routing is relatively suppressed.

### C. Mixed

Both effects contribute.

### D. Broad current-token attenuation

`G_S`, `G_W`, and `G_H` move together, with little partition-specific role difference.

This would contradict the proposed selective routing interpretation and force reconsideration of how the parent `P_S` difference is being localized.

No arbitrary scientific effect threshold should be introduced to choose among these.

Use the exact paired distributions and preregistered summaries.

---

## 28. Claims explicitly prohibited

This stage cannot establish that:

- any in-projection row is causally necessary;
- any input coordinate is causally necessary;
- any channel subset is sufficient;
- strong channels encode a semantic concept;
- the in-projection weights change by role;
- the corr/control phenomenon originates at layer 22;
- a specific input-space direction should be called a learned feature;
- an eigenspace or SVD direction is required;
- the result generalizes beyond the frozen layer/population/window;
- K1 has been established.

No intervention is authorized.

No PCA, SVD, whitening, probe, or learned geometry is authorized.

---

## 29. Stop conditions

Stop if:

- branch is incorrect;
- parent evidence freeze is not an ancestor;
- parent artifact/hash identities mismatch;
- current-token index cannot be authenticated as `X_RF32[0,:]` / `H_RF32[0,:]`;
- bias-free `W_H` boundary cannot be authenticated;
- strong/weak partition differs from `240/1296/0`;
- lag-0 kernel RMS differs from frozen evidence;
- parent current-token metrics cannot be reproduced;
- parent strong-energy mass cannot be reproduced;
- H reconstruction fails;
- exact routing identities fail;
- raw per-item vectors would need to be persisted;
- learned geometry or post-hoc channel selection is proposed;
- unrelated tracked worktree changes exist;
- either unrelated K1 file changes state.

---

## 30. Static-design success criterion

This static design is complete when implementation can proceed without result-dependent scientific choices and the following are fixed:

1. current-token `ΔX_t → ΔH_t` boundary;
2. bias-free `W_H`;
3. parent strong/weak H-channel partition;
4. total/strong/weak transfer definitions;
5. row-gain × squared-directional-alignment factorization;
6. exact channel role-difference contribution `D_j`;
7. all-1536-channel preregistration;
8. parent current-token reproduction;
9. parent strong-energy-mass reproduction;
10. threshold-free concentration/cancellation metrics;
11. fixed downstream-kernel-rank cumulative profile;
12. no raw-vector persistence;
13. no learned geometry;
14. observational/algebraic interpretation boundary.

Completion of this design does not itself authorize scientific model execution.
