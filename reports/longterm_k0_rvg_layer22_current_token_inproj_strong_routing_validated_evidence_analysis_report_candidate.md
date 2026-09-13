# ContraMamba K0-RVG Layer-22 Current-Token In-Projection Strong-Routing
## Validated Evidence Analysis Report Candidate

## 1. Status

**Evidence status:** validated local observational/algebraic execution.

**Base static-design authority commit:**

`d4b3b2fb47c92d2e20b65764178ada4f82c2697b`

**Reconstruction-gate correction authority commit:**

`3df1cfaac5af7a5551307c33380b5dc1f33fecbd`

**Corrected implementation commit / runtime HEAD:**

`cd602f036b03e36e171837d9532541a530799954`

**Failed superseded implementation commit:**

`cc539ff5a3b9d692fa946c0d8e470685ea8ac6e3`

**Parent lag-0 channel-transfer evidence freeze:**

`bfa626261aba575ab7316877bc69ca6e5df38157`

**Parent U-path evidence freeze:**

`dabb9422dbf0e111319828cf027e8dc5d82fe326`

**Run directory:**

`reports/longterm_k0_rvg_layer22_current_token_inproj_strong_routing_cd602f0_v1`

This report interprets only the independently validated artifacts from that run.

It does not authorize training, evaluation, intervention, learned geometry, post-hoc channel search, or a transition to K1.

---

## 2. Frozen scientific question

The bounded question was:

> How does the fixed bias-free layer-22 current-token hidden in-projection map corr and ctrl `ΔX_t` directions into the different `ΔH_t` channel-energy distributions observed at the lag-0 convolution input?

The preceding frozen evidence had already established that:

- the layer-22 lag-0 contribution is the dominant current-token convolution carrier at common-330 `k=2`;
- corr places more normalized `ΔH_t` energy on channels with above-mean lag-0 squared-kernel strength;
- this strong-kernel energy exposure raises normalized lag-0 transfer;
- the remaining unresolved K0 question was therefore upstream of the lag-0 multiplication:
  how the current-token `ΔH_t` distribution itself is constructed.

This stage fixes:

- layer `22`;
- current token only;
- relative coordinate `k=2`;
- common DDSSSSS cohort `330`;
- all `1536` hidden in-projection output channels;
- the frozen downstream strong/weak partition:
  `240 / 1296 / 0`;
- the exact bias-free hidden in-projection `W_H`.

No layer search, lag search, channel subset search, learned geometry, or intervention is part of this stage.

---

## 3. Exact algebraic object

Let:

`x = ΔX_t ∈ R^768`

be the current-token layer-22 mixer-input difference.

Let:

`h = W_H x ∈ R^1536`

be the algebraic hidden in-projection difference.

For output row `w_j` define:

`r_j² = ||w_j||²`.

Define directional alignment:

`A_j(x) = cos²(w_j, x)`.

Then:

`h_j = w_j^T x`

and exactly:

`h_j² / ||x||² = r_j² A_j(x)`.

Define:

`e_j = h_j² / ||x||²`.

Then:

`T_H² = Σ_j e_j`.

For the frozen downstream strong and weak channel partitions:

`T_S² = Σ_(j∈S) e_j`

and:

`T_W² = Σ_(j∈W) e_j`.

For aligned corr/ctrl items define:

`D_j = r_j²(A_corr,j - A_ctrl,j)`.

Because `r_j²` is fixed across roles:

`D_j = e_corr,j - e_ctrl,j`.

Therefore:

`Σ_j D_j = T_H,corr² - T_H,ctrl²`.

Likewise:

`Σ_(j∈S) D_j = T_S,corr² - T_S,ctrl²`

and:

`Σ_(j∈W) D_j = T_W,corr² - T_W,ctrl²`.

This is an exact all-channel decomposition of the role-dependent in-projection routing difference.

There is no learned projection, fitted coefficient, selected subset, or residual explanatory term inside this algebraic object.

---

## 4. Reconstruction-gate correction and numerical semantics

The first runtime preflight of the superseded implementation stopped at:

`0:corr_H_RECON_FAILURE:1.5891726920633263e-06`.

That implementation had incorrectly reused the parent branch-relative reconstruction tolerance:

`1e-6`

on the differenced quantity:

`W_H ΔX_t` versus `ΔH_t`.

The correction authority froze the proper semantics:

1. preserve the parent branch-level reconstruction gate `≤1e-6`;
2. define the scientific algebraic difference in float64:
   `h_alg = W_H(X_m-X_s)`;
3. define the runtime-observed difference:
   `h_obs = H_m-H_s`;
4. define branch errors:
   `e_m = W_HX_m-H_m`,
   `e_s = W_HX_s-H_s`;
5. require the exact finite-precision bridge:
   `h_alg-h_obs = e_m-e_s`;
6. treat the cancellation-sensitive difference-relative residual as diagnostic only;
7. retain the preregistered algebraic↔observed strong-energy bridge tolerance:
   `2e-6`.

No tolerance was widened after viewing the failed result.

The corrected runtime preflight passed before full execution.

---

## 5. Validated artifact identities

### Item metrics

Path:

`reports/longterm_k0_rvg_layer22_current_token_inproj_strong_routing_cd602f0_v1/layer22_current_token_inproj_routing_item_metrics.jsonl`

SHA256:

`f3a34918c02b1715af5ec57221d9d3a34b23afb71849c3962f1aedbe458b14c8`

Row count:

`330`

### Channel summary

Path:

`reports/longterm_k0_rvg_layer22_current_token_inproj_strong_routing_cd602f0_v1/layer22_current_token_inproj_routing_channel_summary.jsonl`

SHA256:

`2cf91c7611cd57cd8bc1deda9269cb7e36b6008af1baab523e8a6b4818752cf2`

Row count:

`1536`

All layer-22 hidden in-projection output channels are represented.

### Kernel-rank cumulative profile

Path:

`reports/longterm_k0_rvg_layer22_current_token_inproj_strong_routing_cd602f0_v1/layer22_current_token_inproj_kernel_rank_cumulative.jsonl`

SHA256:

`dac78490055ffbd0f723bdf07570305d1d51ff48e4550f9f980769355f4772b5`

Row count:

`1536`

### Summary

Path:

`reports/longterm_k0_rvg_layer22_current_token_inproj_strong_routing_cd602f0_v1/summary.json`

SHA256:

`c61f5faabe7a47fc71a516c9035bf739fc7c47f5707041558ab9f7d6f98cc47a`

Schema:

`k0-rvg-layer22-current-token-inproj-routing-summary-v1`

### Execution manifest

Path:

`reports/longterm_k0_rvg_layer22_current_token_inproj_strong_routing_cd602f0_v1/execution_manifest.json`

SHA256:

`91d91b4d52966242e3c2b7d2532d91e3070991eaae8f2dec4f5d5293fe4a83df`

Schema:

`k0-rvg-layer22-current-token-inproj-routing-execution-manifest-v1`

The independent validator recomputed and matched the manifest-recorded output hashes.

No `.partial` run directory remained.

---

## 6. Execution and provenance validity

Validated execution properties:

- runtime HEAD:
  `cd602f036b03e36e171837d9532541a530799954`;
- runtime branch:
  `longterm-k-series-native-state-kinematics`;
- runner SHA256:
  `8d6c45636197ee6cc1d9e6f8c423a1f8fe57fc654ae450a4b3eb6b8657dfaf5b`;
- `model_forward_count = 1344`;
- pair-role plan:
  `672`;
- common cohort:
  `330`;
- source layer:
  `22`;
- relative coordinate:
  `k=2`;
- hidden width:
  `768`;
- intermediate width:
  `1536`;
- parent current-token metrics reproduced:
  `True`;
- parent strong-energy mass reproduced:
  `True`;
- raw vectors persisted:
  `False`;
- raw per-item channel vectors persisted:
  `False`;
- tokenizer invoked:
  `False`;
- logits read:
  `False`;
- task heads executed:
  `False`;
- training:
  `False`;
- causal intervention:
  `False`;
- PCA/SVD/whitening/learned geometry:
  `False`;
- post-hoc layer/lag/channel/item/window search:
  `False`.

The independent artifact validator passed:

`PASS_LAYER22_CURRENT_TOKEN_INPROJ_STRONG_ROUTING_ARTIFACT_VALIDATION`.

---

## 7. Algebraic and numerical validity

Maximum validated residuals:

- parent branch reconstruction relative residual:
  `3.9620320852752387e-07`;
- difference-relative reconstruction residual:
  `4.2112793395426884e-06`
  (**diagnostic only**);
- error-difference identity absolute residual:
  `1.0449974219284286e-14`;
- all-channel `D_j` closure:
  `8.881784197001252e-15`;
- strong-channel `D_j` closure:
  `2.220446049250313e-15`;
- weak-channel `D_j` closure:
  `6.2727600891321345e-15`;
- algebraic↔observed `P_S` bridge:
  `2.3741766519425056e-07`.

The branch-level reconstruction remains below its frozen `1e-6` gate.

The error-difference identity and all `D_j` closures are far below the preregistered `5e-12` algebraic tolerance.

The algebraic↔observed `P_S` bridge is far below the frozen `2e-6` bridge tolerance.

Therefore the fixed `W_H` current-token routing decomposition is numerically closed for the validated population.

---

## 8. Frozen downstream strong/weak partition

The partition inherited from the lag-0 channel-transfer stage is unchanged.

Lag-0 kernel RMS:

`0.24383223809052498`.

Counts:

- strong-kernel channels:
  `240`;
- weak-kernel channels:
  `1296`;
- equal-to-mean channels:
  `0`.

This partition is defined by downstream lag-0 squared-kernel strength, not by the current stage's `D_j` values.

No channel was selected after viewing the new routing results.

---

## 9. Fixed in-projection row gains differ structurally across the downstream partitions

Mean squared hidden in-projection row gain:

- strong partition:
  `13.303531506644353`;
- weak partition:
  `11.511346510774066`.

Strong/weak mean row-gain-squared ratio:

`1.1556885629489901`.

Thus the rows whose outputs feed the downstream strong-kernel partition have about `15.57%` larger mean squared row norm than the weak partition.

This is a fixed operator property.

It is **not** by itself a corr-specific mechanism because the same `W_H` rows are applied to corr and ctrl.

Role dependence can only enter this decomposition through the input-direction alignment term:

`A_j(x)`.

The scientific role difference is therefore the role-dependent alignment/exposure of `ΔX_t` to these fixed rows, weighted by the fixed row gains.

---

## 10. Overall current-token in-projection transfer is only modestly corr-enriched

Current-token total transfer medians:

- corr:
  `T_H = 4.864999167276736`;
- ctrl:
  `T_H = 4.768681757025245`.

Median log enrichment:

`G_H = +0.025515793449323046`.

This corresponds to a median multiplicative ratio of approximately:

`exp(G_H) = 1.02584`.

Paired ordering:

- `T_H,corr > T_H,ctrl`:
  `243/330`.

Thus the total current-token hidden in-projection norm transfer is positively but only modestly corr-enriched.

This total statistic alone hides a much stronger partition-specific redistribution.

---

## 11. Strong-partition routing is universally corr-enhanced in the validated cohort

Strong-partition transfer medians:

- corr:
  `T_S = 2.710264583850882`;
- ctrl:
  `T_S = 2.2249118110107102`.

Median log enrichment:

`G_S = +0.1968593236956881`.

This corresponds to a median multiplicative ratio of approximately:

`exp(G_S) = 1.21757`.

Paired ordering:

- `T_S,corr > T_S,ctrl`:
  `330/330`.

Mean strong-partition transfer-squared difference:

`mean(ΔT_S²) = +2.3029343955594896`.

This is the most population-uniform result in the stage.

At every aligned common-330 item, the fixed in-projection maps the corr current-token input difference to larger algebraic transfer into the downstream strong-kernel channel partition than ctrl.

---

## 12. Weak-partition routing is predominantly lower for corr

Weak-partition transfer medians:

- corr:
  `T_W = 4.048511226571639`;
- ctrl:
  `T_W = 4.193397644573377`.

Median log enrichment:

`G_W = -0.03423699371158165`.

This corresponds to a median multiplicative ratio of approximately:

`exp(G_W) = 0.96634`.

Paired ordering:

- `T_W,corr > T_W,ctrl`:
  `76/330`.

Therefore corr is lower than ctrl on the weak partition for the large majority of aligned items.

Mean weak-partition transfer-squared difference:

`mean(ΔT_W²) = -1.144025070180993`.

The current-token in-projection phenotype is therefore not merely positive routing into strong outputs.

It also contains a counter-directional reduction of weak-partition routing.

---

## 13. The observed `P_S` shift is a mixed redistribution, not a uniform gain

Strong-channel energy mass from the algebraic `W_H ΔX_t` output:

Mean:

- corr:
  `0.3063219128377242`;
- ctrl:
  `0.2202826438512968`.

Absolute mean difference:

`+0.08603926898642739`.

Relative to ctrl, the corr mean is approximately:

`39.06%` larger.

Median:

- corr:
  `0.3075900978133345`;
- ctrl:
  `0.21737790111229088`.

Paired ordering:

- `P_S,corr > P_S,ctrl`:
  `328/330`.

Median log strong-mass enrichment:

`log(P_S,corr/P_S,ctrl) = +0.3470956227935581`.

This is exactly tied to the routing enrichments by:

`log(P_S,corr/P_S,ctrl) = 2(G_S-G_H)`.

The positive strong-mass shift is therefore produced by **strong routing growing much faster than total routing**.

At the same time:

`G_W < G_H`

and the weak-mass enrichment is negative:

`log(P_W,corr/P_W,ctrl) = -0.11731186275025532`.

Hence the `P_S` phenotype is best described as a **mixed partition redistribution**:

- strong-output routing is selectively enhanced;
- weak-output routing is selectively reduced relative to the total;
- the combined effect moves normalized hidden-difference energy toward the downstream strong-kernel partition.

---

## 14. Strong enhancement is partially cancelled by weak suppression in the total transfer

Mean all-channel transfer-squared difference:

`mean(ΔT_H²) = +1.1589093253784961`.

Exact partition components:

- strong:
  `+2.3029343955594896`;
- weak:
  `-1.144025070180993`.

These close exactly:

`+2.3029343955594896 - 1.144025070180993 = +1.1589093253784966`

up to floating-point summation order.

Relative to the small positive net:

- strong positive component:
  approximately `198.72%` of net;
- weak negative component:
  approximately `-98.72%` of net.

These percentages exceed `100%` in magnitude because the two partitions oppose each other.

The correct interpretation is not that the strong partition contributes “198.7% of the mechanism” in an absolute causal sense.

Rather:

**the positive strong-routing shift is roughly twice the surviving net all-channel difference and is then offset by an almost equally large negative weak-routing shift.**

This explains why:

- `G_S` is large and positive;
- `G_W` is negative;
- yet `G_H` is only mildly positive.

---

## 15. The all-channel directional-routing profile is highly cancellation-rich

All-channel aggregate contribution diagnostics:

- `M_abs = 15.311472549580301`;
- `M_net = +1.1589093253784966`;
- `R_cancel = 0.07568895294856948`;
- `N_eff = 184.58354929466057`.

Only about `7.57%` of the absolute aggregate mean channel-contribution mass survives after positive and negative channel contributions cancel.

This means the complete in-projection role difference is strongly signed and internally compensatory.

The effect is not a broad monotone corr increase across all 1536 output channels.

Instead, many positive and negative directional-alignment changes coexist, with a small positive all-channel net.

---

## 16. Strong and weak partitions have different internal organizations

### Strong partition

- `M_abs = 4.920354599977239`;
- `M_net = +2.302934395559489`;
- `R_cancel = +0.46804236336343363`;
- `N_eff = 26.851493630928402`.

The strong partition has a substantial positive net and moderate internal cancellation.

Its effective contribution profile is much more concentrated than the full 1536-channel space.

### Weak partition

- `M_abs = 10.391117949603062`;
- `M_net = -1.1440250701809933`;
- `R_cancel = -0.1100964377201295`;
- `N_eff = 293.02336691353173`.

The weak partition has a negative net with very large absolute contribution mass and strong positive/negative cancellation.

Its role is therefore not a uniform suppression across all weak channels.

Instead, the weak partition contains a broad cancellation-rich redistribution whose surviving net favors ctrl.

The contrast is important:

- strong partition:
  concentrated positive routing advantage;
- weak partition:
  broader cancellation-rich negative routing balance.

---

## 17. Integrated mechanism of the fixed layer-22 current-token in-projection

The validated evidence supports the following hierarchy.

### 17.1 Fixed operator structure

`W_H` is the same for corr and ctrl.

Its output rows feeding the downstream strong-kernel partition have larger mean squared row gain than weak-partition rows:

`13.3035` versus `11.5113`.

This structural asymmetry is role-independent.

### 17.2 Role-dependent input direction

Corr and ctrl provide different current-token `ΔX_t` directions.

Because row gains are fixed across roles, the role-specific contribution is exactly localized to:

`A_corr,j - A_ctrl,j`.

### 17.3 Selective strong routing

The corr direction aligns with the fixed in-projection rows in a way that produces larger strong-partition transfer for:

`330/330`

aligned items.

Median:

`G_S = +0.19686`.

### 17.4 Relative weak suppression

The same corr direction produces lower weak-partition transfer for most aligned items.

Median:

`G_W = -0.03424`.

### 17.5 Modest total gain after cancellation

Strong positive and weak negative components partially cancel.

The surviving total transfer enrichment is only:

`G_H median = +0.02552`.

### 17.6 Downstream consequence

Because strong routing rises much faster than total routing while weak routing is relatively reduced, normalized `ΔH_t` energy shifts toward the frozen downstream strong-kernel partition.

This directly reproduces the previously frozen strong-energy phenotype:

`P_S,corr > P_S,ctrl` in `328/330`.

---

## 18. Frozen scientific conclusion

The validated K0-RVG layer-22 current-token in-projection strong-routing result is:

**At common-330 `k=2`, the fixed bias-free layer-22 hidden in-projection does not produce the corr-specific downstream strong-kernel exposure through a uniform increase in total hidden transfer. Instead, the role difference is a directional-routing redistribution of the current-token `ΔX_t` vector across fixed `W_H` output rows. Corr shows larger strong-partition transfer in all 330 aligned items (`G_S` median `+0.1969`) while weak-partition transfer is lower for most items (`G_W` median `-0.0342`; corr > ctrl only `76/330`). These opposing components partially cancel, leaving only modest total transfer enrichment (`G_H` median `+0.0255`). The exact all/strong/weak `D_j = r_j²(A_corr,j-A_ctrl,j)` identities close to numerical precision. Consequently, the previously frozen corr strong-channel energy advantage (`P_S,corr > P_S,ctrl` in `328/330`) is localized upstream to role-dependent alignment of current-token `ΔX_t` with the fixed hidden in-projection rows: selective strong-output routing plus relative weak-output suppression.**

Equivalently:

**fixed row gains:** structurally larger on downstream strong-kernel rows, but role-independent;

**role-specific source:** corr/ctrl current-token `ΔX_t` directional alignment with those fixed rows;

**strong branch:** universally corr-enhanced across common-330;

**weak branch:** predominantly ctrl-enhanced / corr-suppressed;

**net:** mixed redistribution with strong positive and weak negative components partially cancelling;

**downstream consequence:** normalized `ΔH_t` energy is shifted toward channels that the frozen lag-0 kernel transfers more strongly;

**residual inside fixed current-token `W_H` routing:** none under the exact all-channel decomposition, apart from the explicitly validated finite-precision runtime bridge.

This is an observational/algebraic localization.

It is not a causal intervention result.

---

## 19. Claims not supported

This evidence does not establish that:

- any particular hidden in-projection row or channel is causally necessary;
- any particular row or channel is sufficient;
- the larger strong-partition mean row gain causes the corr/control role difference by itself;
- corr activates every strong channel more than ctrl;
- corr suppresses every weak channel;
- `N_eff` values are exact counts of mechanistically active channels;
- a top-k row subset would reproduce the phenotype;
- the hidden in-projection learns a semantic “strong-kernel” concept;
- the corr/control distinction originates at layer 22;
- the current-token `ΔX_t` directional difference is created by `W_H`;
- the same routing profile generalizes to another layer, coordinate, seed, model, or task;
- K1 has been established.

No intervention was performed.

No learned row/channel selection was performed.

No post-hoc threshold was introduced.

---

## 20. Scientific consequence for the next K0 question

The fixed current-token hidden in-projection mapping is now algebraically closed for this question.

Because:

`D_j = r_j²(A_corr,j-A_ctrl,j)`

and the all/strong/weak sums reproduce the corresponding transfer-squared differences to numerical precision, there is no additional unexplained role-dependent norm-routing term inside the fixed bias-free `W_H` map.

The next unresolved K0 question therefore moves one boundary upstream:

**Why does the layer-22 current-token mixer input `ΔX_t` arrive with the corr-specific direction that aligns preferentially with the fixed strong-output rows of `W_H` and relatively away from the weak-output rows?**

A next K0 stage, if pursued, should localize the construction of the current-token `ΔX_t` direction upstream of the layer-22 hidden in-projection.

It should not repeat:

- the closed lag-0 kernel exposure decomposition;
- the closed current-token `W_H` row-alignment decomposition;
- historical tokenizer provenance;
- K1 execution.

Any transition to K1 remains a separate scientific decision.
