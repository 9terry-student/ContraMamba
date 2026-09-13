# ContraMamba K0-RVG Layer-22 Lag-0 Channel-Conditioned Transfer Localization
## Validated Evidence Analysis Report Candidate

## 1. Status

**Evidence status:** validated local observational/algebraic execution.

**Static-design authority commit:**

`c1d23d469acf37558c794d4b17c95fdb354c2c7f`

**Implementation commit / runtime HEAD:**

`4c19d02d94600e47f39c15bebd839f5a4820a473`

**Parent evidence freeze commit:**

`6b0d282c22d0b1c8f8387d0853205c3d4226a392`

**Run directory:**

`reports/longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_4c19d02_v1`

This report interprets only the validated artifacts from that run.

It does not authorize training, evaluation, intervention, learned geometry, post-hoc channel search, or a transition to K1.

---

## 2. Frozen scientific question

The bounded question was:

> Why does the fixed layer-22 lag-0 depthwise-convolution tap transfer the corr current-token hidden-difference direction more strongly than ctrl?

The parent four-tap decomposition had already frozen that:

- layer 22 is fixed;
- lag 0 is the dominant contribution-magnitude carrier at common-330 `k=2`;
- lag-0 incoming hidden-difference magnitude is already corr-enriched;
- lag-0 fixed-tap transfer contributes an additional positive corr-specific enrichment;
- this stage must explain that additional fixed-tap transfer term only.

No lag search, layer search, window search, or channel subset search is part of this stage.

---

## 3. Exact algebraic object

For channel `j`:

`q_j = k_j h_j`

where:

- `h_j = ΔH_t,j`;
- `k_j` is the fixed layer-22 lag-0 depthwise kernel weight.

Define:

`p_j = h_j² / Σ_r h_r²`.

Then the lag-0 norm transfer is exactly:

`t² = Σ_j p_j k_j²`.

Let:

`μ_k2 = mean_j(k_j²)`,

and define:

`N² = t² / μ_k2`.

Then:

`N² = Σ_j p_j (k_j² / μ_k2)`.

With:

`a_j = k_j² / μ_k2 - 1`,

the normalized transfer excess obeys:

`N² - 1 = Σ_j p_j a_j`.

For aligned corr and ctrl roles:

`Δp_j = p_corr,j - p_ctrl,j`

and:

`c_j = a_j Δp_j`.

Therefore:

`N_corr² - N_ctrl² = Σ_j c_j`.

This identity is exact and contains no fitted parameter, learned projection, selected subset, or residual explanatory term.

Kernel sign and hidden-difference sign do not enter the norm-transfer mechanism.

The entire fixed-tap norm-transfer difference is therefore a question of **where hidden-difference energy is distributed relative to the fixed squared-kernel profile**.

---

## 4. Validated artifact identities

### Item metrics

Path:

`reports/longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_4c19d02_v1/layer22_lag0_channel_transfer_item_metrics.jsonl`

SHA256:

`81396fafb131b1efa5e62877d29ddd9c5b860e718fe94adb1f0c5ddbb8b92f81`

Row count:

`330`

### Channel summary

Path:

`reports/longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_4c19d02_v1/layer22_lag0_channel_transfer_channel_summary.jsonl`

SHA256:

`459af9290b110304160b04148ff46b511aca50ca8bae7849aa80dabe5461b451`

Row count:

`1536`

All layer-22 intermediate channels are represented.

### Kernel-rank cumulative summary

Path:

`reports/longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_4c19d02_v1/layer22_lag0_kernel_rank_cumulative.jsonl`

SHA256:

`ba838da6f12286a6dd7699275d5257a04366015c262f1a070b28f5ef5ace4c50`

Row count:

`1536`

### Summary

Path:

`reports/longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_4c19d02_v1/summary.json`

SHA256:

`ca33ff9a00d2527ce480cbefd4244f7830bc6dbd32298fd6356030cc0ecfc3a3`

Schema:

`k0-rvg-layer22-lag0-channel-transfer-summary-v1`

### Execution manifest

Path:

`reports/longterm_k0_rvg_layer22_lag0_channel_conditioned_transfer_localization_4c19d02_v1/execution_manifest.json`

SHA256:

`5b304a571dfc17af26c21faa185bd7b9b9a74e3432ae1b80e085fa44b8e8a94c`

Schema:

`k0-rvg-layer22-lag0-channel-transfer-execution-manifest-v1`

Manifest-recorded hashes exactly match the recomputed hashes for:

- item metrics;
- channel summary;
- cumulative kernel-rank summary;
- summary JSON.

No `.partial` directory remained.

---

## 5. Execution and provenance validity

Validated execution properties:

- `model_forward_count = 1344`;
- runtime HEAD:
  `4c19d02d94600e47f39c15bebd839f5a4820a473`;
- runtime branch:
  `longterm-k-series-native-state-kinematics`;
- source layer:
  `22`;
- causal lag:
  `0`;
- relative coordinate:
  `k=2`;
- common cohort:
  `330`;
- parent common-k2 lag0 metrics reproduced:
  `True`;
- raw vectors persisted:
  `False`;
- raw per-item channel vectors persisted:
  `False`;
- training:
  `False`;
- intervention:
  `False`;
- post-hoc search:
  `False`.

The runner SHA256 recorded by the manifest is:

`dd1e8ffb0516c4b8392389578815fa5c5c2f05ba826eb37055025b6eda6fae0c`.

---

## 6. Algebraic validity

Maximum validated residuals:

- itemwise channel-contribution closure:
  `1.69309011255336e-15`;
- centered-kernel identity:
  `1.33226762955019e-15`;
- direct transfer bridge:
  `8.32667268468867e-17`;
- aggregate contribution closure:
  `1.11022302462516e-16`.

These are far below the preregistered tolerances.

Therefore the observed lag-0 normalized-transfer difference is exactly accounted for by the all-channel energy-exposure decomposition under numerical precision.

There is no unresolved fixed-lag0 norm-transfer term left inside this decomposition.

---

## 7. Authenticated lag-0 kernel partition

Lag-0 kernel RMS:

`0.243832238090525`

Channels are partitioned only by the algebraic zero point:

`k_j² = mean(k²)`.

Counts:

- strong-kernel channels:
  `240`;
- weak-kernel channels:
  `1296`;
- equal-to-mean channels:
  `0`.

This is not a fitted threshold.

It is the exact sign boundary of:

`a_j = k_j² / mean(k²) - 1`.

---

## 8. Corr shifts hidden-difference energy toward strong-kernel channels

Mean strong-kernel energy mass:

- corr:
  `0.306321914935865`;
- ctrl:
  `0.220282645598935`.

Absolute mean difference:

`0.08603926933693001`.

Relative to ctrl, the corr mean strong-kernel mass is approximately:

`0.390586`

or about `39.06%` larger.

Median strong-kernel energy mass:

- corr:
  `0.3075900794904822`;
- ctrl:
  `0.2173777940609381`.

Paired ordering:

- corr > ctrl:
  `328/330`.

This is the central population-level result.

The corr branch does not merely have a larger total `||ΔH_t||`.

Its **normalized hidden-difference energy distribution** is also shifted toward the channels where the fixed lag-0 kernel has above-mean squared strength.

Because `p_j` is normalized to sum to one, this is an orientation/exposure result in channel-energy space, not a restatement of upstream magnitude enrichment.

---

## 9. The energy redistribution produces the observed transfer advantage

Normalized transfer-squared medians:

- corr:
  `1.986850086431923`;
- ctrl:
  `1.4645596180373468`.

Median paired difference:

`ΔN² = +0.530069198446701`.

Mean paired difference:

`ΔN² = +0.511125337644136`.

Paired ordering:

- `N_corr² > N_ctrl²`:
  `323/330`.

The parent log-transfer enrichment is exactly reproduced:

`G_TAP0 median = +0.15367770768001132`.

Thus the previously frozen positive lag-0 tap-transfer enrichment is not an unexplained property of the convolution operator.

It is localized to the role-dependent distribution of current-token hidden-difference energy across the fixed lag-0 kernel-strength profile.

---

## 10. Strong-kernel channels account for nearly all of the net mean transfer difference

Aggregate all-channel mean contribution:

`M_net = +0.511125337644136`.

Strong-kernel total mean contribution:

`+0.497431650685685`.

Weak-kernel total mean contribution:

`+0.0136936869584509`.

As fractions of the net mean contribution:

- strong-kernel channels:
  `0.973208749`
  ≈ `97.32%`;
- weak-kernel channels:
  `0.026791251`
  ≈ `2.68%`.

Therefore the net transfer advantage is overwhelmingly carried by the preregistered above-mean kernel-strength partition.

The weak-kernel partition contributes only a small positive remainder in aggregate.

This supports:

**preferential corr energy exposure to stronger fixed lag-0 kernel channels is the dominant algebraic explanation of the lag-0 transfer advantage.**

---

## 11. The channel effect is concentrated rather than uniformly diffuse

Threshold-free effective channel count:

`N_eff = 9.84725004781078`.

This statistic is defined from the complete 1536-channel aggregate mean contribution profile.

It should not be interpreted as saying that exactly 10 channels are scientifically active.

Instead, it indicates that the signed contribution magnitude is highly concentrated relative to a uniform 1536-channel distribution.

The correct conclusion is therefore:

**the lag-0 transfer difference is organized through a comparatively concentrated channel-contribution profile, not a broad uniform shift across all 1536 channels.**

No post-hoc top-k channel set is promoted by this result.

The complete channel table and fixed kernel-rank cumulative profile remain the valid evidence.

---

## 12. Strong concentration coexists with substantial signed cancellation

Absolute aggregate contribution mass:

`M_abs = 1.04057192646257`.

Net aggregate contribution:

`M_net = 0.511125337644136`.

Signed cancellation ratio:

`R_cancel = 0.4911965474426256`.

Thus only about:

`49.12%`

of the total absolute mean channel contribution survives after positive and negative channel contributions cancel.

Equivalently, the channel profile is not a purely one-directional redistribution.

Substantial positive and negative channel contributions coexist.

This matters because the scientific picture is not:

> corr simply moves energy into all strong channels and away from all weak channels.

Instead, the exact all-channel geometry contains considerable signed redistribution, while the **net balance** strongly favors the strong-kernel partition.

---

## 13. Integrated mechanism at the fixed lag-0 tap

The validated evidence supports the following hierarchy.

### 13.1 Upstream magnitude component

The parent stage had already shown that lag-0 incoming `||ΔH_t||` is strongly corr-enriched.

That component exists before fixed-tap transfer normalization.

### 13.2 Channel-energy exposure component

After normalizing away total `||ΔH_t||`, corr still differs from ctrl.

Corr places a larger fraction of hidden-difference energy on above-mean `k²` channels:

- mean strong energy:
  corr `0.3063`;
  ctrl `0.2203`;
- corr > ctrl:
  `328/330`.

### 13.3 Exact transfer consequence

This redistribution increases normalized transfer:

- `N²` corr > ctrl in `323/330`;
- mean `ΔN² = +0.5111`;
- median `ΔN² = +0.5301`.

### 13.4 Partition localization

Approximately `97.32%` of the aggregate net mean transfer difference comes from strong-kernel channels.

### 13.5 Internal cancellation

The effect is not monotone at every channel.

`R_cancel ≈ 0.491` shows substantial signed cancellation across the full channel profile.

### 13.6 Concentration

`N_eff ≈ 9.85` indicates a highly concentrated aggregate contribution organization relative to the full 1536-dimensional channel space.

---

## 14. Frozen scientific conclusion

The validated K0-RVG layer-22 lag-0 channel-conditioned transfer result is:

**At common-330 k=2, the additional corr-specific lag-0 fixed-tap transfer advantage is explained by role-dependent channel-energy exposure to the fixed layer-22 lag-0 squared-kernel profile. Corr allocates substantially more normalized current-token hidden-difference energy to above-mean kernel-strength channels than ctrl, and this redistribution raises normalized lag-0 transfer for 323/330 aligned items. The exact all-channel contribution identity closes to numerical precision, and approximately 97% of the aggregate net mean transfer difference is carried by the strong-kernel partition. The aggregate channel effect is comparatively concentrated (`N_eff≈9.85`) but contains substantial positive/negative cancellation (`R_cancel≈0.491`), so the mechanism is concentrated and cancellation-rich rather than a uniform one-directional shift across channels.**

Equivalently:

**upstream magnitude:** corr already has larger lag-0 `ΔH_t`;

**fixed-tap selectivity:** corr distributes that energy toward stronger lag-0 kernel channels;

**exact consequence:** this fully explains the positive lag-0 normalized-transfer difference;

**channel organization:** concentrated, with substantial signed cancellation;

**residual inside fixed lag0 norm transfer:** none under the exact decomposition.

This is an observational/algebraic localization.

It is not a causal intervention result.

---

## 15. Claims not supported

This evidence does not establish that:

- any particular channel is causally necessary;
- any particular channel is sufficient;
- an effective channel count of `9.85` means exactly 10 channels are mechanistically active;
- ablating the strongest contributors would remove the downstream phenotype;
- strong-kernel channels encode a semantic concept;
- kernel sign matters for this norm-transfer effect;
- the corr/control distinction originates in layer 22;
- the channel-energy redistribution is generated by the convolution;
- the same channel profile generalizes to another layer, lag, coordinate, seed, or task;
- K1 has been established.

No intervention was performed.

No learned channel selection was performed.

No post-hoc threshold was introduced.

---

## 16. Scientific consequence for the next K0 question

The lag-0 fixed-tap norm-transfer mechanism is now algebraically closed.

Because:

`N_corr² - N_ctrl² = Σ_j a_j (p_corr,j - p_ctrl,j)`

closes to numerical precision, there is no additional hidden norm-transfer operation to discover inside the fixed lag-0 depthwise tap.

The next unresolved K0 question moves upstream:

**How does layer-22 current-token `ΔH_t` acquire the corr-specific channel-energy distribution that preferentially exposes stronger lag-0 kernel channels?**

A next stage, if pursued, should therefore remain upstream of the depthwise lag-0 multiplication and analyze the construction of `ΔH_t` itself.

It must not repeat the already-closed lag0 kernel exposure decomposition.

It must also not transition to K1 without a separate scientific decision.
