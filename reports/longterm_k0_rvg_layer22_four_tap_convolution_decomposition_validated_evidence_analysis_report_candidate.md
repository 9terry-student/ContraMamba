# ContraMamba K0-RVG Layer-22 Four-Tap Causal-Convolution Decomposition
## Validated Evidence Analysis Report Candidate

## 1. Status

**Evidence status:** validated local observational/algebraic execution.

**Static-design authority commit:**

`7e9e4f5b4ebf0676568822d3aa855278c8c1f192`

**Implementation commit:**

`3dd3791cab718421fd30ef119d44d3d2a4defde9`

**Runtime HEAD:**

`a4c0a48411788cb5bb3ee32fa92e97f91d87de2d`

**Run directory:**

`reports/longterm_k0_rvg_layer22_four_tap_convolution_decomposition_3dd3791_v1`

This report interprets only the validated artifacts from that run.

It does not authorize a causal intervention, training/evaluation, layer search, channel search, K1 transition, or any claim beyond the frozen observational/algebraic boundary.

---

## 2. Validated artifact identities

### Metrics

Path:

`reports/longterm_k0_rvg_layer22_four_tap_convolution_decomposition_3dd3791_v1/layer22_four_tap_convolution_decomposition_metrics.jsonl`

SHA256:

`65d845ecc50021a7dfb1d3b3cd4fd3431843c9d7c8318268f09df0945b4fcf76`

Row count:

`5376`

which equals:

`672 pair-role rows × 8 relative coordinates`.

### Summary

Path:

`reports/longterm_k0_rvg_layer22_four_tap_convolution_decomposition_3dd3791_v1/summary.json`

SHA256:

`620413f7f33a372a082a5ab0e1e938552dc6fa330836768b7e5b3e996f1a9872`

Schema:

`k0-rvg-layer22-four-tap-convolution-decomposition-summary-v1`

### Execution manifest

Path:

`reports/longterm_k0_rvg_layer22_four_tap_convolution_decomposition_3dd3791_v1/execution_manifest.json`

SHA256:

`77e2a06147637f7a233843ed2df2e2ad2f558071785c2ea06bd8c7726fda2e07`

Schema:

`k0-rvg-layer22-four-tap-convolution-decomposition-execution-manifest-v1`

Manifest-recorded runner SHA256:

`bc48b1bcb222dbf75828ad61fb61c0d766a5024e2456d19c8ccee6ccdfcbe088`

The manifest-recorded metrics and summary SHA256 values exactly match the locally recomputed hashes.

No `.partial` directory remained after execution.

---

## 3. Execution and provenance validity

The validated execution completed:

- `model_forward_count = 1344`;
- `parent_h_rf_trajectory_match = True`;
- `parent_delta_c_trajectory_match = True`;
- `raw_vectors_persisted = False`;
- `training_executed = False`;
- `causal_intervention_executed = False`;
- `posthoc_layer_lag_channel_item_or_window_search_executed = False`.

The execution preserved the frozen protocol:

- 336 fixed items;
- 672 pair-role rows;
- common DDSSSSS cohort of 330 items;
- source layer fixed to 22;
- divergence-aligned `k=-1..+6`;
- equal-length prefix execution;
- same frozen model/checkpoint/runtime lineage;
- no tokenizer;
- no logits/task heads;
- no learned geometry;
- no post-hoc search.

### Algebraic / reconstruction gates

Full-execution maxima:

- four-tap reconstruction relative residual:
  `1.4264652867215277e-06`
  against tolerance `2e-5`;
- squared-norm closure relative residual:
  `1.635550143719953e-15`;
- interaction-normalization absolute residual:
  `5.551115123125783e-17`;
- addition-factor identity absolute residual:
  `5.932731466096186e-07`;
- common-330 k2 enrichment identity absolute residual:
  `3.05311331771918e-16`.

All frozen gates pass.

Therefore the four-tap decomposition is valid for scientific interpretation under the frozen design.

---

## 4. Authenticated layer-22 convolution structure

The actual layer-22 depthwise causal-convolution kernel was authenticated at runtime.

Causal lag to kernel RMS:

- lag 0:
  `0.24383223809052498`
- lag 1:
  `0.06307570826918663`
- lag 2:
  `0.016738825616582624`
- lag 3:
  `0.0`

Exact-zero status:

- lag 0: `False`
- lag 1: `False`
- lag 2: `False`
- lag 3: `True`

Thus the earlier layer-23 zero-lag-3 precedent was not assumed.

Instead, layer 22 independently authenticates the same structural fact for lag 3:

**the lag-3 tap is exactly zero.**

It remains part of the preregistered four-tap accounting as a structural-zero term.

---

## 5. Frozen scientific question

The bounded question was:

> Within the fixed layer-22 `H_RF -> C` causal-convolution boundary at common-330 `k=2`, which lag-specific incoming differences, fixed channel-conditioned tap transfers, and constructive/destructive vector interactions account for the corr-selective amplification?

The decomposition is:

`Q_l(t) = K[:,3-l] ⊙ ΔH_(t-l)`

and:

`ΔC_t = Σ_l Q_l(t)`.

The analysis distinguishes:

1. incoming lag-specific `ΔH` magnitude;
2. fixed-tap/channel-conditioned transfer into each `Q_l`;
3. cross-free contribution magnitude;
4. signed vector interaction among the `Q_l`.

---

## 6. Common-330 k2 direct convolution output

Role medians:

- corr `||ΔC||₂`:
  `2.0838235086784174`
- ctrl `||ΔC||₂`:
  `1.3248277045372114`

Paired ordering:

- corr > ctrl:
  `328/330`.

Median paired direct-output enrichment:

`E_C = +0.41831449050989755`.

This reproduces the frozen parent localization result that the layer-22 convolution boundary is strongly corr-selective at `k=2`.

The present audit explains the internal fixed-tap geometry of that result.

---

## 7. Lag-0 is the dominant contribution-magnitude carrier

### 7.1 Incoming lag-0 hidden difference is already strongly corr-enriched

Lag-0 incoming hidden-difference medians:

- corr:
  `5.582606885877793`
- ctrl:
  `3.6309675175334286`

Median paired enrichment:

`E_H0 = +0.41294006625397023`.

Therefore a large fraction of the eventual corr-specific convolution effect is already present in the **current-token hidden difference entering the lag-0 tap**.

This is not a result in which equal incoming differences are subsequently separated only by the convolution kernel.

### 7.2 The lag-0 fixed tap further amplifies corr-specificity

Lag-0 contribution medians:

- corr `||Q0||₂`:
  `1.9458025129361645`
- ctrl:
  `1.0722275214333492`

Median paired contribution enrichment:

`E_Q0 = +0.5675913225723508`.

Lag-0 tap-transfer medians:

- corr:
  `0.3436953644307433`
- ctrl:
  `0.29508331412501293`

Weight-RMS-normalized transfer medians:

- corr:
  `1.409556698171893`
- ctrl:
  `1.210189909405911`

Median paired tap-transfer enrichment:

`G_TAP0 = +0.15367770768001138`.

Because `G_TAP0` is positive, the lag-0 corr advantage is not explained only by larger incoming `||ΔH_t||`.

The fixed lag-0 kernel also transfers the corr hidden-difference direction more strongly than the ctrl direction.

Thus lag 0 exhibits a two-part structure:

**strong upstream/current-token magnitude enrichment + additional fixed-tap/channel-conditioned transfer enrichment.**

---

## 8. Lag-0 dominates the cross-free contribution energy

Median contribution-energy fractions:

### corr

- lag 0:
  `0.8458619923521037`
- lag 1:
  `0.1263210953496163`
- lag 2:
  `0.029056547626712746`
- lag 3:
  `0.0`

### ctrl

- lag 0:
  `0.590529913955016`
- lag 1:
  `0.3353430510109542`
- lag 2:
  `0.05601016833952015`
- lag 3:
  `0.0`

Dominant-`Q` lag counts:

### corr

- lag 0:
  `330/330`
- lag 1:
  `0/330`
- lag 2:
  `0/330`
- lag 3:
  `0/330`

### ctrl

- lag 0:
  `264/330`
- lag 1:
  `66/330`
- lag 2:
  `0/330`
- lag 3:
  `0/330`

Therefore the corr branch is not merely a uniformly larger version of the ctrl four-tap mixture.

At common-330 `k=2`, corr reorganizes the cross-free convolution geometry toward **universal lag-0 dominance**, whereas ctrl retains a substantial minority of lag-1-dominant items.

This supports the statement:

**the dominant magnitude carrier of the layer-22 corr-selective convolution amplification is the current-token / lag-0 contribution.**

---

## 9. Lag-1 is not a positive corr-selective amplifier

Lag-1 incoming hidden-difference medians:

- corr:
  `10.13967228366128`
- ctrl:
  `9.804705211586153`

Median paired incoming enrichment:

`E_H1 = +0.03740544965569577`.

Lag-1 contribution medians:

- corr:
  `0.7438238998576929`
- ctrl:
  `0.8139076888486405`

Median paired contribution enrichment:

`E_Q1 = -0.07194883260194632`.

Tap-transfer medians:

- corr:
  `0.07342614797592074`
- ctrl:
  `0.0815237140859114`

Weight-RMS-normalized transfer medians:

- corr:
  `1.1640954971533861`
- ctrl:
  `1.2924740176994078`

Median paired transfer enrichment:

`G_TAP1 = -0.10792198465032299`.

Thus lag 1 shows the opposite of the lag-0 pattern.

Its incoming hidden difference is slightly corr-enriched, but the fixed lag-1 tap preferentially transfers ctrl rather than corr, reversing the sign at the `Q1` contribution level.

Therefore lag 1 acts as a **relative corr-specificity attenuator**, not the source of the positive convolution amplification.

---

## 10. Lag-2 is small and approximately role-neutral at the transfer level

Lag-2 incoming hidden-difference medians:

- corr:
  `22.91367947156602`
- ctrl:
  `23.70596857713617`

Lag-2 contribution medians:

- corr:
  `0.3492056079105314`
- ctrl:
  `0.31821793271679844`

Tap-transfer medians:

- corr:
  `0.01560230192162955`
- ctrl:
  `0.015763708215872065`

Weight-RMS-normalized transfer medians:

- corr:
  `0.9321025428553867`
- ctrl:
  `0.9417451723886447`

Median paired enrichments:

- `E_H2 = -0.002095301349808994`
- `E_Q2 = +0.026067970124676047`
- `G_TAP2 = +0.0010007503671806348`.

Because these are medians of itemwise quantities, `G_TAP2` must not be reconstructed by subtracting the two reported enrichment medians.

The direct median `G_TAP2` is approximately zero.

Thus lag 2 is not supported as an important role-selective tap-transfer mechanism at common-330 `k=2`.

Its contribution is secondary in magnitude and approximately neutral in transfer selectivity.

---

## 11. Lag-3 is structurally absent

The authenticated layer-22 lag-3 kernel RMS is exactly:

`0.0`.

Therefore:

- `Q3 = 0`;
- lag-3 contribution energy is zero;
- lag-3 tap-transfer and log-enrichment quantities are undefined;
- all lag-3 pairwise interactions are structurally zero.

This is a runtime-authenticated structural fact, not a post-hoc exclusion.

---

## 12. Most corr-selective convolution enrichment exists before vector interaction

Cross-free contribution RSS medians:

- corr:
  `2.119772720046205`
- ctrl:
  `1.3976375955321378`

Paired ordering:

- corr > ctrl:
  `328/330`.

Median paired cross-free enrichment:

`E_RSS = +0.3859722496723491`.

Direct convolution-output enrichment:

`E_C = +0.41831449050989755`.

Median vector-addition enrichment:

`G_ADD = +0.026890058295512442`.

Therefore the majority of the direct convolution-output corr enrichment is already present in the **cross-free fixed-tap contribution magnitudes**.

Vector interaction adds a smaller positive increment.

This supports:

**contribution-magnitude-dominant convolution amplification with interaction-assisted refinement.**

It does not support a model in which the corr effect is primarily created by special constructive cancellation geometry among otherwise similar tap magnitudes.

---

## 13. Vector addition is net destructive in both roles, but less destructive for corr

Median vector-addition factors:

- corr:
  `0.9860579428306913`
- ctrl:
  `0.9542651024612421`.

Both medians are below 1.

Thus vector addition is net destructive, not constructive, in the typical item for both roles.

Median normalized total interactions:

- corr:
  `I_TOTAL = -0.027689699238141092`
- ctrl:
  `I_TOTAL = -0.08937812137391751`.

Paired ordering for addition factor:

- corr > ctrl:
  `290/330`.

Therefore the positive `G_ADD` does not mean corr becomes net constructive.

Instead:

**corr experiences less destructive vector addition than ctrl.**

This relative reduction in destructive interaction provides a modest additional corr advantage on top of the already enriched cross-free tap magnitudes.

---

## 14. The interaction difference is concentrated in the lag-1 × lag-2 pair

Pairwise normalized interaction results:

### lag 0 × lag 1

- corr median:
  `+0.0013843732513591138`
- ctrl median:
  `+0.0008342745659087798`
- median paired difference:
  `+6.448525101073778e-06`
- corr > ctrl:
  `165/330`.

No strong systematic role separation is supported.

### lag 0 × lag 2

- corr median:
  `+0.005527865119445353`
- ctrl median:
  `+0.004847048272292121`
- median paired difference:
  `-9.564329744400765e-05`
- corr > ctrl:
  `163/330`.

Again, no stable systematic role separation is supported.

### lag 1 × lag 2

- corr median:
  `-0.03477610140048297`
- ctrl median:
  `-0.09535609120789323`
- median paired difference:
  `+0.0510785961995193`
- corr > ctrl:
  `291/330`.

This is the clear interaction-level role asymmetry.

Both roles show destructive lag-1 × lag-2 interaction, but the corr branch is substantially **less destructive**.

All interaction pairs involving lag 3 are structurally zero.

Therefore the small positive vector-addition enrichment is principally associated with:

**reduced destructive interference between the lag-1 and lag-2 contributions in corr.**

This is a secondary modifier, not the dominant magnitude carrier.

---

## 15. Integrated interpretation

The layer-22 convolution-stage corr amplification at common-330 `k=2` has a hierarchical structure.

### Primary component: current-token lag-0 magnitude

The strongest contribution comes from lag 0.

The incoming current-token hidden difference is already strongly corr-enriched:

`E_H0 ≈ +0.413`.

### Secondary positive component: lag-0 fixed-tap transfer

The fixed lag-0 kernel further increases corr-specificity:

`G_TAP0 ≈ +0.154`.

This raises the lag-0 contribution enrichment to:

`E_Q0 ≈ +0.568`.

### Countervailing component: lag-1 fixed-tap transfer

Lag 1 does not reinforce corr.

Instead it preferentially transfers ctrl:

`G_TAP1 ≈ -0.108`.

### Near-neutral component: lag 2

Lag-2 fixed-tap transfer is approximately role-neutral.

### Structural-zero component: lag 3

Lag 3 contributes nothing because its kernel is exactly zero.

### Interaction modifier

Most convolution enrichment is already present in cross-free contribution magnitudes:

`E_RSS ≈ +0.386`

versus:

`E_C ≈ +0.418`.

The remaining positive increment:

`G_ADD ≈ +0.0269`

is small and reflects **less destructive**, rather than net constructive, vector addition in corr.

The strongest interaction asymmetry is reduced lag-1 × lag-2 destructive interference.

---

## 16. Frozen scientific conclusion

The validated K0-RVG layer-22 four-tap result is:

**At common-330 k2, the layer-22 corr-selective convolution amplification is contribution-magnitude dominant and is carried primarily by the current-token lag-0 pathway. The lag-0 incoming hidden difference is already strongly corr-enriched, and the fixed lag-0 depthwise tap provides additional role-selective channel-conditioned transfer. Lag 1 counteracts corr selectivity at the fixed-tap transfer level, lag 2 is comparatively small and approximately role-neutral, and lag 3 is structurally zero. Vector interaction is a secondary positive modifier: both roles are net destructive, but corr is less destructive, mainly because lag-1 × lag-2 interference is reduced.**

Equivalently:

**primary carrier:** lag-0/current-token contribution magnitude;

**additional selective gain:** lag-0 fixed-tap transfer;

**countervailing tap:** lag 1;

**secondary interaction effect:** reduced destructive lag-1 × lag-2 interference;

**overall geometry:** cross-free magnitude dominant, interaction assisted.

This is an observational/algebraic decomposition.

It is not evidence that any tap is causally necessary or sufficient.

---

## 17. Claims not supported by this evidence

This evidence does **not** establish that:

- lag 0 is causally necessary;
- lag 0 is sufficient;
- ablating lag 0 would remove the phenotype;
- lag 1 is globally inhibitory outside this frozen population/window;
- lag-1 × lag-2 interference causes the task-level behavior;
- a particular channel subset is causal;
- the original corr/control difference originates in layer 22;
- the convolution is the sole mechanism behind the downstream write/state effect;
- the result generalizes to another layer;
- this establishes a K1 mechanism.

No intervention was performed.

No channel search was performed.

No post-hoc lag selection was used: all four preregistered lags and all six interaction pairs were reported.

---

## 18. Immediate scientific consequence

The next unresolved question is narrower than the completed four-tap audit:

**Why does the fixed lag-0 layer-22 depthwise tap transfer the corr current-token hidden-difference direction more strongly than the ctrl direction?**

The present evidence has already separated:

- incoming lag-0 magnitude enrichment; and
- additional lag-0 fixed-tap/channel-conditioned transfer enrichment.

The next bounded K0 analysis should therefore localize the lag-0 transfer across channels using the exact fixed identity:

`Q0_j = K_j,lag0 * ΔH_t,j`.

A suitable observational/algebraic follow-up would distinguish:

1. whether corr `ΔH_t` energy is preferentially placed on channels with large absolute lag-0 kernel weights;
2. whether signed/channelwise alignment produces the observed lag-0 transfer advantage;
3. whether that channel concentration is specific to corr relative to ctrl.

Such a stage must remain fixed to:

- layer 22;
- lag 0;
- common-330 `k=2`;
- the same frozen checkpoint/runtime/population;
- no learned geometry;
- no channel search after observing results;
- no intervention unless separately authorized later.

This report itself does not authorize that execution.
