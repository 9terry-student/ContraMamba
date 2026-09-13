# ContraMamba K0-RVG Layer-22 Current-Token RMSNorm Routing-Source Decomposition
## Validated Evidence Analysis Report Candidate

## 1. Status

**Evidence status:** validated local observational/algebraic execution.

**Static-design authority freeze:**

`176e47b068961d488f9a57d6181dcf3591cbb9a7`

**Implementation freeze / runtime HEAD:**

`ed005bdd7bcd84a13a9c3bf4738247d867a1b23c`

**Parent current-token in-projection routing evidence freeze:**

`8b494a72d48528c3bdb8985a1907766fded040e0`

**Parent implementation commit:**

`cd602f036b03e36e171837d9532541a530799954`

**Run directory:**

`reports/longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_ed005bd_v1`

This report interprets only the independently validated artifacts from that run.

It does not authorize training, evaluation, causal intervention, learned geometry, post-hoc channel search, or transition to K1.

---

## 2. Frozen scientific question

The bounded question was:

> Why does the layer-22 current-token mixer input `ΔX_t` arrive with the corr-specific direction that the fixed `W_H` routes toward downstream strong-kernel output rows and relatively away from weak-output rows?

More specifically:

> Is the validated strong/weak routing redistribution primarily associated with raw residual-stream difference `ΔR22`, branch-specific RMS scaling contrast `Δs22`, their vector interaction, or a genuinely mixed combination?

The immediately preceding frozen result had already established that fixed layer-22 `W_H` does not merely provide uniform corr amplification.

Instead:

- corr strong-partition transfer is larger for `330/330`;
- corr weak-partition transfer is larger for only `76/330`;
- the strong-energy mass is larger for corr in `328/330`;
- the fixed `W_H` all-channel routing identity closes exactly.

Therefore this stage moved exactly one boundary upstream, to the layer-22 pre-mixer RMSNorm.

---

## 3. Authenticated architecture boundary

The analyzed boundary is:

`R22 -> RMSNorm22 -> X22 -> Mixer22`.

Runtime RMSNorm is:

`variance = mean(R^2)`

`s = rsqrt(variance + eps)`

`X = gamma ⊙ (s R)`.

Frozen dimensions and constants:

- source layer:
  `22`;
- current token only;
- relative coordinate:
  `k=2`;
- common DDSSSSS cohort:
  `330`;
- residual / mixer-input width:
  `768`;
- hidden in-projection output width:
  `1536`;
- RMS epsilon:
  `1e-5`;
- downstream strong/weak partition:
  `240 / 1296 / 0`;
- lag-0 kernel RMS:
  `0.24383223809052498`.

The layer-22 RMSNorm forward output was required to match the already-frozen parent mixer-input `X` exactly in float32.

That runtime boundary identity passed.

---

## 4. Exact RMSNorm decomposition

For matched/swapped branches:

`R_m, R_s ∈ R^768`

with branch RMS scales:

`s_m = rsqrt(mean(R_m^2) + eps)`

`s_s = rsqrt(mean(R_s^2) + eps)`.

Define:

`ΔR = R_m - R_s`

`R_bar = (R_m + R_s)/2`

`Δs = s_m - s_s`

`s_bar = (s_m + s_s)/2`.

The two scientific RMSNorm factors are:

`Q_R = gamma ⊙ (s_bar ΔR)`

and:

`Q_s = gamma ⊙ (R_bar Δs)`.

The symmetric bilinear identity is:

`Q_R + Q_s
 = gamma ⊙ (s_m R_m - s_s R_s)`.

Runtime `X_m` and `X_s` are float32 outputs, so the numerical execution bridge is kept separately.

Define float64 operand replay:

`X_alg,m = gamma ⊙ (s_m R_m)`

`X_alg,s = gamma ⊙ (s_s R_s)`.

Define branch execution errors:

`epsilon_m = X_m - X_alg,m`

`epsilon_s = X_s - X_alg,s`.

Then:

`Q_eps = epsilon_m - epsilon_s`.

The observed difference obeys:

`ΔX_obs = Q_R + Q_s + Q_eps`.

`Q_eps` is a numerical bridge, not a scientific mechanism.

---

## 5. Propagation into the frozen routing operator

The fixed hidden in-projection is:

`W_H ∈ R^(1536 x 768)`.

Define:

`H_R = W_H Q_R`

`H_s = W_H Q_s`

`H_eps = W_H Q_eps`.

For output channel `j`, normalized by:

`D_X = ||ΔX_obs||^2`,

define:

`e_R,j = H_R,j^2 / D_X`

`e_s,j = H_s,j^2 / D_X`

`e_Rs,j = 2 H_R,j H_s,j / D_X`

and the complete numerical contribution:

`e_eps,j =
    (
        H_eps,j^2
        + 2 H_R,j H_eps,j
        + 2 H_s,j H_eps,j
    ) / D_X`.

Then:

`e_total,j
 = e_R,j
 + e_s,j
 + e_Rs,j
 + e_eps,j`.

For aligned corr/ctrl items:

`D_total,j = e_total,corr,j - e_total,ctrl,j`.

The source components are:

`D_R,j`

`D_s,j`

`D_Rs,j`

`D_eps,j`.

Exact identity:

`D_total,j
 = D_R,j
 + D_s,j
 + D_Rs,j
 + D_eps,j`.

The all/strong/weak partition sums of `D_total` reproduce the frozen parent current-token routing contrast.

---

## 6. Validated artifact identities

### Item metrics

Path:

`reports/longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_ed005bd_v1/layer22_current_token_rmsnorm_routing_source_item_metrics.jsonl`

SHA256:

`0dbb9a8177336d7846688f3ded418a6bd6acc8785f42de855be8b0906211990c`

Row count:

`330`

### Channel summary

Path:

`reports/longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_ed005bd_v1/layer22_current_token_rmsnorm_routing_source_channel_summary.jsonl`

SHA256:

`c5258286723280a7f1f478599e0eb76ff472b5aa7c6ec04a5aa0d06765be5c5f`

Row count:

`1536`

### Fixed kernel-rank cumulative profile

Path:

`reports/longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_ed005bd_v1/layer22_current_token_rmsnorm_routing_source_kernel_rank_cumulative.jsonl`

SHA256:

`2b5f7ef31e9c24a2124405c202d483e4cc81958e05edd99c6a949066e0f7476b`

Row count:

`1536`

### Summary

Path:

`reports/longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_ed005bd_v1/summary.json`

SHA256:

`1db1758dd1556f0210e72401135feb1d2af99ff8e13fc5f12b72b623c206c4f1`

### Execution manifest

Path:

`reports/longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_ed005bd_v1/execution_manifest.json`

SHA256:

`e692c17b236451adadc46f0ae73e335ed4a738dc88b8900cc916212fee5943e6`

The independent validator passed:

`PASS_LAYER22_CURRENT_TOKEN_RMSNORM_ROUTING_SOURCE_ARTIFACT_VALIDATION`.

---

## 7. Execution and provenance validity

Validated execution properties:

- runtime HEAD:
  `ed005bdd7bcd84a13a9c3bf4738247d867a1b23c`;
- runner SHA256:
  `7ef61ac389111b56c3c764bd9beae3301a54c2c7e99aea2a1b7e195c79e1a408`;
- pair-role count:
  `672`;
- model forward count:
  `1344`;
- common cohort:
  `330`;
- parent current-token routing reproduced:
  `True`;
- RMSNorm output / parent `X` exact float32 match:
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
- learned geometry:
  `False`;
- post-hoc layer/lag/channel/item/window search:
  `False`.

No `.partial` execution artifact remained.

---

## 8. Numerical validity

Validated maxima:

- RMS branch reconstruction relative residual:
  `0.0`;
- cancellation-sensitive difference-relative diagnostic:
  `9.051130491574617e-07`;
- error-difference identity absolute residual:
  `6.661338147750939e-16`;
- input-energy closure absolute residual:
  `1.3322676295501878e-15`;
- channel-component closure absolute residual:
  `4.6629367034256575e-15`;
- parent partition reproduction absolute residual:
  `8.659739592076221e-15`.

The branch RMS reconstruction is below the frozen `1e-6` gate.

The difference-relative diagnostic is not a blocking gate.

All exact scientific identities close far below the preregistered tolerances.

The numerical bridge is therefore explicitly quantified and negligible relative to the scientific component magnitudes.

---

## 9. Parent routing totals are exactly reproduced

The validated parent mean routing contrasts are reproduced:

### All channels

`Δtotal_all = +1.1589093253784963`

Frozen parent:

`+1.1589093253784961`

### Strong partition

`Δtotal_strong = +2.3029343955594896`

Frozen parent:

`+2.3029343955594896`

### Weak partition

`Δtotal_weak = -1.144025070180993`

Frozen parent:

`-1.144025070180993`

The parent paired ordering is also reproduced:

- `P_S,corr > P_S,ctrl`:
  `328/330`;
- `T_H,corr > T_H,ctrl`:
  `243/330`;
- `T_S,corr > T_S,ctrl`:
  `330/330`;
- `T_W,corr > T_W,ctrl`:
  `76/330`.

Therefore the new RMSNorm-source decomposition is a validated upstream factorization of the already-frozen current-token routing phenotype.

---

## 10. All-channel decomposition is interaction-dominated but strongly cancellation-rich

Mean all-channel components:

- total:
  `+1.1589093253784963`;
- residual-source:
  `-1.1700471099810297`;
- RMS-scale-source:
  `-1.2324588353859254`;
- residual x scale interaction:
  `+3.561415354774301`;
- numerical bridge:
  `-8.402884864338166e-08`.

Exact closure:

`-1.1700471099810297
 -1.2324588353859254
 +3.561415354774301
 -0.00000008402884864338166
 = +1.1589093253784963`

up to floating-point summation.

Absolute scientific component mass, ignoring the negligible numerical bridge, is dominated by the interaction term.

Approximate absolute component shares including the numerical bridge are:

- residual:
  `19.62%`;
- scale:
  `20.67%`;
- interaction:
  `59.72%`;
- numerical bridge:
  `~1.4e-6%`.

The signed component-to-net ratios are approximately:

- residual:
  `-1.010 x net`;
- scale:
  `-1.063 x net`;
- interaction:
  `+3.073 x net`.

Therefore the positive all-channel net is not a broad residual-source gain.

It is the small survivor of large opposing RMSNorm terms.

At the all-channel level, the largest absolute scientific component is the residual x scale interaction.

---

## 11. Strong-partition routing is residual-led with interaction support and scale opposition

Mean strong-partition components:

- total:
  `+2.3029343955594896`;
- residual-source:
  `+1.7958155601281673`;
- RMS-scale-source:
  `-0.4042821980086967`;
- residual x scale interaction:
  `+0.9114010708261777`;
- numerical bridge:
  `-3.738615886503347e-08`.

The sign structure is important:

- residual source:
  positive;
- interaction:
  positive;
- scale source:
  negative.

Thus the raw residual difference already carries a positive strong-routing contrast.

RMS scaling alone opposes this strong advantage.

Residual x scale interaction then restores and amplifies part of the positive strong routing.

Approximate absolute component shares:

- residual:
  `57.72%`;
- scale:
  `12.99%`;
- interaction:
  `29.29%`;
- numerical bridge:
  negligible.

Signed component-to-net ratios:

- residual:
  `+0.780 x net`;
- scale:
  `-0.176 x net`;
- interaction:
  `+0.396 x net`.

The largest absolute scientific component in the strong partition is therefore the raw-residual term.

This partition is **residual-led**, but not residual-only.

---

## 12. Weak-partition suppression is also residual-led, but strongly counteracted by interaction

Mean weak-partition components:

- total:
  `-1.144025070180993`;
- residual-source:
  `-2.965862670109197`;
- RMS-scale-source:
  `-0.8281766373772286`;
- residual x scale interaction:
  `+2.6500142839481224`;
- numerical bridge:
  `-4.664268977834819e-08`.

Again the sign structure is informative.

The raw residual source already strongly favors the validated weak-partition direction:

`corr < ctrl`.

The RMS-scale term has the same negative sign and therefore adds further weak suppression.

However the residual x scale interaction is large and positive.

It cancels most of the combined negative residual and scale terms.

Approximate absolute component shares:

- residual:
  `46.02%`;
- scale:
  `12.85%`;
- interaction:
  `41.12%`;
- numerical bridge:
  negligible.

Signed component-to-net ratios:

- residual:
  `+2.592 x net`;
- scale:
  `+0.724 x net`;
- interaction:
  `-2.316 x net`.

The largest absolute scientific component in the weak partition is still the raw-residual term, but the interaction term is nearly comparable in magnitude.

Therefore the weak routing phenotype is highly cancellation-rich.

---

## 13. The distinctive strong-positive / weak-negative partition signature is already present in the raw residual stream

The most important localization result is the sign pattern of the residual-source component.

For `D_R`:

- strong:
  `+1.7958155601281673`;
- weak:
  `-2.965862670109197`.

This is the same qualitative partition signature as the frozen parent result:

- strong positive;
- weak negative.

Therefore the upstream raw residual difference `ΔR22` already carries the **directional partition bias** that ultimately appears at the normalized mixer input and then through fixed `W_H`.

RMSNorm is not required to create the sign of this partition redistribution from scratch.

This is a stronger localization than merely observing that `||ΔR||` is nonzero.

It uses the exact propagation of the residual-source term through the already-frozen `W_H` and strong/weak partition.

---

## 14. RMSNorm materially reshapes the residual-carried pattern

Although the raw residual source already has the correct strong-positive / weak-negative signature, the final parent routing contrast is not equal to the residual term.

RMSNorm materially reshapes it through two deterministic effects:

1. branch-specific RMS-scale contrast;
2. residual x scale interaction.

The scale term is:

- strong:
  negative;
- weak:
  negative.

The interaction term is:

- strong:
  positive;
- weak:
  positive.

Therefore RMSNorm's two scientific terms do not act as a simple uniform rescaling.

They have different consequences for the two downstream partitions.

In the strong partition:

- scale opposes the residual advantage;
- interaction supports the residual advantage.

In the weak partition:

- scale reinforces residual weak suppression;
- interaction strongly cancels that suppression.

This is a genuinely mixed normalization geometry.

---

## 15. Why the all-channel result alone would be misleading

At the all-channel level:

- residual:
  negative;
- scale:
  negative;
- interaction:
  strongly positive;
- net:
  mildly positive.

If only the all-channel decomposition were examined, one might conclude that RMSNorm interaction creates the phenotype.

That would be incomplete.

The scientific parent phenotype is not merely a positive all-channel norm change.

It is the **strong-positive / weak-negative redistribution**.

For that partition-resolved phenotype:

- the residual source already has the correct directional signature;
- RMSNorm scale and interaction reshape its magnitude through cancellation.

Therefore the correct mechanistic description must remain partition-resolved.

---

## 16. Outcome classification

The preregistered outcome classes included:

- residual-vector transport localized;
- RMS-scale contrast localized;
- residual x scale interaction localized;
- mixed.

The validated result is:

**Outcome D: mixed.**

However it is a structured mixed outcome:

**residual-led partition signature with substantial RMSNorm interaction/cancellation.**

It is not Outcome A in the strict sense because the interaction term is not materially small:

- strong interaction is about half the residual magnitude;
- weak interaction is almost as large as the residual magnitude;
- all-channel interaction is the largest absolute scientific component.

It is not Outcome B because RMS-scale alone is not the largest component in any partition.

It is not Outcome C because interaction does not by itself carry the distinctive strong-positive / weak-negative partition signature:

- interaction is positive in both strong and weak partitions.

The raw residual term is the only individual scientific component among the three whose signs already match both sides of the parent redistribution simultaneously:

- strong positive;
- weak negative.

---

## 17. Numerical bridge is scientifically negligible

Mean numerical bridge components:

- all:
  `-8.402884864338166e-08`;
- strong:
  `-3.738615886503347e-08`;
- weak:
  `-4.664268977834819e-08`.

These are many orders of magnitude smaller than the residual, scale, and interaction terms.

Thus the scientific mixed result is not an artifact of the float32 RMSNorm execution bridge.

The explicit `Q_eps` design successfully separates runtime numerical effects from the scientific decomposition.

---

## 18. Integrated K0 mechanism through this boundary

The validated evidence now supports the following hierarchy.

### 18.1 Raw residual stream

At layer 22, current-token `ΔR22` already contains a directional difference which, after the fixed mean RMS scale, fixed gamma, and fixed `W_H`, yields:

- positive strong-partition routing contrast;
- negative weak-partition routing contrast.

This is the upstream carrier of the qualitative parent partition signature.

### 18.2 Branch-specific RMS scaling

The matched/swapped branch norms generate different reciprocal RMS scales.

The corresponding `Q_s` term is negative in both strong and weak partition contrasts.

It therefore:

- reduces the strong advantage;
- adds weak suppression.

### 18.3 Residual x scale interaction

The symmetric interaction is positive in both partitions.

It therefore:

- restores/supports strong routing;
- strongly cancels weak suppression.

### 18.4 Final normalized mixer input

The final `ΔX22` routing phenotype is the exact combination of these terms.

It cannot be described as a pure residual carry-through or a pure RMSNorm-generated effect.

### 18.5 Fixed hidden in-projection and lag-0 kernel

The now-frozen downstream chain remains:

`ΔR22`
→ layer-22 RMSNorm mixed transformation
→ `ΔX22`
→ fixed `W_H` directional routing
→ corr-enriched strong-kernel channel exposure
→ selective lag-0 convolution transfer
→ downstream U/write/state separation.

This remains an observational/algebraic localization chain, not a causal intervention chain.

---

## 19. Frozen scientific conclusion

The validated K0-RVG layer-22 current-token RMSNorm routing-source result is:

**At common-330 `k=2`, the corr-specific strong-positive / weak-negative current-token `W_H` routing phenotype is already qualitatively present in the raw layer-22 residual-stream difference `ΔR22`. Propagating the residual-source term alone through the fixed RMSNorm mean scale, gamma, and frozen `W_H` gives a positive strong-partition contrast (`+1.7958`) and a negative weak-partition contrast (`-2.9659`), matching the directional signature of the parent routing result. However, RMSNorm materially reshapes this inherited pattern: the branch-specific scale term is negative in both partitions (`-0.4043` strong, `-0.8282` weak), while the residual x scale interaction is positive in both (`+0.9114` strong, `+2.6500` weak). These terms substantially cancel one another, especially in the weak partition and at the all-channel level. The final outcome is therefore mixed rather than residual-only, but the partition signature is residual-led. The explicit numerical bridge is negligible, and all source-component identities and parent reproduction gates close at numerical precision.**

Equivalently:

**source of qualitative strong/weak sign structure:** already present in `ΔR22`;

**RMSNorm role:** deterministic but substantial reshaping/cancellation of that residual-carried pattern;

**strong partition:** residual-led positive routing, interaction support, scale opposition;

**weak partition:** residual-led negative routing, scale reinforcement, large interaction cancellation;

**all channels:** interaction-dominated absolute contribution with strong cancellation;

**numerical bridge:** negligible;

**causal status:** observational/algebraic only.

---

## 20. Claims not supported

This evidence does not establish that:

- RMSNorm is causally necessary for the parent routing phenotype;
- removing RMSNorm would preserve performance;
- the raw residual source alone is sufficient for downstream behavior;
- the scale term is semantically meaningful by itself;
- the interaction term corresponds to an independent learned module;
- individual residual dimensions are causally important;
- individual hidden channels are causally necessary or sufficient;
- another layer, coordinate, seed, task, or model has the same decomposition;
- K1 is established.

No intervention was performed.

No channel subset was learned or selected.

No PCA/SVD/probe was used.

---

## 21. Scientific consequence for the next K0 boundary

The layer-22 RMSNorm boundary is now algebraically closed for this question.

The scale term and interaction are not independent upstream sources.

They are deterministic functions of the same two branch residual vectors:

`R_m`

and:

`R_s`.

Specifically:

- `ΔR`;
- `R_bar`;
- `s_m`;
- `s_s`;
- `Δs`;
- `s_bar`

are all determined by the branch residual-stream states and fixed RMSNorm parameters.

Therefore there is no additional free normalization-side variable to localize after this exact decomposition.

The next unresolved K0 question should move upstream to:

> How is the layer-22 current-token raw residual-stream difference `ΔR22` constructed such that it already carries the strong-positive / weak-negative routing signature before RMSNorm?

The next stage must not repeat:

- fixed `W_H` row-alignment decomposition;
- layer-22 RMSNorm source decomposition;
- lag-0 channel exposure decomposition;
- historical tokenizer provenance;
- K1 execution.

The next static design should localize the construction of current-token `R22` from the preceding block/residual path while preserving the already-frozen common-330 `k=2` population.

---

## 22. Final stage disposition

### Code correctness

PASS.

### Execution success

PASS.

`PASS_LAYER22_CURRENT_TOKEN_RMSNORM_ROUTING_SOURCE_EXECUTION`

### Artifact / provenance validity

PASS.

`PASS_LAYER22_CURRENT_TOKEN_RMSNORM_ROUTING_SOURCE_ARTIFACT_VALIDATION`

### Scientific conclusion

Validated observational/algebraic result:

**mixed RMSNorm transformation with a residual-led strong-positive / weak-negative partition signature and substantial scale/interaction cancellation.**

The next authorized scientific boundary is upstream construction of layer-22 current-token `R22`.

K1 remains out of scope.
