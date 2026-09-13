# ContraMamba K0-RVG Layer-22 Current-Token Residual-Construction Routing-Source Decomposition
## Static Design Candidate

## 1. Status

**Phase:** new K0 scientific static design.

**Immediate parent evidence freeze:**

`5914c28a26ee18442dcfa2b5a4d99901fad3334f`

**Immediate parent implementation / runtime HEAD:**

`ed005bdd7bcd84a13a9c3bf4738247d867a1b23c`

**Immediate parent validated report:**

`reports/longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_validated_evidence_analysis_report_candidate.md`

**Immediate parent report SHA256:**

`dfa3013e701f3cb5561dd8d008624517333b9bb1ea90ae9524bfa21cbbd8d48b`

This document opens exactly one new K0 scientific question upstream of the validated layer-22 RMSNorm routing-source decomposition.

It does not authorize training, task evaluation, causal intervention, learned geometry, tokenizer work, post-hoc channel search, or K1.

A bounded static/runtime preflight after implementation freeze does not require a separate authority document.

---

## 2. Frozen parent result

The immediate parent stage established at common-330, current-token, `k=2`:

1. the fixed layer-22 hidden in-projection routing phenotype is:
   - strong partition:
     corr-enriched;
   - weak partition:
     relatively corr-suppressed;

2. the layer-22 pre-mixer RMSNorm decomposition is exact:

   `ΔX22 = Q_R22 + Q_s22 + Q_eps22`;

3. the parent residual-source component is:

   `Q_R22 = gamma22 ⊙ (s_bar22 ΔR22)`;

4. after propagation through the already-frozen layer-22 hidden in-projection `W_H22`, the mean residual-source routing contrasts are:

   - all channels:
     `-1.1700471099810297`;
   - strong partition:
     `+1.7958155601281673`;
   - weak partition:
     `-2.965862670109197`;

5. therefore the qualitative parent partition signature:

   **strong positive / weak negative**

   is already present in the raw layer-22 residual-stream difference `ΔR22`;

6. RMSNorm materially reshapes this inherited signature, but does not need to create its sign structure from scratch;

7. the explicit RMSNorm numerical bridge is negligible;

8. all parent identities and parent routing totals reproduce at numerical precision.

The parent stage is closed.

This stage must not reopen:

- fixed `W_H22` row-alignment geometry;
- RMSNorm `Q_R/Q_s/Q_eps` factorization;
- lag-0 strong/weak channel partition;
- historical tokenizer provenance.

---

## 3. Exact upstream block boundary

The authenticated Mamba block residual boundary is:

`R_l -> RMSNorm_l -> Mixer_l -> Y_l`

followed by:

`R_(l+1) = R_l + Y_l`.

A frozen K0 precedent directly authenticated this equation at layer 22:

`R23 = R22 + Y22`.

The same frozen block-forward implementation and block class apply one layer upstream.

The new target boundary is therefore:

`R21 -> RMSNorm21 -> Mixer21 -> Y21`

and:

`R22 = R21 + Y21`.

The scientific target remains the **current-token layer-22 residual difference** that feeds RMSNorm22.

No other residual layer is searched.

---

## 4. Scientific question

The bounded question is:

> How is the validated layer-22 current-token raw residual-source routing signature constructed at the immediately preceding residual-addition boundary?

More specifically:

> Is the strong-positive / weak-negative residual-source routing signature inherited primarily from the incoming layer-21 residual difference `ΔR21`, produced primarily by the layer-21 mixer update difference `ΔY21`, created primarily by their downstream vector interaction after the fixed layer-22 residual-source map, or produced by a genuinely mixed combination?

The question is **not** whether `||ΔR22||` is large.

The target is the already-frozen residual-source routing contrast after:

`ΔR22`
→ fixed parent residual-source scale/gamma
→ fixed `W_H22`
→ frozen strong/weak partition.

---

## 5. Population and fixed scope

Scientific population:

- current token only;
- relative coordinate:
  `k=2`;
- common DDSSSSS cohort:
  `330`;
- roles:
  corr and ctrl;
- same matched/swapped branch definitions;
- same checkpoint;
- same handoff;
- same equal-length prefix execution protocol;
- same layer-22 RMSNorm parameters;
- same observed parent branch RMS scales;
- same fixed `W_H22`;
- same downstream lag-0 strong/weak partition:
  `240 / 1296 / 0`.

The runner may execute the frozen parent `672` pair-role plan.

Expected full model forwards:

`1344`.

Scientific summaries are restricted to common-330 `k=2`.

No other layer, coordinate, lag, item subset, seed, or learned subspace may be searched.

---

## 6. Direct runtime operands

For one aligned item and role, capture matched/swapped current-token branch values:

`R21_m, R21_s ∈ R^768`

`Y21_m, Y21_s ∈ R^768`

`R22_m, R22_s ∈ R^768`.

Definitions:

`ΔR21 = R21_m - R21_s`

`ΔY21 = Y21_m - Y21_s`

`ΔR22_obs = R22_m - R22_s`.

The runtime branch equation is:

`R22_b = fl32(R21_b + Y21_b)`

for:

`b ∈ {m,s}`.

The parent layer-22 RMSNorm capture already observes `R22_b` as the input to RMSNorm22.

The new layer-21 block post-hook must match that parent RMSNorm22 input exactly in float32.

This equality is a blocking boundary-identity gate.

---

## 7. Explicit residual-addition numerical bridge

Do not silently treat float32 branch addition as exact float64 addition.

Define float64 branch replay:

`R22_alg,m = R21_m + Y21_m`

`R22_alg,s = R21_s + Y21_s`.

Define branch numerical errors:

`epsilon_add,m = R22_m - R22_alg,m`

`epsilon_add,s = R22_s - R22_alg,s`.

Define the branch-difference numerical bridge:

`Q_add_eps = epsilon_add,m - epsilon_add,s`.

Then:

`ΔR22_obs
 = ΔR21
 + ΔY21
 + Q_add_eps`.

`Q_add_eps` is a numerical execution bridge.

It is not a scientific mechanism.

The branch-level residual-addition reconstruction tolerance remains the frozen precedent value:

`1e-7`.

The cancellation-sensitive relative error of:

`ΔR21 + ΔY21`

versus:

`ΔR22_obs`

is diagnostic-only.

Require the explicit difference-error identity to close at:

`5e-12`

maximum absolute residual.

---

## 8. Link to the frozen parent residual-source map

The parent RMSNorm stage defined:

`Q_R22
 = gamma22 ⊙ (s_bar22 ΔR22_obs)`.

For a fixed item and role, define the observed parent scalar:

`s_bar22 = (s22_m + s22_s)/2`.

The fixed parent residual-source linear map is:

`L_R22(v)
 = gamma22 ⊙ (s_bar22 v)`.

This map is fixed **conditional on the already-observed parent branch values**.

Do not recompute or substitute a different RMS scale.

The parent `s_bar22` used here must exactly reproduce the parent residual-source `Q_R22`.

---

## 9. Exact residual-construction source decomposition in RMSNorm input space

Apply the fixed parent residual-source map to each additive source.

Define:

`Q_in
 = gamma22 ⊙ (s_bar22 ΔR21)`.

Define:

`Q_update
 = gamma22 ⊙ (s_bar22 ΔY21)`.

Define:

`Q_add_eps_scaled
 = gamma22 ⊙ (s_bar22 Q_add_eps)`.

Then:

`Q_R22
 = Q_in
 + Q_update
 + Q_add_eps_scaled`.

This is an exact linear decomposition of the already-frozen parent residual-source term.

Interpretation:

- `Q_in`:
  contribution inherited from the incoming layer-21 residual stream;
- `Q_update`:
  contribution added by the current layer-21 mixer update;
- `Q_add_eps_scaled`:
  numerical bridge from runtime float32 residual addition.

No new fitted parameter is introduced.

---

## 10. Blocking parent `Q_R22` reproduction

The new runner must independently reconstruct the immediate parent residual-source vector from:

`R22_m`

`R22_s`

`gamma22`

`s22_m`

`s22_s`.

Define:

`Q_R22_reconstructed
 = gamma22 ⊙ (s_bar22 (R22_m - R22_s))`.

Require this reconstruction to reproduce the immediate parent scientific operand used in the RMSNorm routing-source stage.

At minimum, reproduce the persisted parent scalar diagnostics:

- `delta_r_l2_corr`;
- `delta_r_l2_ctrl`;
- `q_r_l2_corr`;
- `q_r_l2_ctrl`;
- parent residual-source all/strong/weak item contrasts.

The exact source decomposition must not be interpreted if those bridges fail.

---

## 11. Propagation through fixed `W_H22`

The fixed layer-22 hidden in-projection is:

`W_H22 ∈ R^(1536×768)`.

Define:

`H_in = W_H22 Q_in`

`H_update = W_H22 Q_update`

`H_add_eps = W_H22 Q_add_eps_scaled`.

Define the parent residual-source hidden vector:

`H_R22 = W_H22 Q_R22`.

Require:

`H_R22
 = H_in
 + H_update
 + H_add_eps`

to close at numerical precision.

This stage does not reopen the internal row-gain/alignment decomposition of `W_H22`.

`W_H22` is only a fixed readout operator.

---

## 12. Parent normalization denominator remains fixed

The immediate parent residual-source channel energy was normalized by:

`D_X = ||ΔX22_obs||²`.

This denominator must remain exactly the same.

Do not renormalize source terms by:

- `||ΔR21||`;
- `||ΔY21||`;
- `||Q_in||`;
- `||Q_update||`;
- `||Q_R22||`.

Using the same observed parent denominator is necessary to decompose the already-frozen parent residual-source contribution without changing its meaning.

---

## 13. Exact channelwise decomposition of the parent residual-source energy

For hidden output channel `j`, let:

`h_in,j = H_in[j]`

`h_update,j = H_update[j]`

`h_eps,j = H_add_eps[j]`.

Define:

`e_in,j = h_in,j² / D_X`.

Define:

`e_update,j = h_update,j² / D_X`.

Define additive-source interaction:

`e_cross,j
 = 2 h_in,j h_update,j / D_X`.

Define complete numerical bridge:

`e_eps,j
 = (
     h_eps,j²
     + 2 h_in,j h_eps,j
     + 2 h_update,j h_eps,j
   ) / D_X`.

Then:

`e_R22,j
 = e_in,j
 + e_update,j
 + e_cross,j
 + e_eps,j`.

And:

`e_R22,j
 = H_R22,j² / D_X`.

This is the exact channelwise decomposition of the parent residual-source energy.

---

## 14. Paired corr/ctrl decomposition of the frozen parent residual-source contrast

For aligned corr/ctrl items define:

`D_R22,j
 = e_R22,corr,j
 - e_R22,ctrl,j`.

Define source contrasts:

`D_in,j
 = e_in,corr,j
 - e_in,ctrl,j`.

`D_update,j
 = e_update,corr,j
 - e_update,ctrl,j`.

`D_cross,j
 = e_cross,corr,j
 - e_cross,ctrl,j`.

`D_eps,j
 = e_eps,corr,j
 - e_eps,ctrl,j`.

Require:

`D_R22,j
 = D_in,j
 + D_update,j
 + D_cross,j
 + D_eps,j`.

This is the primary scientific identity.

---

## 15. All/strong/weak partition closure

Use the exact frozen downstream partition:

- all:
  `1536`;
- strong:
  `240`;
- weak:
  `1296`;
- equal:
  `0`.

For each partition:

`P ∈ {all,strong,weak}`

define:

`Delta_in,P = Σ_(j∈P) D_in,j`

`Delta_update,P = Σ_(j∈P) D_update,j`

`Delta_cross,P = Σ_(j∈P) D_cross,j`

`Delta_eps,P = Σ_(j∈P) D_eps,j`.

Require:

`Delta_R22,P
 = Delta_in,P
 + Delta_update,P
 + Delta_cross,P
 + Delta_eps,P`.

The total must reproduce the frozen immediate-parent residual-source contrasts.

Frozen mean targets:

- all:
  `-1.1700471099810297`;
- strong:
  `+1.7958155601281673`;
- weak:
  `-2.965862670109197`.

---

## 16. Why this decomposition directly answers the next K0 question

The parent result established that `ΔR22` already carries the qualitative strong-positive / weak-negative signature.

But `R22` itself is not primitive.

It is the output of the immediately preceding residual addition:

`R22 = R21 + Y21`.

Therefore the narrowest next scientific localization is to ask whether the parent residual-source signature is:

1. already present in incoming `ΔR21`;
2. newly added by `ΔY21`;
3. created mainly by interaction between the two after fixed downstream mapping;
4. genuinely mixed.

This stage answers that exact question without searching deeper layers.

---

## 17. Direct branch-capture policy

Use read-only hooks on **layer 21**:

- layer21 block forward pre-hook:
  capture `R21`;
- layer21 mixer forward hook:
  capture `Y21`;
- layer21 block forward hook:
  capture `R22`.

Simultaneously preserve the immediate parent layer22 RMSNorm pre-hook capture.

At the target token require exact float32 equality between:

- layer21 block output `R22`;
- layer22 RMSNorm input `R22`.

No tensor is modified.

No forward hook may change output values.

---

## 18. Architecture authentication

Before runtime interpretation, authenticate:

- backbone layer count:
  `24`;
- source block index:
  `21`;
- target residual block index:
  `22`;
- layer21 and layer22 block class identity;
- source block forward SHA256:
  `0f808b4d539a496e1681c81d39799d7072fa5c6da381a07cfa19f35fe825b3e4`;
- backbone forward SHA256:
  `3f332bd50e6ea4ff64468c8d748672f90ffc43912d22c9d25d02b3e87d1661a6`;
- Mamba source SHA256:
  `23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`;
- layer21 residual width:
  `768`;
- layer21 mixer output width:
  `768`;
- layer22 residual width:
  `768`;
- layer22 `gamma` shape:
  `(768,)`;
- layer22 `W_H` shape:
  `(1536,768)`;
- frozen strong/weak partition:
  `240 / 1296 / 0`.

No architecture substitution is allowed.

---

## 19. Residual-addition instrumentation precedent

Use as instrumentation precedent:

`scripts/longterm_k0_rvg_layer22_residual_addition_audit.py`.

That runner authenticated at the same frozen Mamba source:

`R23 = R22 + Y22`.

Its generic hook semantics are applicable one layer upstream.

Its scientific result for layer22→23 must **not** be imported as evidence for layer21→22.

Only the source-boundary instrumentation pattern may be reused.

---

## 20. Immediate parent provenance

Authenticate the evidence frozen at:

`5914c28a26ee18442dcfa2b5a4d99901fad3334f`.

Immediate parent run directory:

`reports/longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_ed005bd_v1`

Required exact output hashes:

- item:
  `0dbb9a8177336d7846688f3ded418a6bd6acc8785f42de855be8b0906211990c`;
- channel:
  `c5258286723280a7f1f478599e0eb76ff472b5aa7c6ec04a5aa0d06765be5c5f`;
- cumulative:
  `2b5f7ef31e9c24a2124405c202d483e4cc81958e05edd99c6a949066e0f7476b`;
- summary:
  `1db1758dd1556f0210e72401135feb1d2af99ff8e13fc5f12b72b623c206c4f1`;
- manifest:
  `e692c17b236451adadc46f0ae73e335ed4a738dc88b8900cc916212fee5943e6`.

Immediate parent runner SHA256:

`7ef61ac389111b56c3c764bd9beae3301a54c2c7e99aea2a1b7e195c79e1a408`.

Immediate parent validated report SHA256:

`dfa3013e701f3cb5561dd8d008624517333b9bb1ea90ae9524bfa21cbbd8d48b`.

Any mismatch blocks interpretation.

---

## 21. Parent reproduction gates

For every common-330 item, reproduce from the new capture:

### Layer-22 residual parent metrics

For corr and ctrl:

- `delta_r_l2`;
- observed layer22 branch RMS scale;
- `q_r_l2`.

### Frozen residual-source routing totals

For each item:

- residual-source all-channel contrast;
- residual-source strong contrast;
- residual-source weak contrast.

### Population means

Reproduce:

- all:
  `-1.1700471099810297`;
- strong:
  `+1.7958155601281673`;
- weak:
  `-2.965862670109197`.

No new scientific conclusion is accepted if any parent reproduction gate fails.

---

## 22. Input-space residual-construction diagnostics

For corr and ctrl separately report:

- `||ΔR21||`;
- `||ΔY21||`;
- `||ΔR22||`;
- `||Q_add_eps||`;
- layer21 branch-add reconstruction residual;
- cancellation-sensitive difference-relative diagnostic.

Also report exact residual-space squared-norm decomposition:

`||ΔR22_alg||²
 = ||ΔR21||²
 + ||ΔY21||²
 + 2<ΔR21,ΔY21>`.

Keep this as an upstream geometry diagnostic.

Do not use it alone to decide source attribution.

The primary scientific target remains the frozen residual-source routing contrast.

---

## 23. Item-level persisted metrics

Persist one scalar row per common-330 item.

Required categories:

### Parent bridges

- `delta_r22_l2_corr`;
- `delta_r22_l2_ctrl`;
- `q_r22_l2_corr`;
- `q_r22_l2_ctrl`;
- parent residual-source all/strong/weak totals.

### Residual-construction branch metrics

For corr and ctrl:

- `delta_r21_l2`;
- `delta_y21_l2`;
- `delta_r22_l2`;
- `q_add_eps_l2`;
- branch residual-add reconstruction residual;
- difference-relative diagnostic;
- residual-space cross term;
- residual-space addition ratio.

### Fixed parent residual-source scale

For corr and ctrl:

- `s22_m`;
- `s22_s`;
- `s_bar22`.

These are observed parent values, not new fitted quantities.

### Residual-source routing components

For all/strong/weak:

- `Delta_in`;
- `Delta_update`;
- `Delta_cross`;
- `Delta_eps`;
- `Delta_total`;
- exact closure residual;
- exact parent reproduction residual.

No raw vector is persisted.

---

## 24. Channel aggregate artifact

Persist one row for each of all `1536` frozen output channels.

Required fields:

- channel index;
- frozen lag0 kernel magnitude rank;
- frozen strong/weak partition;
- fixed `W_H22` row-gain squared;
- mean/median `D_R22`;
- mean/median `D_in`;
- mean/median `D_update`;
- mean/median `D_cross`;
- mean/median `D_eps`;
- positive/negative/zero paired counts for each component.

Do not rank channels by new component values.

No new channel subset may be selected.

---

## 25. Fixed kernel-rank cumulative artifact

Use the already-frozen descending lag0 squared-kernel order.

For rank `1..1536`, persist cumulative means for:

- total residual-source contrast;
- incoming residual source;
- layer21 update source;
- incoming×update interaction;
- numerical bridge.

Do not choose a new cutoff.

Do not optimize rank thresholds.

---

## 26. Threshold-free component attribution

For each partition:

`P ∈ {all,strong,weak}`

let the aggregate mean signed components be:

`M_in,P`

`M_update,P`

`M_cross,P`

`M_eps,P`.

Require:

`M_R22,P
 = M_in,P
 + M_update,P
 + M_cross,P
 + M_eps,P`.

Define:

`A_P
 = |M_in,P|
 + |M_update,P|
 + |M_cross,P|
 + |M_eps,P|`.

Report absolute shares when `A_P > 0`.

Also report signed component-to-parent-net ratios.

Ratios may exceed `1` in magnitude under cancellation.

Do not interpret them as probabilities.

No arbitrary dominance threshold is introduced.

---

## 27. Scientific outcome classes

### Outcome A: inherited incoming-residual signature

The incoming source `D_in` carries the main strong-positive / weak-negative signature.

Interpretation:

the layer-22 residual-source pattern is already present before the layer21 mixer update.

This would move the next K0 question one block further upstream.

### Outcome B: layer21 mixer-update source

`D_update` carries the main strong-positive / weak-negative signature.

Interpretation:

the layer21 mixer update is the immediate producer of the parent residual-source routing pattern.

The next K0 question would then enter layer21 mixer internals.

### Outcome C: additive interaction source

`D_cross` is the dominant source of the partition signature.

Interpretation:

neither incoming residual nor layer21 update alone carries the parent routing structure; their downstream vector interaction under the fixed layer22 residual-source map is central.

### Outcome D: mixed

No single component captures the parent signature.

Report the exact partition-resolved mixture and cancellation.

### Numerical-bridge condition

If `D_eps` is non-negligible relative to scientific components, do not promote a source claim.

Do not change tolerances after observing results.

---

## 28. Sign-structure criterion

The immediate parent qualitative signature is:

- strong:
  positive;
- weak:
  negative.

A component may be described as already carrying the qualitative partition signature only if its aggregate mean signs are:

- strong `> 0`;
- weak `< 0`.

This criterion is descriptive, not a significance threshold.

Do not require every item or every channel to share those signs.

---

## 29. Falsification and blocking conditions

Block scientific interpretation if any occurs:

1. layer21 block output `R22` is not exact float32-equal to layer22 RMSNorm input `R22`;
2. branch residual-add reconstruction exceeds `1e-7`;
3. explicit residual-add error-difference identity exceeds `5e-12`;
4. parent `ΔR22` metrics fail reproduction;
5. parent `Q_R22` metrics fail reproduction;
6. parent residual-source all/strong/weak totals fail reproduction;
7. source-component channel identity fails;
8. partition source-component closure fails;
9. strong/weak partition changes;
10. frozen kernel-rank order changes;
11. raw per-item vectors are persisted;
12. training, task evaluation, intervention, tokenizer, logits, PCA/SVD, learned probes, or post-hoc search are invoked.

An unexpected scientific source pattern is not a validation failure.

---

## 30. Tolerances fixed before runtime

Use:

- branch residual-add reconstruction relative tolerance:
  `1e-7`;
- parent scalar relative tolerance:
  `1e-13`;
- parent scalar absolute tolerance:
  `1e-13`;
- vector/error identity maximum absolute tolerance:
  `5e-12`;
- squared-energy relative closure tolerance:
  `1e-12`;
- kernel RMS relative tolerance:
  `1e-13`;
- kernel RMS absolute tolerance:
  `1e-13`.

Difference-relative residual-add reconstruction on the branch difference is diagnostic-only.

No tolerance weakening is allowed after preflight or execution.

---

## 31. Execution plan

After implementation freeze and bounded runtime preflight PASS:

- execute the same frozen `672` pair-role plan;
- matched and swapped branch forward per pair-role;
- expected full forward count:
  `1344`;
- scientific population:
  common-330, current-token `k=2`.

GPU is not required.

Local CPU is preferred.

Kaggle is unnecessary unless local execution becomes technically impossible.

---

## 32. Artifact policy

A later authorized full execution may persist:

1. `330` item scalar rows;
2. `1536` channel aggregate rows;
3. `1536` fixed-kernel-rank cumulative rows;
4. summary JSON;
5. execution manifest.

Do not persist:

- `R21_m/s`;
- `Y21_m/s`;
- `R22_m/s`;
- `ΔR21`;
- `ΔY21`;
- `Q_in`;
- `Q_update`;
- `Q_add_eps_scaled`;
- `H_in`;
- `H_update`;
- `H_add_eps`;
- per-item channel vectors;
- logits;
- task-head outputs.

Temporary vectors exist only in memory for aggregation.

---

## 33. Provenance requirements

Execution manifest must record:

- runtime Git HEAD;
- branch;
- this static-design freeze commit and SHA256;
- immediate parent evidence freeze:
  `5914c28a26ee18442dcfa2b5a4d99901fad3334f`;
- immediate parent implementation:
  `ed005bdd7bcd84a13a9c3bf4738247d867a1b23c`;
- immediate parent runner SHA;
- immediate parent artifact hashes;
- immediate parent report SHA;
- residual-addition instrumentation precedent identity;
- frozen Mamba source SHA;
- block-forward SHA;
- backbone-forward SHA;
- handoff ZIP SHA;
- checkpoint SHA;
- encoder digests;
- forward count;
- layer/coordinate/cohort;
- all tolerances;
- persistence flags;
- training/intervention/search flags;
- output hashes.

Any provenance mismatch blocks interpretation.

---

## 34. Interpretation boundary

Passing this stage may support:

> the validated layer-22 residual-source strong-positive / weak-negative routing signature is algebraically localized to the incoming layer21 residual, the layer21 mixer update, their interaction, or a fixed mixture thereof at the exact `R22 = R21 + Y21` residual-addition boundary.

It may not support:

- causal necessity;
- causal sufficiency;
- semantic meaning of individual dimensions;
- performance claims;
- intervention claims;
- generalization to another layer, seed, task, or model;
- K1.

This remains observational/algebraic localization.

---

## 35. What this stage intentionally does not ask

Do not ask:

- which earlier layer is best;
- which channel subset explains the result;
- whether layer21 should be modified;
- whether residual scaling should be tuned;
- whether another normalization is better;
- whether intervention improves accuracy;
- whether a learned projection separates roles.

Do not open deeper layer21 mixer internals until this residual-construction boundary is validated.

---

## 36. Next-step rule after validated evidence

If incoming `ΔR21` already carries the parent sign structure, move the next K0 boundary one block upstream.

If layer21 mixer update `ΔY21` carries the parent sign structure, move into the narrowest exact layer21 mixer-output source decomposition.

If additive interaction dominates, localize the geometry of that interaction without introducing learned subspaces.

If mixed, follow the largest scientifically relevant unresolved branch while preserving the exact partition-resolved sign structure.

Do not decide that branch before validated evidence exists.

Do not transition to K1 automatically.

---

## 37. Static-design conclusion

The correct next K0 experiment is:

**an exact layer-21 residual-addition source decomposition of the already-validated layer-22 raw-residual routing term, using the authenticated boundary `R22 = R21 + Y21`, an explicit float32 numerical bridge, the frozen layer-22 residual-source RMSNorm map, the frozen `W_H22`, and the frozen strong/weak partition to determine whether the strong-positive / weak-negative residual-source signature is inherited from incoming `ΔR21`, produced by layer21 mixer update `ΔY21`, generated by their downstream interaction, or genuinely mixed.**

This is the narrowest upstream experiment that directly answers the unresolved K0 question.
