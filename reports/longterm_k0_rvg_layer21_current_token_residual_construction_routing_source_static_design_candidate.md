# ContraMamba K0-RVG Layer-21 Current-Token Residual-Construction Routing-Source Decomposition
## Static Design Candidate

## 1. Status

**Phase:** new K0 scientific static design.

**Immediate parent evidence freeze:**

`e78549d64b9628125e619b23dca03b2d025171d3`

**Immediate parent implementation / execution commit:**

`ab3c852b3655b587a7476ea1baca3bcc469e6a9b`

**Immediate parent validated report:**

`reports/longterm_k0_rvg_layer22_current_token_residual_construction_routing_source_validated_evidence_analysis_report_candidate.md`

This stage opens exactly one upstream K0 question:

> How is the already-validated incoming layer21 residual contribution constructed at the immediately preceding residual-addition boundary?

This stage does not authorize:

- training;
- task evaluation;
- logits;
- tokenizer work;
- intervention;
- learned geometry;
- post-hoc layer/channel/item search;
- K1.

A bounded static/runtime preflight after implementation freeze does not require a separate authority document.

---

## 2. Frozen parent result

The immediate parent stage decomposed the layer22 raw-residual routing term at:

`R22 = R21 + Y21`.

It established that the full layer22 residual-source phenotype is mixed, but that the **incoming `ΔR21` term independently carries the same strong-positive / weak-negative partition signature**.

The validated mean incoming-source contrasts are:

### All channels

`-0.2341372085181502`

### Strong partition

`+0.9173468619072845`

### Weak partition

`-1.1514840704254348`

The parent also established:

- incoming carries parent sign structure:
  `True`;
- update carries parent sign structure:
  `True`;
- interaction carries parent sign structure:
  `True`;
- incoming is the largest single component in both strong and weak partitions.

Therefore the next scientifically discriminative unresolved branch is the construction of incoming `ΔR21`.

The parent stage is closed.

This stage must not reopen:

- the `R22 = R21 + Y21` decomposition;
- layer22 RMSNorm factorization;
- fixed layer22 `W_H`;
- lag0 strong/weak partition discovery;
- tokenizer provenance;
- K1.

---

## 3. Parent scientific quantity that must remain fixed

The target is **not** a newly normalized `ΔR21` magnitude.

The target is the exact parent incoming-source routing quantity already defined by the frozen downstream map.

For one item and role, the parent used:

`Q_in21
 = gamma22 ⊙ (s_bar22 ΔR21)`.

Then:

`H_in21
 = W_H22 Q_in21`.

The channel energy contribution was:

`e_in21,j
 = H_in21,j² / ||ΔX22||²`.

Paired corr/ctrl contrast:

`D_in21,j
 = e_in21,corr,j
 - e_in21,ctrl,j`.

The new stage must decompose this exact frozen `D_in21`.

It must preserve:

- observed parent `s_bar22`;
- fixed `gamma22`;
- fixed `W_H22`;
- parent denominator `||ΔX22||²`;
- frozen strong/weak partition `240 / 1296 / 0`;
- fixed lag0 kernel-rank order.

No renormalization is allowed.

---

## 4. Exact upstream residual boundary

The frozen Mamba block semantics are:

`R_l -> RMSNorm_l -> Mixer_l -> Y_l`

followed by:

`R_(l+1) = R_l + Y_l`.

The immediate parent runner already authenticated this block class and forward implementation at layer21 and verified the layer21→22 boundary.

The new candidate boundary is one block upstream:

`R20 -> RMSNorm20 -> Mixer20 -> Y20`

followed by:

`R21 = R20 + Y20`.

The implementation must not assume this boundary solely from indexing.

Before any scientific interpretation, it must authenticate at runtime that:

1. layer20 exists at backbone index `20`;
2. layer20 and layer21 have the same frozen block class;
3. `source_sha(type(layer20).forward)` equals the frozen block-forward SHA:
   `0f808b4d539a496e1681c81d39799d7072fa5c6da381a07cfa19f35fe825b3e4`;
4. layer20 has the expected residual/norm/mixer semantics;
5. layer20 block output is exact float32-equal to the layer21 block input already captured by the frozen parent path;
6. branchwise:
   `R21_b = fl32(R20_b + Y20_b)`
   reconstructs within the fixed residual-addition tolerance.

Failure of any item blocks scientific interpretation.

---

## 5. Scientific question

The bounded question is:

> Is the validated incoming layer21 residual routing signature inherited from incoming `ΔR20`, produced by layer20 mixer update `ΔY20`, generated mainly by their downstream vector interaction under the fixed layer22 parent map, or genuinely mixed?

This question follows the strongest partition-resolved unresolved branch from the parent stage.

It does **not** ask:

- which earlier layer globally matters most;
- which layer should be modified;
- whether layer20 is causally necessary;
- whether another normalization is better;
- whether the pattern improves task performance.

---

## 6. Population and fixed scope

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
- same equal-length prefix protocol;
- same parent layer22 RMSNorm parameters;
- same parent observed branch RMS scales;
- same fixed layer22 hidden in-projection `W_H22`;
- same denominator `||ΔX22||²`;
- same strong/weak partition:
  `240 / 1296 / 0`;
- same fixed lag0 kernel-rank order.

The runner may reuse the frozen `672` pair-role plan.

Expected full model forwards:

`1344`.

Scientific summaries remain restricted to common-330 current-token `k=2`.

No layer search is allowed.

---

## 7. Direct runtime operands

For one aligned item and role, capture matched/swapped current-token branch values:

`R20_m, R20_s ∈ R^768`

`Y20_m, Y20_s ∈ R^768`

`R21_m, R21_s ∈ R^768`.

Definitions:

`ΔR20 = R20_m - R20_s`

`ΔY20 = Y20_m - Y20_s`

`ΔR21_obs = R21_m - R21_s`.

The implementation must also retain the immediate parent capture of:

- `R21`;
- `R22`;
- layer22 RMSNorm input/output;
- parent observed `s22_m`, `s22_s`;
- parent `ΔX22`.

This is necessary to reproduce the frozen parent `D_in21` exactly.

---

## 8. Exact layer-boundary identity gate

The layer20 block post-hook output is the candidate `R21`.

The immediate parent layer21 block pre-hook input is the frozen-parent `R21`.

Require exact float32 equality:

`R21_from_layer20_post
 ==
 R21_from_layer21_pre`.

This is a blocking gate.

The scientific source decomposition is invalid if the two tensors are not exact-equal.

---

## 9. Explicit float32 residual-addition numerical bridge

Do not treat branch residual addition as exact float64 arithmetic.

For branch:

`b ∈ {m,s}`

define float64 replay:

`R21_alg,b
 = R20_b + Y20_b`.

Define branch residual-add execution error:

`epsilon20_b
 = R21_b - R21_alg,b`.

Define matched/swapped difference bridge:

`Q_add20_eps
 = epsilon20_m - epsilon20_s`.

Then:

`ΔR21_obs
 = ΔR20
 + ΔY20
 + Q_add20_eps`.

The numerical bridge is not a scientific mechanism.

Branch-level residual-add reconstruction relative tolerance:

`1e-7`.

The relative difference:

`||ΔR20 + ΔY20 - ΔR21_obs|| / ||ΔR21_obs||`

is diagnostic-only because it is cancellation-sensitive.

Require the explicit error-difference identity at maximum absolute tolerance:

`5e-12`.

---

## 10. Fixed parent downstream map

The frozen target map for `ΔR21` is:

`L_parent(v)
 = gamma22 ⊙ (s_bar22 v)`.

The item/role-specific scalar:

`s_bar22
 = (s22_m + s22_s)/2`

must be the exact parent observed value.

Do not derive a new scale from `R21`.

Do not use RMSNorm21 or RMSNorm20 scaling in the scientific target.

The current question is the construction of the parent `ΔR21` source under the already-frozen layer22 residual-source map.

---

## 11. Exact upstream source decomposition in parent-map input space

Apply the fixed parent map to the two layer20→21 additive sources.

Define:

`Q_r20
 = gamma22 ⊙ (s_bar22 ΔR20)`.

Define:

`Q_y20
 = gamma22 ⊙ (s_bar22 ΔY20)`.

Define:

`Q_eps20
 = gamma22 ⊙ (s_bar22 Q_add20_eps)`.

The exact parent incoming-source vector is:

`Q_in21
 = gamma22 ⊙ (s_bar22 ΔR21_obs)`.

Require:

`Q_in21
 = Q_r20
 + Q_y20
 + Q_eps20`.

This is a linear decomposition of the exact frozen parent `Q_in21`.

---

## 12. Blocking parent `Q_in21` reproduction

The new runner must independently reconstruct:

`Q_in21
 = gamma22 ⊙ (s_bar22 ΔR21_obs)`.

It must reproduce the parent persisted scalar diagnostics associated with the incoming source.

At minimum:

- parent `delta_r21_l2_corr`;
- parent `delta_r21_l2_ctrl`;
- parent `q_in_l2_corr`;
- parent `q_in_l2_ctrl`;
- parent item-level incoming all contrast;
- parent item-level incoming strong contrast;
- parent item-level incoming weak contrast.

Population means must reproduce exactly:

- all:
  `-0.2341372085181502`;
- strong:
  `+0.9173468619072845`;
- weak:
  `-1.1514840704254348`.

No source interpretation is allowed if these parent bridges fail.

---

## 13. Propagation through fixed layer22 `W_H`

Use the same fixed parent operator:

`W_H22 ∈ R^(1536×768)`.

Define:

`H_r20
 = W_H22 Q_r20`

`H_y20
 = W_H22 Q_y20`

`H_eps20
 = W_H22 Q_eps20`

`H_in21
 = W_H22 Q_in21`.

Require exact numerical closure:

`H_in21
 = H_r20
 + H_y20
 + H_eps20`.

This stage does not reinterpret or decompose `W_H22`.

It is a fixed downstream readout operator.

---

## 14. Parent denominator remains unchanged

Use:

`D_X22
 = ||ΔX22_obs||²`.

This is the exact parent denominator.

Do not normalize by:

- `||ΔR20||²`;
- `||ΔY20||²`;
- `||ΔR21||²`;
- `||Q_r20||²`;
- `||Q_y20||²`;
- `||Q_in21||²`.

Changing the denominator would define a new scientific quantity and break parent comparability.

---

## 15. Exact channelwise decomposition of frozen `D_in21`

For output channel `j` define:

`h_r,j
 = H_r20[j]`

`h_y,j
 = H_y20[j]`

`h_eps,j
 = H_eps20[j]`.

Define:

`e_r20,j
 = h_r,j² / D_X22`.

Define:

`e_y20,j
 = h_y,j² / D_X22`.

Define interaction:

`e_ry20,j
 = 2 h_r,j h_y,j / D_X22`.

Define complete numerical bridge:

`e_eps20,j
 = (
     h_eps,j²
     + 2 h_r,j h_eps,j
     + 2 h_y,j h_eps,j
   ) / D_X22`.

Then:

`e_in21,j
 = e_r20,j
 + e_y20,j
 + e_ry20,j
 + e_eps20,j`.

And independently:

`e_in21,j
 = H_in21,j² / D_X22`.

This identity must close channelwise.

---

## 16. Paired corr/ctrl source contrast

For paired items define:

`D_in21,j
 = e_in21,corr,j
 - e_in21,ctrl,j`.

Define upstream source contrasts:

`D_r20,j
 = e_r20,corr,j
 - e_r20,ctrl,j`.

`D_y20,j
 = e_y20,corr,j
 - e_y20,ctrl,j`.

`D_ry20,j
 = e_ry20,corr,j
 - e_ry20,ctrl,j`.

`D_eps20,j
 = e_eps20,corr,j
 - e_eps20,ctrl,j`.

Require:

`D_in21,j
 = D_r20,j
 + D_y20,j
 + D_ry20,j
 + D_eps20,j`.

This is the primary scientific identity.

---

## 17. All/strong/weak partition closure

Use the exact frozen downstream partitions:

- all:
  `1536`;
- strong:
  `240`;
- weak:
  `1296`;
- equal:
  `0`.

For:

`P ∈ {all,strong,weak}`

define:

`Delta_r20,P
 = Σ_(j∈P) D_r20,j`

`Delta_y20,P
 = Σ_(j∈P) D_y20,j`

`Delta_ry20,P
 = Σ_(j∈P) D_ry20,j`

`Delta_eps20,P
 = Σ_(j∈P) D_eps20,j`.

Require:

`Delta_in21,P
 = Delta_r20,P
 + Delta_y20,P
 + Delta_ry20,P
 + Delta_eps20,P`.

The total must reproduce the frozen parent incoming-source contrast per item and in population mean.

---

## 18. Why this is the correct next experiment

The prior stage produced a mixed full `R22` decomposition.

But for the distinctive strong+/weak− partition phenotype:

- incoming `ΔR21` was the largest single component in strong;
- incoming `ΔR21` was the largest single component in weak;
- incoming independently carried the exact qualitative sign structure.

Therefore following `ΔR21` one block upstream is the preregistered scientifically discriminative next step.

This is not arbitrary layer peeling.

It follows the largest partition-resolved unresolved branch.

---

## 19. Direct hook policy

Use read-only hooks on layer20:

- layer20 block forward pre-hook:
  capture `R20`;
- layer20 mixer forward hook:
  capture `Y20`;
- layer20 block forward hook:
  capture candidate `R21`.

Simultaneously preserve the immediate parent layer21 pre-hook capture.

Require exact float32 equality between:

- layer20 block output candidate `R21`;
- immediate-parent layer21 block input `R21`.

No hook may mutate inputs or outputs.

No raw vectors may be persisted.

---

## 20. Runtime architecture authentication

Before scientific calculation, authenticate:

- backbone layer count:
  `24`;
- source block:
  `20`;
- target residual layer:
  `21`;
- layer20 block class == layer21 block class;
- layer21 block class remains the frozen parent class;
- block forward SHA:
  `0f808b4d539a496e1681c81d39799d7072fa5c6da381a07cfa19f35fe825b3e4`;
- backbone forward SHA:
  `3f332bd50e6ea4ff64468c8d748672f90ffc43912d22c9d25d02b3e87d1661a6`;
- frozen Mamba source SHA:
  `23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`;
- layer20 residual width:
  `768`;
- layer20 mixer output width:
  `768`;
- parent `W_H22` shape:
  `(1536,768)`;
- strong/weak partition:
  `240 / 1296 / 0`.

Any mismatch blocks execution interpretation.

---

## 21. Immediate parent provenance

Authenticate evidence freeze:

`e78549d64b9628125e619b23dca03b2d025171d3`.

Immediate parent run directory:

`reports/longterm_k0_rvg_layer22_current_token_residual_construction_routing_source_ab3c852_v1`

Required artifact SHA256 values:

### Item

`063bf6355795c20adde263d8ceb8868be4e5cb42fe5c5fd46da8219cf5b58124`

### Channel

`96a1308a5e384e59cd08e4df6e25d406ff6899e663326ecaff2c1702c8adcdfc`

### Cumulative

`3f7be599d6f505a6f2e679bb24920f7ca09b8577f73289cc457a1138b4547f5d`

### Summary

`364d3a961cc103c2760ea7d70a5547af52502d7e5eadcd2c6421e08ee4893ee0`

### Manifest

`55f992bddc7bdc232ce5a8c2d6555feb82026e6f14bb86424eed23d369368f4b`

Immediate parent runner SHA256:

`973692b8002572e0b229fb5ecf288457b063bae4043ccfeba56fa0c6e7448300`.

The parent validated report must also be authenticated at its frozen evidence commit.

Any parent provenance mismatch blocks interpretation.

---

## 22. Parent item-level reproduction requirements

For every common-330 item reproduce:

### Role scalars

For corr and ctrl:

- `delta_r21_l2`;
- `q_in_l2`;
- observed parent `s22_bar`.

### Parent incoming-source partition contrasts

- `delta_in_all`;
- `delta_in_strong`;
- `delta_in_weak`.

These must match the frozen parent item artifact.

The new run must not merely reproduce the three population means.

Item-level reproduction is a blocking gate.

---

## 23. Parent channel aggregate reproduction

The immediate parent channel artifact persists:

`mean_d_in`

for all `1536` channels.

The new run must independently aggregate its reconstructed `D_in21,j` across common-330 and reproduce the frozen parent channelwise `mean_d_in`.

This is a stronger cross-artifact bridge than partition means alone.

Use preregistered absolute/relative tolerance.

Do not use parent channel values as computational inputs to the new source decomposition.

Use them only as validation targets.

---

## 24. Fixed kernel-rank cumulative reproduction

Use the frozen descending lag0 squared-kernel rank order.

The cumulative sum of reconstructed parent incoming-source channel means must reproduce the immediate parent fixed-rank cumulative incoming profile if available from the parent artifact schema.

If the immediate parent cumulative artifact does not persist an incoming-specific cumulative field, reconstruct the cumulative profile from the authenticated parent channel `mean_d_in` values and require exact agreement with the new total incoming cumulative profile.

No new rank cutoff is selected.

---

## 25. Residual-space diagnostics

For corr and ctrl separately report:

- `||ΔR20||`;
- `||ΔY20||`;
- `||ΔR21||`;
- `||Q_add20_eps||`;
- branch-add reconstruction residual;
- difference-relative diagnostic.

Also report:

`||ΔR20 + ΔY20||²
 = ||ΔR20||²
 + ||ΔY20||²
 + 2<ΔR20,ΔY20>`.

Report:

- residual-space cross term;
- cross fraction;
- addition ratio.

These are diagnostics only.

They do not replace the downstream parent-map routing attribution.

---

## 26. Item artifact

Persist exactly one scalar row per common-330 item.

Required categories:

### Parent bridges

- parent `delta_r21_l2_corr`;
- parent `delta_r21_l2_ctrl`;
- parent `q_in_l2_corr`;
- parent `q_in_l2_ctrl`;
- parent incoming all/strong/weak contrasts.

### Upstream branch metrics

For corr and ctrl:

- `delta_r20_l2`;
- `delta_y20_l2`;
- `delta_r21_l2`;
- `q_add20_eps_l2`;
- `q_r20_l2`;
- `q_y20_l2`;
- `q_eps20_l2`;
- branch-add reconstruction residual;
- difference-relative diagnostic;
- residual-space interaction;
- residual-space addition ratio.

### Fixed parent-map values

For corr and ctrl:

- `s22_matched`;
- `s22_swapped`;
- `s22_bar`.

### Routing source components

For all/strong/weak:

- `Delta_r20`;
- `Delta_y20`;
- `Delta_ry20`;
- `Delta_eps20`;
- `Delta_total_in21`;
- source closure;
- parent item reproduction residual.

No vector is persisted.

---

## 27. Channel aggregate artifact

Persist exactly `1536` rows.

Each row includes:

- channel index;
- frozen lag0 kernel magnitude rank;
- frozen strong/weak partition;
- fixed `W_H22` row-gain squared;
- parent frozen `mean_d_in`;
- new mean/median `D_in21`;
- mean/median `D_r20`;
- mean/median `D_y20`;
- mean/median `D_ry20`;
- mean/median `D_eps20`;
- sign counts.

Require the new `mean D_in21` to reproduce frozen parent `mean_d_in`.

Do not create a new channel ranking.

---

## 28. Cumulative artifact

Persist exactly `1536` rows in frozen lag0 kernel-rank order.

Include cumulative means for:

- total reconstructed parent incoming source;
- `R20` source;
- `Y20` source;
- `R20×Y20` interaction;
- numerical bridge.

No rank threshold is introduced.

No top-k selection is allowed.

---

## 29. Threshold-free component attribution

For:

`P ∈ {all,strong,weak}`

let aggregate means be:

`M_r20,P`

`M_y20,P`

`M_ry20,P`

`M_eps20,P`.

Require:

`M_in21,P
 = M_r20,P
 + M_y20,P
 + M_ry20,P
 + M_eps20,P`.

Define absolute mass:

`A_P
 = |M_r20,P|
 + |M_y20,P|
 + |M_ry20,P|
 + |M_eps20,P|`.

Report absolute shares.

Also report signed component-to-parent-net ratios.

Do not interpret ratios as probabilities.

Do not introduce a dominance threshold.

---

## 30. Scientific outcome classes

### Outcome A: upstream inherited residual

`R20` source is the largest scientifically relevant partition-resolved component and carries the parent strong+/weak− sign structure.

Interpretation:

the layer21 incoming signature is already present one residual boundary earlier.

### Outcome B: layer20 mixer-update source

`Y20` source is the primary partition-resolved carrier.

Interpretation:

layer20 mixer update is the immediate dominant producer of the incoming layer21 routing signature.

### Outcome C: additive interaction

`R20×Y20` interaction is the primary partition-resolved carrier.

Interpretation:

the parent incoming signature is mainly generated by vector interaction at the layer20→21 residual addition.

### Outcome D: mixed

No single source provides a sufficient descriptive account.

Report exact partition-resolved mixture and whether components reinforce or cancel.

### Numerical bridge condition

If the numerical bridge is not negligible relative to scientific components, block promotion of a scientific source claim.

---

## 31. Parent sign-structure criterion

The frozen parent incoming-source qualitative signature is:

- strong:
  positive;
- weak:
  negative.

A scientific component independently carries the parent sign structure only if:

- its strong aggregate mean is `> 0`;
- its weak aggregate mean is `< 0`.

This is descriptive.

It is not a statistical significance threshold.

It does not require itemwise unanimity.

---

## 32. Blocking conditions

Block scientific interpretation if any occurs:

1. layer20 is not the same authenticated block class as layer21;
2. layer20 block forward SHA mismatches;
3. layer20 block output `R21` is not exact-equal to immediate-parent layer21 input `R21`;
4. branch residual-add reconstruction exceeds `1e-7`;
5. explicit add-error identity exceeds `5e-12`;
6. parent `delta_r21_l2` reproduction fails;
7. parent `q_in_l2` reproduction fails;
8. parent item incoming all/strong/weak reproduction fails;
9. parent channel `mean_d_in` reproduction fails;
10. source channel identity fails;
11. partition source closure fails;
12. frozen partition changes;
13. frozen kernel-rank order changes;
14. raw vectors are persisted;
15. tokenizer/logits/task-head/training/intervention/PCA/SVD/learned-probe/post-hoc search occurs.

An unexpected scientific outcome is not a validation failure.

---

## 33. Tolerances fixed before runtime

Use:

- branch residual-add reconstruction relative tolerance:
  `1e-7`;
- parent scalar relative tolerance:
  `1e-13`;
- parent scalar absolute tolerance:
  `1e-13`;
- vector/error identity absolute tolerance:
  `5e-12`;
- squared-energy relative tolerance:
  `1e-12`;
- kernel RMS relative tolerance:
  `1e-13`;
- kernel RMS absolute tolerance:
  `1e-13`.

The cancellation-sensitive difference-relative reconstruction remains diagnostic-only.

No tolerance weakening is allowed after observing runtime results.

---

## 34. Execution plan

After implementation freeze and bounded runtime preflight PASS:

- execute the same frozen `672` pair-role plan;
- matched and swapped branch forward per pair-role;
- expected model forwards:
  `1344`;
- scientific population:
  common-330 current-token `k=2`.

GPU is not required.

Local CPU is preferred.

Kaggle is unnecessary unless local execution becomes technically impossible.

---

## 35. Artifact policy

A later authorized full execution may persist:

1. `330` item scalar rows;
2. `1536` channel aggregate rows;
3. `1536` fixed-rank cumulative rows;
4. summary JSON;
5. execution manifest.

Do not persist:

- `R20_m/s`;
- `Y20_m/s`;
- `R21_m/s`;
- `ΔR20`;
- `ΔY20`;
- `Q_r20`;
- `Q_y20`;
- `Q_eps20`;
- `H_r20`;
- `H_y20`;
- `H_eps20`;
- per-item channel vectors;
- logits;
- task outputs.

Temporary vectors may exist only in memory for aggregation.

---

## 36. Provenance requirements

Execution manifest must record:

- runtime branch;
- runtime Git HEAD;
- this static design freeze commit;
- this static design SHA256;
- immediate parent evidence freeze:
  `e78549d64b9628125e619b23dca03b2d025171d3`;
- immediate parent implementation:
  `ab3c852b3655b587a7476ea1baca3bcc469e6a9b`;
- immediate parent runner SHA;
- immediate parent artifact hashes;
- immediate parent validated report hash;
- frozen Mamba source SHA;
- block-forward SHA;
- backbone-forward SHA;
- handoff ZIP SHA;
- checkpoint SHA;
- encoder digests;
- expected forward count;
- source/target layer indices;
- relative coordinate;
- all tolerances;
- persistence flags;
- forbidden-action flags;
- output hashes.

Any mismatch blocks interpretation.

---

## 37. Scientific interpretation boundary

Passing this stage may support:

> the validated incoming layer21 residual routing signature is algebraically localized at `R21 = R20 + Y20` to the incoming `R20` residual source, layer20 mixer update, their interaction, or a fixed mixture.

It may not support:

- causal necessity;
- causal sufficiency;
- intervention claims;
- task-performance claims;
- semantic interpretation of coordinates;
- model-general claims;
- K1.

This remains observational/algebraic localization.

---

## 38. Next-step rule after validated evidence

If `R20` is the largest partition-resolved carrier and preserves the strong+/weak− signature, move one residual boundary upstream.

If `Y20` is the dominant carrier, enter the narrowest exact layer20 mixer-output source decomposition.

If interaction is dominant, localize the exact geometry of that interaction without learned projections.

If mixed, follow the largest scientifically discriminative unresolved partition-resolved branch.

Do not decide this branch before validated evidence exists.

Do not automatically peel all remaining layers.

Do not transition to K1 automatically.

---

## 39. Static-design conclusion

The correct next K0 experiment is:

**an exact decomposition of the frozen parent incoming-layer21 residual routing term at the immediately preceding `R21 = R20 + Y20` boundary, preserving the frozen layer22 parent map `s_bar22·gamma22·W_H22`, the original `||ΔX22||²` denominator, the 240/1296 strong/weak partition, and fixed kernel-rank order, in order to determine whether the incoming-layer21 strong-positive / weak-negative routing signature is inherited from `ΔR20`, produced by `ΔY20`, generated by their additive interaction, or genuinely mixed.**

This is the narrowest authorized upstream K0 question after the evidence freeze at `e78549d64b9628125e619b23dca03b2d025171d3`.
