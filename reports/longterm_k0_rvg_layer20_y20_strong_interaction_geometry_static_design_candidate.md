# ContraMamba K0-RVG Layer-20→21 Strong-Partition Interaction Geometry Localization
## Static Design Candidate

## 1. Status

**Phase:** new K0 scientific static design.

**Immediate parent evidence freeze:**

`f5ef304701c494b83b070a566265d652c03444dd`

**Immediate parent implementation / execution commit:**

`845827d3d99de5fdf5409b901b7039197cf1c08e`

**Immediate parent validated report:**

`reports/longterm_k0_rvg_layer21_current_token_residual_construction_routing_source_validated_evidence_analysis_report_candidate.md`

This stage opens exactly one scientific question:

> Why is the validated `R20×Y20` interaction contrast so strongly positive in the frozen strong-kernel partition?

The target is the already-frozen strong interaction term from the parent stage.

This stage does not authorize:

- training;
- task evaluation;
- logits;
- tokenizer work;
- intervention;
- learned geometry;
- PCA/SVD;
- layer search;
- channel search;
- item search;
- K1.

A bounded static/runtime preflight after implementation freeze does not require a separate authority document.

---

## 2. Frozen parent result

The immediate parent decomposed the frozen incoming-layer21 source:

`D_in21
 = D_r20
 + D_y20
 + D_ry20
 + D_eps20`

under the unchanged layer22 downstream map.

The validated population means were:

### All channels

- total:
  `-0.2341372085181502`;
- `R20`:
  `-0.7142140003946674`;
- `Y20`:
  `-1.2219687049262402`;
- interaction:
  `+1.7020454571466457`.

### Strong partition

- total:
  `+0.9173468619072845`;
- `R20`:
  `+0.10667586921656537`;
- `Y20`:
  `+0.050076264450526`;
- interaction:
  `+0.760594723140676`.

### Weak partition

- total:
  `-1.1514840704254348`;
- `R20`:
  `-0.8208898696112327`;
- `Y20`:
  `-1.2720449693767664`;
- interaction:
  `+0.9414507340059698`.

The strong interaction term is therefore:

`D_ry20,strong = +0.760594723140676`

and accounts for approximately:

`82.91244617758867%`

of strong absolute scientific component mass.

The parent stage classified the result as Outcome D:

**partition-asymmetric mixed construction with strong-side interaction dominance and weak-side direct-source negativity under positive interaction cancellation.**

The parent stage is closed.

---

## 3. Why this is the correct next K0 target

The long-running K0 chain is localizing the mechanism responsible for corr-enriched **strong-kernel exposure**.

At the parent boundary:

- strong direct `R20` source is small positive;
- strong direct `Y20` source is small positive;
- strong `R20×Y20` interaction is overwhelmingly dominant.

Therefore the narrow next question is not another residual-boundary peel.

It is the exact geometry of the already-validated strong interaction.

The new stage asks whether the corr>ctrl strong interaction advantage is explained primarily by:

1. larger parent-normalized projected `R20` magnitude;
2. larger parent-normalized projected `Y20` magnitude;
3. stronger directional alignment between the two strong subvectors;
4. a genuine mixture.

---

## 4. Frozen scientific target

For item-role branch pair `b`, the parent stage constructed:

`H_r20,b
 = W_H22 Q_r20,b`

and:

`H_y20,b
 = W_H22 Q_y20,b`.

The frozen denominator is:

`D_X22,b
 = ||ΔX22,b||²`.

For strong channel set `S`, the parent strong interaction energy is:

`I_b,strong
 = Σ_(j∈S)
   2 H_r20,b,j H_y20,b,j / D_X22,b`.

The paired interaction contrast is:

`D_ry20,strong
 = I_corr,strong
 - I_ctrl,strong`.

The new stage must reproduce the parent item-level `delta_ry20_strong` exactly.

The population mean must reproduce:

`+0.760594723140676`.

No new normalization, partition, rank cutoff, or channel selection is allowed.

---

## 5. Frozen population and scope

Use exactly:

- current token only;
- relative coordinate:
  `k=2`;
- common DDSSSSS cohort:
  `330`;
- corr and ctrl roles;
- same matched/swapped definitions;
- same checkpoint;
- same handoff;
- same equal-length prefix protocol;
- same layer20 captures;
- same layer22 parent map;
- same `W_H22`;
- same role-specific `D_X22`;
- same frozen strong channel set:
  `240`;
- same weak set:
  `1296`;
- same lag0 kernel-rank identities.

No new channel is admitted.

No strong subset is further subdivided for the primary claim.

---

## 6. Parent-normalized strong subvectors

For role:

`b ∈ {corr, ctrl}`

define the frozen strong subvectors:

`x_b
 = H_r20,b,S / sqrt(D_X22,b)`

and:

`y_b
 = H_y20,b,S / sqrt(D_X22,b)`.

Then:

`x_b, y_b ∈ R^240`.

The exact parent strong interaction is:

`I_b
 = 2 <x_b, y_b>`.

This is numerically identical to the parent strong interaction sum.

No learned projection is introduced.

No basis change is introduced.

The coordinates remain the fixed strong output rows of `W_H22`.

---

## 7. Magnitude and alignment coordinates

Define:

`A_b
 = ||x_b||₂`

`B_b
 = ||y_b||₂`.

For nonzero norms define:

`C_b
 = <x_b,y_b> / (A_b B_b)`.

Thus:

`C_b ∈ [-1,1]`.

The parent strong interaction is exactly:

`I_b
 = 2 A_b B_b C_b`.

If either `A_b` or `B_b` is zero, execution blocks scientific interpretation because the preregistered cosine coordinate is undefined.

No epsilon is added to the denominator.

---

## 8. Paired corr/ctrl notation

For one common item define:

`A_c = A_corr`

`A_t = A_ctrl`

`B_c = B_corr`

`B_t = B_ctrl`

`C_c = C_corr`

`C_t = C_ctrl`.

Define differences:

`ΔA = A_c - A_t`

`ΔB = B_c - B_t`

`ΔC = C_c - C_t`.

Define midpoints:

`A_bar = (A_c + A_t)/2`

`B_bar = (B_c + B_t)/2`

`C_bar = (C_c + C_t)/2`.

Define role interaction magnitudes:

`M_c = 2 A_c B_c`

`M_t = 2 A_t B_t`

and:

`M_bar = (M_c + M_t)/2`.

---

## 9. Exact midpoint identity: first level

Because:

`I = M C`,

the corr–ctrl interaction difference is exactly:

`ΔI
 = I_c - I_t
 = C_bar ΔM + M_bar ΔC`

where:

`ΔM = M_c - M_t`.

This separates:

- a composite magnitude effect;
- an alignment effect.

There is no approximation and no residual term.

---

## 10. Exact midpoint identity: magnitude level

Because:

`M = 2 A B`,

the magnitude difference is exactly:

`ΔM
 = 2 B_bar ΔA
 + 2 A_bar ΔB`.

Substitute into the first-level identity.

Then:

`ΔI
 = Q_A
 + Q_B
 + Q_C`

with:

`Q_A
 = 2 C_bar B_bar ΔA`

`Q_B
 = 2 C_bar A_bar ΔB`

`Q_C
 = M_bar ΔC`.

This is the primary scientific identity.

It is exact and symmetric in the two magnitude factors.

There is no cross residual.

---

## 11. Interpretation of the three exact components

### `Q_A`

Contribution associated with corr–ctrl difference in the **parent-normalized strong `R20` projected magnitude** while holding the symmetric midpoint of the other factors.

### `Q_B`

Contribution associated with corr–ctrl difference in the **parent-normalized strong `Y20` projected magnitude** while holding the symmetric midpoint of the other factors.

### `Q_C`

Contribution associated with corr–ctrl difference in **cosine alignment** between the two parent-normalized strong subvectors.

These are algebraic attribution terms.

They are not causal effects.

---

## 12. Why "parent-normalized magnitude" is the correct wording

`A_b` and `B_b` already include the frozen parent denominator through:

`1 / sqrt(D_X22,b)`.

Therefore this stage must not call them raw `H` magnitudes.

The correct terminology is:

- parent-normalized strong `R20` projected magnitude;
- parent-normalized strong `Y20` projected magnitude.

This preserves exact comparability to the parent interaction target.

A separate decomposition of raw `H` magnitude versus denominator is out of scope.

---

## 13. Exact reconstruction gates

For every common item and role require:

`I_b,strong
 = 2 <x_b,y_b>`

and:

`I_b,strong
 = 2 A_b B_b C_b`.

For every paired item require:

`D_ry20,strong
 = I_corr,strong - I_ctrl,strong`

and:

`D_ry20,strong
 = Q_A + Q_B + Q_C`.

All identities must close within preregistered numerical tolerances.

---

## 14. Parent item-level reproduction

The immediate parent item artifact persists:

`delta_ry20_strong`.

For every common-330 item require the newly reconstructed:

`ΔI`

to reproduce that frozen value.

This is a blocking gate.

Population mean must reproduce:

`+0.760594723140676`.

The new stage is invalid if it only matches the aggregate but not item-level parent values.

---

## 15. Parent strong-channel reproduction

The immediate parent channel artifact persists:

`mean_d_ry20`

for all `1536` channels.

For the frozen `240` strong channels, the new runner must independently reconstruct:

`D_ry20,j
 = 2 x_corr,j y_corr,j
 - 2 x_ctrl,j y_ctrl,j`

and reproduce parent `mean_d_ry20` channel means.

This is a validation bridge.

The new scientific decomposition remains partition-level.

No channel is selected post hoc.

---

## 16. Strong-vector alignment diagnostics

In addition to the exact `A/B/C` decomposition, report threshold-free diagnostics for each role.

### Cosine

`C_b`.

### Positive interaction mass

`P_b
 = Σ_(j∈S) max(2 x_b,j y_b,j, 0)`.

### Negative interaction mass

`N_b
 = Σ_(j∈S) min(2 x_b,j y_b,j, 0)`.

Then:

`I_b = P_b + N_b`.

Also report:

- same-sign channel count:
  number of `j` with `x_b,j y_b,j > 0`;
- opposite-sign channel count:
  number with product `< 0`;
- zero-product count.

These are diagnostics only.

They do not define the primary outcome.

No threshold other than exact sign is introduced.

---

## 17. Why no channelwise cosine decomposition

Each strong output channel is scalar.

A scalar "cosine" would reduce to sign and would discard magnitude structure.

Therefore the primary alignment coordinate is the cosine of the complete frozen 240-dimensional strong subvector.

Channelwise sign statistics are retained only as diagnostics.

---

## 18. Parent denominator and downstream operator remain frozen

Do not alter:

- `s_bar22`;
- `gamma22`;
- `W_H22`;
- `D_X22`;
- strong/weak partition;
- lag0 kernel identities.

Do not use:

- RMSNorm20 scale;
- RMSNorm21 scale;
- a newly fitted normalization;
- per-item unit-vector reweighting beyond the exact cosine definition;
- whitening;
- PCA;
- SVD;
- learned probes.

---

## 19. Runtime capture policy

Reuse the frozen immediate-parent runtime/capture path.

No new model branch is required beyond the same matched/swapped forward protocol.

Reconstruct in memory:

- `H_r20`;
- `H_y20`;
- `ΔX22`;
- strong masks.

Do not persist raw vectors.

No hook may mutate model state or tensors.

Expected full model forwards remain:

`1344`.

---

## 20. Static/runtime authentication

Before scientific calculation authenticate:

- branch and frozen authority;
- immediate parent evidence freeze:
  `f5ef304701c494b83b070a566265d652c03444dd`;
- immediate parent implementation:
  `845827d3d99de5fdf5409b901b7039197cf1c08e`;
- immediate parent runner SHA:
  `788cd3b64883d2d4f8733f0787e0a44de25952e14cf48251b288ac36354388b8`;
- immediate parent item SHA:
  `58be748a721be37971f45bb4f5a2996f240fe91a0cd6ba99f33c0985b7538470`;
- immediate parent channel SHA:
  `23f19995bf06d7fd10048e13411cbd0d4ec75676a99ae9d7032889f24fcb6537`;
- immediate parent cumulative SHA:
  `81845540f4da7d5a43e1dcfa7893177a9ad88b619de8dff4a89e1be10477be4a`;
- immediate parent summary SHA:
  `d65af91038d05bf3c92533ff8b48a0d495bd862dcbb779813bf56d337ab1855e`;
- immediate parent manifest SHA:
  `5268703eb069ad2d385bf15a2f9456adfd583a9fae5fda97c0e646ee05cc2300`;
- immediate parent validated report SHA:
  `38b4bb275cac40cb2b20487f3b0fc378c269cce3a901b49566bc512cec4a0202`.

Also authenticate the frozen Mamba source, block forward, backbone forward, handoff, checkpoint, and encoder identities inherited from the parent manifest.

---

## 21. Strong channel-set authentication

The strong set must contain exactly:

`240`

channels.

For every strong channel require its:

- channel index;
- kernel magnitude rank;
- downstream partition label

to match the frozen immediate-parent channel artifact.

No strong channel may be added or removed.

No equal/weak channel may enter the primary interaction geometry.

---

## 22. Item artifact

Persist exactly one scalar row for each common-330 item.

Required fields include:

### Parent bridge

- parent `delta_ry20_strong`;
- reconstructed `delta_ry20_strong`;
- parent reproduction residual.

### Corr role

- `A_corr`;
- `B_corr`;
- `C_corr`;
- `M_corr`;
- `I_corr`;
- positive interaction mass;
- negative interaction mass;
- same-sign count;
- opposite-sign count;
- zero-product count.

### Ctrl role

Same fields.

### Paired geometry

- `delta_A`;
- `delta_B`;
- `delta_C`;
- `A_bar`;
- `B_bar`;
- `C_bar`;
- `M_bar`;
- `Q_A`;
- `Q_B`;
- `Q_C`;
- reconstructed `delta_I`;
- exact geometry closure residual.

No vector is persisted.

---

## 23. Strong-channel validation artifact

Persist exactly `240` rows, one per frozen strong channel.

Each row includes:

- channel index;
- frozen kernel magnitude rank;
- parent `mean_d_ry20`;
- newly reconstructed mean `D_ry20,j`;
- reproduction residual;
- corr mean signed product:
  `mean(2 x_corr,j y_corr,j)`;
- ctrl mean signed product:
  `mean(2 x_ctrl,j y_ctrl,j)`;
- corr positive/negative/zero sign counts;
- ctrl positive/negative/zero sign counts.

This artifact is a provenance/validation bridge.

It is not used for post-hoc channel selection.

---

## 24. Summary artifact

Report population aggregates of:

- parent strong interaction total;
- reconstructed strong interaction total;
- `Q_A`;
- `Q_B`;
- `Q_C`;
- absolute component mass;
- absolute shares;
- signed component-to-total ratios;
- largest absolute scientific component.

Also aggregate corr and ctrl distributions of:

- `A`;
- `B`;
- `C`;
- `M`;
- `I`;
- positive interaction mass;
- negative interaction mass;
- sign counts.

No statistical significance threshold is introduced.

---

## 25. Exact scientific outcome classes

### Outcome A: `R20` normalized-magnitude dominant

Population mean `Q_A` is the largest absolute scientific component.

Interpretation:

the corr>ctrl strong interaction advantage is primarily associated with stronger parent-normalized `R20` projected magnitude.

### Outcome B: `Y20` normalized-magnitude dominant

Population mean `Q_B` is largest.

Interpretation:

the strong interaction advantage is primarily associated with stronger parent-normalized `Y20` projected magnitude.

### Outcome C: alignment dominant

Population mean `Q_C` is largest.

Interpretation:

the strong interaction advantage is primarily associated with stronger corr directional alignment between the two fixed strong subvectors.

### Outcome D: mixed

No single term gives a sufficient descriptive account.

Report the exact mixture and whether magnitude terms and alignment reinforce or cancel.

No arbitrary dominance threshold is introduced.

---

## 26. Additional directional interpretation

Regardless of largest component, report:

- sign of mean `Q_A`;
- sign of mean `Q_B`;
- sign of mean `Q_C`.

This distinguishes:

- reinforcing magnitude and alignment;
- magnitude advantage opposed by alignment;
- alignment advantage opposed by magnitude;
- mixed cancellation.

Do not infer causality from sign.

---

## 27. Blocking conditions

Block scientific interpretation if any occurs:

1. frozen parent evidence identity mismatches;
2. frozen strong set is not exactly 240 channels;
3. parent item `delta_ry20_strong` reproduction fails;
4. parent strong-channel `mean_d_ry20` reproduction fails;
5. `A` or `B` is zero for any role/item;
6. cosine falls outside numerical `[-1,1]` tolerance;
7. `I = 2<x,y>` closure fails;
8. `I = 2ABC` closure fails;
9. `ΔI = Q_A + Q_B + Q_C` closure fails;
10. raw vectors are persisted;
11. tokenizer/logits/task heads/training/intervention/PCA/SVD/learned geometry executes;
12. any channel/layer/item search is introduced.

An unexpected outcome is not a validation failure.

---

## 28. Numerical tolerances

Use:

- parent scalar relative tolerance:
  `1e-13`;
- parent scalar absolute tolerance:
  `1e-13`;
- vector/identity absolute tolerance:
  `5e-12`;
- channel aggregate reproduction absolute tolerance:
  `5e-12`;
- cosine boundary numerical slack:
  `1e-12`.

Do not weaken tolerances after runtime observation.

---

## 29. Execution policy

After implementation freeze and bounded runtime preflight PASS:

- local CPU execution is preferred;
- GPU remains off;
- same frozen `672` pair-role plan;
- expected model forwards:
  `1344`;
- no Kaggle unless local execution becomes technically impossible.

Execution success alone does not establish the scientific conclusion.

Independent artifact validation remains required.

---

## 30. Artifact policy

A later authorized full execution may persist:

1. `330` item scalar rows;
2. `240` fixed strong-channel validation rows;
3. summary JSON;
4. execution manifest JSON.

No cumulative artifact is required because no new rank-localization question is asked.

Do not persist:

- `x`;
- `y`;
- `H_r20`;
- `H_y20`;
- `ΔX22`;
- per-item channel vectors.

---

## 31. Scientific interpretation boundary

Passing this stage may support:

> the frozen strong `R20×Y20` interaction advantage is algebraically localized to parent-normalized `R20` magnitude, parent-normalized `Y20` magnitude, strong-subvector cosine alignment, or a fixed mixture.

It may not support:

- causal necessity;
- causal sufficiency;
- intervention claims;
- semantic coordinate claims;
- task-performance claims;
- model-general claims;
- K1.

---

## 32. Next-step rule after validated evidence

If `Q_A` dominates, the next K0 branch should localize why the parent-normalized strong `R20` projection magnitude differs.

If `Q_B` dominates, localize the strong `Y20` projected magnitude source.

If `Q_C` dominates, localize the fixed strong-subvector alignment structure without learned geometry.

If mixed, follow the largest scientifically discriminative unresolved term while preserving the frozen strong partition.

Do not automatically peel another residual boundary.

Do not transition to K1 automatically.

---

## 33. Static-design conclusion

The next K0 experiment is:

**an exact, threshold-free decomposition of the validated strong-partition `R20×Y20` interaction contrast into parent-normalized `R20` projected magnitude, parent-normalized `Y20` projected magnitude, and cosine-alignment contributions using the exact midpoint identity on the frozen 240-dimensional strong subspace, while preserving the frozen layer22 downstream map, role-specific `||ΔX22||²` normalization, channel set, checkpoint, handoff, and equal-length prefix protocol.**

This is the narrowest authorized scientific step after the evidence freeze at:

`f5ef304701c494b83b070a566265d652c03444dd`.
