# ContraMamba K0-RVG Frozen-Strong Channel-Mass Breadth / Concentration
## Static Design Candidate

## 1. Status

**Phase:** new K0 scientific static-analysis design.

**Immediate parent evidence freeze:**

`236c15acba1badfcd19ef5bf2a9c236a28f4f399`

**Immediate parent validated result:**

`A1/B1_BROAD_POSITIVE_TOTAL_WITH_MAJORITY_PARENT_CONCORDANCE`

The immediate parent established that the frozen strong-subspace signed-interaction phenotype is broad across items rather than minority-driven.

The next scientific question is orthogonal:

> Across the fixed strong-240 channel population, is the population-level strong interaction contrast distributed broadly across channels, or concentrated into a comparatively small effective set of channels?

A linked question is:

> Is the positive supporting channel mass itself broad, or does a small subset carry most of the supporting mass while many channels merely share the positive sign?

This stage is **model-free**.

No new model forward, checkpoint load, handoff access, tokenizer, logits, task heads, training, intervention, PCA/SVD, learned projection, new channel selection, new item selection, alternate threshold search, or K1 work is authorized.

---

## 2. Scientific motivation

The K0 chain has already established:

1. layer22 post-update state is the dominant state-side carrier;
2. that state contrast is current-write dominated rather than carry dominated;
3. the write contrast is U-dominant with role-dependent U–D cancellation;
4. the layer22 U-path phenotype is selectively amplified by the fixed depthwise convolution;
5. the amplification is primarily lag0/current-token;
6. frozen strong kernel rows receive more normalized current-token hidden-difference energy in corr;
7. layer22 RMSNorm routing is mixed rather than a single-source explanation;
8. layer22 residual construction is mixed;
9. layer21 residual construction contains a strong-side interaction-dominant component;
10. the strong-side `R20×Y20` interaction is overwhelmingly alignment-driven;
11. within that interaction, positive same-sign co-contribution dominates cancellation relief at the population level;
12. that signed-interaction phenotype is broad across the 330-item cohort.

What is not yet known is whether the fixed 240 strong channels themselves share the population effect broadly or whether the effect is concentrated in a small number of channels.

This distinction matters before any deeper channel-localization step.

If the channel mass is broad, further top-channel archaeology would be scientifically misleading.

If it is concentrated, a later frozen-design localization can ask what fixed channel property distinguishes the concentrated support.

---

## 3. Current stage authority

This design becomes the sole scientific authority for the present channel-mass breadth/concentration stage once frozen.

It must be interpreted together with the already-frozen evidence identities below.

No previous exploratory ranking or informal top-channel list may substitute for this design.

---

## 4. Immediate parent evidence identity

Immediate parent evidence freeze:

`236c15acba1badfcd19ef5bf2a9c236a28f4f399`

Immediate parent implementation:

`8df86fa8d0302d9e97ac31e41190b3a613c76896`

Immediate parent validated report:

`reports/longterm_k0_rvg_frozen_strong_sign_contribution_itemwise_breadth_validated_evidence_analysis_report_candidate.md`

Expected report SHA256:

`6de60ce27f7a896a129cbec12208da2d719974f8752cb0403e99debceb64ff2c`

Immediate parent summary:

`reports/longterm_k0_rvg_frozen_strong_sign_contribution_itemwise_breadth_8df86fa_v1/summary.json`

Expected summary SHA256:

`4cbe22ea0e2355632347102fc47f72205b734ed0e5c0008869805fdadfa776bb`

The immediate parent is used to authenticate the current scientific chain and the fact that itemwise breadth has already been validated.

---

## 5. Frozen channel-level scientific input

The scientific input for the new channel distribution analysis is the already-frozen parent strong-channel validation artifact from:

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/layer20_y20_strong_interaction_geometry_channel_validation.jsonl`

Expected SHA256:

`111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942`

Expected rows:

`240`

Expected schema:

`k0-rvg-layer20-y20-strong-interaction-geometry-channel-validation-v1`

Expected downstream partition:

`strong`

The artifact already persists, for each frozen strong channel:

- `channel_index`;
- `kernel_magnitude_rank`;
- `corr_mean_signed_product`;
- `ctrl_mean_signed_product`;
- `parent_mean_d_ry20`;
- `reconstructed_mean_d_ry20`;
- parent reproduction residual;
- corr positive/negative/zero product counts;
- ctrl positive/negative/zero product counts.

No raw vectors are needed.

---

## 6. Frozen coordinate and population scope

Preserve exactly:

- strong channel population:
  `240`;
- common item population:
  `330`;
- current token only;
- relative coordinate:
  `k=2`;
- source block:
  `20`;
- target residual layer:
  `21`;
- downstream parent map:
  layer `22`;
- same frozen kernel-strength partition;
- same frozen matched/swapped role definitions.

No channel may be removed from the primary population.

No item may be removed.

No channel subset may replace the fixed 240-channel population.

---

## 7. Per-channel primary scalar

For frozen strong channel `j`, define:

`d_j = parent_mean_d_ry20_j`.

The channel artifact already validates:

`d_j = corr_mean_signed_product_j - ctrl_mean_signed_product_j`

within frozen numerical tolerance.

Interpretation:

- `d_j > 0`:
  channel `j` supports the positive corr strong-interaction contrast;
- `d_j < 0`:
  channel `j` opposes the positive corr strong-interaction contrast;
- `d_j = 0`:
  exact neutral channel.

The analyzer must use the persisted exact scalar.

No learned or smoothed channel score is permitted.

---

## 8. Population bridge

The channel decomposition must reproduce the frozen strong population mean interaction:

`Σ_j d_j = +0.760594723140676`

for all 240 strong channels.

Because `d_j` is already a per-channel mean over the 330 common items, summing over channels reconstructs the strong interaction population mean.

This is a blocking bridge.

Tolerance:

`5e-12`.

---

## 9. Channel sign breadth

Define:

`N_pos = # {j : d_j > 0}`

`N_neg = # {j : d_j < 0}`

`N_zero = # {j : d_j == 0}`.

All three must sum to:

`240`.

Define:

`F_pos = N_pos / 240`.

Primary sign-breadth classification:

### S1 — positive-channel majority

`N_pos > 120`.

### S2 — not positive-channel majority

`N_pos <= 120`.

The boundary `120` is exactly half of the fixed 240-channel population.

It is not fit from the data.

---

## 10. Signed channel mass

Define positive supporting mass:

`P_ch = Σ_j max(d_j, 0)`.

Define negative opposing mass:

`N_ch = Σ_j min(d_j, 0)`.

Then:

`G_ch = P_ch + N_ch`.

The analyzer must verify:

`G_ch = +0.760594723140676`

within tolerance.

Define opposition ratio:

`R_opp = |N_ch| / P_ch`

when:

`P_ch > 0`.

This reports how much negative channel mass offsets positive channel mass.

---

## 11. Absolute channel mass

Define:

`a_j = |d_j|`.

Define total absolute channel mass:

`A_ch = Σ_j a_j`.

Define normalized absolute mass weight:

`w_abs,j = a_j / A_ch`.

All 240 channels remain in the population.

Zero-mass channels, if any, receive weight zero.

---

## 12. Positive supporting-mass weights

For every channel define:

`p_j = max(d_j, 0)`.

Define:

`P_ch = Σ_j p_j`.

Define positive-mass normalized weight:

`w_pos,j = p_j / P_ch`.

Negative and zero channels receive positive-mass weight zero.

This is not a selected subset.

It is a full-population signed decomposition.

---

## 13. Absolute-mass HHI

Define absolute-mass Herfindahl concentration:

`HHI_abs = Σ_j w_abs,j^2`.

Define effective channel count:

`N_eff_abs = 1 / HHI_abs`.

Range:

`1 <= N_eff_abs <= 240`

when `A_ch > 0`.

Define normalized effective breadth:

`B_eff_abs = N_eff_abs / 240`.

Primary absolute-mass concentration classification:

### C1 — broad absolute channel mass

`N_eff_abs > 120`.

### C2 — concentrated absolute channel mass

`N_eff_abs <= 120`.

Again, `120` is exactly half of the fixed 240-channel population.

No fitted threshold is introduced.

---

## 14. Positive-mass HHI

Define:

`HHI_pos = Σ_j w_pos,j^2`.

Define positive-support effective channel count:

`N_eff_pos = 1 / HHI_pos`.

Define normalized positive effective breadth:

`B_eff_pos = N_eff_pos / 240`.

Primary positive-support concentration classification:

### P1 — broad positive supporting mass

`N_eff_pos > 120`.

### P2 — concentrated positive supporting mass

`N_eff_pos <= 120`.

This is the second primary scientific axis.

---

## 15. Primary two-axis result

The primary stage result combines:

1. sign breadth:
   `S1` versus `S2`;
2. positive supporting-mass breadth:
   `P1` versus `P2`.

Interpretation:

### S1/P1

Positive support is both sign-broad and mass-broad.

### S1/P2

Many channels support corr, but the supporting mass is concentrated in a smaller effective set.

### S2/P1

Not expected under ordinary weight structure, but retained formally.

### S2/P2

Positive support is neither sign-broad nor mass-broad.

The absolute-mass result `C1/C2` is reported as an independent supporting axis.

---

## 16. Fixed top-fraction concentration diagnostics

To make concentration interpretable without defining a new scientific subset, compute cumulative mass captured by fixed population fractions.

Sort channels only for descriptive concentration accounting.

This ranking does not authorize a new channel subset.

For absolute mass, compute cumulative shares for:

- top `1%`:
  fixed count `3` channels;
- top `5%`:
  fixed count `12` channels;
- top `10%`:
  fixed count `24` channels;
- top `25%`:
  fixed count `60` channels;
- top `50%`:
  fixed count `120` channels.

The integer counts are preregistered from the fixed population size 240.

For positive supporting mass, compute the same fixed channel counts over channels ranked by `p_j`.

No threshold may be changed after seeing results.

---

## 17. Half-mass and 80%-mass channel counts

Define:

`K_abs_50`

as the minimum number of channels, ordered by descending `a_j`, required to reach at least 50% of `A_ch`.

Define:

`K_abs_80`

analogously for 80%.

Define:

`K_pos_50`

as the minimum number of channels, ordered by descending `p_j`, required to reach at least 50% of `P_ch`.

Define:

`K_pos_80`

analogously for 80%.

These are descriptive concentration statistics.

They do not create an authorized follow-up subset.

---

## 18. Gini concentration

Compute Gini coefficient over the 240 nonnegative absolute masses:

`a_j`.

Call it:

`Gini_abs`.

Compute Gini coefficient over the 240 nonnegative positive-support masses:

`p_j`.

Call it:

`Gini_pos`.

Use the standard finite-population sorted-vector definition.

Bounds:

`0 <= Gini <= 1`.

Interpretation:

- lower:
  more even;
- higher:
  more concentrated.

No binary scientific outcome is based solely on Gini.

HHI-derived effective counts remain primary.

---

## 19. Channel sign stability across items

The frozen channel artifact contains, per channel:

- corr positive-product count;
- corr negative-product count;
- ctrl positive-product count;
- ctrl negative-product count.

These counts sum to 330 per role because zero-product counts were previously frozen at zero.

For each channel define corr sign occupancy:

`q_corr,j = corr_positive_product_count_j / 330`.

Define ctrl sign occupancy:

`q_ctrl,j = ctrl_positive_product_count_j / 330`.

Define occupancy shift:

`Δq_j = q_corr,j - q_ctrl,j`.

This is secondary.

It asks whether channels with positive mean `d_j` also tend to shift toward more positive signed-product occupancy in corr.

No correlation-based causal claim is permitted.

---

## 20. Occupancy-direction breadth

Classify every channel by:

`Δq_j`.

Report counts:

`N_dq_pos`

`N_dq_neg`

`N_dq_zero`.

Also cross-tabulate sign of:

`d_j`

against sign of:

`Δq_j`.

This distinguishes:

- positive mean contrast accompanied by more-positive occupancy;
- positive mean contrast despite occupancy moving the opposite way;
- negative mean contrast with corresponding occupancy shift.

The cross-tab is descriptive only.

---

## 21. Kernel-magnitude rank use

The frozen artifact includes:

`kernel_magnitude_rank`.

This rank is inherited from the pre-existing strong-partition construction.

It may be used only for **diagnostic rank-distribution summaries**.

Allowed:

- median kernel rank among `d_j > 0`;
- median kernel rank among `d_j < 0`;
- Spearman-free rank bins fixed by the 240-channel strong population:
  top 60 ranks,
  ranks 61–120,
  ranks 121–180,
  ranks 181–240.

Not allowed:

- selecting a new top-k population;
- tuning a rank cutoff from the result;
- claiming kernel magnitude causally determines `d_j`;
- promoting a rank bin as a new primary population.

---

## 22. Fixed kernel-rank quartile diagnostics

Using the frozen rank only, partition the 240 channels into four fixed rank quartiles:

- Q1:
  ranks 1–60;
- Q2:
  ranks 61–120;
- Q3:
  ranks 121–180;
- Q4:
  ranks 181–240.

For each quartile report:

- channel count;
- sum of `d_j`;
- positive mass;
- negative mass;
- absolute mass;
- fraction of total positive mass;
- fraction of total absolute mass.

This is diagnostic.

The primary breadth/concentration outcome remains over all 240 channels.

---

## 23. No post-hoc top-channel scientific claim

Although fixed cumulative concentration diagnostics require sorting by observed mass, no ranked channel list is itself a scientific result.

Do not emit:

- "top mechanistic channels";
- "critical channels";
- "key causal channels";
- a promoted top-k subset.

Any future study focused on a subset of channels requires a new frozen design and must justify the subset independently.

---

## 24. No item heterogeneity reopening

The immediate parent established:

- positive total:
  `295/330`;
- parent-concordant:
  `236/330`;
- positive-component dominant:
  `270/330`.

The present stage does not reopen item subgroup analysis.

All channel statistics are population means over the already-fixed common 330 items.

---

## 25. Parent channel identity checks

For every frozen channel verify:

`reconstructed_mean_d_ry20 = corr_mean_signed_product - ctrl_mean_signed_product`

within:

`5e-12`.

Verify:

`parent_mean_d_ry20 = reconstructed_mean_d_ry20`

within:

`5e-12`.

Verify persisted:

`parent_reproduction_abs_residual <= 5e-12`.

Any failure blocks.

No failing channel may be dropped.

---

## 26. Role sign-count checks

For each frozen channel and each role verify:

`positive_product_count + negative_product_count + zero_product_count = 330`.

Verify all counts are integers in:

`[0, 330]`.

Do not assume zero-product count is zero without checking.

Persist aggregate zero-count totals in the summary.

---

## 27. Frozen provenance authentication

Before analysis authenticate:

1. current branch;
2. current authority design commit;
3. immediate parent evidence freeze `236c15a...` as ancestor;
4. channel evidence freeze `431e8fa...` as ancestor;
5. immediate parent validated report SHA;
6. immediate parent breadth summary SHA;
7. strong-channel validation SHA;
8. exact 240-row count;
9. exact schema for every channel row.

The analyzer must read channel scientific input from the frozen Git object at `431e8fa...`, not an unauthenticated alternate copy.

The current worktree copy must match the frozen Git bytes.

---

## 28. Static implementation boundary

The analyzer must use Python standard library only.

Do not import:

- `torch`;
- `transformers`;
- `numpy`;
- model modules;
- training modules.

Do not open:

- checkpoint files;
- handoff ZIPs.

Do not execute:

- model forwards;
- tokenization;
- logits;
- task heads;
- training;
- interventions;
- PCA;
- SVD;
- learned projections.

---

## 29. Static preflight

Static preflight must:

1. authenticate authority and parent evidence;
2. authenticate the frozen channel artifact;
3. parse exactly 240 rows;
4. validate all channel identities;
5. reproduce the strong interaction total;
6. compute sign breadth and concentration metrics in memory;
7. emit no scientific evidence files;
8. report `model_forward_count = 0`.

No separate preflight authority document is required.

---

## 30. Intended execution outputs

The eventual execution should emit exactly:

### Channel metrics

`frozen_strong_channel_mass_breadth_metrics.jsonl`

One row per frozen strong channel.

Persist:

- channel index;
- kernel magnitude rank;
- `d_j`;
- sign label;
- positive mass;
- absolute mass;
- normalized positive weight;
- normalized absolute weight;
- corr/ctrl sign occupancy;
- occupancy shift;
- fixed rank quartile;
- bridge residuals.

No raw vectors.

### Summary

`summary.json`

Persist:

- sign counts/fractions;
- `P_ch`, `N_ch`, `G_ch`;
- opposition ratio;
- `HHI_abs`, `N_eff_abs`, `B_eff_abs`;
- `HHI_pos`, `N_eff_pos`, `B_eff_pos`;
- `S1/S2`, `P1/P2`, `C1/C2`;
- fixed top-fraction cumulative shares;
- half-mass/80%-mass counts;
- Gini diagnostics;
- occupancy-direction cross-tab;
- rank-quartile diagnostics;
- exact bridge maxima;
- execution-boundary flags.

### Static-analysis manifest

`static_analysis_manifest.json`

Persist:

- authority identity;
- immediate parent evidence identity;
- channel evidence identity;
- implementation identity;
- input/output hashes;
- population scope;
- tolerances;
- forbidden-operation flags.

No additional output class is required.

---

## 31. Primary scientific readout

Primary readout:

`(S sign breadth, P positive-mass effective breadth)`.

Secondary supporting classification:

`C absolute-mass effective breadth`.

The scientific answer should distinguish:

- many positive channels versus few;
- broad positive mass versus concentrated positive mass;
- broad total absolute activity versus concentrated total absolute activity.

These are different questions and must not be conflated.

---

## 32. Interpretation examples

### S1/P1/C1

Many channels support corr and both positive mass and absolute mass are broadly distributed.

Interpretation:

**broad channel-distributed phenotype**.

### S1/P2/C2

Many channels have positive sign, but a small effective set carries most mass.

Interpretation:

**sign-broad but mass-concentrated phenotype**.

### S2/P2/C2

Both sign and mass are concentrated.

Interpretation:

**channel-concentrated phenotype**.

Mixed combinations must be reported literally rather than forced into a single label.

---

## 33. Statistical restraint

This stage is deterministic descriptive accounting over a fixed frozen channel population.

Do not add:

- bootstrap confidence intervals;
- hypothesis tests;
- p-values;
- parametric assumptions;
- arbitrary null models.

The goal is structural localization, not inferential population sampling.

---

## 34. Scientific limits

This stage cannot establish:

- causal importance of any channel;
- necessity of any channel;
- sufficiency of any channel;
- that high `|d_j|` channels would matter under intervention;
- that kernel strength causes channel mass;
- generalization outside the frozen strong-240 set;
- generalization outside common-330 items;
- generalization outside `k=2`;
- generalization outside layer20→21 interaction geometry.

No causal intervention is authorized.

---

## 35. Stop conditions

Block immediately if:

- authority design mismatch;
- immediate parent evidence mismatch;
- channel evidence SHA mismatch;
- row count not 240;
- duplicate channel index;
- duplicate kernel rank;
- rank not exactly a permutation of 1..240 within the strong population;
- per-channel bridge failure;
- role count closure failure;
- population total failure;
- unexpected worktree change;
- any model/checkpoint/handoff dependency appears;
- any K1 file is modified.

Do not relax tolerances to obtain PASS.

---

## 36. Decision value

If positive supporting mass is broad (`P1`), deeper top-channel localization should stop by default because the phenotype is distributed.

If positive supporting mass is concentrated (`P2`), a later K0 stage may be justified to ask which **already-frozen channel property** accounts for concentration, but only under a new static design.

If sign breadth is broad while mass is concentrated (`S1/P2`), the next question should distinguish widespread weak support from concentrated strong support without promoting an arbitrary top-k subset.

---

## 37. No K1 transition

This stage remains within K0-RVG.

The existing unrelated K1 files are not authority for this stage.

No K1 execution, implementation, validation, or interpretation is authorized.

---

## 38. Final design statement

This stage performs a deterministic, model-free, all-channel analysis over the frozen strong-240 channel artifact.

It asks:

**Is the already-validated, itemwise-broad strong interaction phenotype also broad across channels, or is its population mass concentrated into a comparatively small effective set of channels?**

The answer must use all 240 frozen strong channels.

No top-k subset may become the primary population.

No new model execution is scientifically necessary or authorized.
