# ContraMamba K0-RVG Frozen-Strong Alignment Sign / Co-Contribution Localization
## Static Design Candidate

## 1. Status

**Phase:** new K0 scientific static analysis design.

**Immediate parent evidence freeze:**

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`

**Immediate parent implementation / execution commit:**

`0b1168182a265bc405652d2ce519f63310433c3e`

**Immediate parent validated report:**

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_validated_evidence_analysis_report_candidate.md`

This stage introduces **no new model execution**.

The required evidence is already frozen in the parent item and summary artifacts.

The scientific question is:

> Within the frozen strong-240 `R20×Y20` interaction, is corr's larger alignment-driven interaction caused primarily by increased positive same-sign co-contribution, by reduced negative opposite-sign cancellation, or by a mixture of both?

No new forward pass, tokenizer, logits, task heads, training, intervention, PCA/SVD, learned geometry, channel search, item search, or K1 is authorized.

---

## 2. Immediate parent result

The parent evidence established:

`D_ry20,strong = +0.760594723140676`.

Its exact magnitude/alignment decomposition was:

- `Q_A = +0.01818378200815921`;
- `Q_B = +0.009759485141568866`;
- `Q_C = +0.7326514559909479`.

Absolute shares:

- `Q_A`:
  `2.3907320751679763%`;
- `Q_B`:
  `1.2831386866937017%`;
- `Q_C`:
  `96.32612923813832%`.

The validated classification was:

**Outcome C — alignment dominant.**

The next question therefore concerns the fixed-coordinate structure of that strong-subspace interaction/alignment, not another magnitude decomposition.

---

## 3. Why no new model execution is needed

The parent runner already persisted, for every common-330 item and each role:

- `positive_interaction_mass`;
- `negative_interaction_mass`;
- `same_sign_channel_count`;
- `opposite_sign_channel_count`;
- `zero_product_channel_count`;
- role-level strong interaction `I`.

The parent summary also freezes their population aggregates.

Therefore the sign/co-contribution question is a deterministic static transform of already-frozen scientific evidence.

Re-running the model would add no new information and would unnecessarily enlarge the execution surface.

---

## 4. Frozen artifacts

Use only artifacts frozen at:

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`.

Required exact identities:

### Parent item metrics

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/layer20_y20_strong_interaction_geometry_item_metrics.jsonl`

SHA256:

`e7002e03bd170c05ea068e70eb32cc1f7a8b0d4f569ac44e531a42a8a4e1ebfe`

### Parent strong-channel validation

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/layer20_y20_strong_interaction_geometry_channel_validation.jsonl`

SHA256:

`111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942`

### Parent summary

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/summary.json`

SHA256:

`35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9ea9ef66a80798`

### Parent execution manifest

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/execution_manifest.json`

SHA256:

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fe`

### Parent validated report

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_validated_evidence_analysis_report_candidate.md`

SHA256:

`6e33951e0fe02ba6e82da6f59f786034ddbda869d2f0a2f592b9a51467477163`

---

## 5. Frozen population and scope

Preserve exactly:

- common DDSSSSS cohort:
  `330`;
- current token only;
- relative coordinate:
  `k=2`;
- frozen strong partition:
  `240` channels;
- source block:
  `20`;
- target residual layer:
  `21`;
- downstream parent map:
  layer `22`;
- same matched/swapped role definitions;
- same frozen interaction vector definition.

No weak-channel result is promoted in this stage.

No new subset of the 240 strong channels may be selected.

---

## 6. Exact role-level sign decomposition

For one item-role branch `b`, the parent strong-channel interaction coordinate is:

`u_b,j = 2 x_b,j y_b,j`

for every frozen strong channel `j`.

Define positive mass:

`P_b = Σ_j max(u_b,j, 0)`.

Define negative mass:

`N_b = Σ_j min(u_b,j, 0)`.

Then:

`P_b >= 0`

and:

`N_b <= 0`.

The parent interaction is exactly:

`I_b = P_b + N_b`.

No threshold other than exact sign is introduced.

---

## 7. Exact corr–ctrl decomposition

For one paired item define:

`ΔI = I_corr - I_ctrl`.

Define:

`ΔP = P_corr - P_ctrl`.

Define:

`ΔN = N_corr - N_ctrl`.

Then exactly:

`ΔI = ΔP + ΔN`.

Interpretation:

- `ΔP > 0` means corr has more positive same-sign co-contribution mass;
- `ΔN > 0` means corr has less negative opposite-sign cancellation;
- `ΔN < 0` means corr suffers more negative cancellation.

Because `N` itself is negative, a positive `ΔN` is correctly interpreted as **cancellation relief**.

---

## 8. Population target

The parent strong interaction population mean is frozen as:

`mean(ΔI) = +0.760594723140676`.

The new static analysis must reproduce this exactly from:

`mean(ΔP) + mean(ΔN)`.

The parent summary already freezes role-level means:

### Positive interaction mass

corr:

`2.267583155684786`

ctrl:

`1.7361014562993915`.

Therefore provisional exact difference:

`mean(ΔP)
 = 2.267583155684786
 - 1.7361014562993915
 = +0.5314816993853945`.

### Negative interaction mass

corr:

`-0.5449066994977081`

ctrl:

`-0.7740197232529892`.

Therefore provisional exact difference:

`mean(ΔN)
 = -0.5449066994977081
 - (-0.7740197232529892)
 = +0.2291130237552811`.

Their sum is:

`+0.7605947231406756`

up to floating summation representation.

The static analyzer must recompute the means directly from the frozen 330 item rows rather than treating these summary-derived arithmetic values as primary evidence.

---

## 9. Primary scientific components

The two scientific components are:

### Positive co-contribution gain

`G_pos = mean(ΔP)`.

### Negative-cancellation relief

`G_relief = mean(ΔN)`.

The parent total is:

`G_total = mean(ΔI)`.

Exact population closure:

`G_total = G_pos + G_relief`.

---

## 10. Absolute attribution shares

Define:

`M_abs = |G_pos| + |G_relief|`.

Then:

`S_pos = |G_pos| / M_abs`

and:

`S_relief = |G_relief| / M_abs`.

These shares describe the signed-mass decomposition.

They do not replace the parent cosine-alignment result.

If both components are positive, they directly represent reinforcing fractions of the strong interaction advantage.

---

## 11. Count diagnostics

For each role and item, preserve the parent exact counts:

- same-sign channel count;
- opposite-sign channel count;
- zero-product count.

Define paired differences:

`Δn_same
 = n_same,corr - n_same,ctrl`

and:

`Δn_opp
 = n_opp,corr - n_opp,ctrl`.

Because all parent items have zero-product count `0`, the exact identity is:

`Δn_same + Δn_opp = 0`.

These counts are diagnostics only.

They do **not** determine the main scientific classification because signed mass can change without count changes.

---

## 12. Mean per-channel signed-mass diagnostics

To distinguish "more channels" from "stronger contribution per channel", define for every item-role:

`Pbar_b = P_b / n_same,b`

when `n_same,b > 0`.

Define cancellation magnitude:

`K_b = -N_b >= 0`.

Define:

`Kbar_b = K_b / n_opp,b`

when `n_opp,b > 0`.

The parent evidence guarantees nonzero same/opposite counts for all observed rows, but the analyzer must still block if a denominator is zero.

Report paired differences:

`ΔPbar = Pbar_corr - Pbar_ctrl`

and:

`ΔKbar = Kbar_corr - Kbar_ctrl`.

Interpretation:

- positive `ΔPbar`:
  stronger positive co-contribution per same-sign channel;
- negative `ΔKbar`:
  weaker cancellation magnitude per opposite-sign channel.

These are secondary diagnostics.

---

## 13. Channel-level frozen bridge

The parent strong-channel artifact contains, for every frozen strong channel:

- `corr_mean_signed_product`;
- `ctrl_mean_signed_product`;
- their difference `reconstructed_mean_d_ry20`;
- corr/ctrl positive/negative/zero product counts.

The new static analyzer must authenticate all 240 rows and verify:

`corr_mean_signed_product
 - ctrl_mean_signed_product
 = reconstructed_mean_d_ry20`

within frozen tolerance.

No channel is ranked or selected for the primary claim.

Channel-level summaries may report total positive-mean and negative-mean contributions across the fixed 240 rows, but no top-k or thresholded subset is allowed.

---

## 14. Static item-level validation

For every frozen common-330 item verify:

`I_corr = P_corr + N_corr`

`I_ctrl = P_ctrl + N_ctrl`

and:

`reconstructed_delta_I
 = ΔP + ΔN`.

Also reproduce the frozen parent:

`reconstructed_delta_ry20_strong`.

All item-level identities are blocking.

---

## 15. Parent aggregate validation

The analyzer must reproduce from the 330 frozen item rows:

- parent strong interaction mean:
  `+0.760594723140676`;
- corr positive interaction mass mean:
  `2.267583155684786`;
- ctrl positive interaction mass mean:
  `1.7361014562993915`;
- corr negative interaction mass mean:
  `-0.5449066994977081`;
- ctrl negative interaction mass mean:
  `-0.7740197232529892`;
- corr same-sign count mean:
  `140.5818181818182`;
- ctrl same-sign count mean:
  `136.03030303030303`;
- corr opposite-sign count mean:
  `99.41818181818182`;
- ctrl opposite-sign count mean:
  `103.96969696969697`.

These values are validation targets, not free parameters.

---

## 16. Scientific outcome classes

### Outcome A: positive co-contribution dominant

`|G_pos| > |G_relief|`

and `G_pos` is the largest absolute scientific component.

Interpretation:

corr's stronger strong-subspace interaction is primarily associated with increased positive same-sign co-contribution.

### Outcome B: cancellation-relief dominant

`|G_relief| > |G_pos|`

and `G_relief` is the largest absolute scientific component.

Interpretation:

corr's stronger interaction is primarily associated with reduced negative opposite-sign cancellation.

### Outcome C: mixed

Neither term alone gives the primary descriptive account in a scientifically useful way, or the terms materially oppose one another.

Report exact signed mixture.

No arbitrary dominance threshold is introduced.

---

## 17. Directional refinement

Regardless of classification, report whether:

- `G_pos > 0`;
- `G_relief > 0`;
- both reinforce;
- one opposes the other.

Also report count and mean-per-channel diagnostics to determine whether the positive co-contribution difference is associated more with:

- more same-sign channels;
- stronger average same-sign contribution;
- or both.

Likewise determine whether cancellation relief is associated more with:

- fewer opposite-sign channels;
- weaker average cancellation magnitude;
- or both.

These are descriptive decompositions, not causal claims.

---

## 18. Expected provisional pattern

From the already-frozen parent summary alone, the provisional population arithmetic is:

`G_pos ≈ +0.5314816993853945`

`G_relief ≈ +0.2291130237552811`.

Thus both appear to reinforce.

The provisional absolute shares are approximately:

- positive co-contribution gain:
  `69.88%`;
- cancellation relief:
  `30.12%`.

This is **not yet promoted as a new frozen scientific conclusion**.

The static analyzer must independently recompute the decomposition from item-level frozen evidence and validate all bridges.

---

## 19. No new execution boundary

This stage is static only.

The analyzer must not:

- import model code;
- load the checkpoint;
- open the handoff ZIP;
- invoke Transformers;
- invoke a tokenizer;
- execute model forwards;
- use GPU;
- execute task heads;
- read logits;
- train;
- intervene;
- use PCA/SVD;
- fit any learned projection;
- perform post-hoc search.

Any such action is a blocker.

---

## 20. Intended static analyzer outputs

A later implementation may persist only:

1. `330` item-level scalar decomposition rows;
2. one summary JSON;
3. one static-analysis manifest JSON.

A separate 240-channel output file is optional and unnecessary if all required channel bridges are validated read-only against the frozen parent artifact.

No raw vector data exists or is required.

---

## 21. Item output fields

For each item include:

- parent interaction total;
- `P_corr`;
- `P_ctrl`;
- `N_corr`;
- `N_ctrl`;
- `ΔP`;
- `ΔN`;
- reconstructed `ΔI`;
- parent reproduction residual;
- corr/ctrl same-sign counts;
- corr/ctrl opposite-sign counts;
- corr/ctrl zero counts;
- `Δn_same`;
- `Δn_opp`;
- `Pbar_corr`;
- `Pbar_ctrl`;
- `ΔPbar`;
- `Kbar_corr`;
- `Kbar_ctrl`;
- `ΔKbar`;
- exact closure residuals.

---

## 22. Summary output fields

Report:

- `G_total`;
- `G_pos`;
- `G_relief`;
- absolute mass;
- absolute shares;
- signed-to-total ratios;
- largest absolute scientific component;
- role-level positive/negative mass aggregates;
- role-level sign-count aggregates;
- per-channel mean positive contribution diagnostics;
- per-channel mean cancellation magnitude diagnostics;
- maximum item-level closure residual;
- parent aggregate reproduction residuals;
- channel bridge residuals.

---

## 23. Numerical tolerances

Use frozen numerical standards:

- parent scalar relative tolerance:
  `1e-13`;
- parent scalar absolute tolerance:
  `1e-13`;
- identity absolute tolerance:
  `5e-12`;
- channel bridge absolute tolerance:
  `5e-12`.

Do not weaken tolerances after observing the data.

---

## 24. Blocking conditions

Block scientific promotion if any occurs:

1. parent evidence commit mismatch;
2. any parent artifact SHA mismatch;
3. item count is not 330;
4. strong-channel count is not 240;
5. `I != P+N` beyond tolerance;
6. `ΔI != ΔP+ΔN` beyond tolerance;
7. parent item interaction reproduction fails;
8. parent aggregate reproduction fails;
9. channel bridge fails;
10. any new model execution occurs;
11. any learned or post-hoc selection method is introduced.

Unexpected scientific outcome is not a validation failure.

---

## 25. Scientific interpretation boundary

Passing this stage may support:

> corr's stronger frozen-strong interaction/alignment is statically localized to increased positive same-sign co-contribution, reduced negative cancellation, or a validated mixture of both.

It may not support:

- causal necessity;
- causal sufficiency;
- semantic channel claims;
- learned direction claims;
- task-performance claims;
- cross-seed or model-general claims;
- K1.

---

## 26. Next-step rule after validated evidence

If positive co-contribution dominates, the next K0 question should determine whether that increase is mainly count-driven or per-channel-strength-driven across the fixed 240 channels.

If cancellation relief dominates, the next K0 question should determine whether relief is mainly fewer opposite-sign channels or weaker negative mass per opposite-sign channel.

If mixed, follow the largest scientifically discriminative unresolved term while preserving the frozen strong 240 set.

Do not select top-k channels.

Do not transition to K1 automatically.

---

## 27. Static-design conclusion

The next K0 stage is:

**a model-free static decomposition of the already-frozen strong `R20×Y20` interaction into positive same-sign co-contribution gain and negative opposite-sign cancellation relief, using only the parent item/channel/summary artifacts frozen at `431e8faa6e5c82a20d87f532b4ab960fcf641ec2`, with exact item-, aggregate-, and channel-level bridges and no new execution.**
