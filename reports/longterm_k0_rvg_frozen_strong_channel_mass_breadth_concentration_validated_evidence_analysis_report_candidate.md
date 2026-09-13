# ContraMamba K0-RVG Frozen-Strong Channel-Mass Breadth / Concentration
## Validated Evidence Analysis Report Candidate

## 1. Status

**Stage:** validated K0 scientific static analysis.

**Primary validated classification:**

`S1/P2`

with supporting axis:

`C2`.

Expanded labels:

- `S1_POSITIVE_CHANNEL_MAJORITY`;
- `P2_CONCENTRATED_POSITIVE_SUPPORTING_MASS`;
- `C2_CONCENTRATED_ABSOLUTE_CHANNEL_MASS`.

Validated concise interpretation:

**The frozen strong-channel phenotype is sign-broad only in the weak sense of a bare positive-channel majority, while both positive supporting mass and total absolute mass are strongly concentrated into a much smaller effective channel set.**

This is an observational/algebraic static localization.

It is not a causal intervention result.

---

## 2. Scientific question

Across the fixed strong-240 channel population, is the population-level strong interaction contrast distributed broadly across channels, or concentrated into a comparatively small effective set of channels?

Linked question:

Is positive support itself broad in mass, or do many weakly positive channels coexist with a much smaller set carrying most of the positive interaction mass?

---

## 3. Authority chain

### Static design freeze

Commit:

`86287c07546024315d5644de64d9e8d3cd206189`

File:

`reports/longterm_k0_rvg_frozen_strong_channel_mass_breadth_concentration_static_design_candidate.md`

SHA256:

`015867d8fd034391248a4f51aeea01c7f1abd62fecfef5142d635273f71677f1`

Git blob:

`f05209580e6c2885b020c91996dab79482f78589`

### Static analyzer implementation freeze

Commit:

`90737ca6962032e6fdcfeb3cdaa570b4c73790b7`

File:

`scripts/longterm_k0_rvg_frozen_strong_channel_mass_breadth_concentration_static_analysis.py`

SHA256:

`f582b86ebc69435d459f4a0a62255fc81eaea1843df2d04777acf2b7a89baa2b`

---

## 4. Immediate parent evidence

Immediate parent evidence freeze:

`236c15acba1badfcd19ef5bf2a9c236a28f4f399`

Immediate parent validated result:

`A1/B1_BROAD_POSITIVE_TOTAL_WITH_MAJORITY_PARENT_CONCORDANCE`

Immediate parent conclusion:

**The frozen strong-subspace signed-interaction phenotype is broad across the common-330 item cohort rather than minority-driven.**

Key parent counts:

- positive total:
  `295/330`;
- parent-concordant positive-co-contribution-led:
  `236/330`;
- positive-component dominant:
  `270/330`.

The present stage does not reopen itemwise heterogeneity.

It asks the orthogonal channel-population question.

---

## 5. Frozen channel-level input

Channel evidence freeze:

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`

Input file:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/layer20_y20_strong_interaction_geometry_channel_validation.jsonl`

SHA256:

`111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942`

Expected rows:

`240`

Expected schema:

`k0-rvg-layer20-y20-strong-interaction-geometry-channel-validation-v1`

Expected partition:

`strong`.

For each frozen channel `j`, the input persisted:

- channel index;
- frozen kernel-magnitude rank;
- corr mean signed product;
- ctrl mean signed product;
- parent mean channel contribution;
- reconstructed mean channel contribution;
- parent reproduction residual;
- corr sign occupancy counts;
- ctrl sign occupancy counts.

No raw vectors were required.

---

## 6. Frozen scope

The analysis preserves exactly:

- strong channel population:
  `240`;
- common item population:
  `330`;
- relative coordinate:
  `k=2`;
- source block:
  `20`;
- target residual:
  layer `21`;
- downstream map:
  layer `22`;
- frozen matched/swapped role definitions;
- frozen strong partition.

No channel was removed.

No item was removed.

No alternate threshold was searched.

No result-dependent subset was promoted.

---

## 7. Execution boundary

The validated execution is model-free.

Flags:

- `new_model_execution = False`;
- `model_forward_count = 0`;
- `checkpoint_loaded = False`;
- `handoff_opened = False`;
- `transformers_imported = False`;
- `tokenizer_invoked = False`;
- `logits_read = False`;
- `task_heads_executed = False`;
- `training_executed = False`;
- `causal_intervention_executed = False`;
- `pca_svd_or_learned_geometry_executed = False`;
- `posthoc_subset_search_executed = False`;
- `new_item_selection = False`;
- `new_channel_selection = False`;
- `raw_vectors_read_or_persisted = False`.

Therefore the result is a deterministic transform of already-frozen evidence.

---

## 8. Per-channel scalar

For each frozen strong channel `j`:

`d_j = parent_mean_d_ry20_j`.

The frozen channel artifact validates:

`d_j = corr_mean_signed_product_j - ctrl_mean_signed_product_j`.

Interpretation:

- `d_j > 0`:
  supports the positive corr interaction contrast;
- `d_j < 0`:
  opposes the positive corr interaction contrast;
- `d_j = 0`:
  neutral.

The present stage uses all 240 channels.

---

## 9. Population bridge

Validated population reconstruction:

`Σ_j d_j = 0.760594723140676`.

Expected parent total:

`0.760594723140676`.

Residual:

`0.0`.

Thus the 240-channel decomposition exactly reproduces the parent strong interaction population mean at reported precision.

---

## 10. Channel sign breadth

Validated counts:

- positive channels:
  `122`;
- negative channels:
  `118`;
- exact zero:
  `0`.

Positive fraction:

`122 / 240 = 0.5083333333333333`

or:

`50.83333333333333%`.

Since:

`122 > 120`

the preregistered sign-breadth axis is:

`S1_POSITIVE_CHANNEL_MAJORITY`.

However, this majority is minimal:

only two channels above the exact half-population boundary.

Therefore `S1` must not be described as strong sign dominance.

The correct interpretation is:

**bare or marginal positive-channel majority**.

---

## 11. Signed channel mass

Positive supporting mass:

`P_ch = 1.2908845830181823`.

Negative opposing mass:

`N_ch = -0.5302898598775064`.

Net:

`G_ch = 0.7605947231406759`.

Opposition ratio:

`|N_ch| / P_ch = 0.4107957185743516`.

Thus negative channel mass offsets approximately:

`41.07957185743516%`

of the positive supporting mass before netting.

This is substantial opposition.

The positive net is therefore not produced by uniformly positive channels.

---

## 12. Positive supporting-mass concentration

Positive-mass HHI:

`0.034962517441580285`.

Effective positive-support channel count:

`N_eff_pos = 28.60205938176289`.

Normalized effective breadth:

`B_eff_pos = 0.11917524742401203`.

Thus positive support behaves, in HHI-equivalent terms, like approximately:

`28.60`

equally weighted channels out of 240.

That is only:

`11.917524742401203%`

of the frozen strong population.

Because:

`28.60 <= 120`

the preregistered axis is:

`P2_CONCENTRATED_POSITIVE_SUPPORTING_MASS`.

This is a strong concentration result.

---

## 13. Absolute-mass concentration

Absolute-mass HHI:

`0.019044649335064234`.

Effective absolute-mass channel count:

`N_eff_abs = 52.50818654659294`.

Normalized effective breadth:

`B_eff_abs = 0.2187841106108039`.

Thus the full signed activity magnitude behaves like approximately:

`52.51`

equally weighted channels out of 240.

That is:

`21.87841106108039%`

of the strong population.

Because:

`52.51 <= 120`

the supporting axis is:

`C2_CONCENTRATED_ABSOLUTE_CHANNEL_MASS`.

Therefore concentration is not limited to the positive side.

The overall absolute interaction mass is also concentrated.

---

## 14. Primary classification

Primary combined result:

`S1/P2`.

Supporting axis:

`C2`.

Validated interpretation:

**Many channels do not share a dominant positive sign pattern; rather, the sign split is almost balanced. Yet the positive net survives because positive supporting mass is distributed very unevenly, with a comparatively small effective set carrying much of the magnitude.**

A concise label is:

**marginal sign breadth with strong mass concentration**.

---

## 15. Positive-mass fixed cumulative shares

Fixed top-count concentration diagnostics:

### Top 3 channels

Positive mass share:

`0.23197699890411336`

or:

`23.197699890411336%`.

### Top 12 channels

Positive mass share:

`0.5544719379442832`

or:

`55.44719379442832%`.

### Top 24 channels

Positive mass share:

`0.728382799256356`

or:

`72.8382799256356%`.

### Top 60 channels

Positive mass share:

`0.9260954666137191`

or:

`92.60954666137191%`.

### Top 120 channels

Positive mass share:

`0.9998791004380092`

or:

`99.98791004380092%`.

These counts were preregistered as fixed population fractions and are descriptive concentration diagnostics only.

They do not create a new authorized channel subset.

---

## 16. Positive-mass half and 80% counts

Minimum channels required for 50% of positive supporting mass:

`K_pos_50 = 10`.

Minimum channels required for 80%:

`K_pos_80 = 32`.

Therefore:

- `10/240 = 4.1667%` of channels carry at least half of positive support;
- `32/240 = 13.3333%` carry at least 80%.

This is directly consistent with the low:

`N_eff_pos = 28.60`.

---

## 17. Positive-mass Gini

Positive-mass Gini:

`Gini_pos = 0.8433521328282656`.

This is a high inequality value.

The preregistered binary classification is not based on Gini alone, but the Gini strongly agrees with HHI-derived concentration.

Therefore the positive-side concentration is not an artifact of one concentration statistic.

---

## 18. Absolute-mass fixed cumulative shares

### Top 3 channels

Absolute mass share:

`0.16442989998476362`

or:

`16.442989998476362%`.

### Top 12 channels

Absolute mass share:

`0.39302071210181483`

or:

`39.30207121018148%`.

### Top 24 channels

Absolute mass share:

`0.5270317618472005`

or:

`52.70317618472005%`.

### Top 60 channels

Absolute mass share:

`0.746900522315983`

or:

`74.6900522315983%`.

### Top 120 channels

Absolute mass share:

`0.9080836461010506`

or:

`90.80836461010506%`.

Thus total signed activity is also substantially concentrated, though less strongly than positive-only mass.

---

## 19. Absolute-mass half and 80% counts

Minimum channels required for 50% of absolute mass:

`K_abs_50 = 22`.

Minimum channels required for 80%:

`K_abs_80 = 75`.

Thus only:

`22/240 = 9.1667%`

of channels carry at least half of all absolute channel mass.

The 80% absolute-mass threshold requires:

`75/240 = 31.25%`

of channels.

---

## 20. Absolute-mass Gini

Absolute-mass Gini:

`Gini_abs = 0.6573581715076158`.

This is lower than:

`Gini_pos = 0.8433521328282656`

but still indicates substantial inequality.

The positive supporting mass is therefore more concentrated than the total absolute activity.

---

## 21. Occupancy-direction breadth

Per-channel positive-product occupancy shift:

`Δq_j = q_corr,j - q_ctrl,j`.

Counts:

- positive occupancy shift:
  `136`;
- negative occupancy shift:
  `101`;
- zero shift:
  `3`.

Thus occupancy shifts are somewhat more positive than the mean-contrast sign split.

This is secondary descriptive evidence.

---

## 22. Mean-sign × occupancy-shift cross-tab

Validated cross-tab:

### Positive `d_j`, positive `Δq_j`

`115`.

### Positive `d_j`, negative `Δq_j`

`6`.

### Positive `d_j`, zero `Δq_j`

`1`.

### Negative `d_j`, negative `Δq_j`

`95`.

### Negative `d_j`, positive `Δq_j`

`21`.

### Negative `d_j`, zero `Δq_j`

`2`.

Thus most positive-mean channels also shift toward more-positive item occupancy:

`115/122 = 94.2623%`.

Most negative-mean channels shift toward less-positive occupancy:

`95/118 = 80.5085%`.

This supports descriptive sign consistency between mean channel contribution and occupancy direction.

It does not establish causality.

---

## 23. Zero-product count closure

Aggregate corr zero-product count:

`0`.

Aggregate ctrl zero-product count:

`0`.

Therefore every channel-item role entry was strictly signed in the frozen artifact.

No zero occupancy ambiguity affects the cross-tab.

---

## 24. Frozen kernel-rank diagnostic

Median frozen kernel rank among positive-`d_j` channels:

`127.0`.

Median among negative-`d_j` channels:

`113.5`.

These medians do not support a simple monotonic rule that stronger kernel rank automatically implies positive `d_j`.

The fixed-rank quartile diagnostics must therefore be interpreted descriptively.

No rank cutoff is promoted.

---

## 25. Kernel-rank quartile Q1

Ranks:

`1–60`.

Channel count:

`60`.

Signed sum:

`+0.3714366111669356`.

Positive mass:

`0.5357767992062882`.

Negative mass:

`-0.16434018803935257`.

Absolute mass:

`0.7001169872456408`.

Fraction of total positive mass:

`0.4150462452294557`.

Fraction of total absolute mass:

`0.3844315902722897`.

Q1 therefore carries the largest quartile-level positive and absolute mass.

This is diagnostic only.

---

## 26. Kernel-rank quartile Q2

Ranks:

`61–120`.

Signed sum:

`+0.06835624604329985`.

Positive mass:

`0.22451967019841004`.

Negative mass:

`-0.1561634241551102`.

Absolute mass:

`0.38068309435352027`.

Positive-mass fraction:

`0.17392699018332583`.

Absolute-mass fraction:

`0.20903164759342313`.

---

## 27. Kernel-rank quartile Q3

Ranks:

`121–180`.

Signed sum:

`+0.19070830822183946`.

Positive mass:

`0.2963095031137192`.

Negative mass:

`-0.10560119489187973`.

Absolute mass:

`0.4019106980055989`.

Positive-mass fraction:

`0.22953988839259815`.

Absolute-mass fraction:

`0.22068764448866096`.

---

## 28. Kernel-rank quartile Q4

Ranks:

`181–240`.

Signed sum:

`+0.13009355770860104`.

Positive mass:

`0.23427861049976492`.

Negative mass:

`-0.10418505279116388`.

Absolute mass:

`0.3384636632909288`.

Positive-mass fraction:

`0.1814868761946203`.

Absolute-mass fraction:

`0.1858491176456263`.

---

## 29. Rank-quartile interpretation

Q1 carries:

`41.50%`

of positive mass and:

`38.44%`

of absolute mass.

However the remaining mass is not monotonic across quartiles:

Q3 exceeds Q2 in both signed and positive mass.

Together with the positive/negative median ranks, this means the observed concentration cannot be reduced to a simple statement:

**"the strongest kernel-ranked rows are the positive mechanism."**

The correct supported statement is narrower:

**the frozen rank-Q1 group is enriched in mass, but channel-mass concentration is not explained by a monotonic kernel-rank ordering alone.**

No causal kernel-strength claim follows.

---

## 30. Independent artifact validation

External validator:

`validate_frozen_strong_channel_mass_breadth_concentration_artifacts.py`

Validator SHA256:

`f59e46e3b42882e091ee6331a3db4c9955c6df0f0f14f21fee479d3452ff1d43`

Validation marker:

`PASS_FROZEN_STRONG_CHANNEL_MASS_BREADTH_CONCENTRATION_ARTIFACT_VALIDATION`

The validator independently checked:

- analysis HEAD;
- analyzer SHA;
- authority freeze;
- immediate parent evidence freeze;
- channel evidence freeze;
- frozen parent report SHA;
- frozen parent summary SHA;
- frozen channel input SHA;
- 240 channel output rows;
- rank uniqueness and full 1..240 permutation;
- per-channel mass identities;
- summary output hashes;
- sign counts;
- HHI;
- effective counts;
- Gini;
- fixed cumulative shares;
- 50% and 80% channel counts;
- occupancy cross-tab;
- population bridge.

---

## 31. Validated output identities

### Channel metrics

File:

`frozen_strong_channel_mass_breadth_metrics.jsonl`

SHA256:

`006b4dab8da96d1cd35193c5ac74d2e93e506fd01b64bf2afffda19968503b84`

Rows:

`240`.

### Summary

File:

`summary.json`

SHA256:

`c0474b5663d2c25dcf218ce7478c146e92cd4937c2fb584af91d2a968e3874e2`

### Static-analysis manifest

File:

`static_analysis_manifest.json`

SHA256:

`405d049194f830162813582c8364ad2d41e28df18de20c5b4255fb34d7631e3b`

---

## 32. Numerical bridge quality

Maximum formula residual:

`2.7755575615628914e-17`.

Maximum parent bridge residual:

`2.0816681711721685e-17`.

Maximum persisted parent reproduction residual:

`2.0816681711721685e-17`.

Frozen tolerance:

`5e-12`.

All numerical closures pass by a very large margin.

No channel was removed.

---

## 33. Code correctness

Code correctness is supported by:

1. frozen design authority;
2. frozen analyzer implementation;
3. successful static preflight;
4. exact frozen input authentication;
5. per-channel algebraic bridge validation;
6. exact rank-permutation validation;
7. exact population reconstruction;
8. independent artifact validator PASS.

No code correctness blocker remains for this stage.

---

## 34. Execution success

Execution marker:

`PASS_FROZEN_STRONG_CHANNEL_MASS_BREADTH_CONCENTRATION_STATIC_ANALYSIS`.

Execution HEAD:

`90737ca6962032e6fdcfeb3cdaa570b4c73790b7`.

Model forwards:

`0`.

Execution success is established.

---

## 35. Artifact / provenance validity

Independent artifact validation passed.

Output hashes are frozen and independently reproduced.

Artifact/provenance validity is established.

---

## 36. Scientific conclusion

With code correctness, execution success, and artifact/provenance validity established, the stage supports the validated conclusion:

**The strong-240 channel population has only a marginal positive-sign majority (`122/240`), while positive supporting mass is strongly concentrated (`N_eff_pos = 28.60`) and total absolute mass is also concentrated (`N_eff_abs = 52.51`).**

Thus the interaction phenotype is not best described as a broadly distributed channel mechanism.

It is better described as:

**a near-balanced channel-sign field with a strongly unequal mass distribution whose positive side is carried by a much smaller effective channel set.**

---

## 37. Relation to the itemwise breadth result

The immediate parent established that the phenotype is broad across items.

The present result establishes that the same phenotype is concentrated across channels.

These statements are compatible and jointly informative:

- **item dimension:** broad;
- **channel dimension:** mass-concentrated.

Therefore the K0 mechanism now has a two-dimensional structural description:

**broad across examples, concentrated across channels.**

This is a materially sharper localization than either result alone.

---

## 38. Positive support is not sign dominance

Because positive channels are only:

`122/240`

while negative channels are:

`118/240`,

the positive net does not arise from broad sign consensus across channels.

Instead, the decisive asymmetry is in mass.

Positive channels collectively carry:

`1.2908845830181823`

of positive mass against:

`0.5302898598775064`

absolute opposing mass.

Thus magnitude concentration, not sign prevalence, is the dominant channel-level structural fact.

---

## 39. Strongest concentration statement allowed

The strongest supported descriptive statement is:

**Positive interaction support is strongly concentrated in channel mass: 10 channels account for at least half of positive support, 32 account for at least 80%, and the HHI-equivalent positive-support channel count is 28.60 out of 240.**

This does not identify a causal channel set.

It does not authorize intervention on those channels.

It does not promote those ranked channels into a new population.

---

## 40. What is ruled out

The evidence rules out the simple hypothesis that positive interaction support is approximately evenly distributed across the strong-240 channels.

It also rules out the stronger narrative that most strong channels are positive in sign.

The sign split is nearly balanced.

The mass distribution is not.

---

## 41. What remains unresolved

This stage does not establish why the positive mass is concentrated.

It does not distinguish whether concentration is associated with:

- frozen kernel magnitude;
- source-vector geometry;
- role occupancy;
- channel-specific magnitude asymmetry;
- another already-frozen channel property.

The rank-quartile diagnostic shows some enrichment in Q1 but not a monotonic explanation.

Therefore a future stage may ask which pre-existing frozen channel property best accounts for the concentration, but only under a new static design.

---

## 42. No top-k promotion

The fixed top-count diagnostics are concentration summaries only.

Do not infer:

- top 10 are "the mechanism";
- top 32 are "necessary";
- top 24 should become a new authorized subset;
- rank-Q1 channels are causal.

Any follow-up subset-specific analysis requires separate prospective authority.

---

## 43. Scientific limits

This stage cannot establish:

- causal importance of a channel;
- necessity;
- sufficiency;
- intervention effect;
- generalization outside the strong-240 partition;
- generalization outside common-330 items;
- generalization outside `k=2`;
- generalization outside block20→residual21 geometry;
- monotonic causation by kernel magnitude.

The result is descriptive and algebraic.

---

## 44. No K1 transition

This stage remains entirely within K0-RVG.

The unrelated K1 files remain outside scope.

No K1 implementation, execution, or interpretation is authorized by this result.

---

## 45. Decision value for next K0 step

The preregistered decision rule states that `P2` justifies a later K0 stage asking which **already-frozen channel property** accounts for concentration.

That condition is met.

However the follow-up must not simply promote an observed top-k subset.

The correct next question is a full-population property-accounting analysis, using all 240 channels, with prospective metrics fixed before execution.

---

## 46. Final validated classification

Primary:

`S1/P2`

Supporting:

`C2`.

Exact labels:

- `S1_POSITIVE_CHANNEL_MAJORITY`;
- `P2_CONCENTRATED_POSITIVE_SUPPORTING_MASS`;
- `C2_CONCENTRATED_ABSOLUTE_CHANNEL_MASS`.

Key quantities:

- positive channels:
  `122/240`;
- negative channels:
  `118/240`;
- positive effective count:
  `28.60205938176289`;
- absolute effective count:
  `52.50818654659294`;
- positive 50%-mass count:
  `10`;
- positive 80%-mass count:
  `32`;
- absolute 50%-mass count:
  `22`;
- absolute 80%-mass count:
  `75`;
- positive Gini:
  `0.8433521328282656`;
- absolute Gini:
  `0.6573581715076158`.

Final validated descriptive conclusion:

**The K0 strong interaction phenotype is broad across items but channel-mass concentrated: the channel sign field is almost balanced, while a comparatively small effective set carries most positive and absolute interaction mass.**
