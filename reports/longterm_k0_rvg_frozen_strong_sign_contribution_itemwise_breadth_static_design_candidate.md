# ContraMamba K0-RVG Frozen-Strong Sign-Contribution Itemwise Breadth / Consistency
## Static Design Candidate

## 1. Status

**Phase:** new K0 scientific static-analysis design.

**Immediate parent evidence freeze:**

`3f7a9eaa38935fc066f20f81cbce0a9f901909a9`

**Immediate parent implementation freeze:**

`3af43c91d791e2755b977529eb26d65db00c7870`

**Immediate parent validated report:**

`reports/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_validated_evidence_analysis_report_candidate.md`

This stage introduces **no new model execution**.

The immediate parent established the population-level result:

**Outcome A — positive co-contribution dominant.**

The next scientific question is:

> Is the validated population-level positive co-contribution dominance broadly expressed across the frozen common-330 items, or is the positive population mean produced by a minority of large-effect items despite substantial itemwise heterogeneity?

A second, linked question is:

> Are the parent count/strength directions themselves broadly itemwise consistent, or are they only population-average tendencies?

No model forward, checkpoint load, handoff ZIP access, tokenizer, logits, task heads, training, intervention, PCA/SVD, learned geometry, channel search, item search, alternate threshold search, or K1 work is authorized.

---

## 2. Why this question is next

The immediate parent established exact population means:

`G_total = +0.760594723140676`

`G_pos = +0.5314816993853949`

`G_relief = +0.22911302375528114`

with absolute two-component shares:

- positive co-contribution gain:
  `69.87712157544046%`;
- cancellation relief:
  `30.12287842455955%`.

It also established population-average diagnostics:

`mean(Δn_same) = +4.551515151515152`

`mean(ΔPbar) = +0.0035643520982118104`

`mean(ΔKbar) = -0.001921654940814375`.

These means establish the aggregate phenotype but do not by themselves determine its **breadth across items**.

A positive mean may arise from:

1. a broad majority of items sharing the same sign and component structure;
2. a heterogeneous population with a minority of large positive effects;
3. a broad positive total with heterogeneous component dominance;
4. a mixture of reinforcing and opposing component patterns.

The frozen item artifact already contains every scalar required to answer this without re-running the model.

---

## 3. Frozen evidence authority

Use only evidence frozen at:

`3f7a9eaa38935fc066f20f81cbce0a9f901909a9`.

The evidence-freeze commit contains exactly four intended files.

### Parent item metrics

Path:

`reports/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_3af43c9_v1/frozen_strong_alignment_sign_contribution_item_metrics.jsonl`

SHA256:

`cd9d3306c7ef2b2d8f2a759be5e7945ccfdb86f0009981f61247b9ddfcf03974`

Expected rows:

`330`

### Parent summary

Path:

`reports/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_3af43c9_v1/summary.json`

SHA256:

`0696e5887dfbd2fc7eb55d3c726e4e55415858e15522d52efca6ce38ecda1615`

### Parent static-analysis manifest

Path:

`reports/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_3af43c9_v1/static_analysis_manifest.json`

SHA256:

`5e1fdd97236d0e6f2d7b7041207d25af78aba7dae3b15cff9a66d760c365d50f`

### Parent validated report

Path:

`reports/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_validated_evidence_analysis_report_candidate.md`

SHA256:

`a8e4e7fe528e9b01a16b26d7d3508938d7f63bbd4cc57153207c9ae4c58767d6`

No earlier artifact may substitute for these frozen parent outputs.

---

## 4. Frozen population and coordinate scope

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
- existing matched/swapped role definitions;
- existing signed interaction decomposition.

No item may be removed.

No subgroup may be selected for the primary claim.

No channel may be added or removed.

---

## 5. Parent item-level scalar definitions

For every frozen item `i`, read only persisted scalar fields.

Define:

`p_i = ΔP_i`

from:

`delta_P`.

Define:

`r_i = ΔN_i`

from:

`delta_N`.

Define:

`t_i = ΔI_i`

from:

`reconstructed_delta_I`.

The exact parent identity is:

`t_i = p_i + r_i`.

Interpretation:

- `p_i > 0`:
  corr has greater positive same-sign co-contribution mass;
- `r_i > 0`:
  corr has reduced negative cancellation;
- `t_i > 0`:
  corr has a larger net strong-subspace interaction for that item.

No new continuous score is learned.

---

## 6. Parent count/strength diagnostics

For every frozen item `i`, also read:

`c_i = delta_same_sign_channel_count`

`o_i = delta_opposite_sign_channel_count`

`a_i = delta_Pbar`

`k_i = delta_Kbar`.

The parent exact count identity is:

`c_i + o_i = 0`

because zero-product counts are frozen at zero.

Interpretation:

- `c_i > 0`:
  corr has more same-sign participating strong channels;
- `a_i > 0`:
  corr has greater positive mass per same-sign channel;
- `o_i < 0`:
  corr has fewer opposite-sign participating channels;
- `k_i < 0`:
  corr has weaker cancellation magnitude per opposite-sign channel.

These diagnostics remain secondary to the signed component decomposition.

---

## 7. Exact sign classification

For each scalar `x` among:

- `p_i`;
- `r_i`;
- `t_i`;
- `c_i`;
- `a_i`;
- `k_i`;

use the persisted floating value directly.

Classify:

- positive if `x > 0`;
- negative if `x < 0`;
- zero only if `x == 0`.

Do not introduce a post-hoc sign threshold.

The analyzer must separately report any exact-zero count.

No exact-zero item may be silently assigned to a positive or negative class.

---

## 8. Component sign quadrants

Classify every item into exactly one `p_i × r_i` sign quadrant.

### Q++

`p_i > 0` and `r_i > 0`.

Interpretation:

both positive co-contribution gain and cancellation relief reinforce corr.

### Q+-

`p_i > 0` and `r_i < 0`.

Interpretation:

positive co-contribution gain is opposed by greater negative cancellation.

### Q-+

`p_i < 0` and `r_i > 0`.

Interpretation:

cancellation relief is opposed by weaker positive co-contribution.

### Q--

`p_i < 0` and `r_i < 0`.

Interpretation:

both signed components oppose corr.

### Boundary class

Any item with exact `p_i == 0` or exact `r_i == 0`.

No boundary item is merged into another quadrant.

All five counts must sum to `330`.

---

## 9. Per-item component dominance

For each item define:

`m_i = |p_i| - |r_i|`.

Classify:

### Positive-component dominant

`m_i > 0`.

### Cancellation-relief dominant

`m_i < 0`.

### Exact tie

`m_i == 0`.

This is an itemwise descriptive classification.

It does not replace the population-level parent outcome.

---

## 10. Parent-concordant item definition

Define a **parent-concordant positive-co-contribution-led item** as an item satisfying all of:

`t_i > 0`

`p_i > 0`

`|p_i| > |r_i|`.

This definition permits either sign of `r_i`.

It asks whether the item's positive net strong-interaction contrast is primarily carried by the same component that dominates at the population level.

Define indicator:

`C_i = 1`

when all three conditions hold, else:

`C_i = 0`.

The primary concordance count is:

`N_concordant = Σ_i C_i`.

The primary concordance fraction is:

`F_concordant = N_concordant / 330`.

---

## 11. Total-effect breadth

Define:

`N_total_pos = # {i : t_i > 0}`

`N_total_neg = # {i : t_i < 0}`

`N_total_zero = # {i : t_i == 0}`.

Define:

`F_total_pos = N_total_pos / 330`.

This is the primary breadth statistic for the sign of the net parent phenotype.

A positive population mean is considered **majority-broad** if:

`N_total_pos > 165`.

It is considered **minority-driven in sign** if:

`N_total_pos <= 165`

while the frozen population mean remains positive.

The `165` boundary is not a fitted threshold.

It is exactly one half of the fixed `330`-item population.

---

## 12. Positive-component breadth

Define:

`N_p_pos = # {i : p_i > 0}`

`N_p_neg = # {i : p_i < 0}`

`N_p_zero = # {i : p_i == 0}`.

Define:

`F_p_pos = N_p_pos / 330`.

This answers whether the dominant population component itself has broadly positive itemwise sign.

---

## 13. Cancellation-relief breadth

Define:

`N_r_pos = # {i : r_i > 0}`

`N_r_neg = # {i : r_i < 0}`

`N_r_zero = # {i : r_i == 0}`.

Define:

`F_r_pos = N_r_pos / 330`.

Because `r_i = ΔN_i`, positive `r_i` means itemwise cancellation relief.

---

## 14. Both-components-reinforce breadth

Define:

`N_both = # {i : p_i > 0 and r_i > 0}`.

Define:

`F_both = N_both / 330`.

This is exactly the Q++ fraction.

It determines how often the population-level "both components reinforce" result also holds itemwise.

---

## 15. Component-dominance breadth

Define:

`N_p_dom = # {i : |p_i| > |r_i|}`

`N_r_dom = # {i : |r_i| > |p_i|}`

`N_tie = # {i : |p_i| == |r_i|}`.

Define:

`F_p_dom = N_p_dom / 330`.

This asks whether population-level positive-component dominance is also the majority itemwise dominance relation, independent of total sign.

---

## 16. Parent-concordance majority test

The strongest direct breadth diagnostic is:

`N_concordant`.

Classify parent-concordance as:

### Majority-concordant

`N_concordant > 165`.

### Not-majority-concordant

`N_concordant <= 165`.

Again, `165` is fixed by the cohort size and is not tuned from results.

---

## 17. Two-axis scientific classification

The stage uses two primary axes.

### Axis A — total-effect breadth

**A1: majority-broad positive total**

`N_total_pos > 165`.

**A2: minority-driven positive total**

`N_total_pos <= 165`

despite:

`mean(t_i) > 0`.

### Axis B — parent-component concordance

**B1: majority parent-concordant**

`N_concordant > 165`.

**B2: not majority parent-concordant**

`N_concordant <= 165`.

The final scientific description combines both axes.

Examples:

- `A1/B1`:
  broad positive total with majority positive-co-contribution-led concordance;
- `A1/B2`:
  broad positive total but heterogeneous component dominance;
- `A2/B1`:
  logically possible but unusual under this definition;
- `A2/B2`:
  positive population mean driven by a minority / heterogeneous item structure.

No stronger threshold such as 60%, 70%, or 80% is introduced.

---

## 18. Cross-item positive/negative mass balance

For any frozen per-item scalar sequence `x_i`, define:

`X_plus = Σ_i max(x_i, 0)`

`X_minus = Σ_i min(x_i, 0)`.

Then:

`Σ_i x_i = X_plus + X_minus`.

Compute this for:

- `t_i`;
- `p_i`;
- `r_i`.

Report:

`|X_minus| / X_plus`

when `X_plus > 0`.

This quantifies how much negative itemwise mass offsets positive itemwise mass without selecting a subset.

It is a secondary breadth diagnostic.

---

## 19. Median bridge

Recompute medians directly from all 330 item rows for:

- `t_i`;
- `p_i`;
- `r_i`;
- `c_i`;
- `a_i`;
- `k_i`.

The analyzer must reproduce the parent summary medians for:

`delta_P`

`delta_N`

`reconstructed_delta_I`

`delta_same_sign_channel_count`

`delta_Pbar`

`delta_Kbar`.

This is a blocking bridge between the frozen parent summary and the new static analysis.

---

## 20. Count-direction breadth

For the parent count mechanism report:

`N_count_support = # {i : c_i > 0}`

`N_count_oppose = # {i : c_i < 0}`

`N_count_zero = # {i : c_i == 0}`.

Since:

`o_i = -c_i`

the opposite-sign count direction is algebraically redundant.

Do not double-count it as independent evidence.

---

## 21. Positive-strength breadth

Report:

`N_Pbar_support = # {i : a_i > 0}`

`N_Pbar_oppose = # {i : a_i < 0}`

`N_Pbar_zero = # {i : a_i == 0}`.

This asks how broadly the per-same-sign-channel positive-strength advantage holds itemwise.

---

## 22. Cancellation-strength breadth

Because negative `k_i = ΔKbar_i` means weaker cancellation magnitude for corr, define:

`N_Kbar_relief = # {i : k_i < 0}`

`N_Kbar_worse = # {i : k_i > 0}`

`N_Kbar_zero = # {i : k_i == 0}`.

This asks how broadly the per-opposite-sign-channel cancellation-relief direction holds.

---

## 23. Joint count/strength support patterns

For the positive side classify each item using:

`c_i`

and:

`a_i`.

Report the four sign combinations:

- `c_i > 0`, `a_i > 0`;
- `c_i > 0`, `a_i < 0`;
- `c_i < 0`, `a_i > 0`;
- `c_i < 0`, `a_i < 0`;

plus any exact-zero boundary.

This determines whether positive-component support is jointly broad in both participation and strength.

For the cancellation side classify using:

`o_i < 0`

and:

`k_i < 0`.

Because `o_i = -c_i`, this uses no new independent count variable.

---

## 24. No correlation-based causal claim

This stage does not use correlation between item diagnostics as a primary scientific statistic.

In particular, it does not claim that:

- more same-sign channels cause larger `ΔP`;
- larger `ΔPbar` causes larger `ΔP`;
- weaker `ΔKbar` causes larger `ΔN`.

The stage is limited to breadth, sign consistency, dominance, and exact mass accounting.

---

## 25. Parent population bridges

The analyzer must reproduce from the frozen 330 rows:

`mean(t_i) = +0.760594723140676`

`mean(p_i) = +0.5314816993853949`

`mean(r_i) = +0.22911302375528114`

`mean(c_i) = +4.551515151515152`

`mean(a_i) = +0.0035643520982118104`

`mean(k_i) = -0.001921654940814375`.

All are blocking.

---

## 26. Parent identity bridges

For every item verify:

`t_i = p_i + r_i`

within absolute tolerance:

`5e-12`.

Also verify:

`c_i + o_i = 0`

exactly.

Verify all parent reproduction residuals remain within:

`5e-12`.

No row failing an identity may be dropped.

Any failure blocks the stage.

---

## 27. Frozen artifact authentication

The analyzer must authenticate before analysis:

1. current branch;
2. the immediate parent evidence freeze as an ancestor;
3. exact parent item SHA256;
4. exact parent summary SHA256;
5. exact parent manifest SHA256;
6. exact parent report SHA256.

It must read scientific inputs from the frozen parent evidence commit, not from an unauthenticated alternate file.

The working-tree copy must match the frozen Git object bytes.

---

## 28. Static-only implementation boundary

The analyzer must be implementable with Python standard library only.

Do not import:

- `torch`;
- `transformers`;
- `numpy`;
- model modules;
- training modules.

Do not open:

- checkpoint files;
- handoff ZIP files.

Do not execute:

- model forwards;
- tokenization;
- logits;
- task heads;
- training;
- interventions.

---

## 29. Static preflight

Static preflight must:

1. authenticate the parent evidence freeze;
2. authenticate all four frozen parent artifact SHA256 values;
3. parse all `330` item rows;
4. verify item identities;
5. reproduce parent means and medians;
6. compute all classifications in memory;
7. emit no scientific evidence files;
8. report `model_forward_count = 0`.

No separate preflight authority document is required.

---

## 30. Intended execution outputs

The eventual static execution should emit exactly:

### Item classification rows

`frozen_strong_sign_contribution_itemwise_breadth_classification.jsonl`

One row per frozen item.

Persist only:

- stable item identity;
- `p_i`, `r_i`, `t_i`;
- sign labels;
- quadrant;
- dominance label;
- parent-concordant flag;
- count/strength direction labels;
- exact bridge residuals.

No raw vectors.

### Summary

`summary.json`

Persist:

- all counts and fractions;
- quadrant counts;
- dominance counts;
- parent-concordance count/fraction;
- cross-item positive/negative mass balances;
- mean/median bridges;
- final two-axis classification;
- execution-boundary flags.

### Static-analysis manifest

`static_analysis_manifest.json`

Persist:

- authority/evidence identities;
- implementation identity;
- input/output hashes;
- population scope;
- tolerances;
- zero model-forward count;
- forbidden-operation flags.

No other output class is required.

---

## 31. Primary scientific readout

The primary readout is not another population mean.

It is the pair:

`(Axis A total-effect breadth, Axis B parent-component concordance)`.

This directly answers whether the parent Outcome A is:

- broad across items;
- broad in total but heterogeneous in component structure;
- or minority-driven / heterogeneous.

---

## 32. Secondary scientific readouts

Secondary readouts are:

- `F_p_pos`;
- `F_r_pos`;
- `F_both`;
- `F_p_dom`;
- component sign quadrants;
- count-direction breadth;
- positive-strength breadth;
- cancellation-strength breadth;
- positive/negative cross-item mass balance.

These refine but do not replace the primary two-axis result.

---

## 33. Interpretation constraints

Allowed language:

- broad;
- majority;
- minority-driven in sign;
- heterogeneous;
- parent-concordant;
- positive-component dominant;
- cancellation-relief dominant;
- reinforcing;
- opposing;
- itemwise;
- population-average.

Disallowed language without intervention:

- causal;
- causes;
- necessary;
- sufficient;
- mechanism proven;
- channel count drives;
- per-channel strength drives.

---

## 34. No subgroup promotion

Even if one quadrant is scientifically interesting, the analyzer must not promote a newly selected item subset as a new primary population.

No follow-up execution may be defined from "top items", "largest items", error cases, or a post-hoc percentile without a new frozen design.

All primary claims remain over all `330` items.

---

## 35. No K1 transition

This design remains entirely within the K0-RVG line.

The existing unrelated K1 files remain outside scope.

No K1 implementation, test, execution, or interpretation is authorized by this stage.

---

## 36. Expected decision value

If the result is `A1/B1`, the K0 chain gains evidence that the positive co-contribution phenotype is not merely a population-average artifact but is broadly expressed itemwise.

If the result is `A1/B2`, the parent total effect is broad but the internal signed-component organization is heterogeneous.

If the result is `A2/B2`, the positive population mean is substantially minority-driven and future K0 work should prioritize explaining heterogeneity before deeper channel localization.

This decision determines whether a later channel-distribution analysis is scientifically justified.

---

## 37. Stop conditions

Stop and block if any of the following occurs:

- parent evidence commit mismatch;
- parent artifact SHA mismatch;
- row count not `330`;
- duplicate stable item identity;
- item identity failure above tolerance;
- parent mean or median bridge mismatch;
- unexpected raw-vector dependency;
- model/checkpoint/handoff dependency;
- unauthorized worktree change;
- any K1 file modification.

Do not weaken tolerances to obtain PASS.

---

## 38. Final design statement

This stage is a **model-free itemwise breadth/consistency localization** of the already-frozen K0 strong sign-contribution result.

It asks one narrow question:

**Is the validated positive co-contribution-dominant population phenotype broad across the fixed 330-item cohort, or is it primarily an aggregate produced by heterogeneous / minority item behavior?**

The answer must come entirely from frozen evidence at:

`3f7a9eaa38935fc066f20f81cbce0a9f901909a9`.

No new model execution is scientifically necessary or authorized.
