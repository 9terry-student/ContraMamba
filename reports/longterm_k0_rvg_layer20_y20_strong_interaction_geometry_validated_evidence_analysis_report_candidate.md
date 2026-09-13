# ContraMamba K0-RVG Layer-20→21 Strong-Partition Interaction Geometry
## Validated Evidence Analysis Report Candidate

## 1. Status

**Stage:** validated K0 observational/algebraic scientific evidence.

**Static design freeze:**

`2b239beb646c681f2032b6cfcaca15a34d25ad34`

**Implementation / execution commit:**

`0b1168182a265bc405652d2ce519f63310433c3e`

**Runner:**

`scripts/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_audit.py`

**Runner SHA256:**

`0c487e3d29236e298f8c74cbdbcdfce3599af587dfd27a6e5fcf169df64eade5`

**Run directory:**

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1`

**Execution marker:**

`PASS_LAYER20_Y20_STRONG_INTERACTION_GEOMETRY_EXECUTION`

**Independent artifact-validation marker:**

`PASS_LAYER20_Y20_STRONG_INTERACTION_GEOMETRY_ARTIFACT_VALIDATION`

This report interprets only the validated common-330, current-token `k=2`, frozen strong-240 evidence.

It introduces no new model execution, tokenizer work, training, logits, task evaluation, intervention, learned geometry, PCA/SVD, layer/channel/item search, or K1 evidence.

---

## 2. Scientific question

The immediate parent stage established that the incoming-layer21 strong routing source is dominated by the `R20×Y20` interaction term:

`D_ry20,strong = +0.760594723140676`.

The present stage asks:

> Why is that strong interaction contrast positive: larger parent-normalized `R20` projected magnitude, larger parent-normalized `Y20` projected magnitude, stronger directional alignment, or a mixture?

The target is exactly the already-frozen parent strong interaction term.

---

## 3. Frozen strong-space geometry

For each item-role branch:

`x = H_r20,strong / sqrt(D_X22)`

`y = H_y20,strong / sqrt(D_X22)`.

Define:

`A = ||x||₂`

`B = ||y||₂`

`C = cos(x,y)`

`M = 2AB`.

Then:

`I = 2<x,y> = 2ABC = MC`.

For one corr/ctrl item pair:

`ΔI = I_corr - I_ctrl`.

Using exact symmetric midpoint identities:

`ΔI = Q_A + Q_B + Q_C`

with:

`Q_A = 2 C_bar B_bar ΔA`

`Q_B = 2 C_bar A_bar ΔB`

`Q_C = M_bar ΔC`.

This decomposition is exact and has no residual scientific term.

---

## 4. Execution validity

Full execution completed with:

`model_forward_count = 1344`.

Frozen scope remained:

- common items:
  `330`;
- strong channels:
  `240`;
- weak channels:
  `1296`;
- relative coordinate:
  `k=2`;
- source block:
  `20`;
- target residual layer:
  `21`;
- parent map layer:
  `22`;
- lag0 kernel RMS:
  `0.24383223809052498`.

The runner reproduced both the parent item target and parent strong-channel target.

---

## 5. Independent artifact validation

Independent validation passed.

It independently checked:

- exact four-file output set;
- no `.partial` sibling;
- runtime HEAD;
- runner SHA and blob;
- static-design identity;
- immediate-parent evidence identities;
- manifest provenance;
- output hashes;
- `330` item rows;
- `240` strong-channel validation rows;
- exact parent item `delta_ry20_strong` reproduction;
- exact parent strong-channel `mean_d_ry20` reproduction;
- `I=2<x,y>`;
- `I=2ABC`;
- exact midpoint decomposition;
- summary-to-item/channel aggregation;
- forbidden-action flags.

Therefore execution success, artifact/provenance validity, and scientific interpretation are closed separately.

---

## 6. Validated artifact identities

### Item metrics

`e7002e03bd170c05ea068e70eb32cc1f7a8b0d4f569ac44e531a42a8a4e1ebfe`

### Strong-channel validation

`111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942`

### Summary

`35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9ea9ef66a80798`

### Execution manifest

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fe`

---

## 7. Numerical validity

Validated maxima:

- parent item reproduction residual:
  `9.992007221626409e-16`;
- parent strong-channel mean reproduction residual:
  `2.0816681711721685e-17`;
- exact `ΔI = Q_A + Q_B + Q_C` residual:
  `6.661338147750939e-16`;
- role-level `I=2ABC` residual:
  `4.440892098500626e-16`.

All are far below the preregistered blocking tolerances.

The scientific identity therefore closes numerically.

---

## 8. Parent target reproduction

The frozen parent strong interaction mean was:

`+0.760594723140676`.

The reconstructed mean is exactly:

`+0.760594723140676`.

Thus the new geometry decomposition exactly targets the already-validated parent quantity.

---

## 9. Exact population decomposition

Population mean components:

### `Q_A`: parent-normalized strong `R20` projected magnitude

`+0.01818378200815921`

### `Q_B`: parent-normalized strong `Y20` projected magnitude

`+0.009759485141568866`

### `Q_C`: strong-subvector cosine alignment

`+0.7326514559909479`

### Total

`+0.760594723140676`

All three terms reinforce the parent positive interaction contrast.

However, their magnitudes are highly unequal.

Absolute component shares:

- `Q_A`:
  `2.3907320751679763%`;
- `Q_B`:
  `1.2831386866937017%`;
- `Q_C`:
  `96.32612923813832%`.

Largest absolute scientific component:

**`Q_C`**.

---

## 10. Role geometry

### Corr

Mean parent-normalized strong `R20` magnitude:

`A_corr = 1.5174351424881483`

Median:

`1.5035177131637223`.

Mean parent-normalized strong `Y20` magnitude:

`B_corr = 1.2868907438993054`

Median:

`1.2659509718331587`.

Mean cosine alignment:

`C_corr = 0.4364279014925435`

Median:

`0.43812188210006314`.

Mean strong interaction:

`I_corr = 1.7226764561870782`

Median:

`1.6658289327373261`.

### Ctrl

Mean parent-normalized strong `R20` magnitude:

`A_ctrl = 1.4829015640324512`

Median:

`1.4714676719186803`.

Mean parent-normalized strong `Y20` magnitude:

`B_ctrl = 1.2592471445236473`

Median:

`1.2172924762260067`.

Mean cosine alignment:

`C_ctrl = 0.23896035194570278`

Median:

`0.24390478373667812`.

Mean strong interaction:

`I_ctrl = 0.9620817330464022`

Median:

`0.816955164262678`.

---

## 11. Magnitude differences are real but small

Role means imply:

`A_corr - A_ctrl ≈ +0.03453`

and:

`B_corr - B_ctrl ≈ +0.02764`.

Therefore corr does have somewhat larger parent-normalized projected magnitudes on both source branches.

That contribution is not zero.

But the exact population decomposition shows that these two magnitude terms together account for only about:

`3.67%`

of total absolute scientific component mass.

Therefore magnitude cannot explain the parent strong interaction phenotype by itself.

---

## 12. Alignment difference is dominant

The mean cosine difference is approximately:

`0.43643 - 0.23896 = +0.19747`.

Both roles are positively aligned on average.

However, corr alignment is substantially stronger.

The exact alignment term contributes:

`Q_C = +0.7326514559909479`.

This alone accounts for:

`96.33%`

of total absolute scientific component mass.

Therefore the corr>ctrl strong interaction advantage is overwhelmingly an **alignment effect**.

---

## 13. Scientific outcome classification

The validated scientific outcome is:

# Outcome C: alignment dominant

A more precise statement is:

# frozen-strong-subspace directional-alignment dominance with small reinforcing source-magnitude contributions

This is not merely "mixed with alignment largest."

The exact decomposition is overwhelmingly concentrated in `Q_C`.

---

## 14. Why not Outcome A

Outcome A would require the `R20` normalized-magnitude term `Q_A` to provide the primary account.

Instead:

`Q_A = +0.01818378200815921`

with only:

`2.39%`

absolute share.

Outcome A is rejected.

---

## 15. Why not Outcome B

Outcome B would require the `Y20` normalized-magnitude term `Q_B` to dominate.

Instead:

`Q_B = +0.009759485141568866`

with only:

`1.28%`

absolute share.

Outcome B is rejected.

---

## 16. Why Outcome C

Outcome C is supported because:

- `Q_C` is the largest component;
- `Q_C` is positive;
- `Q_C` contributes `+0.73265` of the `+0.76059` total;
- `Q_C` carries `96.33%` of absolute component mass;
- corr cosine alignment is much higher than ctrl alignment;
- `Q_A` and `Q_B` are both small secondary positive contributions.

Thus the validated strong interaction phenotype is best described as alignment-dominant.

---

## 17. Mechanistic localization

The immediate parent stage showed:

- strong direct `R20` source:
  small positive;
- strong direct `Y20` source:
  small positive;
- strong `R20×Y20` interaction:
  dominant positive.

The present stage localizes that interaction one step further:

> The dominant positive strong interaction is not mainly because the two source vectors are much larger in corr. It is mainly because, after the already-frozen downstream layer22 map and normalization, the `R20` and `Y20` strong subvectors point much more coherently in the same direction for corr than for ctrl.

This is a geometric localization under a frozen operator.

It is observational/algebraic, not causal.

---

## 18. Integrated chain

The validated K0 chain now contains the following localization:

`ΔR20` and `ΔY20`
→ projected through frozen layer22 parent map
→ restricted to frozen strong 240 rows
→ both individually small positive contributors
→ but their joint corr alignment is much stronger than ctrl
→ large positive `R20×Y20` interaction
→ strong-positive incoming `R21` routing source
→ downstream residual accumulation / RMSNorm reshaping / fixed-row routing
→ corr-enriched strong-kernel exposure
→ selective lag0 transfer
→ U/write/state separation.

The new evidence identifies the strong interaction as principally a directional-coherence phenomenon.

---

## 19. What is not established

This evidence does not establish:

- causal necessity of alignment;
- causal sufficiency of alignment;
- why the alignment arises internally inside `R20` or `Y20`;
- semantic meaning of the aligned direction;
- task performance effect;
- cross-model or cross-seed generality;
- K1 claims.

No intervention was performed.

No learned basis was introduced.

No PCA/SVD was used.

No post-hoc channel subset was selected.

---

## 20. Code correctness, execution, artifact validity, science

### Code correctness

Static preflight passed.

Implementation freeze:

`0b1168182a265bc405652d2ce519f63310433c3e`.

### Execution success

Full execution marker:

`PASS_LAYER20_Y20_STRONG_INTERACTION_GEOMETRY_EXECUTION`.

### Artifact/provenance validity

Independent validation marker:

`PASS_LAYER20_Y20_STRONG_INTERACTION_GEOMETRY_ARTIFACT_VALIDATION`.

### Scientific conclusion

Only after those gates passed is Outcome C accepted.

---

## 21. Validated scientific conclusion

> **At common-330 current-token `k=2`, the previously validated strong-partition `R20×Y20` interaction contrast (`+0.760594723`) is overwhelmingly explained by stronger corr directional alignment between the frozen parent-normalized strong `R20` and `Y20` projected subvectors. The exact midpoint decomposition yields `Q_A=+0.01818`, `Q_B=+0.00976`, and `Q_C=+0.73265`, with `Q_C` accounting for `96.33%` of absolute scientific component mass. Corr and ctrl differ only modestly in source magnitudes, while mean cosine alignment rises from about `0.239` in ctrl to `0.436` in corr. The correct classification is therefore Outcome C: alignment dominant, with small reinforcing magnitude contributions.**

This conclusion is observational/algebraic, not causal.

---

## 22. Compact quantitative summary

| Quantity | Value |
|---|---:|
| Parent strong interaction | +0.760594723 |
| `Q_A` | +0.018183782 |
| `Q_B` | +0.009759485 |
| `Q_C` | +0.732651456 |
| `Q_A` share | 2.39% |
| `Q_B` share | 1.28% |
| `Q_C` share | 96.33% |
| corr mean `A` | 1.517435142 |
| ctrl mean `A` | 1.482901564 |
| corr mean `B` | 1.286890744 |
| ctrl mean `B` | 1.259247145 |
| corr mean `C` | 0.436427901 |
| ctrl mean `C` | 0.238960352 |
| corr mean `I` | 1.722676456 |
| ctrl mean `I` | 0.962081733 |

---

## 23. Next K0 scientific branch after evidence freeze

The preregistered rule for Outcome C is to localize the fixed strong-subvector alignment structure without learned geometry.

The next stage should therefore ask:

> What exact fixed-coordinate structure makes `x_corr` and `y_corr` substantially more aligned than `x_ctrl` and `y_ctrl` across the frozen strong 240 rows?

The next design should not return to generic magnitude analysis.

It should remain inside the frozen strong subspace and decompose the dot-product/cosine structure without learned projections.

A narrow next candidate is a sign/co-contribution localization that separates:

- same-sign positive-product channels;
- opposite-sign negative-product channels;
- whether corr's higher cosine arises primarily from increased positive co-contribution, reduced negative cancellation, or both;

while preserving the exact frozen 240-channel set and without post-hoc selecting a subset.

Do not transition to K1.

---

## 24. Freeze recommendation

Freeze exactly:

1. item metrics JSONL;
2. strong-channel validation JSONL;
3. summary JSON;
4. execution manifest JSON;
5. this validated evidence analysis report.

Do not include the unrelated K1 files.

---

## 25. Final status

**Execution:** PASS.

**Independent artifact validation:** PASS.

**Parent item target reproduction:** PASS.

**Parent strong-channel reproduction:** PASS.

**Exact midpoint decomposition:** PASS.

**Scientific classification:** Outcome C — alignment dominant.

**Refined interpretation:** frozen-strong-subspace directional-alignment dominance with small reinforcing source-magnitude contributions.

**Next K0 target after evidence freeze:** fixed-coordinate source of corr's stronger strong-subvector alignment.

**K1:** not authorized.
