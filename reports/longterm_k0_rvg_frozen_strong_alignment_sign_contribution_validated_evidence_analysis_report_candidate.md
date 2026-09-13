# ContraMamba K0-RVG Frozen-Strong Alignment Sign / Co-Contribution
## Validated Evidence Analysis Report Candidate

## 1. Status

**Stage:** validated K0 scientific static analysis.

**Scientific question:**

Within the frozen strong-240 `R20×Y20` interaction, is corr's larger alignment-driven interaction caused primarily by increased positive same-sign co-contribution, by reduced negative opposite-sign cancellation, or by a mixture of both?

**Validated outcome:**

**Outcome A — positive co-contribution dominant.**

Refined descriptive conclusion:

**The frozen strong-subspace interaction advantage is dominated by increased positive same-sign co-contribution, with a smaller but substantial reinforcing contribution from reduced negative opposite-sign cancellation. The positive-gain side is supported by both more same-sign participating channels and stronger positive mass per same-sign channel; the cancellation-relief side is supported by both fewer opposite-sign channels and weaker cancellation magnitude per opposite-sign channel.**

This is an observational/algebraic localization over already-frozen artifacts.

It is not a causal intervention result.

---

## 2. Authority chain

### Original static design freeze

Commit:

`fe73f67e84176f5b7d468287d8084a8836e9a240`

File:

`reports/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_static_design_candidate.md`

SHA256:

`4298eecd25dce651f772d249c2b6e7c9830a447bd3595a344ca28ced94d76909`

Git blob:

`7e9beaa78bad01b7ae4564de09635e0cdcc36b75`

### Provenance correction amendment freeze

Commit:

`76ed124a3db439c6371bda4c5170a56455789b76`

File:

`reports/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_static_design_provenance_correction_amendment_candidate.md`

SHA256:

`f5b6266d03d669f0580f8d40221e973ffff0f3a3ce1f3333fd722ff45bc290a5`

Git blob:

`4bae34c23ce4045e1b43dbcef5249710708416e2`

The amendment corrected exactly one malformed parent execution-manifest SHA literal.

No scientific semantics changed.

### Static analyzer implementation freeze

Commit:

`3af43c91d791e2755b977529eb26d65db00c7870`

File:

`scripts/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_static_analysis.py`

SHA256:

`6cb326e15f8f9b78aae2209f08672f6773fe5443669ccf6492448b2b8dc1e1bf`

Git blob:

`3588a7bfa9e21694044629572d0d47ffdd622906`

---

## 3. Immediate parent evidence

Parent evidence freeze:

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`

Parent implementation:

`0b1168182a265bc405652d2ce519f63310433c3e`

Parent run:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1`

The parent established the strong-partition interaction contrast:

`D_ry20,strong = +0.760594723140676`

with alignment attribution:

- `Q_A = +0.01818378200815921`;
- `Q_B = +0.009759485141568866`;
- `Q_C = +0.7326514559909479`.

Absolute shares:

- `Q_A = 2.3907320751679763%`;
- `Q_B = 1.2831386866937017%`;
- `Q_C = 96.32612923813832%`.

The validated parent conclusion was:

**Outcome C — alignment dominant.**

The present stage does not replace that result.

It localizes the fixed-coordinate signed interaction structure inside that already-validated strong-subspace alignment phenotype.

---

## 4. Frozen parent artifact identities

The static analyzer authenticated the following artifacts directly from parent evidence commit:

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`.

### Item metrics

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/layer20_y20_strong_interaction_geometry_item_metrics.jsonl`

SHA256:

`e7002e03bd170c05ea068e70eb32cc1f7a8b0d4f569ac44e531a42a8a4e1ebfe`

### Strong-channel validation

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/layer20_y20_strong_interaction_geometry_channel_validation.jsonl`

SHA256:

`111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942`

### Summary

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/summary.json`

SHA256:

`35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9ea9ef66a80798`

### Execution manifest

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/execution_manifest.json`

Corrected exact SHA256:

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee`

### Validated evidence report

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_validated_evidence_analysis_report_candidate.md`

SHA256:

`6e33951e0fe02ba6e82da6f59f786034ddbda869d2f0a2f592b9a51467477163`

---

## 5. Frozen scientific scope

The present analysis preserves exactly:

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
- existing strong interaction coordinate.

No new channel subset was selected.

No item subset was selected.

No new layer, lag, token, window, or projection was searched.

---

## 6. Execution boundary

This stage introduced no new model execution.

Validated execution flags:

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
- `raw_vectors_read_or_persisted = False`.

Therefore all new evidence in this stage is a deterministic static transform of already-frozen parent evidence.

---

## 7. Exact signed decomposition

For one item-role branch `b` and fixed strong channel `j`:

`u_b,j = 2 x_b,j y_b,j`.

Positive same-sign co-contribution mass:

`P_b = Σ_j max(u_b,j, 0)`.

Negative opposite-sign contribution mass:

`N_b = Σ_j min(u_b,j, 0)`.

Therefore:

`I_b = P_b + N_b`.

For one paired corr/ctrl item:

`ΔP = P_corr - P_ctrl`.

`ΔN = N_corr - N_ctrl`.

`ΔI = I_corr - I_ctrl`.

Exact paired identity:

`ΔI = ΔP + ΔN`.

Because `N <= 0`, a positive `ΔN` means the corr branch is less negatively cancelled.

This is termed:

**cancellation relief**.

---

## 8. Validated static execution

Run directory:

`reports/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_3af43c9_v1`

Execution marker:

`PASS_FROZEN_STRONG_ALIGNMENT_SIGN_CONTRIBUTION_STATIC_ANALYSIS`

Execution HEAD:

`3af43c91d791e2755b977529eb26d65db00c7870`

Item count:

`330`

Strong-channel count:

`240`

Model forward count:

`0`

---

## 9. Primary scientific result

Validated total interaction contrast:

`G_total = +0.760594723140676`

Positive co-contribution gain:

`G_pos = +0.5314816993853949`

Cancellation relief:

`G_relief = +0.22911302375528114`

Population closure:

`G_total = G_pos + G_relief`

up to floating representation.

Both scientific components are positive.

Therefore both reinforce the parent positive interaction contrast.

---

## 10. Absolute component attribution

Absolute component share for positive co-contribution gain:

`0.6987712157544046`

or:

`69.87712157544046%`.

Absolute component share for cancellation relief:

`0.3012287842455955`

or:

`30.12287842455955%`.

Largest absolute scientific component:

`positive_co_contribution_gain`.

Hence the preregistered outcome is:

**OUTCOME_A_POSITIVE_CO_CONTRIBUTION_DOMINANT**

or:

**Outcome A — positive co-contribution dominant.**

The result is not pure positive-mass gain.

Cancellation relief contributes materially, but remains secondary.

---

## 11. Role-level positive interaction mass

corr mean positive interaction mass:

`2.267583155684786`

ctrl mean positive interaction mass:

`1.7361014562993915`

Difference:

`+0.5314816993853949`

Therefore corr carries substantially more positive same-sign signed interaction mass in the fixed strong-240 subspace.

---

## 12. Role-level negative interaction mass

corr mean negative interaction mass:

`-0.5449066994977081`

ctrl mean negative interaction mass:

`-0.7740197232529892`

Difference:

`+0.22911302375528114`

Because the negative interaction mass is less negative for corr, this is correctly interpreted as:

**reduced opposite-sign cancellation**.

Thus corr benefits not only from more positive co-contribution but also from weaker net cancellation.

---

## 13. Same-sign participation count

corr mean same-sign channel count:

`140.5818181818182`

ctrl mean same-sign channel count:

`136.03030303030303`

Paired mean difference:

`+4.551515151515152`

Relative to the ctrl mean, this descriptive difference is approximately:

`+3.3459567832479394%`.

This count diagnostic supports increased same-sign participation in corr.

Count alone, however, does not determine the scientific outcome.

The per-channel mass diagnostic is required.

---

## 14. Opposite-sign participation count

corr mean opposite-sign channel count:

`99.41818181818182`

ctrl mean opposite-sign channel count:

`103.96969696969697`

Paired mean difference:

`-4.551515151515152`

Relative to the ctrl mean, this descriptive difference is approximately:

`-4.377732439522006%`.

Because zero-product channel count is zero, the paired count identity closes exactly:

`Δn_same + Δn_opp = 0`.

This means corr's extra same-sign participation corresponds one-for-one to fewer opposite-sign participating channels inside the fixed 240-channel partition.

---

## 15. Positive mass per same-sign channel

corr mean:

`0.016067288107596122`

ctrl mean:

`0.01250293600938431`

Paired mean difference:

`ΔPbar = +0.0035643520982118104`

Relative to the ctrl mean, this descriptive difference is approximately:

`+28.50812077688408%`.

Therefore the positive co-contribution advantage is not merely due to having more same-sign channels.

The average positive contribution per same-sign channel is also substantially larger in corr.

This establishes a two-part descriptive structure:

1. more same-sign participating channels;
2. stronger positive contribution per same-sign channel.

---

## 16. Cancellation magnitude per opposite-sign channel

Define cancellation magnitude:

`K_b = -N_b >= 0`.

corr mean cancellation magnitude per opposite-sign channel:

`0.005438501798898937`

ctrl mean:

`0.007360156739713312`

Paired mean difference:

`ΔKbar = -0.001921654940814375`

Relative to the ctrl mean, this descriptive difference is approximately:

`-26.108886111700197%`.

Therefore cancellation relief is also not purely a count effect.

corr has:

1. fewer opposite-sign participating channels;
2. weaker cancellation magnitude per remaining opposite-sign channel.

This is a genuine secondary signed-mass phenotype inside the frozen strong partition.

---

## 17. Combined count-plus-strength interpretation

The positive side is jointly associated with:

- `+4.551515151515152` same-sign channels on average;
- `+0.0035643520982118104` positive mass per same-sign channel.

The negative side is jointly associated with:

- `-4.551515151515152` opposite-sign channels on average;
- `-0.001921654940814375` cancellation magnitude per opposite-sign channel.

Therefore the validated static structure is not adequately described by:

- channel count alone;
- per-channel strength alone;
- cancellation relief alone.

The dominant pattern is:

**greater positive same-sign co-contribution through both broader participation and stronger per-channel contribution, supplemented by reduced opposite-sign cancellation through both fewer participating channels and weaker per-channel cancellation.**

---

## 18. Relation to parent alignment result

The immediate parent established:

`Q_C = +0.7326514559909479`

and:

`96.32612923813832%`

of the parent magnitude/alignment attribution was carried by the cosine-alignment component.

The present result shows how the fixed-coordinate signed interaction underlying that strong-subspace phenotype is organized:

- the larger corr interaction is mainly produced by larger positive same-sign co-contribution;
- a smaller but substantial part comes from cancellation relief.

This does not imply that the original alignment result was caused by sign-count changes.

The present analysis is a signed-coordinate decomposition of already-frozen strong-subspace interaction evidence.

It should therefore be interpreted as a structural localization beneath the parent alignment phenotype, not as an intervention on alignment itself.

---

## 19. Independent artifact validation

External validator:

`validate_frozen_strong_alignment_sign_contribution_artifacts.py`

Validated validator SHA256:

`c57e8ee2de8898442556c3bac968321d88dd5d8f40d324f515cf0be52c90dca0`

Validation marker:

`PASS_FROZEN_STRONG_ALIGNMENT_SIGN_CONTRIBUTION_ARTIFACT_VALIDATION`

Validated execution identity:

`analysis_git_head = 3af43c91d791e2755b977529eb26d65db00c7870`

Validated analyzer SHA256:

`6cb326e15f8f9b78aae2209f08672f6773fe5443669ccf6492448b2b8dc1e1bf`

---

## 20. Validated output identities

### Item output

File:

`frozen_strong_alignment_sign_contribution_item_metrics.jsonl`

SHA256:

`cd9d3306c7ef2b2d8f2a759be5e7945ccfdb86f0009981f61247b9ddfcf03974`

Rows:

`330`

### Summary output

File:

`summary.json`

SHA256:

`0696e5887dfbd2fc7eb55d3c726e4e55415858e15522d52efca6ce38ecda1615`

### Static-analysis manifest

File:

`static_analysis_manifest.json`

SHA256:

`5e1fdd97236d0e6f2d7b7041207d25af78aba7dae3b15cff9a66d760c365d50f`

---

## 21. Independent numerical validation

The external validator independently reproduced:

`G_total = 0.760594723140676`

`G_pos = 0.5314816993853949`

`G_relief = 0.22911302375528114`

Absolute shares:

- positive co-contribution:
  `0.6987712157544046`;
- cancellation relief:
  `0.3012287842455955`.

Validated outcome:

`OUTCOME_A_POSITIVE_CO_CONTRIBUTION_DOMINANT`

The external validator also reproduced:

`delta_same_sign_channel_count_mean = +4.551515151515152`

`delta_opposite_sign_channel_count_mean = -4.551515151515152`

`delta_Pbar_mean = +0.0035643520982118104`

`delta_Kbar_mean = -0.001921654940814375`

---

## 22. Exact identity validation

Maximum role-level closure residual:

`8.881784197001252e-16`

Maximum paired decomposition residual:

`1.1657341758564144e-15`

Maximum parent reproduction residual:

`0.0`

Maximum count-difference closure residual:

`0.0`

Maximum parent channel bridge residual from execution:

`2.7755575615628914e-17`

All are far below the frozen blocking tolerances.

Therefore:

- role sign decomposition closes;
- paired decomposition closes;
- parent total is reproduced exactly;
- same/opposite count identity closes;
- parent strong-channel bridge remains valid.

---

## 23. Code correctness

Code correctness evidence consists of:

1. frozen implementation at:
   `3af43c91d791e2755b977529eb26d65db00c7870`;
2. static preflight PASS;
3. exact authority authentication;
4. exact parent artifact SHA authentication;
5. synthetic sign-decomposition closure;
6. deterministic no-model execution boundary;
7. independent artifact validator PASS.

No code correctness blocker remains for this stage.

---

## 24. Execution success

Static execution completed with:

`PASS_FROZEN_STRONG_ALIGNMENT_SIGN_CONTRIBUTION_STATIC_ANALYSIS`

No model forward was executed.

No checkpoint was loaded.

No handoff ZIP was opened.

The static run emitted exactly the intended evidence classes:

- item-level scalar metrics;
- summary;
- static-analysis manifest.

Execution success is established.

---

## 25. Artifact / provenance validity

Independent artifact validation passed.

The validator confirmed:

- analysis HEAD;
- analyzer SHA;
- authority identities;
- amendment identities;
- parent evidence freeze;
- all five frozen parent artifact hashes;
- output file hashes;
- 330 item rows;
- population identities;
- scientific outcome fields;
- forbidden execution flags.

Therefore artifact/provenance validity is established.

---

## 26. Scientific conclusion

With code correctness, execution success, and artifact/provenance validity established, the stage supports the following validated scientific conclusion:

**Within the frozen strong-240 `R20×Y20` interaction at layer20→21, corr's larger interaction is primarily associated with increased positive same-sign co-contribution rather than cancellation relief. Positive co-contribution contributes approximately 69.88% of the absolute two-component signed-mass attribution, while reduced negative cancellation contributes approximately 30.12%. Both reinforce the positive corr advantage.**

The positive side is structurally associated with both:

- more same-sign channels;
- larger positive mass per same-sign channel.

The cancellation-relief side is structurally associated with both:

- fewer opposite-sign channels;
- smaller cancellation magnitude per opposite-sign channel.

A concise mechanistic description is therefore:

**frozen-strong positive co-contribution dominance with secondary cancellation relief, jointly expressed through participation-count and per-channel-strength shifts.**

---

## 27. Scientific limits

This result does not establish:

- that sign pattern causes the downstream behavior;
- that changing same-sign count would change the final task decision;
- that changing per-channel signed mass would causally alter performance;
- that the phenotype generalizes outside the frozen common-330 cohort;
- that the result applies outside `k=2`;
- that the result applies outside the frozen strong-240 partition;
- that the result is layer-global;
- that the result defines a new learned subspace;
- that the result supersedes the parent alignment decomposition.

No causal intervention was performed.

The conclusion is observational/algebraic.

---

## 28. No weak-partition promotion

This stage does not promote any weak-partition result.

The scientific claim is restricted to the frozen strong-240 partition inherited from the parent stage.

No comparison to a newly selected weak subset is authorized by this evidence.

---

## 29. No post-hoc subset claim

No top-k channels were selected.

No item subgroup was selected.

No alternate threshold was introduced.

No channel ranking was used to define the primary result.

The fixed 240 strong channels were preserved exactly.

Therefore the primary conclusion is not the result of post-hoc subset search.

---

## 30. Final classification

**Validated outcome: Outcome A — positive co-contribution dominant.**

Primary component:

`G_pos = +0.5314816993853949`

Secondary reinforcing component:

`G_relief = +0.22911302375528114`

Parent total:

`G_total = +0.760594723140676`

Absolute attribution:

- positive co-contribution gain:
  `69.87712157544046%`;
- cancellation relief:
  `30.12287842455955%`.

Validated descriptive refinement:

**positive co-contribution dominance with secondary cancellation relief, with both count and per-channel-strength structure on both sides of the signed decomposition.**
