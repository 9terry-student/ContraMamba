# ContraMamba K0-RVG Strong-Side Directional-Alignment Causal Falsification
## Static Design Candidate

## 1. Status

**Phase:** prospective K0 causal falsification / intervention design.

**Immediate authority / stopping-boundary freeze:**

`f5cf85692bf8900f019a06979c54e637144b2d68`

**Immediate authority file:**

`reports/longterm_k0_rvg_native_state_mechanism_synthesis_stopping_boundary_report_candidate.md`

**Immediate authority Git blob:**

`37c9c9ee2384a3d409ad4fd27482299217cf8c0e`

This document is a **prospective static scientific design**. It does not contain new model evidence.

No implementation, model execution, intervention execution, training, task evaluation, logits readout, Kaggle run, or K1 work is authorized until this design is reviewed and frozen.

---

## 2. Why this stage exists

The frozen K0 synthesis closed routine observational decomposition and required the next K0 stage, if any, to answer a different class of question:

> Does a pre-specified perturbation falsify or support a central functional link in the frozen mechanism chain?

The preferred next evidence class is:

**minimal causal falsification of the strong-side directional-alignment mechanism.**

The present design therefore does not ask for another scalar correlate, another top-k list, another channel subgroup, or another descriptive factorization.

It asks whether a pre-specified representation-level perturbation that removes the already-frozen corr-vs-ctrl strong-side directional-alignment excess causes the already-frozen downstream native write/state phenotype to decrease.

---

## 3. Immediate frozen scientific parent

The immediate mechanism parent is the validated layer20→21 strong-partition interaction geometry.

**Evidence freeze:**

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`

**Static design freeze:**

`2b239beb646c681f2032b6cfcaca15a34d25ad34`

**Implementation / execution commit:**

`0b1168182a265bc405652d2ce519f63310433c3e`

**Runner:**

`scripts/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_audit.py`

**Runner SHA256:**

`0c487e3d29236e298f8c74cbdbcdfce3599af587dfd27a6e5fcf169df64eade5`

**Runner Git blob:**

`007d9ec21487ccfe7c56d6410d1dac6f832a9a22`

Validated artifact identities:

- item metrics SHA256:
  `e7002e03bd170c05ea068e70eb32cc1f7a8b0d4f569ac44e531a42a8a4e1ebfe`
- strong-channel validation SHA256:
  `111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942`
- summary SHA256:
  `35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9a51467477163`
- execution manifest SHA256:
  `1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee`
- validated report SHA256:
  `6e33951e0fe02ba6e82da6f59f786034ddbda869d2f0a2f592b9a51467477163`

Frozen population decomposition:

- `Q_A = +0.01818378200815921`
- `Q_B = +0.009759485141568866`
- `Q_C = +0.7326514559909479`
- total `= +0.760594723140676`

Absolute component shares:

- R20-magnitude term: `2.3907320751679763%`
- Y20-magnitude term: `1.2831386866937017%`
- directional-alignment term: `96.32612923813832%`

Frozen role means:

- `A_corr = 1.5174351424881483`
- `A_ctrl = 1.4829015640324512`
- `B_corr = 1.2868907438993054`
- `B_ctrl = 1.2592471445236473`
- `C_corr = 0.4364279014925435`
- `C_ctrl = 0.23896035194570278`

The present experiment targets the **directional-alignment excess**, not the already-small magnitude components.

---

## 4. Frozen downstream native-state endpoint

The downstream endpoint is not newly invented.

It is reused from the validated layer-22 carry/write factorization.

**Evidence freeze:**

`21b36fa4579bee53644fa7f3da84cdc947ddef5b`

**Implementation commit:**

`378ac88cf3413dc46c9c7fa153ef60785fd370a4`

**Run:**

`reports/longterm_k0_rvg_layer22_carry_write_factorization_378ac88_v1`

Artifact identities:

- item metrics SHA256:
  `c9aa47f5cff0b8d4bd7ca81dfcef886e47fef02df9d6ca4e8e2dc2419367f2ad`
- summary SHA256:
  `b17b8bc0a6aa679f0f18e0eb3f064d1d9e50b5bea135ecb026836f4fd28461f4`

Relevant row schema:

`k0-rvg-layer22-carry-write-factorization-row-v1`

Exact baseline fields to be bridged:

- `local_template_index`
- `stable_item_id`
- `role`
- `relative_coordinate`
- `in_common_ddsssss_cohort`
- `delta_s22_post_l2`
- `delta_w_l2`
- `delta_carry_l2`

At common-330, `k=2`, frozen population means include:

- corr `delta_s22_post_l2 = 9.952563381180072`
- ctrl `delta_s22_post_l2 = 4.499298567842198`
- corr `delta_w_l2 = 9.89307668899622`
- ctrl `delta_w_l2 = 4.409111767725006`
- corr `delta_carry_l2 = 1.0152018733903254`
- ctrl `delta_carry_l2 = 0.8274459872454784`

Frozen itemwise role count at `k=2`:

- post-state corr > ctrl: `322/330`
- write corr > ctrl: `322/330`
- carry corr > ctrl: `330/330`

These values are baseline evidence only. The intervention execution must reproduce the frozen per-item baseline rows rather than trusting only these aggregates.

---

## 5. Scientific question

The frozen observational evidence says that the strong-side layer20→21 interaction contrast is overwhelmingly directional-alignment dominated.

The causal question is:

> If the corr strong-side directional alignment is minimally equalized to the paired ctrl alignment while preserving the corr source magnitudes and the branch midpoint, does the downstream layer-22 native write/post-state excess decrease?

A secondary specificity question is:

> Is the downstream effect of alignment equalization larger than the effect of transplanting only the small frozen source-magnitude differences while preserving corr alignment?

---

## 6. Exact causal claim under test

The prospective claim is deliberately narrow:

**Within the frozen common-330, k=2, layer20→21→layer22 pathway, the corr excess in strong-side R20×Y20 directional alignment makes a causal contribution to the downstream layer-22 current-write/post-state excess.**

This design does **not** test:

- causal necessity of any individual channel;
- causal sufficiency of any channel set;
- task accuracy;
- final decisions or logits;
- generalization to other seeds, datasets, layers, architectures, or coordinates;
- whether R20 or Y20 is the ultimate upstream origin;
- whether the strong-240 partition is globally optimal;
- whether the concentrated effective channel subset is causal.

---

## 7. Intervention boundary

The intervention boundary is:

**layer-22 mixer x-branch, immediately after the hidden in-projection split and before the depthwise convolution, at the current k=2 token.**

The intervention applies only to the fixed **strong-240** intermediate channels inherited from the frozen lag-0 kernel partition.

The z/gate branch is untouched.

All weak/equal channels are untouched.

All other tokens are untouched.

All other layers are untouched.

No weights are changed.

No upstream residual, RMSNorm, or source activations are changed.

This is therefore a **representation-level intervention at the already-localized strong routing boundary**.

It must not be interpreted as an intervention on the biological/semantic “origin” of R20 or Y20.

---

## 8. Why the strong-240 population is used

The terminal channel-mass stage showed that positive mass is concentrated across channels.

However, this causal experiment does **not** promote a top-k subset into a new scientific population.

The fixed strong-240 partition is retained because:

1. it was defined before the terminal concentration result;
2. it is the exact population used in the frozen interaction geometry;
3. the causal question is about the already-frozen strong-side alignment mechanism;
4. selecting the top 10, top 28, top 32, or another post-hoc concentrated subset would re-open channel archaeology and confound the causal test with data-dependent selection.

Therefore all 240 frozen strong channels are the intervention domain.

---

## 9. Baseline source vectors

For each common item `i` and role `r ∈ {corr, ctrl}`, reconstruct the exact frozen geometry quantities using the authenticated parent runner.

Let:

- `R20_m`, `R20_s` be matched/swapped layer-20 residual vectors;
- `Y20_m`, `Y20_s` be matched/swapped layer-20 update vectors;
- `X22_m`, `X22_s` be matched/swapped layer-22 RMSNorm outputs;
- `s_m`, `s_s` be matched/swapped RMS scales;
- `gamma` be the frozen layer-22 RMSNorm weight;
- `W_hidden` be the hidden half of the layer-22 in-projection;
- `S` be the fixed strong-240 mask.

Define:

`ΔR20 = R20_m - R20_s`

`ΔY20 = Y20_m - Y20_s`

`ΔX22 = X22_m - X22_s`

`s_bar = (s_m + s_s) / 2`

`q_R = gamma * (s_bar * ΔR20)`

`q_Y = gamma * (s_bar * ΔY20)`

`h_R = W_hidden q_R`

`h_Y = W_hidden q_Y`

`d = ||ΔX22||₂`

Strong normalized source vectors:

`x = h_R[S] / d`

`y = h_Y[S] / d`

Then:

`A = ||x||₂`

`B = ||y||₂`

`C = cos(x,y)`

`I = 2 <x,y>`

The implementation must reproduce the frozen per-item `A`, `B`, `C`, and `I` values before any intervention.

---

## 10. Primary intervention: paired ctrl-cosine transplant

The primary intervention is **corr only**.

For common item `i`, use the paired corr and ctrl roles sharing the same `stable_item_id`.

Let corr baseline vectors be:

`x_c`, `y_c`

with:

`A_c = ||x_c||₂`

`B_c = ||y_c||₂`

`C_c = cos(x_c,y_c)`

Let the paired ctrl baseline cosine be:

`C_t = C_ctrl`.

Define the corr R20 unit direction:

`u = x_c / A_c`

Decompose the corr Y20 direction into components parallel and orthogonal to `u`.

Let:

`v_raw = y_c / B_c - C_c u`

`v = v_raw / ||v_raw||₂`

The target Y20 vector is:

`y* = B_c [ C_t u + sqrt(1 - C_t^2) v ]`

with numerical boundary handling only for values within a preregistered floating-point slack of `[-1,1]`.

This construction:

- preserves `A_c`;
- preserves `B_c`;
- preserves the corr R20 direction;
- preserves the corr Y20-side norm;
- changes only the corr R20–Y20 cosine;
- sets that cosine to the paired ctrl baseline cosine;
- keeps the target in the original corr `span{x_c, y_c}`;
- introduces no learned direction and no data-fitted basis.

The normalized correction is:

`δ_norm = y* - y_c`

The pre-convolution strong-channel correction in runtime x-branch units is:

`δ_h = d_c * δ_norm`.

---

## 11. Branch-symmetric application

Let the baseline corr layer-22 x-branch strong activation at the intervention token be:

- matched: `H_m[S]`
- swapped: `H_s[S]`

Apply:

`H'_m[S] = H_m[S] + 0.5 δ_h`

`H'_s[S] = H_s[S] - 0.5 δ_h`

Therefore:

`0.5(H'_m[S] + H'_s[S]) = 0.5(H_m[S] + H_s[S])`

and:

`H'_m[S] - H'_s[S] = H_m[S] - H_s[S] + δ_h`.

Thus the branch midpoint is exactly preserved while only the matched-minus-swapped strong difference is changed.

This symmetry is mandatory.

A one-sided matched-only or swapped-only perturbation is not authorized by this design.

---

## 12. Primary manipulation checks

For every item, before downstream interpretation, the implementation must verify:

1. corr baseline source metrics reproduce frozen `A_corr`, `B_corr`, `C_corr`, `I_corr`;
2. ctrl baseline source metrics reproduce frozen `A_ctrl`, `B_ctrl`, `C_ctrl`, `I_ctrl`;
3. intervention keeps `A_corr` unchanged;
4. intervention keeps `B_corr` unchanged;
5. intervention target cosine equals paired baseline `C_ctrl`;
6. the strong mask contains exactly 240 channels;
7. only strong-240 current-token x-branch values are modified;
8. z/gate branch is byte-identical to baseline capture at the hook;
9. weak/equal x-branch channels are byte-identical at the hook;
10. all earlier tokens are byte-identical at the hook;
11. branch midpoint preservation closes numerically;
12. the applied runtime delta matches the intended float64 construction after runtime-dtype casting within preregistered tolerance.

If any mandatory manipulation check fails, the scientific run is invalid.

---

## 13. Component-specific control: paired magnitude transplant

A second corr-only intervention is included as a mechanistic specificity control.

It targets the small frozen magnitude components while preserving corr directional alignment.

For corr baseline:

`x_c`, `y_c`, `A_c`, `B_c`, `C_c`

and paired ctrl magnitudes:

`A_t = A_ctrl`

`B_t = B_ctrl`

define:

`x_M = (A_t / A_c) x_c`

`y_M = (B_t / B_c) y_c`.

This preserves the corr directions and therefore:

`cos(x_M, y_M) = C_c`.

The normalized correction is:

`δ_norm,M = (x_M - x_c) + (y_M - y_c)`.

Runtime correction:

`δ_h,M = d_c * δ_norm,M`.

Apply the same branch-symmetric rule:

`H^M_m[S] = H_m[S] + 0.5 δ_h,M`

`H^M_s[S] = H_s[S] - 0.5 δ_h,M`.

The magnitude control is not perturbation-norm matched to the alignment intervention.

Its role is narrower:

**it is a decomposition-aligned mechanistic control corresponding to the already-frozen small `Q_A + Q_B` components.**

The itemwise L2 size of both interventions must be reported so that downstream effect size is not interpreted without perturbation-size context.

---

## 14. No-op / baseline control

The execution must include a baseline pass through the same runner with zero intervention.

The baseline run must reproduce, for every common-330 item and each role at `k=2`, the frozen carry/write artifact values:

- `delta_s22_post_l2`
- `delta_w_l2`
- `delta_carry_l2`

The bridge is itemwise, not aggregate-only.

The baseline also must reproduce the frozen geometry item metrics.

A failure to reproduce frozen baseline evidence blocks interpretation of all intervention results.

---

## 15. Downstream recurrence capture

After the layer-22 x-branch perturbation, the remainder of the frozen model forward is executed unchanged.

The existing native recurrence observer semantics are reused.

For matched and swapped branches, capture at layer 22 / `k=2`:

- `S_prev`
- `G`
- `W`
- `S_post`

with the frozen runtime identity:

`S_post32 = fl(G32 * S_prev32 + W32)`.

Define:

`Carry32 = G32 * S_prev32`.

For each condition, compute matched-minus-swapped vector norms exactly as in the frozen carry/write stage:

- `delta_s22_post_l2`
- `delta_w_l2`
- `delta_carry_l2`.

No new state metric replaces these primary frozen endpoints.

---

## 16. Primary endpoint

The **primary endpoint** is the common-330 population mean reduction in corr post-state divergence under the alignment transplant.

For item `i`, let:

- `S_c0(i)` = baseline corr `delta_s22_post_l2`
- `S_t0(i)` = baseline ctrl `delta_s22_post_l2`
- `S_cA(i)` = alignment-transplanted corr `delta_s22_post_l2`
- `S_cM(i)` = magnitude-transplanted corr `delta_s22_post_l2`.

Define baseline paired excess:

`G_S0 = mean_i [S_c0(i) - S_t0(i)]`.

Alignment-intervention paired excess:

`G_SA = mean_i [S_cA(i) - S_t0(i)]`.

Magnitude-control paired excess:

`G_SM = mean_i [S_cM(i) - S_t0(i)]`.

Primary alignment reduction:

`R_SA = G_S0 - G_SA = mean_i [S_c0(i) - S_cA(i)]`.

Magnitude-control reduction:

`R_SM = G_S0 - G_SM = mean_i [S_c0(i) - S_cM(i)]`.

Descriptive closure fraction, only if `G_S0 > 0`:

`F_SA = R_SA / G_S0`.

No arbitrary minimum closure percentage is preregistered.

---

## 17. Write bridge endpoint

The key mechanistic bridge endpoint is the frozen current-write norm.

Analogously define:

- `W_c0(i)`
- `W_t0(i)`
- `W_cA(i)`
- `W_cM(i)`

from `delta_w_l2`.

Then:

`G_W0 = mean_i [W_c0(i) - W_t0(i)]`

`G_WA = mean_i [W_cA(i) - W_t0(i)]`

`G_WM = mean_i [W_cM(i) - W_t0(i)]`

`R_WA = mean_i [W_c0(i) - W_cA(i)]`

`R_WM = mean_i [W_c0(i) - W_cM(i)]`.

This endpoint tests whether any post-state effect is expressed through the already-frozen write-dominant mechanism.

---

## 18. Carry specificity endpoint

For `delta_carry_l2`, define the same baseline and intervention quantities:

`R_CA = mean_i [C_c0(i) - C_cA(i)]`

`R_CM = mean_i [C_c0(i) - C_cM(i)]`.

Carry is supporting specificity evidence only.

Because x-branch intervention can alter the layer-22 gate dynamics downstream of convolution/x-projection, carry is **not required to be invariant**.

The expected write-dominant pattern is instead:

`|R_WA| > |R_CA|`

when the primary intervention produces a nonzero effect.

This comparison is supporting, not the primary causal decision rule.

---

## 19. Itemwise breadth diagnostics

The paired ctrl cosine is the item-specific intervention target.

Therefore some items may have:

`C_corr > C_ctrl`

while others may have:

`C_corr < C_ctrl`.

The intervention is never selectively disabled based on that sign.

Report separately:

- count with `C_corr > C_ctrl`;
- count with `C_corr < C_ctrl`;
- count equal;
- within each sign stratum, counts of `S_cA < S_c0`, `S_cA > S_c0`, and equal;
- within each sign stratum, counts of `W_cA < W_c0`, `W_cA > W_c0`, and equal.

For the magnitude control report the unconditional state/write increase/decrease/equal counts.

Also report paired medians of:

- `S_c0 - S_cA`
- `S_c0 - S_cM`
- `W_c0 - W_cA`
- `W_c0 - W_cM`.

These are breadth diagnostics only.

They are not used as an additional primary threshold because the item-specific transplant can legitimately increase or decrease alignment depending on the paired baseline geometry.

---

## 20. Primary outcome classes

The result is classified prospectively.

### Outcome A — alignment-specific causal contribution supported

All must hold:

1. `R_SA > 0`;
2. `R_WA > 0`;
3. `R_SA > R_SM`;
4. `R_WA > R_WM`;
5. all mandatory manipulation checks pass.

Interpretation:

**paired ctrl-cosine equalization causally reduces the frozen corr write/post-state phenotype more than the frozen magnitude-component control.**

This supports a causal contribution of strong-side directional alignment.

It does not establish individual-channel necessity, sufficiency, or task-level causality.

### Outcome B — causal perturbation effect without alignment specificity

All mandatory manipulation checks pass, both:

- `R_SA > 0`
- `R_WA > 0`

but at least one component-specific comparison fails:

- `R_SA <= R_SM`, or
- `R_WA <= R_WM`.

Interpretation:

**the alignment transplant affects the downstream write/state phenotype, but the result does not isolate directional alignment as the privileged causal component over the preregistered magnitude control.**

### Outcome C — chain-discordant causal effect

All mandatory manipulation checks pass and exactly one of:

- `R_SA > 0`
- `R_WA > 0`

holds.

Interpretation:

**the intervention changes one frozen downstream component in the expected direction but not the other.**

If `R_WA > 0` and `R_SA <= 0`, the stronger alignment→post-state link is falsified while a narrower alignment→write effect remains.

If `R_SA > 0` and `R_WA <= 0`, the post-state change is not expressed through the previously frozen write-dominant bridge, so the proposed causal chain is not supported as specified.

### Outcome D — direct causal falsification

Mandatory manipulation checks pass and both:

- `R_SA <= 0`
- `R_WA <= 0`.

Interpretation:

**paired ctrl-cosine equalization does not reduce either the frozen write or post-state phenotype.**

The central causal-contribution claim is falsified under the pre-specified intervention.

### Invalid

Any mandatory baseline-authentication, manipulation, provenance, or output-integrity gate fails.

No scientific classification is allowed.

---

## 21. Why there is no arbitrary percentage threshold

The frozen evidence is deterministic, not a noisy stochastic training comparison.

This experiment therefore uses:

- exact baseline reproduction;
- exact manipulation-target checks;
- direction of downstream change;
- comparison against the decomposition-aligned magnitude control;
- itemwise majority breadth.

No post-hoc “20%”, “50%”, or other effect-size cutoff is introduced.

Closure fractions are reported descriptively.

---

## 22. Population and pairing

Scientific population:

- exactly the frozen common DDSSSSS cohort;
- exactly `330` stable items;
- paired corr and ctrl role per item;
- exact `stable_item_id` equality required;
- exact local-template identity required.

Relative coordinate:

`k = 2`.

Source block:

`20`.

Target residual layer:

`21`.

Intervention / downstream mixer layer:

`22`.

No other item, role, layer, or coordinate enters the primary experiment.

---

## 23. Runtime protocol

Use the existing authenticated equal-length matched/swapped prefix construction.

The implementation should preserve the frozen causal prefix protocol used by the parent stack.

The intervention is activated only at the layer-22 current `k=2` token.

The baseline and intervention conditions must use identical token sequences and model parameters.

No tokenizer call is allowed during scientific execution if the frozen parent stack can supply the authenticated token IDs, consistent with prior K0 runs.

---

## 24. Full model-forward budget

For each of 330 items:

### Baseline source / endpoint reconstruction

- corr matched
- corr swapped
- ctrl matched
- ctrl swapped

`4 × 330 = 1320` forwards.

### Alignment-transplant corr intervention

- corr matched
- corr swapped

`2 × 330 = 660` forwards.

### Magnitude-transplant corr control

- corr matched
- corr swapped

`2 × 330 = 660` forwards.

Maximum planned scientific execution:

`2640` model forwards.

No additional scientific condition is authorized in the same run.

---

## 25. Preflight budget

After implementation is frozen, a bounded runtime preflight may execute at most:

- `2` fixed common items;
- all baseline/alignment/magnitude conditions needed to exercise the hook;
- maximum `16` model forwards.

The preflight is operational validation only.

It produces no scientific evidence artifact and requires no separate preflight authority document.

The preflight item identities must be selected by a fixed deterministic rule, such as the first two sorted `local_template_index` values, not by observed effect size.

---

## 26. Compute environment

Training/evaluation:

**NOT ALLOWED.**

Task heads / logits:

**NOT READ.**

Kaggle:

**NOT REQUIRED / NOT AUTHORIZED BY DEFAULT.**

GPU:

**NOT REQUIRED / NOT AUTHORIZED BY DEFAULT.**

The intended execution is local deterministic inference/capture using the already-frozen checkpoint and runtime identity.

If the local runtime cannot execute the bounded design safely, that is an operational blocker to resolve before changing the scientific design.

---

## 27. Frozen checkpoint and runtime identities

Checkpoint:

`files/reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0/selected_checkpoint.pt`

Checkpoint SHA256:

`4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c`

Handoff ZIP SHA256:

`96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861`

Encoder canonical digest:

`48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597`

Encoder raw-concat digest:

`968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae`

Mamba source SHA256:

`23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`

HF model:

`state-spaces/mamba-130m-hf`

Frozen revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

The implementation must authenticate the same runtime lineage used by the parent K0 evidence.

---

## 28. Numerical construction requirements

All intervention geometry must be constructed in float64 on detached CPU tensors from authenticated baseline captures.

The actual correction is cast only when applied to the runtime x-branch tensor.

Mandatory numerical blockers include:

- non-finite source vector;
- zero `A` or `B`;
- degenerate orthogonal component preventing definition of `v`;
- cosine outside `[-1,1]` beyond floating-point boundary slack;
- strong mask count not exactly 240;
- item/role pairing mismatch;
- applied correction mismatch beyond preregistered runtime-cast tolerance;
- midpoint-preservation failure;
- baseline frozen-evidence reproduction failure.

No silent fallback direction is allowed for a degenerate geometry item.

A degenerate item blocks the run and requires design review rather than ad hoc basis selection.

---

## 29. Baseline authentication requirements

Before intervention execution, authenticate:

1. current branch:
   `longterm-k-series-native-state-kinematics`;
2. frozen stopping-boundary commit is an ancestor;
3. frozen geometry evidence commit is an ancestor;
4. frozen carry/write evidence commit is an ancestor;
5. frozen geometry runner SHA/blob;
6. frozen geometry item/summary/manifest identities;
7. frozen carry/write item/summary identities;
8. frozen checkpoint/handoff identities;
9. worktree cleanliness except explicitly known unrelated untracked K1/validator files if still present;
10. implementation runner tracked and byte-identical to its frozen implementation commit.

No provenance mismatch is a warning. It is a blocker.

---

## 30. Required baseline bridges

The runner must independently reproduce the following before accepting intervention data.

### Geometry bridge

For all common-330 paired roles:

- `A`
- `B`
- `C`
- `I`
- parent strong interaction target.

### Native-state bridge

For all common-330 paired roles at `k=2`:

- `delta_s22_post_l2`
- `delta_w_l2`
- `delta_carry_l2`.

The per-item frozen artifact is the authority for this bridge.

Aggregate agreement alone is insufficient.

---

## 31. Required output set

A successful scientific execution writes exactly three evidence files:

1. `strong_alignment_causal_falsification_item_metrics.jsonl`
2. `summary.json`
3. `execution_manifest.json`

No raw activation vectors are persisted.

No checkpoint copy is persisted.

No logits are persisted.

No token text is persisted if it is not already part of authenticated parent metadata.

Temporary `.partial` files must be atomically removed/renamed on success and absent from a valid final run directory.

---

## 32. Item artifact content

Each item row must contain at minimum:

Identity:

- schema version
- local template index
- stable item ID
- common-cohort flag
- source/target/intervention layers
- relative coordinate.

Frozen baseline geometry bridge:

- `A_corr`, `A_ctrl`
- `B_corr`, `B_ctrl`
- `C_corr`, `C_ctrl`
- `I_corr`, `I_ctrl`
- reproduction residuals.

Alignment manipulation:

- target cosine
- realized cosine
- `A` preservation residual
- `B` preservation residual
- midpoint residual
- runtime correction L2
- applied-correction residual.

Magnitude control manipulation:

- target `A_ctrl`
- target `B_ctrl`
- realized A/B
- realized cosine
- cosine preservation residual
- midpoint residual
- runtime correction L2
- applied-correction residual.

Native endpoint values:

- baseline corr/ctrl state/write/carry norms;
- alignment corr state/write/carry norms;
- magnitude corr state/write/carry norms;
- itemwise baseline-minus-intervention changes.

No raw 240-vectors are persisted.

---

## 33. Summary artifact content

The summary must include:

- exact population size;
- exact forward count;
- baseline aggregate bridges;
- geometry manipulation-check maxima;
- primary state gaps/reductions;
- write gaps/reductions;
- carry changes;
- closure fractions;
- intervention L2 distributions;
- itemwise reduction counts;
- magnitude-control comparison;
- primary outcome classification;
- explicit forbidden-action flags.

The summary may report means, medians, minima, maxima, and counts.

It must not create new post-hoc channel groups.

---

## 34. Execution manifest content

The manifest must include:

- runtime branch;
- runtime git HEAD;
- design freeze commit and design file identity;
- implementation commit;
- runner SHA256 and Git blob;
- stopping-boundary freeze identity;
- geometry parent evidence identities;
- carry/write parent evidence identities;
- checkpoint/handoff/runtime identities;
- exact model-forward count;
- output file SHA256 hashes;
- `training_executed = false`;
- `task_heads_executed = false`;
- `logits_read = false`;
- `tokenizer_invoked` truthfully recorded;
- `causal_intervention_executed = true`;
- `raw_vectors_persisted = false`;
- `k1_executed = false`.

---

## 35. No scientific search during execution

The runner must not:

- choose channels by observed intervention effect;
- tune cosine targets;
- tune intervention strength;
- choose items by observed response;
- sweep layers;
- sweep coordinates;
- sweep top-k channel counts;
- compare multiple rotations;
- search for a better control;
- inspect task outcomes to refine the perturbation.

The only target is the paired ctrl cosine fixed before downstream intervention outcomes are observed.

---

## 36. Why paired ctrl cosine is the target

The target is not zero alignment and not random alignment.

Zeroing/orthogonalizing the corr interaction would introduce a larger and less natural perturbation than required to test the frozen role contrast.

The paired ctrl cosine is preferred because it asks the minimal role-equalization question:

> What happens if corr keeps its source magnitudes but loses exactly the directional-alignment advantage that distinguishes it from its paired ctrl role?

This is the narrowest intervention directly matched to the frozen observational claim.

---

## 37. Why the intervention is corr-only

The central frozen phenotype is an excess of corr over ctrl.

Changing both roles would make the downstream comparison harder to interpret and would not directly test removal of the corr excess.

Therefore:

- ctrl remains an unperturbed paired reference;
- corr receives the alignment transplant;
- corr also receives the separate magnitude control.

This preserves a fixed comparator and minimizes the number of intervention conditions.

---

## 38. Interpretation boundary

A positive result establishes only a causal contribution at the tested representation boundary.

It does not prove that:

- upstream R20 or Y20 semantics are causal;
- the same effect exists outside common-330;
- the same effect exists outside k=2;
- the same effect controls logits or task decisions;
- the concentrated channel subset is necessary;
- the mechanism generalizes.

A negative result is scientifically useful.

If the manipulation succeeds but the downstream write/state phenotype does not decrease, the observational alignment dominance is not sufficient evidence for the proposed functional link.

---

## 39. Independent validation requirement

Because this stage crosses from observational decomposition into intervention evidence, independent artifact validation is mandatory before any evidence freeze.

The validator must independently check:

- exact output set;
- no `.partial` residue;
- all provenance identities;
- item count;
- forward count;
- baseline geometry reproduction;
- baseline native-state reproduction;
- paired identity;
- manipulation algebra;
- midpoint preservation;
- cosine transplant;
- magnitude-control preservation;
- summary recomputation from item rows;
- output hashes;
- forbidden-action flags.

The validator must not import the execution runner's scientific aggregation functions.

---

## 40. Evidence separation

The workflow must keep separate:

1. implementation correctness;
2. execution success;
3. artifact/provenance validity;
4. scientific classification.

A PASS execution marker alone is not evidence freeze.

A successful intervention that fails baseline reproduction is invalid.

A valid artifact with Outcome D remains a scientifically successful falsification result.

---

## 41. Stop conditions

Stop immediately and do not interpret if:

- branch/provenance identity mismatches;
- frozen parent artifacts drift;
- baseline item bridges fail;
- source geometry is degenerate for any required item;
- manipulation checks fail;
- unexpected worktree changes appear;
- output directory already exists;
- model forward count exceeds the authorized budget;
- task logits/heads are touched;
- raw vectors are persisted;
- a new channel selection or hyperparameter choice is introduced.

---

## 42. What this design authorizes after freeze

After this design itself is frozen, it authorizes only:

1. implementation of one bounded causal-intervention runner;
2. independent verification of that implementation because intervention semantics are high-risk;
3. bounded no-evidence preflight;
4. if implementation and preflight pass, one full local scientific execution under the exact `2640`-forward budget;
5. independent artifact validation;
6. a validated evidence report and evidence freeze if all gates pass.

No training/evaluation is authorized.

No K1 transition is authorized.

No follow-up intervention is authorized by default.

---

## 43. Implementation verification class

This is a high-risk change because it alters model hidden-state semantics during forward execution.

Therefore implementation requires:

- one implementer;
- one independent verifier.

The verifier must specifically inspect:

- hook location;
- x-branch versus z/gate separation;
- strong-mask identity;
- current-token targeting;
- branch-symmetric application;
- float64 construction / runtime cast;
- no-op baseline path;
- no accidental persistence of raw vectors;
- forward budget accounting;
- absence of task-head/logit access.

---

## 44. Expected run naming

After implementation is frozen at commit `<IMPLEMENTATION_SHA>`, use:

`reports/longterm_k0_rvg_strong_alignment_causal_falsification_<shortimpl>_v1`

where `<shortimpl>` is the first seven hexadecimal characters of the frozen implementation commit.

Do not reuse a run directory from another commit.

---

## 45. Expected execution marker

On successful full execution:

`PASS_K0_RVG_STRONG_ALIGNMENT_CAUSAL_FALSIFICATION_EXECUTION`

This marker means only that the authorized run completed and internal gates passed.

It does not by itself state Outcome A/B/C/D.

---

## 46. Scientific stopping rule after this experiment

After validated causal classification:

- Outcome A: freeze the causal contribution result and stop K0 by default; do not immediately search individual concentrated channels.
- Outcome B: freeze the non-specific causal perturbation result; do not rescue alignment specificity with post-hoc retuning.
- Outcome C: freeze the write-only / post-state-falsification result; reconsider the downstream link only under a new scientific design.
- Outcome D: freeze the causal falsification; do not continue observational decomposition to “explain away” the negative result.
- Invalid: repair only the demonstrated implementation/provenance defect; do not change the scientific target unless the frozen design itself is shown defective.

---

## 47. Final design decision

The next K0 scientific test is:

**paired ctrl-cosine transplantation of the corr strong-240 R20×Y20 directional alignment at the layer-22 pre-convolution x-branch, with branch midpoint and corr source magnitudes preserved, followed by the existing frozen layer-22 write/post-state endpoints.**

The paired magnitude transplant is the single pre-specified component-specific control.

This is a minimal falsification test of the central frozen alignment mechanism without reopening routine decomposition or post-hoc channel selection.

**Status: STATIC DESIGN CANDIDATE — NOT YET FROZEN.**
