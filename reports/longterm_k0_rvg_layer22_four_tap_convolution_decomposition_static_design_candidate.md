# ContraMamba K0-RVG Layer-22 Four-Tap Causal-Convolution Decomposition — Static Design Candidate

## 1. Authority and phase

**Phase:** scientific static design only.

This document defines the next bounded K0-RVG observational/algebraic audit after the validated layer-22 U-path source-localization evidence freeze.

It authorizes only:

- definition of the exact layer-22 four-tap causal-convolution decomposition;
- definition of its fixed scalar metrics;
- static validation of provenance, tensor boundaries, algebra, population, and falsification rules.

It does **not** authorize:

- scientific model execution;
- runtime preflight;
- training or evaluation;
- Kaggle/GPU use;
- causal intervention;
- tokenizer execution;
- logits or task-head reads;
- PCA, SVD, whitening, learned probes, or fitted geometry;
- post-hoc layer, lag, channel, item, or window search;
- D-path decomposition;
- K1 work.

Implementation may begin only after this static design is reviewed and frozen.

---

## 2. Repository boundary

Repository:

`9terry-student/ContraMamba`

Branch:

`longterm-k-series-native-state-kinematics`

Current frozen parent evidence commit:

`dabb9422dbf0e111319828cf027e8dc5d82fe326`

Expected unrelated local files, which must remain untouched and untracked:

- `scripts/longterm_k1_native_state_kinematics.py`
- `tests/test_longterm_k1_native_state_kinematics.py`

No historical tokenizer/P1/P2 provenance work is reopened by this stage.

---

## 3. Frozen parent evidence identities

### 3.1 Layer-22 U-path source-localization implementation

Implementation commit:

`5f08eefd82195aed12052c625891608b43fe2f29`

Runner path:

`scripts/longterm_k0_rvg_layer22_u_path_source_localization_audit.py`

Runner SHA256:

`1cfafd365fff2dbb10fb2d3ccd610874eb849cc3ddb53a4424e6a3ba21dd8528`

### 3.2 Frozen execution artifacts

Run directory:

`reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1`

Metrics path:

`reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1/layer22_u_path_source_localization_metrics.jsonl`

Metrics SHA256:

`735f66534774a8d760f758ba17427af2de0f817340729f3619dddaae2464ef37`

Summary path:

`reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1/summary.json`

Summary SHA256:

`75c91fe3c0a047abfd0a74d37d12988dd7c34029c74768e7a86f56290995cb38`

Execution manifest path:

`reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1/execution_manifest.json`

Execution manifest SHA256:

`a91721745e008c6e15a00ec4fc938d2d9de5bb81034fd7888c451219bc8c2c70`

### 3.3 Frozen validated-evidence report

Path:

`reports/longterm_k0_rvg_layer22_u_path_source_localization_validated_evidence_analysis_report_candidate.md`

SHA256:

`b110225009ed63d7cc1ad31c9c8dbf1217051833cf9c37f991c371be751beb24`

Git blob at the evidence-freeze commit:

`e370ad275a7645bc3a77d4e089cfe69447f4e49d`

### 3.4 Provenance chain

The parent execution records:

- runtime branch:
  `longterm-k-series-native-state-kinematics`;
- runtime HEAD:
  `5f08eefd82195aed12052c625891608b43fe2f29`;
- model forward count:
  `1344`;
- parent `delta_w` trajectory reproduction:
  `True`;
- raw vectors persisted:
  `False`;
- training executed:
  `False`;
- causal intervention executed:
  `False`.

The parent evidence freeze commit is:

`dabb9422dbf0e111319828cf027e8dc5d82fe326`

This stage must authenticate that commit and the exact frozen parent artifacts before any implementation can be treated as valid.

---

## 4. Structural precedent

The existing layer-23 four-tap runner is a structural/algebraic precedent only.

Path:

`scripts/longterm_k0_rvg_four_tap_convolution_decomposition_audit.py`

SHA256:

`49cd2708bf7ba3ef9f6beb5987f845d585cb9f991e459939c86cb303e0ce5070`

Git blob:

`83928f80349aab50f0c077fb4b2d1f4349a08d41`

The precedent establishes the exact difference identity:

`Q_l(t) = K_(3-l) ⊙ ΔH_(t-l)`

`ΔC_t = Q_0(t) + Q_1(t) + Q_2(t) + Q_3(t)`

where `⊙` is channelwise multiplication.

It also establishes the squared-norm identity:

`||ΔC_t||₂² = Σ_l ||Q_l||₂² + 2 Σ_(l<m) <Q_l, Q_m>`.

Only the algebra and implementation pattern are inherited.

**No layer-23 scientific result is layer-22 evidence.**

In particular, the layer-23 precedent's exact-zero lag-3 tap must **not** be assumed for layer 22.

---

## 5. Frozen scientific frontier

The validated parent result at common-330 `k=2` is:

1. the corr-specific U-path phenotype is already present at layer-22 `X_RF`;
2. hidden in-projection relatively attenuates corr specificity;
3. the dominant positive layer-22 stage enrichment occurs at `H_RF -> C`;
4. SiLU adds a smaller downstream amplification.

Frozen common-330 `k=2` parent values include:

- `E_H = +0.02387519243067802`;
- `E_C = +0.41831449050989755`;
- `G_HC = +0.3674020536829949`;
- `T_HC corr > ctrl = 326/330`;
- `delta_C corr > ctrl = 328/330`.

Therefore the next bounded question is not whether layer-22 convolution is selective.

That localization is already frozen.

The next question is which **fixed four-tap contribution geometry inside that convolution** accounts for the selective amplification.

---

## 6. Scientific question

**Within the fixed layer-22 `H_RF -> C` causal-convolution boundary at common-330 `k=2`, which lag-specific incoming differences, fixed channel-conditioned tap transfers, and constructive/destructive vector interactions account for the corr-selective amplification?**

This question is observational/algebraic.

It does not ask whether any tap is causally necessary or sufficient.

---

## 7. Fixed native convolution identity

For target token `t`, let the authenticated layer-22 hidden receptive field be:

`H_RF(t) = [H_t, H_(t-1), H_(t-2), H_(t-3)]`.

For matched and swapped branches define:

`ΔH_(t-l) = H^m_(t-l) - H^s_(t-l)`.

Let the actual frozen layer-22 depthwise causal-convolution kernel be:

`K[:, 0], K[:, 1], K[:, 2], K[:, 3]`

with the authenticated PyTorch causal mapping:

- lag `0` -> kernel index `3`;
- lag `1` -> kernel index `2`;
- lag `2` -> kernel index `1`;
- lag `3` -> kernel index `0`.

Define for each lag `l ∈ {0,1,2,3}`:

`Q_l(t) = K[:, 3-l] ⊙ ΔH_(t-l)`.

Because the convolution bias is identical between matched and swapped branches, it cancels in the branch difference.

Therefore:

`ΔC_t = Σ_(l=0)^3 Q_l(t)`.

This identity must be reconstructed against the directly captured parent pre-activation convolution difference.

---

## 8. Kernel authentication requirements

Before scientific interpretation, implementation must authenticate the actual layer-22 convolution structure:

- source layer exactly `22`;
- intermediate width exactly `1536`;
- depthwise Conv1d;
- `in_channels = 1536`;
- `out_channels = 1536`;
- `groups = 1536`;
- kernel size exactly `4`;
- causal padding consistent with the frozen slow-path source;
- actual kernel tensor shape consistent with `(1536, 1, 4)`.

For every causal lag, record:

`kernel_weight_rms_l = sqrt(mean(K[:,3-l]^2))`.

Also record whether each tap is exact zero.

No tap is excluded from the pre-registered analysis.

If a tap is exact zero, its `Q_l` is structurally zero and that fact must be reported rather than used to redefine the analysis.

---

## 9. Frozen population and coordinate protocol

The audit must preserve the parent execution protocol:

- 336 fixed items;
- 672 pair-role rows;
- corr and ctrl roles;
- common DDSSSSS cohort of 330 items;
- source layer fixed to 22;
- divergence-aligned coordinates `k=-1..+6`;
- equal-length prefix execution truncated through `k+6`;
- same frozen model/checkpoint/runtime lineage;
- same matched/swapped definitions;
- no layer search;
- no lag selection after observing results;
- no channel search;
- no item filtering beyond the frozen common-330 cohort.

The primary scientific population is:

**common-330 aligned corr/control pairs at `k=2`.**

The full 336 population and full `k=-1..+6` trajectories remain validation/context outputs.

---

## 10. Primary lag-specific measurements

For each pair-role-coordinate row and each causal lag `l`, record:

### 10.1 Incoming lag magnitude

`delta_h_lag_l_l2 = ||ΔH_(t-l)||₂`.

### 10.2 Fixed-tap contribution magnitude

`q_l_l2 = ||Q_l(t)||₂`.

### 10.3 Tap transfer

When `delta_h_lag_l_l2 > 0`:

`t_l = q_l_l2 / delta_h_lag_l_l2`.

If the denominator is zero:

- do not add epsilon;
- record the transfer as undefined.

### 10.4 Weight-RMS-normalized tap transfer

When both:

- `delta_h_lag_l_l2 > 0`;
- `kernel_weight_rms_l > 0`;

define:

`n_l = t_l / kernel_weight_rms_l`.

Otherwise record it as undefined.

This diagnostic distinguishes simple kernel-scale differences from direction/channel-conditioned transfer through the fixed tap.

It is not a learned attribution.

---

## 11. Cross-free tap magnitude geometry

Define:

`RSS² = Σ_l ||Q_l||₂²`

`RSS = sqrt(RSS²)`.

For `RSS² > 0`, define per-lag contribution-energy fractions:

`F_l = ||Q_l||₂² / RSS²`.

If `RSS² = 0`, all `F_l` values are undefined rather than epsilon-regularized.

Also record:

`dominant_q_lag = argmax_l ||Q_l||₂`

with deterministic lowest-lag tie handling if an exact tie occurs.

These quantities describe contribution magnitude before vector interactions are applied.

---

## 12. Pairwise vector-interaction geometry

For every lag pair `a < b`, define the signed pairwise cross term:

`Cross_ab = 2 <Q_a, Q_b>`.

The total interaction is:

`Cross_total = Σ_(a<b) Cross_ab`.

The squared-norm closure is:

`||Σ_l Q_l||₂² = RSS² + Cross_total`.

For `RSS² > 0`, define:

`I_ab = Cross_ab / RSS²`

and:

`I_total = Cross_total / RSS² = Σ_(a<b) I_ab`.

Interpretation:

- positive `I_ab`: constructive pairwise interaction;
- negative `I_ab`: destructive pairwise interaction;
- zero: orthogonal or structurally zero interaction.

No absolute-value replacement is permitted.

The sign is scientifically meaningful.

---

## 13. Vector-addition factor

Define:

`A = ||ΔC||₂ / RSS`

when `RSS > 0`.

If `RSS = 0`, `A` is undefined.

Under exact algebra:

`A² = 1 + I_total`.

This identity must be numerically checked.

Interpretation:

- `A > 1`: net constructive addition;
- `A < 1`: net destructive addition;
- `A = 1`: net-zero interaction contribution to squared norm.

This is a descriptive vector-geometry diagnostic, not a causal effect.

---

## 14. Common-330 k2 corr-vs-ctrl enrichment diagnostics

The parent U-path stage used paired itemwise log enrichments.

The same paired convention is retained.

For every lag `l`, where both role values are strictly positive:

`E_H_l = log(delta_h_lag_l_l2_corr / delta_h_lag_l_l2_ctrl)`

`E_Q_l = log(q_l_l2_corr / q_l_l2_ctrl)`.

When the corresponding tap transfers are defined and positive:

`G_l = E_Q_l - E_H_l`

which must agree numerically with:

`G_l = log(t_l_corr / t_l_ctrl)`.

`G_l` measures corr-selective channel-conditioned transfer through that fixed tap beyond the incoming lag-specific magnitude.

No epsilon is permitted.

Undefined cases must be counted.

---

## 15. Cross-free versus interaction enrichment

Where role values are strictly positive, define:

`E_RSS = log(RSS_corr / RSS_ctrl)`

`E_C = log(delta_c_l2_corr / delta_c_l2_ctrl)`.

Define the vector-addition enrichment:

`G_ADD = E_C - E_RSS`.

When both role addition factors are defined and positive, this must equal:

`G_ADD = log(A_corr / A_ctrl)`.

Thus the observed corr-vs-ctrl convolution output enrichment is separated into:

1. corr selectivity already present in the magnitudes of fixed-tap contributions, summarized by `E_RSS`; and
2. additional corr selectivity introduced by constructive/destructive vector addition, summarized by `G_ADD`.

This is an exact norm-geometry decomposition of the observed fixed-tap contributions.

It is not causal attribution.

---

## 16. Pairwise interaction role diagnostics

Because `I_ab` can be signed, no log transform is used.

For each aligned common-330 `k=2` item define:

`Delta_I_ab = I_ab_corr - I_ab_ctrl`

when both values are defined.

Also define:

`Delta_I_total = I_total_corr - I_total_ctrl`.

Required summaries include:

- role medians for each `I_ab`;
- paired `corr > ctrl`, `corr < ctrl`, and equal counts;
- median `Delta_I_ab`;
- role medians for `I_total`;
- paired counts for `I_total`;
- median `Delta_I_total`.

These diagnostics localize which lag-pair interactions become more constructive or less destructive in corr relative to ctrl.

No interaction pair may be selected post hoc and then presented as if pre-registered.

All six lag pairs must be reported.

---

## 17. Required common-330 k2 summaries

For each lag `0..3`, report:

- corr and ctrl medians of `delta_h_lag_l_l2`;
- corr > ctrl counts for `delta_h_lag_l_l2`;
- corr and ctrl medians of `q_l_l2`;
- corr > ctrl counts for `q_l_l2`;
- corr and ctrl medians of `t_l`;
- corr > ctrl counts for `t_l`;
- corr and ctrl medians of `n_l`;
- corr > ctrl counts for `n_l`;
- corr and ctrl medians of `F_l`;
- role counts of `dominant_q_lag`;
- medians and undefined counts of `E_H_l`, `E_Q_l`, and `G_l`.

For the combined magnitude geometry, report:

- corr and ctrl medians of `RSS`;
- paired corr > ctrl count for `RSS`;
- median `E_RSS`.

For interaction geometry, report:

- all six `I_ab` role medians and paired counts;
- all six median `Delta_I_ab`;
- `I_total` role medians and paired counts;
- median `Delta_I_total`;
- `A` role medians and paired counts;
- median `G_ADD`.

For the direct parent boundary, report:

- corr and ctrl medians of `delta_c_l2`;
- paired corr > ctrl count;
- median `E_C`.

All undefined ratios/logs must be explicitly counted.

---

## 18. Required full-trajectory outputs

For all pair-role rows at every `k=-1..+6`, preserve scalar trajectory summaries for:

- `delta_c_l2`;
- `RSS`;
- `A`;
- `I_total`;
- all four `delta_h_lag_l_l2`;
- all four `q_l_l2`;
- all four `t_l`;
- all four `n_l`;
- all four `F_l`;
- all six signed `I_ab`;
- reconstruction residual;
- squared-norm closure residual.

This prevents a k2-only implementation from silently losing the frozen divergence-aligned context.

No interpretation outside the frozen primary `k=2` population is automatically authorized.

---

## 19. Reconstruction and algebraic gates

### 19.1 Parent boundary reproduction

The audit must reproduce the frozen parent layer-22 trajectories for:

- `delta_h_rf_l2`;
- `delta_c_l2`.

The reproduction tolerance must be frozen in implementation before runtime execution and must not be tuned after observing data.

The expected convention is the same strict summary-level reproduction used by prior K0 audits unless implementation evidence shows that an exact-byte bridge requires a separately documented numerical convention.

### 19.2 Four-tap direct reconstruction

For every row:

`Q_sum = Q_0 + Q_1 + Q_2 + Q_3`.

Compare `Q_sum` to the directly observed `ΔC`.

Inherited structural-precedent tolerance:

`FOUR_TAP_RECON_REL_TOL = 2e-5`.

### 19.3 Squared-norm closure

Check:

`||Q_sum||₂² = RSS² + Cross_total`

within a fixed floating-point closure tolerance frozen before runtime execution.

### 19.4 Interaction normalization closure

For `RSS² > 0`, check:

`I_total = Σ_(a<b) I_ab`.

### 19.5 Addition-factor closure

For `RSS > 0`, check:

`A² = 1 + I_total`

within a fixed numerical tolerance.

### 19.6 Enrichment identities

Where defined, check:

`G_l = E_Q_l - E_H_l = log(t_l_corr / t_l_ctrl)`.

Also check:

`G_ADD = E_C - E_RSS = log(A_corr / A_ctrl)`.

These identities must be enforced itemwise before a scientific PASS is possible.

---

## 20. k=-1 negative control

At `k=-1`, matched and swapped paths must be exact-identical at the parent-observed layer-22 boundaries.

Therefore:

- every `delta_h_lag_l_l2 = 0`;
- every `q_l_l2 = 0`;
- `RSS = 0`;
- `delta_c_l2 = 0`.

All ratio/log quantities whose denominators vanish must be recorded as undefined.

No epsilon substitution is allowed.

Failure of the exact k=-1 identity blocks scientific interpretation.

---

## 21. Scientific decision rules

No arbitrary effect threshold is authorized.

Interpretation must use:

- full distributions;
- paired ordering counts;
- itemwise enrichment medians;
- full trajectory context;
- exact algebraic closure.

### 21.1 Incoming-lag magnitude explanation

If corr-selective `E_Q_l` is largely already present in `E_H_l`, with little or no positive `G_l`, then that tap's corr selectivity is primarily transported from incoming lag-specific `ΔH` magnitude rather than selectively generated by fixed channel weighting.

Permitted conclusion:

**corr-specificity is already present in the incoming lag-specific hidden difference and is transmitted through the fixed tap.**

### 21.2 Fixed-tap/channel-conditioned transfer explanation

If a lag shows a reproducible positive `G_l` and corr-selective `t_l`/`n_l`, then the fixed tap selectively transfers the corr hidden-difference direction more strongly than the ctrl direction.

Permitted conclusion:

**the fixed layer-22 tap provides role-selective channel-conditioned transfer for that causal lag.**

This is not a causal necessity claim.

### 21.3 Contribution-magnitude-dominant convolution amplification

If the direct `E_C` increase is already largely present in `E_RSS`, while `G_ADD` is near zero or negative relative to the observed distributions, then the convolution amplification is principally associated with the magnitudes of the fixed-tap contributions rather than special constructive interaction.

### 21.4 Interaction-assisted amplification

If `G_ADD` is reproducibly positive and the signed `I_ab` diagnostics identify systematic corr-vs-ctrl shifts toward more constructive or less destructive addition, then vector interaction contributes additional corr-selective convolution amplification beyond cross-free tap magnitudes.

### 21.5 Interaction attenuation

If `G_ADD` is negative, vector interaction attenuates corr selectivity relative to the cross-free tap-magnitude geometry.

This does not negate tap-specific contribution enrichment.

### 21.6 Mixed mechanism

If both one or more `G_l` terms and `G_ADD` show reproducible role selectivity, report a mixed geometry:

**lag-specific fixed-tap transfer plus vector-interaction modulation.**

Do not force a unique tap or interaction pair.

### 21.7 Unresolved result

If no lag-specific magnitude/transfer pattern and no interaction pattern accounts for the frozen convolution enrichment in a stable way, preserve the result as unresolved.

Do not respond by introducing learned geometry, channel search, or an expanded window.

---

## 22. Interpretation boundary

Permitted language includes:

- “carried by incoming lag-specific magnitude”;
- “selectively transferred through a fixed tap”;
- “dominant contribution magnitude”;
- “constructive/destructive vector interaction”;
- “interaction-assisted”;
- “interaction-attenuated”;
- “associated with the observed layer-22 convolution geometry”.

Not permitted from this audit alone:

- “tap `l` is causally necessary”;
- “tap `l` is sufficient”;
- “removing the tap would eliminate the effect”;
- “the convolution caused the original corr/control distinction”;
- “a channel subset is causal”;
- “this establishes a K1 mechanism”.

No intervention is authorized.

---

## 23. Artifact policy for a later authorized execution

If implementation and execution are later separately frozen and authorized, evidence artifacts must:

- contain scalar metrics only;
- never persist raw vectors;
- record exact authority/parent/precedent identities;
- record exact runtime commit and runner SHA256;
- record actual layer-22 kernel structure and weight RMS by lag;
- record exact-zero status for every tap;
- record output hashes in the manifest;
- use atomic `.partial` output behavior;
- preserve no-tokenizer/no-logits/no-task-head/no-training/no-intervention flags;
- preserve the no-search boundary.

---

## 24. Stop conditions

Stop before implementation or runtime use if any of the following occurs:

- branch is not `longterm-k-series-native-state-kinematics`;
- `dabb9422dbf0e111319828cf027e8dc5d82fe326` is not an ancestor;
- any frozen parent artifact identity mismatches;
- the frozen validated-evidence report identity mismatches;
- the structural precedent identity mismatches;
- unrelated tracked worktree changes exist;
- either existing K1 untracked file changes state;
- layer-22 convolution is not authenticatable as the expected depthwise four-tap operator;
- direct parent `H_RF`/`C` boundaries cannot be reproduced without modifying model semantics;
- a proposed metric requires a learned transform;
- a proposed interpretation requires post-hoc lag/channel/window selection.

---

## 25. Static-design success criterion

This static-design stage is complete when one implementation can be written without result-dependent scientific choices and the following are fixed:

1. parent evidence/provenance;
2. layer-22 four-tap kernel orientation;
3. all four lag-specific `Q_l` terms;
4. incoming-lag, tap-transfer, contribution-energy, and interaction metrics;
5. common-330 `k=2` paired enrichment diagnostics;
6. full `k=-1..+6` trajectory outputs;
7. parent-boundary reproduction;
8. four-tap and squared-norm closure gates;
9. k=-1 exact negative control;
10. falsification/interpretation rules;
11. no-search/no-intervention boundary.

Completion of this document does **not** authorize model forward execution.
