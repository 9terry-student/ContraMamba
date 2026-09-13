# ContraMamba K0-RVG Layer-22 U-Path Source Localization — Static Design Candidate

## 1. Authority and phase

**Phase:** scientific static design only.

This document authorizes definition and static validation of the next bounded K0-RVG observational audit.

It does **not** authorize:
- scientific model execution;
- training or evaluation;
- Kaggle/GPU use;
- causal intervention;
- tokenizer execution;
- logits or task-head reads;
- PCA, SVD, whitening, learned probes, or fitted geometry;
- post-hoc layer/window search;
- K1 work;
- reopening historical tokenizer/P1/P2 provenance.

Implementation of a runner may begin only after this static design is frozen.

## 2. Repository boundary

Repository:

`9terry-student/ContraMamba`

Branch:

`longterm-k-series-native-state-kinematics`

Parent evidence freeze:

`2228d4601b55d08310392812ef8cc6fb1feb1ba0`

Expected unrelated local files, which must remain untouched and untracked:

- `scripts/longterm_k1_native_state_kinematics.py`
- `tests/test_longterm_k1_native_state_kinematics.py`

## 3. Frozen parent evidence identities

### Layer-22 write-factor runner

Path:

`scripts/longterm_k0_rvg_layer22_write_factor_decomposition_audit.py`

SHA256:

`96e5a7911a26d589a693c3e26b2b6889f152238c3da6c868820544786663c4d0`

Git blob:

`f51e9dd471d27f5b60ee0fc67663b8fa68e29ee2`

### Frozen parent summary

Path:

`reports/longterm_k0_rvg_layer22_write_factor_decomposition_bd2331b_v1/summary.json`

SHA256:

`99f917f77b6950d099b0a14004e9c001aa89dde8ef7366b1f43878458d916ab4`

Git blob:

`dbe695082379269d2309dc9aedec2563016d13a5`

### Frozen parent execution manifest

Path:

`reports/longterm_k0_rvg_layer22_write_factor_decomposition_bd2331b_v1/execution_manifest.json`

SHA256:

`fd370690205d1d3245e24fff251534c143885ae912daa372226fc9aad573c433`

Git blob:

`b42a9b9ed862774458b1388e40229e26c7f95407`

### Frozen parent metrics

Path:

`reports/longterm_k0_rvg_layer22_write_factor_decomposition_bd2331b_v1/layer22_write_factor_decomposition_metrics.jsonl`

SHA256:

`68ad2fb9354a176d63391735d33d102f5393ce3fb6f04b1a7690fa33fb1d1f8b`

Git blob:

`90f3169881bddbaca17a23d7b7010a9c2350b6e0`

## 4. Structural precedent identities

These runners are implementation and measurement precedents only. Their layer-23 scientific results are not evidence for layer 22.

### Post-convolution U factorization precedent

Path:

`scripts/longterm_k0_rvg_postconv_u_factorization_audit.py`

SHA256:

`0f3e9527ff3d5a3c91e646547b1ab17e134e799130833100446bd976e4a4d052`

Git blob:

`acab0c14e7e7ceb7216c244e3f99a93333309b05`

### Four-tap convolution precedent

Path:

`scripts/longterm_k0_rvg_four_tap_convolution_decomposition_audit.py`

SHA256:

`49cd2708bf7ba3ef9f6beb5987f845d585cb9f991e459939c86cb303e0ce5070`

Git blob:

`83928f80349aab50f0c077fb4b2d1f4349a08d41`

### Hidden-branch in-projection precedent

Path:

`scripts/longterm_k0_rvg_hidden_inproj_transfer_audit.py`

SHA256:

`65810137bebb768c12bb11fd279a630d6a6a1f1a6116b3b12f00a0c286ea8d64`

Git blob:

`4a87718b919d3164a87c9d9d704751f5389f897d`

## 5. Frozen scientific frontier

The validated layer-22 recurrent boundary is:

`W = D * U`

where:
- `D = discrete_B`;
- `U = conv-activated hidden_states`;
- `W = deltaB_u`.

The frozen parent audit established that, for the common-330 corr population at `k=2`, the U-associated symmetric component is the dominant magnitude component of the write difference:

- `Q_U > Q_D`: `310/330`;
- median `Q_U` energy fraction: approximately `0.844`;
- median `Q_D` energy fraction: approximately `0.156`;
- corr > ctrl `Q_U`: `329/330`;
- corr > ctrl `Q_D`: `330/330`.

The same evidence also establishes strong destructive U–D vector interaction at corr `k=2`.

Therefore this stage must not claim that U is a causal mechanism or that D is irrelevant.

The next bounded question is where the observed corr-k2 `ΔU` structure arises along the native layer-22 U path.

## 6. Scientific question

**Within layer 22, is the corr-k2 U separation already present at the mixer input / hidden-branch receptive field, or is it selectively amplified across hidden in-projection, fixed depthwise causal convolution, or SiLU activation?**

The goal is source localization, not performance optimization and not causal attribution.

## 7. Frozen native forward boundaries

For target token `t`, define the layer-22 path:

`X_t → H_t → C_t → U_t`

where:

- `X_t` is the layer-22 mixer input before `in_proj`;
- `H_t` is the hidden half of the bias-free `in_proj`;
- `C_t` is the pre-activation output of the depthwise causal convolution;
- `U_t = SiLU(C_t)` is the conv-activated hidden state entering the authenticated write boundary.

Because the convolution has temporal receptive field, define:

`X_RF(t) = [X_t, X_(t-1), X_(t-2), X_(t-3)]`

`H_RF(t) = [H_t, H_(t-1), H_(t-2), H_(t-3)]`

The fixed forward identities to authenticate are:

`H_lag = W_H X_lag`

for every available receptive-field lag, with exact causal zero-padding treatment where required;

`C_t = depthwise_causal_conv(H_RF(t)) + bias`

using the actual frozen layer-22 kernel and bias;

`U_t = SiLU(C_t)`.

The observed `U_t` must be bridged back to the same layer-22 recurrence-frame U operand used by the frozen parent `W=D*U` audit.

## 8. Population and coordinate protocol

The new audit must preserve the frozen parent protocol:

- 336 fixed items;
- 672 pair-role rows;
- corr and ctrl roles;
- common DDSSSSS cohort of 330 items;
- divergence-aligned relative coordinates `k=-1..+6`;
- equal-length prefix execution truncated through `k+6`;
- same model/checkpoint/runtime lineage;
- source layer fixed to 22;
- no layer search;
- no window search.

The common-330 `k=2` population is the primary scientific population.

Full-336 and full `k=-1..+6` trajectories remain validation/context outputs.

## 9. Primary raw measurements

For each pair-role-coordinate row, record raw native differences:

- `delta_x_rf_l2 = ||ΔX_RF||₂`;
- `delta_h_rf_l2 = ||ΔH_RF||₂`;
- `delta_c_l2 = ||ΔC||₂`;
- `delta_u_l2 = ||ΔU||₂`.

Diagnostics only:

- `delta_x_current_l2 = ||ΔX_t||₂`;
- `delta_h_current_l2 = ||ΔH_t||₂`.

The receptive-field norms, not the current-token-only norms, are the denominators for convolution-path localization.

No learned transformation of these vectors is permitted.

## 10. Stage-transfer measurements

When the denominator is strictly positive, define:

`T_XH = ||ΔH_RF||₂ / ||ΔX_RF||₂`

`T_HC = ||ΔC||₂ / ||ΔH_RF||₂`

`T_CU = ||ΔU||₂ / ||ΔC||₂`

If a denominator is zero, do not add epsilon. Mark that transfer undefined for that row.

These are observational norm-transfer diagnostics through frozen native operators. They are not causal effects.

## 11. Corr-vs-ctrl enrichment diagnostics

For common-330 aligned corr/control pairs at `k=2`, report paired counts and distribution summaries for every raw boundary norm and every defined transfer.

Where both role values are strictly positive, define:

`E_X = log(delta_x_rf_l2_corr / delta_x_rf_l2_ctrl)`

`E_H = log(delta_h_rf_l2_corr / delta_h_rf_l2_ctrl)`

`E_C = log(delta_c_l2_corr / delta_c_l2_ctrl)`

`E_U = log(delta_u_l2_corr / delta_u_l2_ctrl)`

Derived stage enrichment changes:

`G_XH = E_H - E_X`

`G_HC = E_C - E_H`

`G_CU = E_U - E_C`

Equivalent transfer form:

`G_XH = log(T_XH_corr / T_XH_ctrl)`

`G_HC = log(T_HC_corr / T_HC_ctrl)`

`G_CU = log(T_CU_corr / T_CU_ctrl)`

when the required terms are defined and positive.

No arbitrary scientific threshold for “large” amplification is authorized.

Interpretation must use effect distributions, paired ordering, and the full trajectory rather than a tuned cutoff.

## 12. Required reconstruction and bridge gates

Before scientific interpretation, the implementation must validate:

### Hidden in-projection reconstruction

Reconstruct layer-22 hidden-branch vectors from the captured mixer inputs and frozen hidden-half `in_proj` weight.

Inherited precedent tolerance:

`INPROJ_RECON_REL_TOL = 1e-6`

### Causal convolution reconstruction

Reconstruct pre-activation `C_t` from the captured four-token `H_RF`, the actual frozen layer-22 depthwise convolution kernel, causal padding, and bias.

Inherited precedent tolerance:

`CONV_RECON_REL_TOL = 2e-5`

Do not assume that any individual layer-22 convolution tap is zero merely because a layer-23 precedent had such a property.

### Activation reconstruction

Reconstruct:

`U_reconstructed = SiLU(C)`

and compare it to the directly captured layer-22 U operand.

Inherited precedent tolerance:

`ACTIVATION_REL_TOL = 1e-6`

### Parent-U bridge

The U observed at the post-convolution/activation boundary must numerically reproduce the U operand captured at the authenticated layer-22 recurrent write frame.

The implementation must define and statically freeze the exact bridge metric and tolerance before runtime execution.

### Parent trajectory reproduction

The new audit must reproduce the frozen parent `delta_w_l2` trajectory from the authenticated layer-22 write-factor evidence within the existing parent-reproduction tolerance convention.

A successful model process without this reproduction is not valid scientific evidence.

## 13. k=-1 negative control

At `k=-1`, matched and swapped paths must be exact-identical at every observed native boundary:

- `X_RF`;
- `H_RF`;
- `C`;
- `U`.

Accordingly all corresponding L2 differences must be exactly zero.

Failure of this negative control blocks scientific interpretation.

## 14. Required k2 paired summaries

For the common-330 aligned corr/control population at `k=2`, the summary must include at minimum:

- corr > ctrl counts for `delta_x_rf_l2`;
- corr > ctrl counts for `delta_h_rf_l2`;
- corr > ctrl counts for `delta_c_l2`;
- corr > ctrl counts for `delta_u_l2`;
- role-wise medians of all four raw boundary norms;
- role-wise medians of `T_XH`, `T_HC`, `T_CU`;
- distributions/medians of `E_X`, `E_H`, `E_C`, `E_U`;
- distributions/medians of `G_XH`, `G_HC`, `G_CU`;
- undefined-count reporting for every ratio/log metric.

No post-hoc metric selection is allowed after seeing results.

## 15. Scientific decision rules

The audit localizes the earliest supported source boundary as follows.

### Upstream-present result

If corr-specific enrichment is already strong at `X_RF`, while downstream stage gains do not selectively increase it, conclude only:

**The layer-22 U-path phenotype is already present at the mixer-input receptive field and is not generated by the measured layer-22 in-projection / convolution / activation stages.**

This points further upstream but does not authorize a new upstream execution automatically.

### Hidden in-projection amplification

If the principal selective enrichment increase occurs across `X_RF → H_RF`, localize the observed amplification to direction-conditioned transfer through the frozen hidden-branch in-projection.

### Causal-convolution amplification

If the principal selective enrichment increase occurs across `H_RF → C`, localize the observed amplification to fixed depthwise causal temporal mixing.

### SiLU amplification

If the principal selective enrichment increase occurs across `C → U`, localize the observed amplification to SiLU operating-point transmission.

### Mixed serial amplification

If multiple stages show substantial, reproducible enrichment increases, report a mixed/serial localization.

Do not force a unique stage.

### Failure to localize

If the U phenotype is not explained by the measured upstream magnitude and stage transfers, preserve the unresolved result.

Do not invent a learned metric or expand the layer/window search.

The next scientifically justified branch may then revisit the frozen D/U interaction structure.

## 16. Interpretation boundary

All results remain observational/algebraic.

Permitted language:

- “already present at”;
- “associated with”;
- “amplified across”;
- “localized to the observed native boundary”;
- “consistent with fixed-stage transmission”.

Not permitted from this audit alone:

- “caused by”;
- “necessary”;
- “sufficient”;
- “intervention establishes”;
- “U is the causal mechanism”.

## 17. Artifact policy

If later implementation/execution is separately authorized, evidence artifacts must:

- persist scalar metrics and provenance only;
- not persist raw vectors;
- record exact parent and precedent identities;
- record source/runtime identity;
- distinguish code correctness, execution success, artifact validity, and scientific conclusion;
- use atomic `.partial` output behavior;
- preserve all no-tokenizer/no-logits/no-training/no-intervention flags.

## 18. Stop conditions

Stop before implementation if:

- current branch is not `longterm-k-series-native-state-kinematics`;
- parent freeze `2228d4601b55d08310392812ef8cc6fb1feb1ba0` is not an ancestor;
- any frozen parent or structural-precedent identity mismatches;
- unrelated tracked worktree changes exist;
- either existing K1 untracked file changes state;
- exact layer-22 tensor-boundary instrumentation cannot be defined without modifying model semantics;
- a proposed measurement requires training, a learned transform, or post-hoc search.

## 19. Static-design success criterion

This static-design stage is complete when:

1. the authority file is frozen at exact repository bytes;
2. an implementation task can be written without scientific ambiguity;
3. the exact layer-22 `X_RF → H_RF → C → U` boundaries, measurements, reconstruction gates, negative control, population, and falsification rules above require no result-dependent choices.

No scientific model forward is authorized by completion of this document.
