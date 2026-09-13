# ContraMamba K0-RVG Layer-22 U-Path Source Localization
## Validated Evidence Analysis Report Candidate

## 1. Status

**Evidence status:** validated local observational execution.

**Implementation commit:**

`5f08eefd82195aed12052c625891608b43fe2f29`

**Static-design authority commit:**

`497598f22aeccf628d1e0ada4a2e5f9aa67f4c94`

**Run directory:**

`reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1`

This report interprets only the validated artifacts from that run. It does not authorize a new execution, intervention, training/evaluation, layer search, or K1 transition.

---

## 2. Validated artifact identities

### Metrics

Path:

`reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1/layer22_u_path_source_localization_metrics.jsonl`

SHA256:

`735f66534774a8d760f758ba17427af2de0f817340729f3619dddaae2464ef37`

### Summary

Path:

`reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1/summary.json`

SHA256:

`75c91fe3c0a047abfd0a74d37d12988dd7c34029c74768e7a86f56290995cb38`

Schema:

`k0-rvg-layer22-u-path-source-localization-summary-v1`

### Execution manifest

Path:

`reports/longterm_k0_rvg_layer22_u_path_source_localization_5f08eef_v1/execution_manifest.json`

SHA256:

`a91721745e008c6e15a00ec4fc938d2d9de5bb81034fd7888c451219bc8c2c70`

Schema:

`k0-rvg-layer22-u-path-source-localization-execution-manifest-v1`

Manifest-recorded runtime HEAD:

`5f08eefd82195aed12052c625891608b43fe2f29`

Manifest-recorded runner SHA256:

`1cfafd365fff2dbb10fb2d3ccd610874eb849cc3ddb53a4424e6a3ba21dd8528`

The manifest-recorded metrics and summary hashes exactly match the locally recomputed hashes.

No `.partial` run directory remained after execution.

---

## 3. Execution validity

The validated execution completed:

- `model_forward_count = 1344`
- `parent_delta_w_trajectory_match = True`
- `raw_vectors_persisted = False`
- `training_executed = False`
- `causal_intervention_executed = False`

The run preserved the frozen population and protocol:

- 336 fixed items;
- 672 pair-role rows;
- common DDSSSSS cohort of 330 items;
- layer 22 only;
- divergence-aligned `k=-1..+6`;
- equal-length prefix execution;
- no tokenizer;
- no logits/task heads;
- no learned geometry;
- no post-hoc layer/window search.

### Reconstruction / bridge gates

Full-execution maxima:

- hidden in-projection reconstruction relative residual:
  `3.9620320852752387e-07`
  against tolerance `1e-6`;
- causal-convolution reconstruction relative residual:
  `9.541348939169357e-08`
  against tolerance `2e-5`;
- SiLU activation reconstruction relative residual:
  `8.443059482136118e-08`
  against tolerance `1e-6`;
- parent-U bridge relative residual:
  `8.443059482136118e-08`
  against tolerance `1e-6`.

The frozen parent write trajectory was reproduced exactly at the summary level required by the parent bridge:

`parent_delta_w_trajectory_match = True`.

Therefore the observed `X_RF -> H_RF -> C -> U` chain is valid for scientific interpretation under the frozen static design.

---

## 4. Frozen scientific question

The bounded question was:

> Within layer 22, is the corr-k2 U separation already present at the mixer-input / hidden-branch receptive field, or is it selectively amplified across hidden in-projection, fixed depthwise causal convolution, or SiLU activation?

The analysis is observational/algebraic. It does not establish causal necessity or sufficiency.

---

## 5. Common-330 k2 results

### 5.1 Raw boundary norms

Role medians:

| Boundary | corr median | ctrl median | corr > ctrl |
|---|---:|---:|---:|
| `||ΔX_RF||₂` | 6.5695676411901776 | 5.9885469573074355 | 307 / 330 |
| `||ΔH_RF||₂` | 25.78866570866662 | 25.962574207190883 | 215 / 330 |
| `||ΔC||₂` | 2.0838235086784174 | 1.3248277045372114 | 328 / 330 |
| `||ΔU||₂` | 1.255049349854127 | 0.7707117346424772 | 328 / 330 |

The role-paired enrichment medians are:

- `E_X = 0.09803846737059453`
- `E_H = 0.02387519243067802`
- `E_C = 0.41831449050989755`
- `E_U = 0.479825035451787`

These are medians of itemwise log-ratios and must not be replaced by the log of role medians.

### 5.2 Stage transfers

Role medians:

| Stage | corr median | ctrl median | corr > ctrl |
|---|---:|---:|---:|
| `T_XH = ||ΔH_RF|| / ||ΔX_RF||` | 4.060946731273972 | 4.317034936185259 | 18 / 330 |
| `T_HC = ||ΔC|| / ||ΔH_RF||` | 0.08387592261067447 | 0.0560378098963565 | 326 / 330 |
| `T_CU = ||ΔU|| / ||ΔC||` | 0.6015414395345926 | 0.5666909723645825 | 235 / 330 |

Itemwise stage-enrichment medians:

- `G_XH = -0.05262506978156998`
- `G_HC = +0.3674020536829949`
- `G_CU = +0.0634248789066637`

The runner also enforced the itemwise identities connecting `G` values to transfer log-ratios and the telescoping `X -> U` enrichment relation before the execution could PASS.

---

## 6. Interpretation

### 6.1 The corr-k2 phenotype is already present upstream of the layer-22 mixer transformations

At the earliest measured boundary, `X_RF`, corr exceeds ctrl in `307/330` aligned items and the median itemwise enrichment is positive:

`E_X = +0.0980`.

Therefore the final `ΔU` phenotype cannot be described as being newly generated inside the layer-22 hidden in-projection, convolution, or activation path.

The earliest observed evidence in this audit is already present at the layer-22 mixer-input receptive field.

This localizes part of the phenomenon upstream of the measured layer-22 path, but does not by itself identify an earlier layer or mechanism.

### 6.2 Hidden in-projection does not selectively amplify corr-k2 separation

The hidden in-projection substantially increases absolute vector norm for both roles, but it does not preferentially amplify corr.

Evidence:

- `T_XH corr > ctrl` in only `18/330`;
- `T_XH corr < ctrl` in `312/330`;
- median `G_XH = -0.0526`;
- raw `E_X = +0.0980` falls to `E_H = +0.0239`.

Thus the bias-free hidden-half in-projection is a strong absolute transfer operator but a **relative corr-specificity attenuator** at common-330 k2.

It is not supported as the source of the corr-specific U amplification.

### 6.3 The dominant layer-22 internal selective amplification occurs across the causal convolution

The strongest stage-localized role selectivity is the `H_RF -> C` transition.

Evidence:

- `T_HC corr > ctrl` in `326/330`;
- `delta_C corr > ctrl` in `328/330`;
- median `G_HC = +0.3674`;
- median enrichment rises from
  `E_H = +0.0239`
  to
  `E_C = +0.4183`.

This is the dominant positive stage-enrichment change among the measured layer-22 U-path stages.

Therefore the validated observational result supports:

**The fixed layer-22 depthwise causal-convolution stage is the principal layer-22 internal selective amplifier of the corr-k2 U-path phenotype.**

This is a localization statement about observed native transfer. It is not a causal-intervention claim.

### 6.4 SiLU adds a smaller downstream amplification

The `C -> U` activation stage also shows role-selective amplification, but it is substantially smaller than the convolution-stage effect.

Evidence:

- `T_CU corr > ctrl` in `235/330`;
- `delta_U corr > ctrl` in `328/330`;
- median `G_CU = +0.0634`;
- enrichment rises from
  `E_C = +0.4183`
  to
  `E_U = +0.4798`.

Thus SiLU contributes a **modest additional corr-selective amplification** after the larger convolution-stage amplification.

It should not be described as the primary layer-22 source.

---

## 7. Frozen scientific conclusion

The validated K0-RVG layer-22 U-path result is:

**At common-330 k2, the corr-specific U-path phenotype is already present at the layer-22 mixer-input receptive field, is relatively attenuated rather than generated by the hidden in-projection, is strongly and selectively amplified across the fixed depthwise causal convolution, and receives a smaller additional amplification across SiLU.**

Equivalently:

**earliest observed phenotype: upstream-present at `X_RF`; dominant layer-22 internal selective amplification: `H_RF -> C`; secondary amplification: `C -> U`.**

This is therefore a **mixed serial localization** result rather than either:

- an “upstream-only” result; or
- a “convolution creates the phenotype from nothing” result.

The principal layer-22 internal mechanism candidate for the observed U-path magnification is the causal-convolution transfer geometry.

---

## 8. Claims not supported by this evidence

This evidence does **not** establish that:

- the convolution is causally necessary or sufficient;
- the convolution creates the original corr/control distinction;
- SiLU is irrelevant;
- the hidden in-projection is globally suppressive outside this fixed population/window;
- a particular convolution lag or tap is responsible;
- a particular channel subset causes the effect;
- the effect generalizes to another layer;
- the result establishes a K1 mechanism.

No intervention was performed.

No post-hoc search is authorized from this report.

---

## 9. Immediate scientific consequence

The next bounded K0 question follows directly from the localization result:

**Within the fixed layer-22 `H_RF -> C` causal-convolution boundary at common-330 k2, which lag-specific fixed-kernel contributions and their vector interactions account for the corr-selective amplification?**

The appropriate next analysis is a layer-22 four-tap convolution decomposition using the exact identity

`ΔC_t = Σ_l K_(3-l) ΔH_(t-l)`

with the actual authenticated layer-22 kernel and without assuming any tap is zero.

That future stage should remain observational/algebraic first and preserve the same fixed cohort, divergence coordinate, model/checkpoint, and no-search boundary.

This report itself does not authorize that execution.
