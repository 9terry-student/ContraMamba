# ContraMamba Gen4 — Three-Scale Readout-Alignment Descriptive Synthesis

Status: STATIC_DESCRIPTIVE_SYNTHESIS_ONLY

Frozen synthesis base commit:

`707f1a5fcd8867d73ced758cb3650df6f13cf030`

This report adds:

- no model execution;
- no training;
- no forward or backward pass;
- no new p-value;
- no three-scale trend test;
- no monotonicity test;
- no interpolated zero-crossing estimate;
- no scale-threshold optimization;
- no plane reselection;
- no cohort replacement or response-guided filtering.

It only places already-frozen readout-alignment results side by side under the
three-scale contextual-synthesis boundary declared in the prospective 130M plan.

## 1. Frozen source artifacts

### Mamba-130M

Primary analysis:

`reports/reason_router_gen4_mamba130m_readout_alignment_analysis_v1/readout_alignment_analysis.json`

Frozen analysis SHA256:

`6ae8866d4c39cd1ffceb490c77ba9f8f207a14bce26059d96dcf845fa1946184`

Pair-value SHA256:

`8012e07d2a53b90d9d271ca4e4aa93af26f78c2fdfeaf0d149e38ff602c232b4`

Result freeze commit:

`8a2043a261c2ff7a01ff3c24855aca8640e1ca87`

### Mamba-370M / Mamba-1.4B

Primary analysis:

`reports/reason_router_gen4_mamba370m14b_readout_alignment_analysis_v1/readout_alignment_analysis.json`

Frozen analysis SHA256:

`c5568b73e85d4e2f9b1ce52cc923533ca80435ea96d1e7484b248c907e99265b`

Pair-value SHA256:

`f566f704b11d6b9574ac381f8584c44f3be191183ddb0e65eb3cb2e315f62779`

The frozen Study-B result is:

`READOUT_ALIGNMENT_SCALE_SIGN_REVERSAL_SUPPORTED`

## 2. Three-scale descriptive table

| scale | population | checkpoint SHA256 | selected plane | response-blind control | mean Delta_L | fraction Delta_L > 0 |
|---|---|---|---|---|---:|---:|
| Mamba-130M | `xg1_fact_2701..3000` | `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f` | `P3` | `P5` | `+0.0011203660365321265` | `0.5366666666666666` |
| Mamba-370M | `xg1_fact_4801..5100` | `9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72` | `P3` | `P5` | `+0.0005089563518854174` | `0.5766666666666667` |
| Mamba-1.4B | `xg1_fact_4801..5100` | `915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a` | `P5` | `P4` | `-0.0011954475058862238` | `0.35333333333333333` |

## 3. Descriptive synthesis

The frozen local first-order task-margin readout has positive mean alignment at
Mamba-130M and Mamba-370M, and negative mean alignment at Mamba-1.4B.

At 130M, the prospectively specified one-scale test already established:

`MAMBA130M_READOUT_ALIGNMENT_POSITIVE_SUPPORTED`

At 370M versus 1.4B, the independent prospective Study-B analysis already
established:

`READOUT_ALIGNMENT_SCALE_SIGN_REVERSAL_SUPPORTED`

The three-scale table therefore adds contextual breadth to the frozen readout
evidence: the tested smaller checkpoints have positive mean local readout alignment,
whereas the tested 1.4B checkpoint has negative mean alignment under its own
frozen scale-local selected/control geometry.

The recurring object is not a fixed principal-plane rank. The selected/control
identity is `P3/P5` at 130M and 370M, but `P5/P4` at 1.4B. Interpretation should
therefore remain scale-local: the analysis concerns the relation between each
frozen scale-local causal displacement and its downstream task-margin readout.

## 4. Cohort boundary

Mamba-130M uses `xg1_fact_2701..3000`.

Mamba-370M and Mamba-1.4B use `xg1_fact_4801..5100`.

Accordingly, this report does not treat the three rows as a matched three-scale
sample. In particular, it does not infer a monotone scaling law from the three
means and does not interpret their numerical ordering as a calibrated scaling
curve.

The 370M-versus-1.4B sign-reversal claim remains supported by its own frozen,
matched-pair prospective Study-B design. The 130M result is a separately frozen
prospective result on its own matched behavioral-bridge population.

## 5. Interpretation boundary

Supported descriptive statement:

> Across the three frozen checkpoints, local task-margin readout alignment is
> positive at Mamba-130M and Mamba-370M and negative at Mamba-1.4B, while the
> relevant selected/control causal geometry remains scale-local rather than tied
> to a universal principal-plane rank.

This synthesis does not establish:

- a monotonic relationship between parameter count and `Delta_L`;
- a parameter-count threshold at which the sign must reverse;
- a zero-crossing location between 370M and 1.4B;
- universality outside the tested checkpoints and populations;
- a common calibrated magnitude scale across the 130M and 370M/1.4B cohorts;
- complete causal mediation of behavioral effects.

## 6. Companion 130M explanatory evidence

The separately frozen 130M descriptive readout-versus-behavior merge at commit
`707f1a5fcd8867d73ced758cb3650df6f13cf030` reports substantial pair-level
agreement:

- Pearson `corr(Delta_L, D_BEH) = 0.617548086733788`;
- Spearman rank correlation `= 0.8324696941077123`;
- sign agreement `= 247/300 = 0.8233333333333334`;
- mean residual `D_BEH - Delta_L = +0.007211825619501071`.

These values are descriptive only and are not used to define or test a three-scale
trend. They indicate that the 130M local first-order readout captures meaningful
pair-level ordering and sign information while materially underestimating the
average behavioral effect magnitude.

## 7. Stop rule

No additional three-scale inference is authorized by this synthesis.

Do not add:

- a trend p-value;
- monotonicity significance testing;
- interpolation or regression across parameter count;
- a zero-crossing estimate;
- threshold optimization;
- post-hoc cohort or plane changes

in response to the observed three-scale pattern.
