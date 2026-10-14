# ContraMamba Gen5 — Cross-seed bypass geometry: LOSO and actual donor-operator-span scientific interpretation

**Status:** `STATIC_TWO_JSON_REVIEW_COMPLETE / PARTIALLY_SHARED_WRITE_GEOMETRY / LIMITED_OPERATOR_SPAN / CAUSAL_AND_TASK_CLAIMS_NOT_PROMOTED`
**Date:** 2026-10-14 (Asia/Seoul)
**Repository/branch:** `9terry-student/ContraMamba` / `gen5-causal-role-state-ownership`
**Read-only analysis HEAD:** `b7212f8273d74737b4a3ffa81cefbad47d0021d0`
**Historical Phase3A execution HEAD:** `d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

## 1. Scope, immutable inputs, and provenance

This report interprets **two user-executed, CPU-only, read-only static audits**. The author separately read and SHA256-verified the uploaded JSON files and checked their numeric and identity consistency; it did **not** independently rerun the historical PyTorch checkpoint computation. There was no new training, model forward, evaluation, backward, tokenizer use, GPU run, or confirmatory-data access.

- LOSO JSON: `gen5_cross_seed_bypass_loso_b7212f8.json` SHA256 `b706fb8f616d2eff05cbe0d4b03045e1de8f248c165e355e3676794b0dfe51fb`.
- Actual donor-operator JSON: `gen5_cross_seed_donor_operator_span_b7212f8.json` SHA256 `4addef7cfd4d652c249ed6ce200f3634a34624851f879c3a4afa1578638e1a2a`.
- First static audit script SHA256 `0fb097c5042ff1636f12d2b652e7b3e618d24dd62f9754bb86f997632f47cfe7`.
- Second static audit script SHA256 `900020a2b7fec501a1e1f3e496de5115d21a0ab4084f19e2700891e6880c69c2`.
- Population: the same frozen Phase3A `G5-C0`, `P0`, final-step correction checkpoints at seeds **6201, 6202, 6203**. The dev population and training contract are inherited rather than resampled.
- Exact checkpoint paths are `reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed{6201,6202,6203}/P0/final_correction.pt`.

| Seed | Frozen checkpoint SHA256 |
|---|---|
| 6201 | `157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf` |
| 6202 | `1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214` |
| 6203 | `c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770` |

The second JSON identifies the first by its exact SHA256, both report the same HEAD/seeds/historical execution/pressure, and per-seed checkpoint and `A_theta.weight`/`B_theta.weight` tensor identities agree. The user-executed program reports checkpoint SHA validation PASS and zero GPU/forward/evaluation/training; the JSON independently reproduces those identities but is **not** by itself an independent rerun of the source files.

## 2. Objects and noninterchangeable estimands

For source seed `s`, define the frozen full correction operator `M_s = B_s A_s`, where `B_s` has shape `[24576,2]` and `A_s` has shape `[2,768]`. Avoid materializing the dense `24576 x 768` matrix: all operator Frobenius products may be evaluated as `sum((B_s.T @ B_t) * (A_s @ A_t.T))` in float64.

Let `D_s` denote the **other two** training seeds, fixed independently of the held-out seed. Their union output and input subspaces, of dimensions at most four each, are `U_{-s}=span(B_t:t in D_s)` and `V_{-s}=span(A_t.T:t in D_s)`. The previously completed static LOSO quantities are:

- **Output B coverage**: `C_B(s)=||P_U Q_{B_s}||_F^2/2`, for an orthonormal 2-column basis `Q_{B_s}`.
- **Read A coverage**: `C_A(s)=||P_V Q_{A_s}||_F^2/2`, likewise.
- **Free Cartesian operator coverage**: `C_cart(s)=||P_U M_s P_V||_F^2/||M_s||_F^2`. It permits arbitrary new pairings of the donor input and output bases and hence does **not** measure the span of the two actually learned operators.

The second audit measures **actual donor-operator-span coverage**:

`C_actual(s)=||Proj_span{M_t:t in D_s}(M_s)||_F^2 / ||M_s||_F^2`.

The optimal two donor coefficients are obtained **using the held-out operator itself**, by solving the donor Gram linear system. Thus `C_actual` describes optimal *ex-post representability* in a fixed donor-defined space. It does not constitute out-of-sample prediction, a pre-trained transfer operator, new task performance, or causal substitution. Its donor operator span is a subset of the Cartesian operator span, so `C_actual <= C_cart` is an essential numeric check.

## 3. Held-out quantitative results

| Held-out seed | `C_B` | `C_A` | `C_cart` | `C_actual` | `C_cart - C_actual` |
|---|---:|---:|---:|---:|---:|
| 6201 | 0.559733601411 | 0.185269472972 | 0.293052718821 | 0.270998321158 | 0.022054397662 |
| 6202 | 0.540653431704 | 0.193764816998 | 0.327411238799 | 0.296947544602 | 0.030463694196 |
| 6203 | 0.617834912713 | 0.186222695844 | 0.293999270693 | 0.274846819424 | 0.019152451270 |
| **Descriptive mean** | **0.572740648609** | **0.188418995271** | **0.304821076104** | **0.280930895061** | **0.023890181043** |

The mean unexplained held-out operator Frobenius energy is **0.719069104939**. The ratio `C_actual / C_cart` is `0.924743`, `0.906956`, `0.934855` by seed, with unweighted mean `0.922185`. This ratio says that the actual donor operators account for most of what is representable *within the considerably larger Cartesian donor space*, not that they explain 92% of the full held-out operator.

The user-supplied result JSON reports per-seed Cartesian replay matching the first audit. Direct independent JSON review reproduced that equality (absolute discrepancies `<=5.6e-17`), the mean fractions, per-seed difference arithmetic, and all SHA256 links. No arbitrary numerical cutoff or inferential p-value was defined.

## 4. Pairwise signed operator geometry and Gram stability

| Operator seed pair | Signed Frobenius cosine |
|---|---:|
| 6201–6202 | 0.458617459155 |
| 6201–6203 | 0.430704529705 |
| 6202–6203 | 0.463140095978 |

All three signed cosines are positive, approximately `0.431–0.463`, giving a nonzero, directionally consistent pairwise alignment within this frozen operator coordinate frame. Squared pairwise cosines lie in `0.1855–0.2145`: each *single* donor direction accounts for only about 19–21% of the held-out operator energy, while its optimal **two-donor** span accounts for about 27–30%. Positive alignment should not be called identical operators or an independently replicated common causal direction.

The coefficients minimizing squared Frobenius error are positive for both donors across every held-out seed:

| Held-out | Donor order | Held-out-optimal coefficients | Donor Gram condition number |
|---|---|---|---:|
| 6201 | 6202, 6203 | 0.32893127, 0.28248069 | 2.727550 |
| 6202 | 6201, 6203 | 0.31910402, 0.33244839 | 2.514665 |
| 6203 | 6201, 6202 | 0.27197422, 0.32994087 | 2.694297 |

These moderate donor Gram condition numbers argue against an ill-conditioned, near-collinear two-donor fit as the source of the apparent coverage, **within the reported numeric regime**. They do not certify checkpoint correctness beyond the authenticated source/scope, causal interpretability, or statistical significance.

The three-by-three **unnormalized** operator Gram eigenvalues are `[13.270270699494152, 3.953949875046905, 3.7029166951135783]`. They are all positive, yielding rank three at the displayed numerical scale. As proportions of total source-operator squared Frobenius energy, these eigenmodes explain `63.4118%`, `18.8939%`, and `17.6943%`. The latter two are substantial; the result is incompatible with exact one-dimensional operator collapse. The leading fraction is **in-sample and norm-weighted**; it does not demonstrate prospective held-out generalization or uniquely localize a common mechanistic direction.

## 5. Principal-angle structure from the preceding LOSO JSON

| Pair | Output principal angles | Read principal angles |
|---|---|---|
| 6201–6202 | 22.89°, 87.20° | 57.25°, 88.37° |
| 6201–6203 | 25.07°, 63.31° | 58.30°, 89.97° |
| 6202–6203 | 20.73°, 69.67° | 57.15°, 88.43° |

There is stronger partial sharing in the write/output geometry, particularly one relatively aligned direction across every pair, than in read/input geometry, where the second direction is nearly orthogonal across all three pairs. The LOSO output coverage is substantially larger than input coverage, but coverage fractions have different ambient dimensions and refer to *different geometric objects*. Their numerical ratio must not be treated as a causal effect size.

The pooled six-column spectrum from the first JSON is descriptive and in-sample; it is **not** an independent cross-seed test. All claims of seed dependence refer only to the three frozen Phase3A P0 seeds.

## 6. Scientific interpretation and retained negative results

**Supported bounded description:** `PARTIAL_SHARED_WRITE_GEOMETRY / POSITIVELY_ALIGNED_BUT_NONIDENTICAL_OPERATORS / LIMITED_ACTUAL_DONOR_SPAN / HETEROGENEOUS_SEED_REALIZATION`.

**Not supported:** a single near-identical, rank-one, universally shared correction operator; complete operator reconstruction from the other two seeds; prospective unseen-seed operator prediction; donor-space task sufficiency; common latent semantic owner; ownership effectiveness; or optimizer-induced R22 contention.

The current results are consistent with a family of partially shared output routes coupled to substantially seed-dependent read and operator realizations. They do not establish *why* those geometries recur, whether the overlap is task-functional, whether the unexplained 72% is functionally relevant, or whether different operators have the same downstream image.

Keep separate the established earlier evidence and its boundaries:

1. Frozen Phase1B: R22 has bounded causal-role necessity/restoration evidence at the authenticated native site; this is not a universal semantic label.
2. Frozen Phase2: `GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED` (matched M1-minus-C1 confirmatory p-value `0.36700787067950813`).
3. Frozen Phase3A: `GEN5_PHASE3_CAUSAL_ROLE_CONTENTION_NOT_ESTABLISHED`; Phase3B remains blocked.
4. Frozen Stages A–D: optimization-path bypass of R22 with partial, anisotropic downstream functional convergence in the tested task setting; not full rank-two downstream equivalence.
5. Frozen Stage E SCALEMATCH: the same-seed oracle `Q_B` and `A_free` four-parameter control recovered approximately 99.07% of unrestricted task gain but **did not** reconstruct the original rank-two source core; this does not demonstrate cross-seed transfer.
6. v10 fine-anchor/static-gap: strong universal W3/W4 standardized reversal falsified on common support; `STRICT_ZERO_OPPORTUNITY=FAIL_RETAINED`, `FORMAL_E0_E3=NOT_PROMOTED`. No equivalence is inferred between fine-token `RESIDUAL` and native-state R22.

The present static computation is **descriptive geometrical analysis**; it is neither task evaluation nor a causal intervention, and the three seeds are not independent task-level replication. No p-value is computed or implied.

## 7. Next scientific question: shared geometry versus shared task function

The current geometry program is **closed at its pre-specified descriptive scope**. Further eigenvalue, threshold, seed, layer, or subspace sweeps are not warranted solely to rescue a universal shared-operator interpretation.

A **new, prospectively specified** functional question remains meaningful:

> Does the geometrically shared donor-operator component carry disproportionate task-functional gain, despite representing only about 28% of the frozen held-out correction operator's squared Frobenius norm?

A minimal future falsifiable design would freeze, *before any functional outcomes are read*:

- For each `s`, the already defined donors `D_s`, their actual two-operator span, and the geometry-only, label-blind projection `U_s = Proj_span{M_t:t in D_s}(M_s)`.
- The exact orthogonal operator residual `V_s=M_s-U_s`, and the frozen full operator `M_s` / zero-correction reference.
- Read-only, same-architecture downstream comparisons `FULL`, `SHARED_ONLY`, `RESIDUAL_ONLY`, `ZERO`, with controls that distinguish unequal perturbation norms, nonlinear interactions, and reparameterization effects. Define endpoint, permitted data split and criterion *before* execution. A prospective extension may assess matched local native-role integrity separately rather than conflating it with task CE.
- No data-derived selection of donor seeds, coefficients, rank, surface, checkpoint, task, threshold, or favorable subgroup. The held-out operator is used only for label-blind geometry projection, so this is still **not** prospective unseen-operator prediction. Predefine what constitutes positive/negative outcomes and retain all three seeds, without pseudo-replicate p-values.

If the shared component produces no distinctive task effect under valid controls, the **task-functional shared-core** hypothesis is unsupported, regardless of the geometric overlap. If task gain is largely retained with `SHARED_ONLY` and not `RESIDUAL_ONLY` under predeclared controls, this supports a *bounded task-functional core*, not ownership privilege or R22 contention. Nonlinearity may make effects non-additive; do not interpret the geometric decomposition as an additive causal loss attribution.

**Authorization boundary:** this report proposes that *future design only*. It does **not** authorize new model forward, evaluation, training, intervention, GPU, Kaggle, or confirmatory-data use. The exact native-role contention hypothesis, if pursued later, requires a separate prospective manipulation qualification and matched R22/C22 controls. The Phase3A failure cannot be overwritten or bypassed.

## 8. Final disposition

- `FIRST_LOSO_JSON_IDENTITY=PASS_REVIEWED`
- `DONOR_SPAN_JSON_IDENTITY=PASS_REVIEWED`
- `FROZEN_CHECKPOINT_SOURCE_IDENTITIES=PASS_REPORTED_BY_USER_EXECUTION`
- `CARTESIAN_REPLAY_AND_ARITHMETIC=PASS_REVIEWED`
- `ACTUAL_DONOR_OPERATOR_COVERAGE_MEAN=0.280930895061`
- `UNEXPLAINED_OPERATOR_ENERGY_MEAN=0.719069104939`
- `PAIRWISE_OPERATOR_ALIGNMENT=POSITIVE_MODERATE_NOT_IDENTICAL`
- `SINGLE_COMMON_OPERATOR_CLAIM=NOT_SUPPORTED`
- `TASK_FUNCTIONAL_CORE=NOT_TESTED`
- `OWNERSHIP_CONTENTION=NOT_ESTABLISHED`
- `INDEPENDENT_CONFIRMATION=NO`
- `TRAINING=NO; EVALUATION=NO; GPU=NO; CODEX=NO`
- `NEXT=PROSPECTIVE_STATIC_FUNCTIONAL_CORE_FALSIFICATION_DESIGN_ONLY`

**Archival boundary:** documentation-only candidate. The two machine-readable JSONs and helper scripts were generated outside the repository and are not automatically imported or promoted to authoritative scientific artifacts. Any future repository archive should keep this report's descriptive/provisional status, stage only the explicit file, and perform a clean `cm ship` manual review. No automatic commit/push or training/evaluation promotion follows from this report.
