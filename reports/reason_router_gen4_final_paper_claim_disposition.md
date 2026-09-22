# ContraMamba Gen4 Final Paper Claim Disposition

## Status

`STATIC_FINAL_PAPER_CLAIM_FREEZE`

Evidence base HEAD:

`b5b5436a7df4c01385262ac0ce327d264aa33b38`

This artifact freezes the paper-facing claim set after the three-scale
readout-ownership correction and factor-2 root-cause closure.

No new experiment is authorized by this document.

Disposition labels:

- `KEEP`: retain as a main-text scientific claim under its existing scope.
- `CORRECT`: retain the result, but replace the stated semantics or magnitude wording.
- `DROP`: remove from the paper as a scientific claim; do not rescue by new experiments.
- `APPENDIX`: retain only as supporting, robustness, diagnostic, null, or provenance evidence.

## 1. Main claim disposition

| ID | Claim | Disposition | Frozen paper wording / action |
|---|---|---|---|
| C1 | A scale-local dominant causal role recurs across tested native-Mamba scales while the surrounding residual realization reorganizes. | KEEP | Use the bounded `CORE-STABLE / RESIDUAL-PLASTIC` formulation. Do not claim same-numbered plane identity or a universal scaling law. |
| C2 | The 130M and 370M readout means are positive while the 1.4B readout mean is negative; the matched 370M-vs-1.4B sign reversal is supported. | CORRECT | Keep the sign/order conclusion, but call stored values `Delta_L_owned` / gradient-ownership-weighted readouts. When discussing the numerical forward derivative use `Delta_L_forward = 2 * Delta_L_owned`. |
| C3 | Mamba-130M has positive local readout alignment. | CORRECT | Keep the conclusion and original t/p. Stored mean `+0.001120366036532127` is ownership-weighted; forward-equivalent mean is `+0.002240732073064253`. |
| C4 | Mamba-370M vs Mamba-1.4B shows a readout-alignment sign reversal. | CORRECT | Keep the paired t=`10.771333954019786`, p=`2.2238330789610916e-23`, and sign gates. Do not present owned-gradient magnitudes as ordinary forward derivatives. |
| C5 | Downstream behavioral relevance recurs uniformly with scale. | DROP | Frozen behavioral bridge supports Mamba-370M but not Mamba-1.4B. Use `SCALE_SPECIFIC_BEHAVIORAL_BRIDGE_ONLY`. |
| C6 | Internal causal relevance reaches the downstream decision margin at 370M, while the same positive bridge is not established at 1.4B. | KEEP | This is the correct behavioral-bridge claim. |
| C7 | The frozen causal intervention transfers to natural-language AVeriTeC gold-evidence inputs at 130M and 370M. | KEEP | Retain with the existing narrow three-class/gold-evidence scope; no retrieval, benchmark-superiority, or 1.4B transfer claim. |
| C8 | The tested 370M fixed-mirror intervention provides useful steering. | DROP | Utility was not established; 0 corrections, 0 damages, 0 net accuracy change. |
| C9 | Explanatory causal structure and useful control are distinct empirical questions. | KEEP | Retain as a synthesis statement supported by causal-transfer evidence plus failed fixed-mirror steering utility. |
| C10 | The canonical 1.4B site is stronger than the single prospectively fixed adjacent +1 site under the matched test. | KEEP | Retain the one-shot site-specificity claim only; do not generalize to a global layer optimum. |
| C11 | The factor near two reveals unexplained nonlinear amplification or a finite-displacement gain regime. | DROP | Root cause is the intentional `G3-GROUP-D-HALF` gradient-ownership graph. |
| C12 | The factor near two is caused by fused CUDA Mamba backward semantics. | DROP | Independent slow/reference backend reproduces the same owned-gradient mismatch; backend-specific defect is not supported. |
| C13 | For the frozen D-half readout, `D_BEH(alpha) ~= alpha * Delta_L_forward` locally. | CORRECT | This is the corrected first-order interpretation. Equivalent historical notation is `D_BEH(alpha) ~= 2 * alpha * Delta_L_owned`. |
| C14 | Same-numbered principal planes are semantically identical across model scales. | DROP | Plane identities are scale-local rank labels; only the role/procedure is compared. |
| C15 | There is a universal monotonic scaling law in readout magnitude, causal effect, or downstream usefulness. | DROP | Cross-scale magnitude calibration and monotonic usefulness are not established. |
| C16 | ContraMamba establishes architecture-independent or transformer-general causal universality. | DROP | Evidence is limited to the tested Mamba checkpoints, sites, populations, and endpoints. |

## 2. Appendix-only evidence

| ID | Evidence / claim | Disposition | Reason |
|---|---|---|---|
| A1 | Positive small-alpha factor-2 curve down to alpha `0.03125`. | APPENDIX | Important root-cause validation, but no longer a standalone mystery or main claim. |
| A2 | Symmetric gradient-consistency finite-difference diagnostic. | APPENDIX | Establishes forward-vs-owned-gradient mismatch and rules out wrong-class switching/positive-side curvature explanations. |
| A3 | Fast-vs-slow reference-backend diagnostic. | APPENDIX | Rules out backend-specific factor-2 explanations; supporting root-cause evidence. |
| A4 | Small-epsilon P3 spectral robustness at `0.0125` and `0.00625`. | APPENDIX | Robustness result; does not establish an epsilon-to-zero limit or new causal claim. |
| A5 | Residual decomposition / coefficient census / generator-family localization. | APPENDIX | Mechanistic support for residual plasticity; useful figures/tables, but secondary to the core-stable/residual-plastic claim. |
| A6 | Precursor-v4 analytic local differential susceptibility Stage A. | APPENDIX | Primary null: `LOCAL_DIFFERENTIAL_SUSCEPTIBILITY_TEMPORAL_PRECEDENCE_NOT_SUPPORTED`, p=`0.29444632184455377`. It is not ownership-scaled because it uses direct `MambaForCausalLM` VJP. |
| A7 | O0b narrow sufficiency-sensitive precursor clue. | APPENDIX | Exploratory narrow clue, not a hallucination detector or general precursor result. |
| A8 | O0c terminal-localized recurrent-state separation. | APPENDIX | Broad native precursor was not supported; terminal-localized observation is heterogeneous and post-hoc limited. |
| A9 | Detailed factor-2 provenance, snapshot, kernel, and backend checks. | APPENDIX | Reproducibility/provenance support only. |

## 3. Claims explicitly removed from the main paper

The following narratives must not appear as positive main-text claims:

1. `factor-2 = mysterious Mamba gain`;
2. `factor-2 = fused-kernel backward bug`;
3. `factor-2 = ordinary finite-displacement curvature`;
4. `readout magnitude is directly comparable across scales without qualification`;
5. `increasing scale monotonically strengthens behavioral usefulness`;
6. `fixed P3 mirror steering is useful`;
7. `same-numbered plane rank has a universal semantic identity`;
8. `Precursor-v4 predicts future unsupported commitment`;
9. `broad native hallucination precursor is established`;
10. `the current evidence establishes transformer or architecture-wide universality`.

No new experiment may be introduced to rescue any of these dropped claims.

## 4. Paper-facing corrected quantitative table

| Scale | Paper quantity to report | Value |
|---|---|---:|
| Mamba-130M | owned-gradient mean | `+0.001120366036532127` |
| Mamba-130M | forward-equivalent mean | `+0.002240732073064253` |
| Mamba-370M | owned-gradient mean | `+0.0005089563518854174` |
| Mamba-370M | forward-equivalent mean | `+0.001017912703770835` |
| Mamba-1.4B | owned-gradient mean | `-0.001195447505886224` |
| Mamba-1.4B | forward-equivalent mean | `-0.002390895011772448` |
| 370M - 1.4B | owned-gradient paired mean difference | `0.001704403857771641` |
| 370M - 1.4B | forward-equivalent paired mean difference | `0.003408807715543282` |

Paper convention:

- main text may show both owned and forward-equivalent columns once;
- subsequent sign/rank discussion may use `Delta_L_owned` if explicitly labeled;
- behavioral first-order discussion should use `Delta_L_forward`;
- historical immutable artifacts keep their original field name `Delta_L`.

## 5. Recommended main-text claim set

The paper should be organized around four claims only:

1. **Scale-local causal core with residual plasticity.**
   A dominant causal role recurs under the frozen reconstruction procedure, while
   the residual realization reorganizes across tested Mamba scales.

2. **Corrected three-scale downstream readout geometry.**
   Gradient-ownership-weighted readout alignment is positive at 130M and 370M
   and negative at 1.4B; the corresponding forward-equivalent derivatives are
   exactly twice the stored magnitudes under the shared D-half arm, without
   changing sign/rank inference.

3. **Behavioral relevance is real but scale-dependent.**
   The internal causal structure reaches the downstream decision margin at 370M,
   but the same positive behavioral bridge is not established at 1.4B; natural-
   language causal transfer is nevertheless supported at 130M and 370M.

4. **Mechanistic validity does not imply universal control utility.**
   Small-epsilon robustness and one-shot site specificity support the internal
   mechanism, while the preregistered fixed-mirror steering intervention did not
   establish practical utility.

Everything else is supporting evidence, boundary, null result, or appendix material.

## 6. New-experiment stop rule

Effective after this claim freeze:

`NEW_EXPERIMENTS_FOR_CURRENT_PAPER = STOP`

Do not add:

- more alpha values;
- smaller epsilon values;
- another backend comparison;
- another factor-2 cohort;
- another scale solely to smooth the cross-scale story;
- another neighboring layer/site;
- another steering transform to rescue the failed fixed-mirror utility result;
- another precursor VJP variant to rescue the null;
- post-hoc plane re-selection;
- new subgroup tests;
- new p-values for descriptive residual analyses.

Permitted work is limited to:

- manuscript writing;
- figure/table generation from frozen evidence;
- deterministic semantic relabeling already specified by the correction artifact;
- citation/related-work completion;
- reproducibility packaging;
- formatting and submission preparation.

## 7. Final paper state

`PAPER_CLAIM_SET_FROZEN = YES`

`CURRENT_PAPER_NEW_SCIENTIFIC_EXECUTION = CLOSED`

`NEXT_ACTION = WRITE_AND_PACKAGE_EXISTING_EVIDENCE`
