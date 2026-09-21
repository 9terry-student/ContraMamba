# ContraMamba Gen4 — Next Mechanistic Program Prospective Freeze

## Status

`PROSPECTIVE_RESEARCH_PROGRAM_FREEZE`

Branch:

`gen4-mamba370m-core-replication`

Parent HEAD at freeze:

`e95fb499a544144dfbff6cf032ae871295542494`

This document freezes three next mechanistic questions and their interpretation boundaries.

It does **not** authorize scientific execution, statistical testing, training, or a Kaggle run by itself.

The completed five-experiment post-synthesis program remains closed. This document does not reopen or rescue any closed primary endpoint.

The three frozen studies are:

1. **A — Cross-layer causal transport**
2. **B — Cross-scale task-gradient / readout alignment**
3. **C — Intervention manifold-deviation reviewer-defense audit**

The preferred scientific order is:

`A -> B -> C`

Study C may begin with read-only static reuse of already frozen artifacts, but any new model execution for C requires a separate bounded execution design.

---

# A. Cross-layer causal transport

## A1. Scientific question

Experiment 5 established a sharp one-block site-specificity pattern at Mamba-1.4B:

- canonical triplet `(33,34,35)`;
- adjacent `+1` triplet `(34,35,36)`;
- fixed causal candidate `P5`;
- canonical `mean(D_CAN) = +9.102783197942576e-09`;
- adjacent `mean(D_ADJ) = -1.551069575441687e-09`;
- paired specificity `mean(S) = +1.0653852773384264e-08`;
- final result:
  `MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_SUPPORTED`.

The unresolved mechanistic question is:

> Why does the causal effect change sign one block downstream?

The frozen transport hypothesis is:

`P5_canonical_35 --J_35_to_36(q)--> transported_P5_36(q)`

and the target comparison is against the independently reconstructed site-local plane:

`P5_adjacent_36`.

This is not a layer sweep.

No other layer, token, plane, row family, checkpoint, or adjacent direction may be selected from transport outcomes.

## A2. Frozen local map

At both sites, use the already frozen intervention coordinate:

- `mixer.in_proj` output;
- content/x half only;
- target token `A_IDENTITY + 2`;
- gate half fixed;
- non-target tokens fixed at the local boundary;
- full ambient content coordinate used for transport comparison.

For frozen row `q`, define:

`Phi_q : R^4096 -> R^4096`

where the input is the block-35 target-token content half and the output is the block-36 target-token content half before block-36 depthwise convolution.

The row-conditioned Jacobian is:

`J_q = d Phi_q / d x35_q,tau`.

A global input-independent `T_35_to_36` is not an authorized substitute.

## A3. Frozen source and target planes

Scatter the canonical site-local P5 basis through its frozen strong indices into the common ambient coordinate:

`U35 = [u35_plus, u35_minus] in R^(4096 x 2)`.

Scatter the adjacent P5 basis through its own frozen strong indices:

`U36 = [u36_plus, u36_minus] in R^(4096 x 2)`.

For each row:

`W_q = J_q U35`.

If the transported span has numerical rank two:

`Q_q = orth(W_q)`.

The canonical and adjacent compressed coordinates must never be dot-compared directly before this ambient scatter.

## A4. Frozen technical-resolution ladder

The original direct forward-mode path has already been tested and closed:

`torch.func.jvp` on the exact authenticated fast-CUDA path

-> `FORWARD_JVP_UNSUPPORTED`

because the causal-conv custom `torch.autograd.Function` does not provide the functorch `setup_context` protocol required by that transform.

This is a technical backend result, not a scientific transport result.

The fallback order is now frozen.

### A4.1 Next gate: reverse-over-reverse exact autodiff

For:

`y = Phi(x)`

and desired direction `v`, introduce auxiliary `w` and compute:

`g_x = grad_x(w^T y) = J^T w`

then:

`s = v^T g_x = w^T J v`

and therefore:

`grad_w(s) = J v`.

The next bounded gate asks only whether this exact `Jv` is technically obtainable through the same frozen local map using ordinary reverse-mode plus double backward.

The gate must preserve:

- same Mamba-1.4B model/checkpoint;
- same exact authenticated kernel bytes;
- same frozen row `xg2_fact_301`;
- same cell `C2_NAME`;
- same anchor `A_IDENTITY`;
- same target offset `+2`;
- same block35 -> block36 local map;
- same canonical P5 plus/minus directions;
- frozen model parameters;
- no Experiment-5 XG1 response access;
- no principal angles;
- no projector overlap;
- no Procrustes result;
- no p-value;
- no population transport conclusion.

If double backward is unsupported, that is a technical failure only.

### A4.2 Second fallback: exact reference autodiff vs fixed-epsilon fast-CUDA finite difference

Only if reverse-over-reverse exact autodiff fails technically, a separate one-row equivalence gate may compare:

`reference exact-autodiff Jv`

against:

`Jv_hat(epsilon) = [Phi(x + epsilon v) - Phi(x - epsilon v)] / (2 epsilon)`

on the authenticated fast-CUDA path.

The reference is defined by mathematical semantics, not by CPU identity:

> an unfused/reference implementation of the exact same frozen local map that is differentiable by standard autodiff.

The finite-difference radius is prospectively fixed to:

`epsilon = 0.025`

because this is the already frozen Mamba-1.4B Experiment-5 intervention scale at the same intervention semantics.

No epsilon sweep is allowed.

No result-based epsilon selection is allowed.

No tolerance relaxation after observing the equivalence result is allowed.

If this fixed estimator-equivalence gate fails, the finite-difference route closes.

## A5. Frozen population if a technical route passes

The first scientific transport measurement uses the already frozen response-free geometry populations only:

- XG2: `xg2_fact_301..xg2_fact_600`;
- XG4: `xg4_fact_301..xg4_fact_600`;
- exactly 300 source pairs per family;
- total: `N = 600` row-conditioned maps;
- row/cell: `C2_NAME`;
- anchor: `A_IDENTITY`;
- target offset: `+2`.

The selection is inherited from the response-free geometry program and the one-row technical gate.

No XG1 Experiment-5 response is used to select transport rows.

## A6. Frozen transport outputs

For every row `q`, save at minimum:

- `||J_q u35_plus||_2`;
- `||J_q u35_minus||_2`;
- singular values of `W_q`;
- numerical rank of `W_q`;
- condition number of the transported two-vector span when rank two;
- singular values `sigma_1 >= sigma_2` of `Q_q^T U36`;
- principal angles:
  `theta_k = arccos(sigma_k)`;
- normalized projector overlap:
  `Omega_q = 0.5 * ||Q_q^T U36||_F^2`;
- orthogonal Procrustes residual between `Q_q` and `U36`.

The first population transport measurement is descriptive/mechanistic.

It adds no p-value unless a later prospective inferential design is frozen before transport outcomes are observed.

## A7. Frozen interpretation branches

Three qualitatively distinct outcomes must remain separate.

### Preserved geometry

`healthy rank + high alignment`

supports the interpretation that the transported canonical P5 remains close to the adjacent P5 geometry.

If the behavioral sign still differs, the remaining explanation moves toward local causal/readout semantics rather than loss of subspace orientation.

### Reoriented geometry

`healthy rank + low alignment`

supports cross-block causal-subspace reorientation/reorganization.

### Anisotropic collapse/compression

`rank loss or severe ill-conditioning`

must not be mislabeled as simple plane rotation.

It instead identifies anisotropic transport/compression as the primary mechanistic candidate.

No post-result threshold search is allowed to manufacture one of these labels.

## A8. Optional transported-basis adjacent response

A later one-shot response experiment may test the transported basis at the adjacent site.

It is **not** authorized by this document.

It must be prospectively designed after the transport measurement path itself is validated and before any new response is observed.

A result-adaptive choice of layer, token, plane, epsilon, row subset, or transported basis is prohibited.

---

# B. Cross-scale task-gradient / readout alignment

## B1. Scientific question

Experiment 1 established a scale-specific behavioral contrast on the same frozen XG1 population.

Mamba-370M:

- selected causal plane: `P3`;
- control plane: `P5`;
- `mean(D_BEH) = +0.0011206856990853946`;
- Holm-adjusted `p = 1.4957180487214262e-08`;
- behavioral bridge supported.

Mamba-1.4B:

- selected causal plane: `P5`;
- control plane: `P4`;
- `mean(D_BEH) = -0.002012885312239329`;
- Holm-adjusted `p = 0.9999999999753244`;
- positive bridge not supported.

The open mechanistic question is:

> Does the downstream task-margin gradient align differently with the scale-local causal correction geometry at 370M versus 1.4B?

This study explains the already observed scale-dependent behavioral coupling.

It is not an independent replication of Experiment 1.

## B2. Frozen scale-local objects

Use exactly the already frozen scale-local causal objects.

Mamba-370M:

- selected plane: `P3`;
- response-blind control: `P5`.

Mamba-1.4B:

- selected plane: `P5`;
- response-blind control: `P4`.

Both inherit the already frozen homologous intervention site:

- source block `33`;
- target residual layer `34`;
- intervention layer `35`;
- anchor `A_IDENTITY`;
- target offset `+2`;
- `mixer.in_proj` content-half intervention semantics.

Results from Study A may not be used to change these B objects.

## B3. Frozen population

Use the exact Experiment-1 behavioral bridge population:

`xg1_fact_4801..xg1_fact_5100`

with the same frozen cells and pair aggregation used by the behavioral bridge.

No new scale-specific rescue cohort is allowed.

No row selection may depend on gradient magnitude, behavioral effect, correctness, or sign.

## B4. Gradient object

For each scale `s` and frozen row `q`, at the unmodified native forward state define the task margin:

`m_y`

using the same correct-class margin semantics as the frozen behavioral bridge.

Let `x_s,q` denote the scale-local intervention tensor at the frozen target token.

Define:

`g_s,q = grad_(x_s,q) m_y`.

Model parameters remain frozen.

The derivative is with respect to the local activation only.

No parameter update or training is allowed.

## B5. Frozen readout-alignment outputs

For selected plane `U_sel` and control plane `U_ctrl`, save:

- `||g||_2`;
- `||Pi_sel g||_2`;
- `||Pi_ctrl g||_2`;
- normalized projection fractions relative to `||g||_2`;
- directional coordinates `U_sel^T g`;
- directional coordinates `U_ctrl^T g`.

For the exact frozen state-dependent behavioral corrections:

`C_sel(h)` and `C_ctrl(h)`

also save:

- raw local-linear selected prediction:
  `L_sel = g^T C_sel(h)`;
- raw local-linear control prediction:
  `L_ctrl = g^T C_ctrl(h)`;
- primary differential local-linear prediction:
  `Delta_L = L_sel - L_ctrl`;
- cosine alignment between `g` and each nonzero correction vector.

`Delta_L` is the key readout differential because, to first order, it predicts the selected-versus-control task-margin contrast.

## B6. Prospective sign question

The mechanistic sign pattern of interest is prospectively fixed from the already known Experiment-1 behavioral outcomes:

- Mamba-370M: positive `Delta_L`;
- Mamba-1.4B: negative `Delta_L`.

Any statistical test of this pattern must be declared before gradient outcomes are inspected.

This program document does not execute those tests.

If inferential testing is later authorized, it must be labeled:

> prospective mechanistic follow-up conditioned on an already known behavioral sign pattern,

not independent confirmation of the behavioral result.

## B7. Interpretation boundary

If the scale-local causal planes remain strong but `Delta_L` reverses sign across scale, the supported mechanistic interpretation becomes:

> downstream task-readout differential geometry is scale-dependent even when a scale-local internal causal role remains well-defined.

This does not imply universal plane identity across scale.

It does not imply monotonic scaling.

It does not imply that Study A and Study B share one global coordinate system.

---

# C. Intervention manifold-deviation reviewer-defense audit

## C1. Motivation

Recent mechanistic-interpretability work has emphasized that causal activation interventions may move representations away from the model's natural representation distribution.

Grant, Han, Tartaglini, and Potts, *Addressing divergent representations from causal interventions on neural networks*, ICLR 2026, distinguish:

- potentially harmless divergence;
- pernicious divergence that can recruit hidden pathways or dormant behavior.

ContraMamba therefore freezes a reviewer-defense audit of intervention displacement.

This is a rigor/faithfulness audit, not a novelty experiment.

## C2. First priority: static reuse

First inspect already frozen intervention artifacts only.

No new model forward is performed if the required native/intervened state quantities are already stored.

The audit may reuse frozen correction vectors and state captures from the completed causal studies, but may not reopen their scientific decisions.

If existing artifacts are insufficient for a metric, report it as unavailable rather than reconstructing it from an incompatible representation.

## C3. If new capture is required

A separate bounded capture design is required before new model execution.

Any new capture must preserve:

- exact frozen models/checkpoints;
- exact frozen intervention sites;
- exact frozen populations;
- exact frozen conditions;
- no intervention magnitude tuning;
- no response-guided row selection;
- no new steering optimization;
- no new causal-plane discovery.

The purpose is only to record native and intervened representations needed for the audit.

## C4. Frozen primary displacement metrics

At the intervention tensor, let:

`h_native`

be the native representation and:

`h_int = h_native + delta_h`

the intervened representation.

Primary displacement:

`R_rel = ||delta_h||_2 / max(||h_native||_2, 1e-12)`.

This must be reported in both:

- full content-half coordinates;
- the frozen strong-coordinate subspace used by the intervention,

when both are available.

## C5. Frozen nearest-native-state metric

For a frozen native reference set `H_native`, excluding the same row `q`, define:

`d_int(q) = min_(r != q) ||h_int(q) - h_native(r)||_2`

and:

`d_native(q) = min_(r != q) ||h_native(q) - h_native(r)||_2`.

Define the local-density-normalized divergence ratio:

`R_NN(q) = d_int(q) / max(d_native(q), 1e-12)`.

This is a descriptive manifold-deviation proxy.

It is not proof that the representation is on- or off-manifold.

## C6. Mahalanobis-like metrics

No covariance-regularized or Mahalanobis-like metric is primary in this freeze because the high-dimensional covariance problem creates avoidable regularization choices.

Such a metric may be added only if its covariance estimator, dimensionality reduction, and regularization are frozen before intervention-distance outcomes are observed.

No regularization search is allowed.

## C7. Claim boundary

A small displacement does not prove mechanistic faithfulness.

A large displacement does not automatically prove a pernicious intervention.

The Grant et al. distinction between harmless and pernicious divergence is functional, not purely geometric.

Therefore Study C may support only claims such as:

- intervention magnitude is small/large relative to native activation scale under the frozen metric;
- intervened states remain near/far from the frozen native-state neighborhood under the frozen nearest-neighbor metric;
- divergence differs across already frozen intervention conditions or scales, if prospectively compared.

It may not claim that hidden pathways were or were not activated without additional functional evidence.

---

# Program-level ordering and anti-rescue rules

## Fixed order

Primary mechanistic order:

1. resolve Study A measurement operator technically;
2. execute/freeze Study A transport measurement;
3. execute Study B readout-alignment measurement;
4. perform Study C static audit and only then decide whether bounded new capture is necessary.

## No outcome-adaptive cross-contamination

Study A outcomes may not change:

- Study B scale-local plane identities;
- Study B intervention layer/token;
- Study B population.

Study B outcomes may not change:

- Study A transport plane/site/population.

Study C outcomes may not be used to retune A or B interventions.

## Closed branches remain closed

This program does not reopen:

- Precursor-v4;
- NAME native-state branch;
- the closed Gen4 x K directional-alignment transport branch;
- the closed XG2/XG4 local-Jacobian branch;
- the completed post-synthesis Experiments 1-5.

## Supplementary-only candidate

The remaining Experiment-2 evidence-length/truncation decomposition is not part of the primary three-study program.

If performed, it is a supplementary static analysis only:

- consumed evidence length vs `D_EXT`;
- truncated vs non-truncated descriptive comparison;
- no rescue of the already frozen external-transfer conclusion.

---

# Current next action

The current active scientific line remains Study A.

The direct `torch.func.jvp` path is closed technically.

The next bounded implementation target is:

`same Jv via reverse-over-reverse exact autodiff on one frozen row`.

This document itself authorizes no execution.

`SCIENTIFIC_EXECUTION_AUTHORIZED = FALSE`

`STATISTICAL_TESTING_AUTHORIZED = FALSE`

`TRAINING_AUTHORIZED = FALSE`
