# ContraMamba Gen4 Methods Manuscript Draft v1

## Status

- Status: STATIC MANUSCRIPT METHODS DRAFT
- Evidence HEAD: `7fef44bd6955298e886cdf81e55b1d30f0a3c08e`
- Scientific execution: CLOSED
- New model execution: NONE
- New statistical tests: NONE
- New p-values: NONE

This document converts the frozen experimental program into manuscript-facing
Methods prose. It does not create a new scientific endpoint or authorize a
new experiment.

---

## 2.1 Models, populations, and prospective analysis structure

We study frozen pretrained Mamba checkpoints at 130M, 370M, and 1.4B
parameters. The causal analyses operate on independently reconstructed internal
geometry at each checkpoint. Principal-plane vectors, strong-channel indices,
selected ranks, and control ranks are not transferred from one checkpoint to
another. Equal local rank labels therefore do not imply a shared coordinate
identity across models.

The synthetic causal program separates geometry construction from response
evaluation. XG2 and XG4 provide the frozen geometry used to construct paired
local subspaces, whereas the structurally independent XG1 generator provides
prospective response populations. XG1 source pairs contain the same fixed
six-cell structural design and are generated deterministically. Prospective
validation stages use disjoint source-pair ranges, with no response-dependent
pair removal, substitution, or reordering.

All Gen4 causal studies evaluate frozen checkpoints. Scientific interventions,
local gradients, and downstream readouts do not update model parameters.
Exact model, tokenizer, checkpoint, runtime, artifact, and cohort identities
are retained in the frozen provenance artifacts accompanying each experiment.

The main inferential unit throughout the prospective studies is the frozen
item or source-pair population. Cross-checkpoint tests involving 370M and
1.4B use matched source pairs where explicitly specified. These analyses do
not treat the three checkpoints as random samples from a population of
independently trained model seeds.

## 2.2 Local recurrent-state geometry and susceptibility

Within each checkpoint, the analysis uses the frozen target-token intervention
coordinate and the predeclared within-layer strong-channel mask. The mask is
constructed independently at each checkpoint using the same frozen
strong-channel rule rather than transferring channel identities across scales.

For each of XG2 and XG4, the frozen direction rows are normalized to unit norm.
Their float64 CPU uncentered second moment is eigendecomposed, and the five
largest-eigenvalue eigenvectors define the corresponding top-five local
subspace. The frozen projector-contrast construction then pairs the two
subspaces into five checkpoint-local principal planes. All sign
canonicalization and orthonormality rules are deterministic.

For a unit probe direction `w`, local directional susceptibility is measured
with the centered finite difference

`J_i(w) = [F_i(+epsilon; w) - F_i(-epsilon; w)] / (2 epsilon)`.

The primary finite-difference scale is

`epsilon = 0.025`.

For the 130M projector-contrast analysis, each paired plane contains a positive
and negative contrast mode. For plane `k`, with frozen projector-separation
weight `s_k`, the plane contribution is

`C_k(i) = (s_k / 5) * [J_i(PPk+)^2 - J_i(PPk-)^2]`.

The broader five-dimensional susceptibility endpoint used for causal
necessity and restoration is defined from the independently frozen XG2 and
XG4 bases:

`E_XG2(i,c) = (1/5) * sum_j J_i(v_XG2,j; c)^2`,

`E_XG4(i,c) = (1/5) * sum_j J_i(v_XG4,j; c)^2`,

`Q_i(c) = E_XG2(i,c) - E_XG4(i,c)`.

No basis is refit or rotated using XG1 responses.

## 2.3 Prospective causal validation at Mamba-130M

Static localization identified principal pair 3, PP3, as the dominant shared
positive projector-contrast candidate at 130M. PP3 was frozen before
prospective XG1 validation.

External-generator transport was first evaluated on a fresh 300-pair XG1
population. A second non-overlapping 300-pair population tested specificity
against PP5, selected without XG1 responses as the principal plane with
maximum frozen projector separation:

`PP5 = argmax_k sin(theta_k)`.

PP5 is therefore a response-blind hard geometric control, not a random
direction baseline.

Necessity was tested on a third fresh 300-pair XG1 population. Let `h` denote
the native strong-coordinate target state and let `u3+`, `u3-` be the frozen
orthonormal PP3 modes. Native PP3 coefficients are

`a = <h, u3+>`

and

`b = <h, u3->`.

Complete PP3 neutralization removes the native PP3 component:

`T3(h) = h - a u3+ - b u3-`.

The matched PP5 control transfers the same coefficients into the orthonormal
PP5 plane:

`T5(h) = h - a u5+ - b u5-`.

The primary necessity contrast compares the broad endpoint `Q` under these two
conditions rather than testing PP3 with its own PP3-specific response measure.
This avoids a structurally circular necessity endpoint.

Restoration was tested on a fourth disjoint 300-pair XG1 population. Define
the PP3-neutralized background

`B = h - (a u3+ + b u3-)`.

The selected restoration returns the exact removed native component:

`R3 = B + a u3+ + b u3- = h`.

The matched replacement inserts the same coefficients in PP5:

`R5 = B + a u5+ + b u5-`.

Because PP3 and PP5 are orthonormal and use the same coefficient pair, the
added components have equal Euclidean norm. The resulting comparison is a
local matched restoration/replacement test; it is not a claim that PP3 is
globally sufficient for model behavior.

Finite-epsilon robustness was assessed descriptively at `0.025`, `0.0125`,
and `0.00625` using the frozen plane construction. These checks add no new
inferential test and are not extrapolated to an `epsilon -> 0` limit.

## 2.4 Cross-checkpoint recurrence and cross-block geometric transport

For each larger checkpoint, the geometry is reconstructed independently.
On a scale-specific discovery population, define

`G_s,k(i) = Q_restored(s,k,i) - Q_neutralized(s,k,i)`.

The scale-local dominant candidate is the unique maximizer of the mean
discovery response:

`k*_s = unique argmax_k mean_i G_s,k(i)`.

A response-blind control is then selected from the remaining frozen geometry
without using the confirmation outcome. On a disjoint confirmation population
the core endpoint is

`D_CORE_s(i) = Q_restored(s,k*_s,i) - Q_control(s,c*_s,i)`.

This procedure tests recurrence of a causal role under a common scientific
protocol. It does not require the selected principal-plane rank to be the same
across checkpoints.

All planes excluding the selected scale-local component form the residual
mechanism for descriptive characterization. Residual signed effects,
coefficient-mass allocation, dominant residual rank, and generator-family
coupling are reported descriptively. No cross-checkpoint residual significance
test is assigned post hoc.

At Mamba-1.4B, site specificity is tested with a single prospectively fixed
adjacent site. The canonical source/target/intervention triplet is

`(33, 34, 35)`

and the architecture-predeclared downstream `+1` triplet is

`(34, 35, 36)`.

The adjacent direction was fixed from the validated instrumentation boundary
rather than from causal responses, and no additional neighboring site is
searched after observing the result.

To measure cross-block geometric reorientation, the frozen canonical P5
subspace is transported through the adjacent block using the frozen local
Jacobian-vector transport procedure. The transported rank-two subspace is
compared with the independently reconstructed adjacent P5 plane using
principal angles and projector overlap on response-free XG2/XG4 rows.

Functional relevance is then evaluated on the frozen 300-pair XG1 adjacent-site
response population. The prospectively defined paired transport endpoint is

`G = D_TRANSPORT - D_ADJ`.

A separate mandatory sign gate asks whether

`mean(D_TRANSPORT) > 0`.

The relative-shift test and the positive-restoration gate are kept distinct.
Ratios of canonical, adjacent, and transported mean responses are not
interpreted as mediated or explained percentages.

## 2.5 Downstream task readout and behavioral localization

The principal cross-checkpoint readout study compares Mamba-370M and
Mamba-1.4B on the same `xg1_fact_4801..xg1_fact_5100` population of 300
source pairs. Each pair contributes the two frozen behavioral cells
`C0_SHAM` and `C2_NAME`, which are averaged within pair.

For a row with correct class `y`, the native downstream task margin is

`m = z_y - max_{c != y} z_c`.

The local gradient is evaluated at the unmodified native forward state. The
upstream graph is cut at the frozen intervention tensor, which is treated as a
local differentiation leaf; the downstream computation remains intact and
model parameters receive no gradients.

Let `h` be the native strong-coordinate activation and let
`U_sel=[u_sel+,u_sel-]` and `U_ctrl=[u_ctrl+,u_ctrl-]` denote the already
frozen selected and response-blind control planes. Native selected
coefficients are reused in both component constructions:

`C_sel(h) = a u_sel+ + b u_sel-`,

`C_ctrl(h) = a u_ctrl+ + b u_ctrl-`.

The per-row local directional readout is

`Delta_L_row = g^T [C_sel(h) - C_ctrl(h)]`.

The two cells are averaged within source pair to obtain `Delta_L_s,q`.
The primary matched cross-checkpoint endpoint is

`R_q = Delta_L_370M,q - Delta_L_1.4B,q`.

The frozen `G3-GROUP-D-HALF` computation graph uses an intentional
edge-specific backward ownership factor of one half on the relevant D edge
while preserving the forward computation. Consequently, the historically
stored quantity and the numerical forward directional derivative satisfy

`Delta_L_owned = 0.5 * Delta_L_forward`,

or equivalently

`Delta_L_forward = 2 * Delta_L_owned`.

This deterministic positive rescaling changes numerical derivative units but
does not change signs, ranks, correlations, t statistics, or p-values.

Direct behavioral coupling uses the same frozen selected-versus-control
component construction. Because exact selected-component restoration returns
the native state, the behavioral contrast can be written as

`D_BEH = M_native - M_control`.

For stagewise localization, downstream margins are recorded after the frozen
layer-35 intervention under `native`, `dominant_neutralized`, and
`dominant_control` conditions. At each stage `k`,

`A_sel(k) = M_native(k) - M_neutralized(k)`

`B_ctrl(k) = M_control(k) - M_neutralized(k)`

and

`D(k) = M_native(k) - M_control(k) = A_sel(k) - B_ctrl(k)`.

Stagewise quantities are descriptive and are not assigned additional
inferential tests or interpreted as identifying a unique mediator.

## 2.6 Natural-language external transfer and steering boundary

External transfer uses the official AVeriTeC development source with annotated
gold evidence. The source contains 500 examples. The frozen compatible
three-class cohort excludes `Conflicting Evidence/Cherrypicking`, leaving
462 examples:

- `Refuted -> REFUTE`;
- `Not Enough Evidence -> NOT_ENTITLED`;
- `Supported -> SUPPORT`.

The experiment is a causal-transfer study rather than an end-to-end
fact-verification benchmark evaluation. Retrieval and question generation are
not executed. Gold question-answer evidence is serialized deterministically;
the textual justification field is not used as model input.

The frozen input contract uses claim text, an EOS separator, and gold evidence
within a maximum sequence length of 128 tokens. Model-specific tokenizer and
anchor eligibility are verified before scientific response execution, and no
favorable subset is selected after tokenization or model response.

For each compatible example, the external causal conditions are
`native`, `dominant_neutralized`, and `dominant_control` using the
checkpoint-local selected and coefficient-matched response-blind control
planes. The external endpoint is the corresponding correct-class task-margin
contrast under the frozen causal comparison.

The completed 130M/370M external tests form their predeclared transfer family.
The later Mamba-1.4B experiment is a separate prospective negative-sign
extension and does not reopen or modify that family.

Prediction-level control utility is tested separately with the preregistered
Mamba-370M fixed-mirror steering rule. Its utility endpoints count
native-wrong-to-correct transitions, native-correct-to-wrong transitions, and
net accuracy change. Causal margin transfer and prediction-level steering
utility are therefore treated as distinct empirical endpoints.

## 2.7 Statistical analysis and inferential scope

Prospective causal contrasts use the test and direction frozen before their
corresponding response populations are inspected. Positive mean causal
contrasts are generally evaluated with one-sided Student t tests on the
predeclared item or paired-difference endpoint. Matched cross-checkpoint
comparisons use the matched item differences directly.

Multiplicity correction is applied only where a prospective family explicitly
defines multiple primary tests, including the frozen two-test Holm families.
Descriptive residual decompositions, finite-epsilon profiles, geometric
transport summaries, and stagewise localization do not acquire post-hoc
p-values.

No failed historical gate is rescued by redefining its endpoint, tail,
population, plane, layer, token, or control. Likewise, the three checkpoint
means are not fit with a model-size scaling regression or used to estimate a
parameter-count threshold.

The matched Mamba-370M-versus-Mamba-1.4B readout inference concerns variation
across 300 matched source pairs for two fixed pretrained checkpoints. It does
not quantify variation across independently trained model seeds.

## 2.8 Reproducibility and provenance

Every scientific stage freezes its cohort identity, model/checkpoint identity,
tokenizer and anchor contract, geometry or intervention objects, statistical
endpoint, and artifact hashes before downstream interpretation. Raw execution
artifacts are separated from static inferential analyses and manuscript
synthesis.

The manuscript figures and Main Table 1 are generated only from the frozen
artifacts bound in
`reports/reason_router_gen4_main_figure_plot_data_manifest_v1.json`.
The renderer performs no model execution and verifies the bound source
artifacts remain byte-identical during rendering.

The current manuscript construction phase performs no new model execution,
training, endpoint selection, statistical test, or p-value computation.

---

## Evidence mapping for manuscript verification

Primary frozen design and method sources:

- `reports/reason_router_gen4_pp3_xg1_external_transport_scope.md`
- `reports/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_design.md`
- `reports/reason_router_gen4_pp3_necessity_design.md`
- `reports/reason_router_gen4_pp3_restoration_sufficiency_design.md`
- `reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_hypothesis_design.md`
- `reports/reason_router_gen4_one_shot_adjacent_site_specificity_design.md`
- `reports/reason_router_gen4_study_b_readout_alignment_prospective_design.md`
- `reports/reason_router_gen4_cross_scale_stagewise_behavioral_coupling_localization_design.md`
- `reports/reason_router_gen4_averitec_gold_evidence_external_transfer_static_feasibility_audit.md`
- `reports/reason_router_gen4_averitec_mamba14b_negative_sign_transfer_prospective_plan.md`
- `reports/reason_router_gen4_three_scale_readout_owned_forward_correction.md`
- `reports/reason_router_gen4_factor2_gradient_ownership_root_cause_closure.md`

The evidence-mapping block is an internal drafting aid and should not appear
in the submitted manuscript.

`METHODS_MANUSCRIPT_DRAFT_V1_READY = YES`
