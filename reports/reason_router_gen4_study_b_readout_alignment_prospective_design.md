# ContraMamba Gen4 Study B — Cross-Scale Task-Gradient / Readout-Alignment Prospective Design

Status: PROSPECTIVE_DESIGN_HASH_FREEZE
Execution authorized by this file: NO
Training authorized: NO
Current Study-A outcome required to define this design: NO

## 1. Independence from Study A

Study B is fixed independently of Study A response outcomes.

Study A results may not change:
- scale-local selected plane identities;
- response-blind control plane identities;
- intervention block/layer/token;
- Study-B population;
- task-margin definition;
- Study-B primary endpoint or primary test.

Study A may affect only later joint interpretation/synthesis.

## 2. Scientific question

Does the downstream correct-class task-margin readout couple with the already-frozen
scale-local causal geometry in opposite directions at Mamba-370M and Mamba-1.4B?

This is a prospective mechanistic follow-up conditioned on the already-known
Experiment-1 behavioral sign pattern. It is not an independent replication of that
behavioral result.

## 3. Frozen scale-local objects

Mamba-370M:
- selected plane: P3
- response-blind control plane: P5

Mamba-1.4B:
- selected plane: P5
- response-blind control plane: P4

For both scales:
- source block: 33
- target residual layer: 34
- intervention layer: 35
- anchor: A_IDENTITY
- target offset: +2
- intervention coordinate: mixer.in_proj content half at the frozen target token
- frozen scale-local strong mask
- no plane/control reselection

## 4. Frozen population and row aggregation

Population:
- xg1_fact_4801..xg1_fact_5100
- N = 300 matched source pairs

Rows per pair:
- C0_SHAM, correct label SUPPORT (class 2)
- C2_NAME, correct label NOT_ENTITLED (class 1)

No row subset is selected by gradient magnitude, prediction correctness, sign,
behavioral effect, or any Study-A result.

Cell-level quantities are averaged within each source pair exactly as in Experiment 1.

## 5. Native task margin and local gradient

For row r with correct class y:

m_r = z_y - max_{c != y} z_c

The gradient is evaluated at the unmodified native forward state.

Let x_s,r be the block-35 target-token content-half activation at the frozen
mixer.in_proj intervention coordinate for scale s.

Define:

g_s,r = d m_r / d x_s,r

Implementation semantics:
- all model parameters are frozen and require no parameter gradients;
- the upstream graph is cut at the intervention tensor;
- the intervention tensor is treated as a local leaf for differentiation;
- the downstream model graph remains intact;
- no correction/intervention is applied during gradient measurement;
- save the full content-half gradient only as needed for audit, and gather the
  frozen strong coordinates for all plane/readout calculations;
- exact wrong-class argmax used by max() is recorded; exact ties are a blocking
  technical condition, not a basis for row removal.

## 6. Frozen selected/control geometry at each native row

In strong coordinates, let U_sel=[u_sel,+, u_sel,-] and
U_ctrl=[u_ctrl,+, u_ctrl,-].

For native strong-coordinate activation h:

a = h^T u_sel,+
b = h^T u_sel,-

Define the selected component:

C_sel(h) = a u_sel,+ + b u_sel,-

Define the response-blind coefficient-matched control component:

C_ctrl(h) = a u_ctrl,+ + b u_ctrl,-

These are the two component objects whose difference equals the
dominant-restored versus dominant-control displacement used by Experiment 1:

h_restored - h_control = C_sel(h) - C_ctrl(h)

No coefficient fitting is performed.

## 7. Per-row readout-alignment outputs

For every scale and row save:

- ||g||_2
- ||Pi_sel g||_2
- ||Pi_ctrl g||_2
- ||Pi_sel g||_2 / max(||g||_2, 1e-12)
- ||Pi_ctrl g||_2 / max(||g||_2, 1e-12)
- U_sel^T g (two coordinates)
- U_ctrl^T g (two coordinates)
- ||C_sel(h)||_2
- ||C_ctrl(h)||_2
- L_sel = g^T C_sel(h)
- L_ctrl = g^T C_ctrl(h)
- Delta_L_row = L_sel - L_ctrl
- cosine(g, C_sel(h)) when C_sel is nonzero
- cosine(g, C_ctrl(h)) when C_ctrl is nonzero
- correct-class ID and active wrong-class argmax ID
- finite/shape/provenance audits

Delta_L_row is the first-order prediction of the Experiment-1
restored-versus-control task-margin effect at that native row.

## 8. Pair-level aggregation

For each scale s and source pair q:

Delta_L_s,q =
    0.5 * (
        Delta_L_s,q,C0_SHAM
        +
        Delta_L_s,q,C2_NAME
    )

All other row-level outputs are summarized descriptively with the same two-cell
pairing where applicable.

## 9. Primary cross-scale endpoint

Define the matched-pair scale contrast:

R_q = Delta_L_370M,q - Delta_L_1.4B,q

Primary hypothesis:

H0: E[R] <= 0
H1: E[R] > 0

Primary test:
- one-sided paired / one-sample Student t-test on {R_q}
- N = 300
- alpha = 0.05
- exactly one primary p-value
- no multiplicity correction is needed because there is exactly one primary test

## 10. Mandatory sign gates

A significant positive R alone is not sufficient for a sign-reversal claim.

The mechanistic scale-sign pattern is supported only if all three conditions hold:

1. primary p < 0.05
2. mean(Delta_L_370M) > 0
3. mean(Delta_L_1.4B) < 0

If the primary contrast is significant but either sign gate fails, report
cross-scale separation without claiming the prospectively specified sign reversal.

## 11. Secondary descriptive outputs

No additional inferential p-values are primary or confirmatory.

Report descriptively:
- per-scale mean, SD, median, quartiles, min/max of Delta_L
- per-scale fraction Delta_L > 0
- distributions of gradient norm and selected/control projection fractions
- directional-coordinate summaries
- selected/control cosine summaries
- pairwise cross-scale correlation of Delta_L, if finite
- numerical stability / zero-gradient / tie counts

After raw gradient evidence is frozen, a CPU-only static merge may read the already
frozen Experiment-1 D_BEH values and report, descriptively only:
- Pearson correlation between Delta_L_s and D_BEH_s
- mean and distribution of first-order residual D_BEH_s - Delta_L_s

These descriptive comparisons are conditioned on already-known Experiment-1 behavior
and add no p-value.

## 12. Raw-execution information boundary

The GPU raw measurement must not read:
- Experiment-1 D_BEH pair values;
- Experiment-1 behavioral inference p-values;
- Study-A transported-response results;
- Study-A response p-values.

It may use only the frozen structural population, labels, checkpoints, geometry,
token/anchor metadata, and scale-local selected/control planes required to execute
the native gradient measurement.

Raw GPU execution computes no p-value and no Study-B scientific conclusion.

## 13. Technical gate before scientific execution

Because Study B requires local backward propagation through the exact frozen
downstream path, implementation must first pass a bounded technical gate.

The gate:
- uses one non-Study-B diagnostic row when available;
- otherwise may use one Study-B row only if no numeric gradient/alignment value is
  surfaced or retained, and only capability booleans are emitted;
- checks exact model/checkpoint/runtime identity;
- checks local activation leaf construction;
- checks finite downstream margin;
- checks gradient shape and finiteness;
- checks no parameter gradient is created;
- checks exact selected/control plane identities;
- does not compute Delta_L population summaries or any p-value.

A technically unsupported backward path closes that implementation route only; it
does not alter the scientific endpoint or permit a different cohort/plane/test.

## 14. Preferred execution shape

Preferred single scientific run:
- GPU0: Mamba-370M, all 300 pairs x 2 cells = 600 native forward/backward rows
- GPU1: Mamba-1.4B, all 300 pairs x 2 cells = 600 native forward/backward rows

Total:
- 1200 native full-model forwards
- 1200 local backward evaluations from task margin to the intervention tensor
- zero training steps
- zero parameter updates
- zero intervention-condition forwards

This scale-per-GPU layout keeps both scales at one execution HEAD and avoids
cross-scale semantic drift.

## 15. Raw artifacts

The raw run should freeze exactly:

1. readout_alignment_items.jsonl
2. raw_readout_alignment_summary.json
3. artifact_manifest.json
4. SHA256SUMS.txt

Each row records scale, pair, cell, model/checkpoint identity, native margin identity,
gradient/readout outputs, plane/control identity, and boundary flags.

## 16. Static primary analysis after raw freeze

Only after the four raw artifacts are frozen:

- verify exact pair/cell coverage for both scales;
- aggregate row Delta_L to pair Delta_L;
- verify matched pair order;
- construct R_q;
- execute exactly one one-sided Student t-test;
- apply the two predeclared sign gates;
- optionally merge frozen Experiment-1 D_BEH for descriptive-only correlations and
  first-order residuals;
- emit no rescue subgroup, scale-specific alternative test, row filter, or second
  primary p-value.

## 17. No-rescue rules

Prohibited after any Study-B gradient outcome is observed:
- row filtering by norm, correctness, sign, margin, or Delta_L;
- layer/token changes;
- scale-local plane or control changes;
- normalization changes to replace the raw Delta_L primary;
- alternative robust/nonparametric test as a replacement primary;
- additional primary p-values;
- new cohort;
- epsilon/intervention tuning;
- use of Study-A results to alter Study B.

## 18. Interpretation boundary

If the primary paired contrast passes and both sign gates pass, Study B supports:

"the local downstream task-readout differential associated with the frozen
selected-versus-control causal geometry reverses sign across Mamba-370M and
Mamba-1.4B on the matched Experiment-1 population."

This does not establish:
- universal plane identity across model scale;
- monotonic scaling;
- causal equivalence between Study A transport geometry and Study B readout geometry;
- independent replication of Experiment-1 behavior.

If the primary contrast or sign gates fail, the prospectively specified
readout-sign-reversal explanation is not established. No rescue analysis is allowed.
