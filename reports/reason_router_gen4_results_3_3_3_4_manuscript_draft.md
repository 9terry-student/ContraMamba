# ContraMamba Gen4 Results Draft — Sections 3.3–3.4

## Status

- Status: STATIC MANUSCRIPT PROSE DRAFT
- Evidence HEAD: `3f1015c71ab951bad728c81235cc01f5b12303f9`
- Scientific execution: CLOSED
- New model execution: NONE
- New statistical tests: NONE
- New p-values: NONE

This document converts already-frozen evidence into paper-facing Results prose.
It does not create or authorize a new scientific claim.

---

## 3.3 Cross-block transport only partially accounts for local functional change

The preceding results show that the dominant causal role can recur even as its
local geometric realization changes. We next asked whether this geometric
reorientation is functionally relevant rather than merely representational.
Mamba-1.4B provides a controlled setting for this test because the canonical
causal site and a prospectively fixed adjacent site can be compared under the
same rank-aligned measurement procedure.

We first established that the causal response is site-specific. On a fresh
\(N=300\) XG1 population, the canonical \((33,34,35)\) site produced a positive
P5-versus-P4 core contrast,

\[
D_{\mathrm{CAN}} = 9.10\times10^{-9},
\]

whereas the architecture-predeclared adjacent \(+1\) site \((34,35,36)\)
produced a negative mean contrast,

\[
D_{\mathrm{ADJ}} = -1.55\times10^{-9}.
\]

The paired canonical-minus-adjacent specificity contrast had mean

\[
S = 1.07\times10^{-8},
\]

with \(t(299)=18.46\), one-sided
\(p=1.34\times10^{-51}\). The canonical effect was positive for 0.777 of
pairs, while the adjacent response was positive for only 0.0067. This
prospective comparison establishes specificity relative to one fixed adjacent
site; it does not identify a global optimum over layers or intervention sites.

We then measured how the canonical P5 subspace transforms across the
intervening block. On 600 response-free XG2/XG4 rows, the transported
two-dimensional subspace remained rank two but was strongly reoriented relative
to the local adjacent P5 plane. The pooled principal angles were approximately
\(86.45^\circ\) and \(89.22^\circ\), and mean projector overlap was
\(0.00231\). These quantities are descriptive and were measured without access
to the subsequent XG1 response endpoint. They show that the canonical causal
plane is not propagated to the adjacent site as an approximately identical
Euclidean subspace.

To test whether that reorientation matters functionally, we transported the
canonical P5 geometry to the adjacent intervention site and evaluated the
result on the same frozen \(N=300\) response population. The transported
response remained negative on average,

\[
D_{\mathrm{TRANSPORT}} = -5.51\times10^{-10},
\]

but was shifted upward relative to the native adjacent response. The
prospectively defined paired contrast

\[
G
=
D_{\mathrm{TRANSPORT}}-D_{\mathrm{ADJ}}
\]

had mean

\[
1.00\times10^{-9},
\]

with \(t(299)=18.58\), one-sided
\(p=4.38\times10^{-52}\), and was positive for 0.867 of pairs.

Thus transporting the canonical geometry causally moves the adjacent response
toward the canonical direction. However, it does not recover the canonical
effect: the transported mean remains negative, whereas the canonical mean is
positive and an order of magnitude larger. The mandatory positive-restoration
gate therefore fails.

This result supports a bounded interpretation. Cross-block geometric
reorientation contributes causally to the local functional difference between
the canonical and adjacent sites, but geometric transport alone does not fully
account for that difference. We therefore do not interpret ratios of the mean
effects as a mediated or "explained" percentage, and we leave the remaining
functional discrepancy as an unresolved component rather than attributing it
post hoc to an additional mechanism.

---

## 3.4 Downstream readout reverses in a matched 370M-versus-1.4B comparison

The first implication tested by Sections 3.2–3.3 concerned geometric
invariance. We next tested a distinct implication: if a causal role recurs,
does its downstream task alignment remain oriented in the same direction?

We measured the local directional readout of each checkpoint's independently
frozen selected-versus-control causal displacement into the downstream
correct-class task margin. For a native hidden state \(h\), local task gradient
\(g\), selected component \(C_{\mathrm{sel}}(h)\), and response-blind control
component \(C_{\mathrm{ctrl}}(h)\), the stored directional readout is

\[
\Delta L_{\mathrm{owned}}
=
g^\top
\left(
C_{\mathrm{sel}}(h)-C_{\mathrm{ctrl}}(h)
\right)
\]

under the frozen edge-specific gradient-ownership graph.

All three readout studies used the same `G3-GROUP-D-HALF` D-edge ownership
factor. The historical stored values therefore satisfy

\[
\Delta L_{\mathrm{owned}}
=
\frac{1}{2}\Delta L_{\mathrm{forward}},
\]

so that

\[
\Delta L_{\mathrm{forward}}
=
2\Delta L_{\mathrm{owned}}.
\]

This correction changes the numerical derivative scale but not sign, ranking,
correlation, or any of the frozen t statistics and p-values.

The primary cross-checkpoint test was prospectively fixed before Study-B
execution and compared Mamba-370M and Mamba-1.4B on the same 300 source pairs.
The scale-local selected/control geometry was P3/P5 at 370M and P5/P4 at 1.4B;
plane ranks were not reselected using the readout outcome.

The mean ownership-weighted directional readout was positive at Mamba-370M,

\[
\overline{\Delta L}_{370\mathrm{M},\mathrm{owned}}
=
+5.09\times10^{-4},
\]

but negative at Mamba-1.4B,

\[
\overline{\Delta L}_{1.4\mathrm{B},\mathrm{owned}}
=
-1.20\times10^{-3}.
\]

The prospectively defined paired endpoint

\[
R_q
=
\Delta L_{370\mathrm{M},q}
-
\Delta L_{1.4\mathrm{B},q}
\]

had mean \(1.70\times10^{-3}\) and paired SD
\(2.74\times10^{-3}\). The one-sided paired test gave

\[
t(299)=10.77,
\qquad
p=2.22\times10^{-23}.
\]

Both mandatory sign gates also passed:
the 370M mean was positive and the 1.4B mean was negative. The frozen
prospective criterion therefore supports opposite mean downstream readout
alignment between the two checkpoints.

Under the deterministic ownership correction, the corresponding numerical
forward directional derivatives are

\[
\overline{\Delta L}_{370\mathrm{M},\mathrm{forward}}
=
+1.02\times10^{-3}
\]

and

\[
\overline{\Delta L}_{1.4\mathrm{B},\mathrm{forward}}
=
-2.39\times10^{-3},
\]

with paired mean difference \(3.41\times10^{-3}\). Because this transformation
is a common positive factor of two, the paired t statistic, p-value, signs, and
rank-based conclusions are unchanged.

Mamba-130M provides a separately frozen contextual checkpoint. Its mean
ownership-weighted readout is positive
(\(+1.12\times10^{-3}\); forward-equivalent
\(+2.24\times10^{-3}\)), with the separately pre-specified positive-mean test
supported. We do not combine these three checkpoint means into a scaling
regression, monotonicity test, or estimate of a parameter-count threshold:
130M was evaluated on a different frozen population, whereas the
370M-versus-1.4B result is the matched cross-checkpoint inferential comparison.

The statistical unit of the primary reversal test is therefore the matched item
population for two fixed pretrained checkpoints. The result does not quantify
variation across independently trained model seeds and does not establish a
universal model-size phase transition.

Within that scope, the result directly answers the paper's second central
question. Recurrence of a scale-local internal causal role does not guarantee
preservation of downstream functional orientation. Across the matched
Mamba-370M and Mamba-1.4B checkpoints, the internally recurrent role couples to
the downstream task margin with opposite mean local alignment.

---

## Evidence mapping for manuscript verification

Section 3.3 is grounded in:

- `reports/reason_router_gen4_mamba14b_adjacent_site_specificity_analysis_runs/g4k-mamba14b-adjacent-site-specificity-analysis-xg1-5101-5400-5d651f6/adjacent_site_specificity_analysis.md`
- `reports/reason_router_gen4_mamba14b_p5_cross_block_population_transport_runs/g4k-mamba14b-p5-crossblock-population-859831c-2t4/transport_summary.json`
- `reports/reason_router_gen4_mamba14b_transported_p5_adjacent_response_static_analysis_v1/primary_analysis.json`

Section 3.4 is grounded in:

- `reports/reason_router_gen4_study_b_readout_alignment_prospective_design.md`
- `reports/reason_router_gen4_mamba370m14b_readout_alignment_analysis_v1/readout_alignment_analysis.json`
- `reports/reason_router_gen4_three_scale_readout_owned_forward_correction.md`
- `reports/reason_router_gen4_three_scale_readout_alignment_synthesis.md`

The evidence-mapping block is an internal drafting aid and should not appear in
the submitted manuscript.

`RESULTS_3_3_3_4_DRAFT_READY_FOR_REVIEW = YES`
