# ContraMamba Gen4 Results Draft — Sections 3.5–3.6

## Status

- Status: STATIC MANUSCRIPT PROSE DRAFT
- Evidence HEAD: `0ac43dd`
- Scientific execution: CLOSED
- New model execution: NONE
- New statistical tests: NONE
- New p-values: NONE

This document converts already-frozen evidence into paper-facing Results prose.
It does not create or authorize a new scientific claim.

---

## 3.5 Behavioral consequences are checkpoint-dependent

The readout analysis in Section 3.4 establishes that the same scale-local causal
role can couple to the downstream task margin with opposite local alignment.
We next asked whether this difference is observable under direct intervention
rather than only through a local gradient readout.

On the prospectively shared XG1 population of 300 source pairs, Mamba-370M
showed a positive restored-versus-control behavioral contrast,

\[
D_{\mathrm{BEH}}
=
M_{\mathrm{dominant\ restored}}
-
M_{\mathrm{dominant\ control}}.
\]

The mean effect was

\[
\overline{D}_{\mathrm{BEH},370\mathrm{M}}
=
+1.12\times10^{-3},
\]

with SD \(3.33\times10^{-3}\),
\(t(299)=5.82\), Cohen's \(d_z=0.336\), and
Holm-adjusted \(p=1.50\times10^{-8}\).
The behavioral contrast was positive for 0.58 of matched pairs.
Thus the internally identified causal component reaches the downstream
correct-class decision margin at Mamba-370M.

The corresponding Mamba-1.4B mean had the opposite sign,

\[
\overline{D}_{\mathrm{BEH},1.4\mathrm{B}}
=
-2.01\times10^{-3},
\]

with SD \(5.11\times10^{-3}\),
\(t(299)=-6.82\), and Cohen's
\(d_z=-0.394\). Because the prospectively specified primary family tested for
a positive behavioral bridge, the 1.4B positive-bridge criterion did not pass.
We therefore use the bounded conclusion
**scale-specific behavioral bridge only** rather than treating the two
checkpoints as samples from a monotonic scaling trend.

This behavioral divergence is not restricted to the final output layer.
A frozen stagewise downstream analysis localized the first persistent
checkpoint opposition for the entitlement-sensitive C2 condition immediately
after the intervention block, at `post_block_35`: the 370M contrast was
positive while the 1.4B contrast was negative. The full pair-averaged
opposition became persistent only at `post_block_47`. At the final normalized
readout, the corresponding contrasts were

\[
D_{370\mathrm{M}}=+1.12\times10^{-3}
\]

and

\[
D_{1.4\mathrm{B}}=-2.01\times10^{-3}.
\]

The stagewise pattern is therefore consistent with two descriptive components:
an early entitlement-sensitive routing divergence and a late consolidation of
the aggregate downstream orientation. Final normalization further exposes the
difference, particularly through the 1.4B control contribution, but the
evidence does not identify final normalization—or any other single stage—as a
unique causal mediator.

A previously frozen Mamba-130M behavioral study provides an additional
contextual checkpoint: its restored-versus-control behavioral effect was also
positive on its own fresh population
(mean \(D_{\mathrm{BEH}}=8.33\times10^{-3}\),
one-sided \(p=2.45\times10^{-17}\)).
We do not combine the three studies into a model-size trend test because their
populations and prospective inferential families differ.

Together with the readout reversal in Section 3.4, these interventions show
that recurrence of an internal causal role does not by itself determine how
that role participates in the final task decision. Within the tested
checkpoints, internal causal recurrence and downstream behavioral alignment are
empirically separable properties.

---

## 3.6 External transfer and the boundary between explanation and control

The preceding analyses use synthetic XG1 inputs designed for controlled causal
measurement. We therefore tested whether the frozen causal intervention also
produces a measurable task-margin effect on natural-language fact-verification
inputs.

The external-transfer study used the official AVeriTeC development source,
restricted prospectively to 462 examples with compatible
`Refuted`, `Supported`, or `Not Enough Evidence` labels. Inputs contained the
claim and gold evidence only. The experiment did not perform retrieval and is
not an evaluation of end-to-end AVeriTeC benchmark performance.

At Mamba-130M, the frozen selected-versus-control intervention produced a
positive mean correct-class-margin effect,

\[
\overline{D}_{\mathrm{EXT}}
=
+2.31\times10^{-4},
\]

with Cohen's \(d_z=0.120\) and Holm-adjusted
\(p=0.0104\). Mamba-370M showed a smaller but likewise supported positive mean,

\[
\overline{D}_{\mathrm{EXT}}
=
+4.56\times10^{-5},
\]

with \(d_z=0.106\) and Holm-adjusted
\(p=0.0114\). These two tests formed the prospectively frozen external-transfer
family. Thus the internally identified causal component transfers to
natural-language gold-evidence inputs at both checkpoints under the frozen
margin-level endpoint.

The later prospective Mamba-1.4B extension produced a negative point estimate,

\[
\overline{D}_{\mathrm{EXT}}
=
-7.85\times10^{-5},
\]

with \(d_z=-0.067\), \(t(461)=-1.44\), and one-sided
\(p=0.0757\). Its negative-sign gate passed, but the pre-specified alpha gate
did not. Negative external transfer at 1.4B is therefore not established.
The three checkpoints should not be interpreted as a statistically established
external sign-reversal curve.

The AVeriTeC results establish margin-level causal transfer rather than
benchmark improvement. At 1.4B, for example, all three frozen intervention
conditions produced the same argmax prediction on all 462 examples. The
130M/370M external effects are also heterogeneous by source label in descriptive
analyses, and no post-hoc subgroup inference is used here.

Finally, we tested whether a causally validated direction automatically
provides a useful control direction. At Mamba-370M, a preregistered fixed-mirror
steering intervention was evaluated on 2,799 AVeriTeC examples. Of these,
331 were natively correct and 2,468 were natively incorrect. The intervention
produced

\[
C=0
\]

native-wrong-to-correct transitions and

\[
D=0
\]

native-correct-to-wrong transitions. Native and steered accuracy were therefore
identical (\(0.1183\)), with zero correction rate, zero damage rate, and zero
net accuracy change. Because there were no discordant prediction pairs, the
pre-specified exact utility test was not estimable and its success gates did
not pass.

This null does not show that Mamba is generally unsteerable, nor does it
invalidate the causal interpretation of the identified component. It instead
marks a narrower empirical boundary: a direction can carry reproducible causal
information, influence downstream margins, and transfer to natural-language
inputs without yielding useful prediction-level control under a fixed
intervention rule.

The appropriate synthesis is therefore that **mechanistic validity and control
utility are distinct empirical questions**. For applications such as model
editing or steering, recurrence of an apparently analogous causal role across
checkpoints is not sufficient reason to assume that either its downstream
alignment or a fixed intervention direction will transfer unchanged.

---

## Main Table 1 draft — external transfer and control boundary

| Evidence | Checkpoint | N | Primary outcome | Standardized effect / utility | Frozen conclusion |
|---|---:|---:|---:|---:|---|
| AVeriTeC gold-evidence transfer | 130M | 462 | \(D_{\mathrm{EXT}}=+2.31\times10^{-4}\) | \(d_z=+0.120\), Holm \(p=0.0104\) | Positive transfer supported |
| AVeriTeC gold-evidence transfer | 370M | 462 | \(D_{\mathrm{EXT}}=+4.56\times10^{-5}\) | \(d_z=+0.106\), Holm \(p=0.0114\) | Positive transfer supported |
| AVeriTeC gold-evidence transfer | 1.4B | 462 | \(D_{\mathrm{EXT}}=-7.85\times10^{-5}\) | \(d_z=-0.067\), one-sided \(p=0.0757\) | Negative transfer not established |
| Fixed-mirror steering | 370M | 2799 | corrections \(=0\), damages \(=0\) | net accuracy change \(=0\) | Useful fixed-mirror steering not established |

This table reports causal margin transfer and a fixed steering utility test.
It does not report retrieval performance, benchmark superiority, or a general
steerability claim.

---

## Evidence mapping for manuscript verification

Section 3.5 is grounded in:

- `reports/reason_router_gen4_mamba370m14b_behavioral_bridge_runs/g4k-mamba370m14b-behavioral-bridge-xg1-4801-5100-2gpu-2b41f28-retry1/behavioral_bridge_analysis.json`
- `reports/reason_router_gen4_stagewise_behavioral_coupling_localization_result.md`
- `reports/reason_router_gen4_averitec_external_transfer_scale_family_correction.md`

Section 3.6 is grounded in:

- `reports/reason_router_gen4_averitec_external_transfer_static_post_result_analysis.md`
- `reports/reason_router_gen4_averitec_three_scale_external_transfer_synthesis.md`
- `reports/reason_router_gen4_averitec_370m_fixed_mirror_steering_analysis_328ae53/steering_analysis.md`

The evidence-mapping block is an internal drafting aid and should not appear in
the submitted manuscript.

`RESULTS_3_5_3_6_DRAFT_READY_FOR_REVIEW = YES`
