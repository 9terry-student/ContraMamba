# ContraMamba Gen4 — Integrated Results Manuscript v1

## Status

- Status: STATIC INTEGRATED MANUSCRIPT PROSE
- Evidence HEAD: a55c7e935f2539c49a43edf538e00a7a83951226
- Scientific execution: CLOSED
- New model execution: NONE
- New statistical tests: NONE
- New p-values: NONE

This file integrates the already-frozen Results drafts without expanding the scientific claim set.

## 3.1 Identifying a causal recurrent-state role

We first asked whether the local recurrent-state geometry contains a
reproducible causal component rather than merely a direction that covaries with
the observed susceptibility contrast. In Mamba-130M, a static decomposition of
the frozen XG2/XG4 response geometry localized the dominant shared positive
contrast component to principal pair 3 (PP3). Importantly, substantial response
remained in the other principal planes, so this localization did not imply a
one-dimensional mechanism. We therefore treated PP3 as a candidate distributed
causal component and evaluated it prospectively on fresh populations.

The first prospective test examined whether the PP3 geometry transported beyond
the generator families used to localize it. On an independent XG1 population
of 300 examples, the frozen PP3 contrast remained positive
(mean \(C_{\mathrm{PP3}}=7.12\times10^{-8}\),
SD \(=6.50\times10^{-8}\);
\(t(299)=18.97\), one-sided
\(p=1.52\times10^{-53}\)).
Thus the localized PP3 susceptibility geometry transported prospectively to a
generator population that was not used to select the plane.

Transport alone does not establish specificity: a highly separated principal
plane could produce a large response without carrying a distinctive causal
role. We therefore prospectively selected PP5 as a response-blind geometric
control because it had the largest projector-separation magnitude among the
frozen principal planes. On a second, non-overlapping XG1 holdout
(\(N=300\)), PP3 again produced a larger susceptibility contrast than this
max-separation control. Mean contrast was
\(7.12\times10^{-8}\) for PP3 and
\(4.82\times10^{-8}\) for PP5, yielding
mean \(D_{\mathrm{SPEC}}=2.30\times10^{-8}\)
(\(t(299)=7.66\), one-sided
\(p=1.34\times10^{-13}\)).
The effect is therefore not adequately explained by principal-plane separation
alone. This comparison does not establish dominance over every arbitrary or
random direction.

We next tested whether PP3 contributes causally to the observed effect. On a
third fresh XG1 population, selectively removing PP3 attenuated the frozen
susceptibility contrast more strongly than an intervention-magnitude-matched
PP5 coefficient-transfer control. The prospectively defined necessity contrast

\[
D_{\mathrm{NEC}}
=
(Q_0-Q_3)-(Q_0-Q_5)
=
Q_5-Q_3
\]

had mean \(4.74\times10^{-8}\)
(SD \(=3.42\times10^{-8}\)),
with \(t(299)=23.98\) and one-sided
\(p=6.22\times10^{-72}\).
This supports PP3 as a local necessary contributor relative to the matched
control intervention; it does not establish global behavioral necessity.

A complementary restoration experiment tested whether reinstating the native
PP3 component after its removal preferentially recovered the susceptibility
contrast. On another disjoint \(N=300\) XG1 population, restoring the exact
native PP3 component recovered more contrast than inserting an
equal-coefficient, equal-addition-norm PP5 replacement. The prospective
restoration contrast had mean
\(D_{\mathrm{SUF}}=4.08\times10^{-8}\)
(SD \(=3.51\times10^{-8}\)),
with \(t(299)=20.11\) and one-sided
\(p=9.01\times10^{-58}\).
We therefore describe PP3 as locally restoration-sufficient relative to the
matched PP5 replacement. This is not sufficiency in an otherwise empty state,
nor does it establish sufficiency for downstream task behavior.

Finally, we tested whether the identification of PP3 was fragile to the finite
perturbation scale used in the local susceptibility measurement. The spectral
ordering was unchanged at
\(\epsilon=0.025\), \(0.0125\), and \(0.00625\):
PP3 remained the unique dominant plane and its mean contribution remained
larger than PP5 at both smaller perturbations. The normalized five-plane
profiles were nearly identical, with pairwise cosine similarities above
\(0.99994\). These descriptive checks support finite-\(\epsilon\) robustness
of the PP3 localization, but they do not establish an
\(\epsilon\rightarrow0\) limit or a new causal ranking.

Together, prospective transport, geometric-control specificity, matched-control
necessity, restoration sufficiency, and small-\(\epsilon\) robustness support
PP3 as a reproducible local causal contributor to the Mamba-130M recurrent-state
susceptibility mechanism. At the same time, the persistent contribution of
secondary planes requires a distributed-mechanism interpretation rather than a
claim that PP3 is the sole causal component.

---

## 3.2 Causal roles recur while residual geometry reorganizes

We next asked what, if anything, remains invariant when the same causal
reconstruction procedure is applied at larger Mamba checkpoints. We deliberately
did not require a fixed principal-plane rank across models. Principal planes are
constructed independently within each checkpoint, so equal rank labels do not
imply semantic identity. The prospective invariant was instead functional: does
each independently reconstructed model contain a scale-local component that
plays the same dominant causal role relative to a response-blind matched
control?

At Mamba-370M, the frozen discovery procedure selected P3 against the
response-blind P5 control. On the fresh confirmation cohort, the dominant
contrast was positive with
mean \(D_{\mathrm{DOM}}=3.97\times10^{-8}\)
(SD \(=2.64\times10^{-8}\)),
\(t(299)=26.06\), raw one-sided
\(p=3.19\times10^{-79}\), and Holm-adjusted
\(p=6.39\times10^{-79}\). The effect was positive for 94% of matched
examples. This establishes recurrence of the dominant component at 370M.
It does not retroactively rescue the earlier 370M joint criterion, whose
separate positive-residual endpoint failed and remains not established.

The same question was then tested prospectively at Mamba-1.4B using disjoint
discovery and confirmation populations. Independent reconstruction selected P5
as the scale-local dominant candidate and P4 as the geometry-only response-blind
control. On the fresh \(N=300\) confirmation population,

\[
D_{\mathrm{CORE}}
=
Q_{\mathrm{restored}}(\mathrm{P5})
-
Q_{\mathrm{control}}(\mathrm{P4})
\]

had mean \(9.28\times10^{-9}\)
(SD \(=9.89\times10^{-9}\)),
with \(t(299)=16.25\), one-sided
\(p=2.62\times10^{-43}\), Cohen's
\(d_z=0.94\), and a positive-pair fraction of 0.783.
The pre-specified 1.4B core criterion therefore passed.

The dominant rank itself is consequently not the invariant:
the selected component is P3 at 370M but P5 at 1.4B. What recurs is the
scale-local dominant causal role under the same reconstruction and
matched-control procedure. We refer to this bounded observation as
**core-stable**. It does not imply a universal dominant rank, semantic identity
of same-numbered planes, or architecture-independent universality.

The surrounding residual geometry shows a different pattern. At 130M, the
earlier frozen characterization placed approximately 0.78 of the residual
coefficient-energy mass in local P1, with positive raw
native-neutralization attenuation across the four tested residual planes.
At 370M, residual organization shifted strongly toward local P5:
P5 carried approximately 0.695 of residual coefficient energy, and the signed
residual profile contained a large negative P5 contribution together with a
positive P4 contribution. At 1.4B, the organization changed again. Residual
coefficient mass was concentrated primarily in P2
(\(\approx0.574\)) and secondarily in P3
(\(\approx0.318\)); P2 became the dominant negative residual contributor,
while P4 was also negative and the smaller P3 term was positive.

The residual therefore remains structured rather than disappearing as the
dominant role recurs. At both 370M and 1.4B, the aggregate residual was closely
tracked by the signed sum of its constituent plane effects
(\(r=0.963\) and \(r=0.995\), respectively), arguing against an interpretation
in which the residual is dominated by a large aggregate-only interaction.
What changes is which local planes carry the state mass and signed causal
effect, together with the generator-family coupling that produces those
effects. In particular, the strongest negative residual mode at 1.4B is
generated by a different XG2/XG4 contribution pattern from the major negative
370M modes.

These residual comparisons are descriptive; no cross-checkpoint significance
test is assigned to the residual reorganization itself. Their role is to
characterize how the realization surrounding the prospectively confirmed core
changes across the tested checkpoints. The resulting pattern is therefore
best summarized as **core-stable / residual-plastic**: a scale-local dominant
causal role recurs through Mamba-1.4B, while its surrounding residual
realization reorganizes in signed effects, coefficient-mass distribution,
dominant residual rank, and generator-family coupling. This formulation does
not imply a monotonic scaling law or fixed geometric coordinates across model
checkpoints.

---

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

---

## Integrated Results scope

The Results support two headline conclusions:

1. recurrence of a scale-local causal role does not imply preservation of its geometric realization;
2. recurrence of that causal role does not imply preservation of downstream functional alignment.

Sections 3.5–3.6 provide behavioral, external-transfer, and control-utility validation or boundary evidence rather than additional headline novelty claims.

The integrated Results do not establish a universal scaling law, a model-size phase transition, model-seed generalization, architecture-independent universality, complete mediation by geometric transport, or general steering failure.

INTEGRATED_RESULTS_V1_READY = YES
