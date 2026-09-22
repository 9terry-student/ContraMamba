# ContraMamba Gen4 Results Draft — Sections 3.1–3.2

## Status

- Status: STATIC MANUSCRIPT PROSE DRAFT
- Evidence HEAD: `37dc29180015c84106b4c3c376e0770813d1c0ae`
- Scientific execution: CLOSED
- New statistical tests: NONE
- New p-values: NONE
- New model execution: NONE

This draft converts already-frozen evidence into paper-facing Results prose.
It does not create or authorize a new scientific claim.

---

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

## Evidence mapping for manuscript verification

Section 3.1 is grounded in:

- `reports/reason_router_gen4_pp3_specificity_mechanism_synthesis.md`
- `reports/reason_router_gen4_pp3_necessity_validated_result_dc85079.md`
- `reports/reason_router_gen4_pp3_restoration_sufficiency_validated_result_53b6cce.md`
- `reports/reason_router_gen4_small_epsilon_robustness_analysis_199fd28/small_epsilon_robustness_analysis.md`

Section 3.2 is grounded in:

- `reports/reason_router_gen4_core_stable_residual_plastic_cross_scale_synthesis.md`

The evidence-mapping block is an internal drafting aid and should not appear in
the submitted manuscript.

`RESULTS_3_1_3_2_DRAFT_READY_FOR_REVIEW = YES`
