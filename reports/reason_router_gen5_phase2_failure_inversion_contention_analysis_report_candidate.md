# ContraMamba Gen5 Phase 2
## Failure-Inversion and Ownership-Contention Analysis Report Candidate

### Status

`PHASE2_TERMINAL_NEGATIVE_MECHANISM_INTERPRETED`

Frozen Phase 2 confirmatory conclusion:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED`

This report does not rescue or reinterpret that confirmatory decision.

---

## 1. Purpose

The completed Phase 2 assay established that protecting the prospectively
authenticated R22 native causal-role realization from learned WRITE22
correction did not produce a confirmatory advantage over the matched C22
control.

The present analysis asks a different descriptive mechanistic question:

`WHY_WAS_THE_OWNERSHIP_ADVANTAGE_NEAR_ZERO?`

No new training, model forward, p-value, cohort selection, or confirmatory
decision was performed.

---

## 2. Phase 1B causal evidence remains positive

The Phase 2 terminal negative does not invalidate the prior R22 causal
evidence.

Frozen prior results include:

- matched R22 local necessity over C22 control: supported;
- mean D_NEC22:
  `5.687471435811073e-08`
- one-sided p:
  `5.199019171379918e-33`

and:

- matched R22 restoration over C22 replacement: supported;
- mean D_SUF22:
  `1.3295631928032442e-07`
- mean Q_RR:
  `2.3810608805673197e-07`
- mean S_R:
  `1.6531488477786262e-07`
- one-sided p:
  `1.8381607091158128e-45`

Therefore:

`R22_CAUSAL_ROLE_NOT_ESTABLISHED = FALSE`

The Phase 2 failure concerns ownership privilege, not R22 causal validity.

---

## 3. Phase 2 confirmatory negative

Validated fresh ownership assay:

- n = `300`
- CUDA scientific forwards = `48000`
- mean Ibar_M1 =
  `1.3327600706416034e-07`
- mean Q_RR_M1 =
  `2.3840579959434754e-07`
- mean S_R_M1 =
  `1.6427901626504933e-07`
- mean D_OWN =
  `9.5939370052090783e-12`
- one-sided confirmatory p =
  `0.36700787067950813`

The M1-minus-C1 ownership advantage was only:

`7.1985477480507535e-05`

of the absolute mean M1 I_A signal, or approximately:

`0.00719854774805 percent`

The Phase 2 decision therefore remains terminal negative.

---

## 4. Unrestricted correction showed minimal natural R22 contention

For the unrestricted G5-C0 correction, the final learned correction-map
energy fraction lying in R22 was:

- seed 5201: `5.894709817e-04`
- seed 5202: `1.301594137e-04`
- seed 5203: `1.773147266e-04`

The corresponding B output-subspace energy fractions were:

- seed 5201: `5.633822939121e-04`
- seed 5202: `1.363829601981e-04`
- seed 5203: `1.723988837949e-04`

The largest principal cosine between unrestricted C0 B output space and
R22 was only:

- seed 5201: `0.025276566896348027`
- seed 5202: `0.019355751353810523`
- seed 5203: `0.01824444631884581`

Thus the unrestricted learned correction naturally occupied a subspace
almost orthogonal to R22.

This is evidence that the task optimization pressure did not naturally
compete strongly for the authenticated R22 realization.

---

## 5. Ownership constraints changed learned geometry

The terminal negative cannot be explained by all three arms learning
identical correction maps.

Same-seed effective-map comparisons gave M1-versus-C1 cosines:

- seed 5201: `0.8934108121`
- seed 5202: `0.9629870419`
- seed 5203: `0.9244736535`

with relative map differences:

- seed 5201: `0.4620533397`
- seed 5202: `0.2722238129`
- seed 5203: `0.3889817069`

Therefore:

`CORRECTION_GEOMETRY_IDENTICAL_ACROSS_ARMS = FALSE`

The constraints materially altered the learned effective correction
geometry.

---

## 6. Task optimization was nearly indifferent to the ownership constraint

All matched arms began each seed at exactly the same initial loss.

Final losses:

seed 5201:
- C0: `0.603882133961`
- C1: `0.603876829147`
- M1: `0.603773891926`

seed 5202:
- C0: `0.603896677494`
- C1: `0.603978574276`
- M1: `0.603956639767`

seed 5203:
- C0: `0.605430901051`
- C1: `0.605719089508`
- M1: `0.605474114418`

Cross-seed mean final loss:

- C0: `0.604403237502`
- C1: `0.604524830977`
- M1: `0.604401548704`

The final 3-way CE therefore admitted similarly successful solutions
under all three ownership geometries.

---

## 7. Fresh causal-role endpoint was nearly arm invariant

Fresh-assay mean I_A:

- C0: `1.3323498654355599e-07`
- C1: `1.3326641312715514e-07`
- M1: `1.3327600706416034e-07`

Itemwise I_A correlations:

- M1 vs C1:
  `0.999993741710`
- C0 vs C1:
  `0.999994484438`
- C0 vs M1:
  `0.999992981428`

M1-minus-C1 descriptive contrast:

- mean:
  `9.5939370052090783e-12`
- standard deviation:
  `4.8858889598293707e-10`
- positive fraction:
  `0.500000000`
- median:
  `1.0011031089316927e-12`

This pattern does not resemble a large heterogeneous effect hidden by
simple mean cancellation.

Instead, causal-role integrity was nearly invariant across substantially
different learned correction geometries.

---

## 8. Mechanistic interpretation

The evidence supports the bounded interpretation:

`OWNERSHIP_CONSTRAINT_WAS_EFFECTIVELY_NON_BINDING_WITH_RESPECT_TO_THE_AUTHENTICATED_CAUSAL_ROLE`

More explicitly:

1. R22 remained causally necessary and restorable.
2. The unrestricted correction naturally placed extremely little energy
   in R22.
3. R22/C22 protection changed the learned correction geometry.
4. Task loss remained nearly unchanged across those geometries.
5. The fresh R22 causal-role endpoint also remained nearly unchanged.

Therefore the failed ownership advantage is best explained by lack of
optimization-time contention for R22, rather than by absence of the R22
causal role itself.

This is descriptive mechanistic interpretation, not a new confirmatory
test.

---

## 9. Dataset/objective interpretation

The Phase 2 training problem used the broad controlled-v5 intervention
distribution and final 3-way cross-entropy only.

That corpus spans multiple intervention families and failure modes.
The objective contains no explicit requirement that a learned correction
must compete for the prospectively authenticated R22 operational causal
role.

The semantic dataset label `role_swap` must not be identified with R22.
R22 is an operationally validated native causal role, not a named
semantic role coordinate.

Accordingly, the relevant limitation is:

`TRAINING_PRESSURE_DID_NOT_ESTABLISH_R22_CONTENTION`

rather than:

`DATASET_INVALID`

or:

`R22_NOT_CAUSAL`

---

## 10. New scientific question

The next scientific question is not a rescue of the failed Phase 2
claim.

Proposed new hypothesis:

`H_G5_2_CONTENTION`

A causal-role-grounded ownership constraint becomes mechanistically
relevant only when the learning problem creates genuine optimization
pressure that would otherwise modify or occupy the authenticated
causal-role realization.

Primary question:

`IS_CONTENTION_FOR_THE_CAUSAL_ROLE_A_NECESSARY_CONDITION_FOR_OWNERSHIP_TO_HAVE_AN_EFFECT?`

This must be treated as a new stage.

The completed Phase 2 terminal negative remains immutable.

---

## 11. Boundaries for the next stage

Not authorized by this report:

- new training;
- new evaluation;
- GPU execution;
- owner-rank sweep;
- layer search;
- plane search;
- cohort rescue;
- post-hoc subgroup selection;
- reuse of Phase 2 p-value as a selection criterion.

The next stage should first define a static contention-identification
design that distinguishes:

1. no contention;
2. genuine causal-role contention;
3. matched control contention.

A later execution may proceed only after that design is prospectively
frozen.

---

## 12. Final disposition

R22 causal necessity: `SUPPORTED`

R22 causal restoration: `SUPPORTED`

Phase 2 ownership advantage: `NOT_ESTABLISHED`

Unrestricted R22 correction contention: `VERY_LOW`

Cross-arm correction geometry identity: `FALSE`

Cross-arm causal-role endpoint separation: `NEAR_ZERO`

Best current mechanistic interpretation:

`CAUSAL_IMPORTANCE_DOES_NOT_IMPLY_OPTIMIZATION_TIME_CONTENTION_OR_OWNERSHIP_PRIVILEGE`

Next stage:

`GEN5_PHASE3_CAUSAL_ROLE_CONTENTION_IDENTIFICATION_STATIC_DESIGN`
