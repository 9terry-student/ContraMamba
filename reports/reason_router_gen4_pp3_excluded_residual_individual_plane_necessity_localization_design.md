# Gen4 PP3-Excluded Residual Individual-Plane Necessity Localization — Prospective Design

## Status

`PROSPECTIVE_DESIGN_ONLY_NO_EXECUTION_AUTHORITY`

This document freezes the next scientific question after completion of:

- PP3 transport / specificity / local necessity / restoration-sufficiency;
- prospective XG2-like PP3-excluded residual-template transport;
- aggregate local necessity of the complete PP3-excluded residual subspace.

It authorizes no model execution, no statistical inference, no Kaggle run,
no training, and no downstream evaluation.

## Prior validated synthesis

Current synthesis commit:

`7cfb70950d8506b24d84c57c68d26f7d0804e642`

Current combined mechanism picture:

`shared PP3 causal core + structured XG2-like secondary residual geometry with aggregate local necessity on XG1`

The aggregate residual necessity result applies to the complete frozen
PP3-excluded residual subspace:

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`

It does not establish individual necessity of P1, P2, P4, or P5.

## Scientific question

Without selecting any secondary plane based on observed outcomes, which, if any,
of the four already frozen PP3-excluded principal planes makes an individually
detectable local necessity contribution to the same frozen susceptibility
endpoint relative to an equal-norm orthogonal matched perturbation in that
same plane?

The four planes are all primary members of one pre-specified family:

`K = {P1, P2, P4, P5}`

No plane is selected, screened, promoted, dropped, or weighted using outcomes.

## Fresh prospective population

Use the next non-overlapping deterministic XG1 range:

`xg1_fact_1801..xg1_fact_2100`

Pair count:

`N = 300`

Expected six-cell rows:

`1800`

Static preparation must establish before any scientific model forward:

- deterministic regeneration identity for all previously frozen XG1 cohorts;
- no pair-ID overlap with `xg1_fact_001..xg1_fact_1800`;
- no exact claim overlap with prior XG1 cohorts;
- no exact evidence overlap with prior XG1 cohorts;
- no exact `(claim, evidence)` overlap with prior XG1 cohorts;
- tokenizer/anchor eligibility for all 300 pairs;
- exact frozen principal-plane vector identities;
- exact XG2/XG4 basis identities.

Failure of any fresh-population or provenance gate blocks execution.

## Frozen geometry

Residual plane order:

`[P1, P2, P4, P5]`

For each plane `Pk`, let its frozen orthonormal basis vectors be:

`p_k+`, `p_k-`

For a branch-local native hidden state `h`, define native coordinates:

`a_k = <h, p_k+>`

`b_k = <h, p_k->`

Native component in that plane:

`c_k = a_k p_k+ + b_k p_k-`

Quarter-turn vector:

`r_k = -b_k p_k+ + a_k p_k-`

Because `p_k+` and `p_k-` are orthonormal:

`||c_k||_2 = ||r_k||_2`

and

`<c_k, r_k> = 0`

These identities must be validated statically and audited during execution.

## Conditions

Exactly nine conditions are frozen:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_control`
4. `p2_neutralized`
5. `p2_quarter_turn_control`
6. `p4_neutralized`
7. `p4_quarter_turn_control`
8. `p5_neutralized`
9. `p5_quarter_turn_control`

The native condition is shared across all four plane tests and is executed once
per item.

For each plane `Pk`:

Treatment correction:

`delta_N,k = -c_k`

so the native component in `Pk` is neutralized.

Matched-control correction:

`delta_C,k = -r_k`

so the correction has the same L2 norm as treatment, lies in the same frozen
plane, and is orthogonal to the native component.

Both corrections must preserve PP3 coordinates up to the frozen numerical
tolerance.

Because the principal-plane basis is orthonormal, the correction for `Pk` does
not directly alter the coordinates of the other frozen principal planes except
for permitted runtime cast residual.

## Frozen susceptibility endpoint

Use exactly the established broad endpoint:

`E_XG2 = (1/5) * sum_j J(v_XG2,j; c)^2`

`E_XG4 = (1/5) * sum_j J(v_XG4,j; c)^2`

`Q = E_XG2 - E_XG4`

with frozen probe epsilon:

`epsilon = 0.025`

For the shared native condition:

`Q0 = Q(native)`

For each plane `Pk`:

`QN,k = Q(Pk neutralized)`

`QC,k = Q(Pk quarter-turn control)`

Native attenuation:

`A_N,k = Q0 - QN,k`

Matched-control attenuation:

`A_C,k = Q0 - QC,k`

Primary per-plane causal contrast:

`D_k = A_N,k - A_C,k = QC,k - QN,k`

The canonical stored form is:

`D_k = QC,k - QN,k`

Do not require bitwise equality between mathematically equivalent subtraction
paths.

## Confirmatory family

Exactly four primary confirmatory hypotheses are frozen, one per plane:

For each `k in {1,2,4,5}`:

`H0,k: mean(D_k) <= 0`

`H1,k: mean(D_k) > 0`

Each raw test is:

- one-sample Student t-test;
- one-sided greater;
- `N = 300`;
- `df = 299`.

Exactly four raw confirmatory p-values are permitted.

No additional inferential p-values are permitted.

## Multiplicity control

The four raw p-values form one family.

Control familywise error at:

`FWER alpha = 0.05`

using the pre-specified Holm step-down procedure across exactly:

`{P1, P2, P4, P5}`

No plane may be removed from the family after outcomes are observed.

No uncorrected per-plane declaration is allowed.

No alternate correction method may be substituted after outcomes are observed.

## Positive gates

Shared gate:

`mean(Q0) > 0`

For an individual plane `Pk` to be declared supported, all of the following
must hold:

1. shared gate `mean(Q0) > 0`;
2. `mean(A_N,k) > 0`;
3. `mean(D_k) > 0`;
4. the one-sided test for `D_k` is rejected by the frozen Holm procedure.

Supported planes, if any, are reported as an unordered set.

Do not rank planes by p-value, t statistic, effect size, or apparent importance.

Family-level positive label:

`PP3_EXCLUDED_INDIVIDUAL_RESIDUAL_PLANE_NECESSITY_LOCALIZATION_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

is used if at least one plane satisfies all frozen plane-level gates.

If no plane satisfies all gates, use:

`PP3_EXCLUDED_INDIVIDUAL_RESIDUAL_PLANE_NECESSITY_LOCALIZATION_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

A negative family-level result does not invalidate the already established
aggregate residual necessity result.

## Forward budget

Each condition uses the established ten susceptibility directions:

- five frozen XG2 directions;
- five frozen XG4 directions.

Each direction uses exactly four scientific forwards.

Therefore:

`40 forwards / condition / pair`

Nine conditions give:

`360 forwards / pair`

For `N = 300`:

`108000 scientific model forwards`

Baseline model forwards:

`0`

Recommended fixed two-GPU split:

- GPU 0: `xg1_fact_1801..xg1_fact_1950`, 150 pairs, `54000` forwards;
- GPU 1: `xg1_fact_1951..xg1_fact_2100`, 150 pairs, `54000` forwards.

No DDP.

No NCCL.

Each worker loads the model/checkpoint once.

## Raw-run boundary

The raw runner must record per-item:

- `Q0`;
- for each P1/P2/P4/P5:
  - `QN,k`;
  - `QC,k`;
  - `A_N,k`;
  - `A_C,k`;
  - `D_k`;
- branch-local native coefficients `(a_k, b_k)`;
- treatment/control correction norms;
- treatment/control correction dot product;
- PP3 coefficient drift;
- post-treatment target-plane residual projection;
- scientific forward counts.

The raw runner must not compute:

- t statistics;
- p-values;
- Holm decisions;
- supported-plane set;
- family-level scientific label.

Raw artifact fields must state:

`primary_inference_executed = false`

`multiplicity_correction_executed = false`

`scientific_conclusion = null`

## Interpretation if positive

A supported plane `Pk` may be described only as:

an individually detectable local necessity contributor to the frozen
layer-17 / target-token susceptibility endpoint relative to its own
pre-specified equal-norm orthogonal within-plane matched control.

This does not establish:

- sufficiency of that plane;
- exclusivity or dominance of that plane;
- additivity across planes;
- absence of interactions among residual planes;
- causal necessity of the XG2-like residual-template orientation;
- behavioral or downstream-task necessity;
- checkpoint-, layer-, token-, model-, or architecture-wide universality.

Multiple supported planes indicate distributed individual contributions under
the tested native-background interventions; they do not imply an additive
decomposition of the aggregate residual effect.

## Interpretation if negative

If no plane survives the frozen familywise procedure, conclude only that
individual plane-level necessity localization was not established under this
design.

Do not reinterpret that outcome as evidence that the aggregate residual
necessity result was false.

A negative result is compatible with:

- distributed sub-threshold contributions;
- interactions across planes;
- necessity that emerges only at aggregate subspace level.

No rescue analysis, alternative grouping, subgroup search, alternate tail,
effect-based plane selection, or additional p-value is authorized by a negative
result.

## Prohibited analyses

The following are outside this design:

- selecting one or two planes because they look strongest;
- ranking P1/P2/P4/P5;
- dropping a plane from Holm correction;
- using the XG2-like residual template as a hidden-state intervention vector;
- constructing a template-weighted hidden-state correction from q-space
  coordinates;
- subgroup mining;
- tail switching;
- alternative epsilon sweeps;
- alternate endpoint sweeps;
- rescue experiments;
- additional p-values;
- PP3 re-testing under minor variants;
- training;
- backward passes;
- task-head evaluation;
- logits-based downstream interpretation.

## Falsification logic

The design is intentionally capable of failing.

The aggregate residual mechanism is individually localized only to planes that
satisfy all frozen gates after Holm correction.

No plane is promoted merely because its raw mean or uncorrected p-value looks
large.

## Current authority

Design freeze: `YES`

Static preparation after design freeze: `YES`

Scientific model execution: `NO`

Kaggle scientific run: `NO`

Training/backward: `NO`

Primary inference: `NO`

Commit/push: manual only.
