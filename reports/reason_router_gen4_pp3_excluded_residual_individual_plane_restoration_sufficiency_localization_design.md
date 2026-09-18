# Gen4 PP3-Excluded Residual Individual-Plane Restoration Sufficiency Localization — Prospective Design

## Status

`DESIGN_ONLY_STATIC_PREPARATION_ALLOWED_NO_IMPLEMENTATION_NO_SCIENTIFIC_EXECUTION`

This document freezes the next scientific question after completion of:

- PP3 transport, specificity, local necessity, and restoration sufficiency;
- PP3-excluded XG2-like residual-template transport;
- aggregate necessity of the complete PP3-excluded residual subspace;
- individual local necessity localization across P1, P2, P4, and P5.

It authorizes deterministic static preparation only.

It does not authorize model execution, checkpoint loading, GPU inference,
Kaggle scientific execution, statistical inference, training, backward,
task-head evaluation, or post-hoc endpoint modification.

## Prior validated synthesis

Current synthesis commit:

`335e3317b87d55caccf3ee7e01390c20c292735e`

Current bounded mechanism picture:

`shared PP3 causal core + structured XG2-like PP3-excluded residual geometry + aggregate residual local necessity + individually localized local necessity across P1/P2/P4/P5`

The unresolved question is individual restoration sufficiency for the four
PP3-excluded residual planes.

The completed necessity result does not establish individual restoration
sufficiency.

## Scientific question

For each of the four already frozen PP3-excluded residual principal planes,
starting from that plane's exact native-component-neutralized local state, does
restoring the exact removed native plane component recover the frozen
susceptibility endpoint more strongly than adding a pre-specified equal-norm
orthogonal quarter-turn replacement in the same plane?

All four planes form one prospectively frozen confirmatory family:

`K = {P1, P2, P4, P5}`

No plane may be selected, screened, dropped, promoted, weighted, or assigned a
different sample size using the previously observed necessity effect sizes,
t statistics, or p-values.

The supported set, if any, is unordered.

## Fresh prospective population

Use the next deterministic non-overlapping XG1 range:

`xg1_fact_2101..xg1_fact_2400`

Pair count:

`N = 300`

Expected six-cell rows:

`1800`

Static preparation must establish before any scientific model forward:

- deterministic cohort identity;
- zero pair-ID overlap against `xg1_fact_001..xg1_fact_2100`;
- zero exact claim overlap against previously used XG1 cohorts;
- zero exact evidence overlap against previously used XG1 cohorts;
- zero exact `(claim, evidence)` overlap against previously used XG1 cohorts;
- tokenizer/anchor eligibility for all 300 pairs;
- exact frozen principal-plane vector identities;
- exact frozen XG2/XG4 endpoint basis identities.

No outcome-dependent exclusion, replacement, or substitution is permitted.

Failure of any population, tokenizer, geometry, or provenance gate blocks
implementation and execution.

## Frozen geometry

Residual plane order:

`[P1, P2, P4, P5]`

For each `Pk`, let the already frozen orthonormal plane basis be:

`p_k+`, `p_k-`

For the branch-local native target-token strong-channel state `h`:

`a_k = <h, p_k+>`

`b_k = <h, p_k->`

Native plane component:

`c_k = a_k p_k+ + b_k p_k-`

Frozen within-plane quarter-turn component:

`r_k = -b_k p_k+ + a_k p_k-`

The design requires:

`||c_k||_2 = ||r_k||_2`

and:

`<c_k, r_k> = 0`

No response-guided rotation, reorientation, sign selection, or alternative
matched vector is permitted.

PP3 remains untouched.

The other residual planes remain untouched by each individual-plane
intervention except for permitted runtime numerical residual.

## Frozen conditions

Exactly nine scientific conditions are frozen:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_replacement`
4. `p2_neutralized`
5. `p2_quarter_turn_replacement`
6. `p4_neutralized`
7. `p4_quarter_turn_replacement`
8. `p5_neutralized`
9. `p5_quarter_turn_replacement`

The native condition is shared across all four plane tests and is executed once
per item.

For each plane `Pk`:

### Neutralized background

`B_k = h - c_k`

Direct final-state correction:

`delta_B,k = -c_k`

### Exact native restoration

`R_k = B_k + c_k = h`

The exact restoration state is therefore the shared native condition.

Direct final-state correction relative to native `h`:

`delta_R,k = 0`

No separate restoration scientific condition is needed beyond `native`.

### Matched quarter-turn replacement

`C_k = B_k + r_k`

Equivalently:

`C_k = h - c_k + r_k`

Direct final-state correction:

`delta_C,k = -c_k + r_k`

Relative to the common neutralized background:

`R_k - B_k = c_k`

`C_k - B_k = r_k`

Therefore:

`||R_k - B_k||_2 = ||C_k - B_k||_2`

and the matched replacement is orthogonal to the exact removed native
component within the same frozen plane.

## Frozen susceptibility endpoint

Retain exactly the established broad endpoint.

For condition `c`:

`E_XG2(c) = (1/5) * sum_j J(v_XG2,j; c)^2`

`E_XG4(c) = (1/5) * sum_j J(v_XG4,j; c)^2`

`Q(c) = E_XG2(c) - E_XG4(c)`

Use exactly the frozen five XG2 directions followed by the frozen five XG4
directions.

Finite-difference epsilon remains:

`epsilon = 0.025`

Shared native endpoint:

`Q0 = Q(native)`

For each plane:

`Q_B,k = Q(B_k)`

`Q_C,k = Q(C_k)`

Exact native-restoration endpoint:

`Q_R,k = Q(R_k) = Q0`

## Restoration quantities

For each plane `Pk`:

Exact native restoration gain:

`S_k = Q_R,k - Q_B,k = Q0 - Q_B,k`

Matched replacement gain:

`S_C,k = Q_C,k - Q_B,k`

Primary restoration-specific contrast:

`D_SUF,k = S_k - S_C,k`

Therefore:

`D_SUF,k = Q0 - Q_C,k`

The common neutralized-background term cancels algebraically from the primary
contrast but remains a required observed quantity for the restoration
interpretation.

The canonical stored primary contrast is:

`D_SUF,k = Q0 - Q_C,k`

The raw validator must also verify the equivalent expanded form within the
frozen numerical tolerance.

## Confirmatory family

Exactly four primary confirmatory hypotheses are frozen.

For each `k in {1,2,4,5}`:

`H0,k: mean(D_SUF,k) <= 0`

`H1,k: mean(D_SUF,k) > 0`

Each raw test is:

- one-sample Student t-test;
- one-sided greater;
- `N = 300`;
- `df = 299`.

Exactly four raw confirmatory p-values are permitted.

No fifth p-value is permitted.

No subgroup, alternate-tail, rescue, plane-comparison, or interaction p-value
is permitted.

## Multiplicity control

The four raw p-values form one family:

`{P1, P2, P4, P5}`

Control familywise error using:

`Holm step-down`

with:

`FWER alpha = 0.05`

No plane may be removed from the family after outcomes are observed.

No uncorrected per-plane scientific declaration is permitted.

No alternate correction procedure may be substituted after outcomes are known.

## Positive gates

Shared gate:

`mean(Q0) > 0`

For an individual plane `Pk` to be declared restoration-sufficiency supported,
all of the following must hold:

1. shared `mean(Q0) > 0`;
2. `mean(S_k) > 0`;
3. `mean(D_SUF,k) > 0`;
4. its one-sided test is rejected by the frozen Holm procedure.

Supported planes are reported only as an unordered set.

Do not rank planes by:

- p-value;
- t statistic;
- mean contrast;
- effect magnitude;
- apparent scientific importance.

If at least one plane satisfies all frozen gates, the family-level positive
label is:

`PP3_EXCLUDED_INDIVIDUAL_RESIDUAL_PLANE_RESTORATION_SUFFICIENCY_LOCALIZATION_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

If no plane satisfies all frozen gates:

`PP3_EXCLUDED_INDIVIDUAL_RESIDUAL_PLANE_RESTORATION_SUFFICIENCY_LOCALIZATION_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

A negative restoration result does not invalidate the already established
individual necessity result for any plane.

## Exact scientific forward budget

Each condition uses:

- 5 frozen XG2 susceptibility directions;
- 5 frozen XG4 susceptibility directions.

Each direction uses exactly:

`4 scientific model forwards`

Therefore:

`40 forwards / condition / pair`

Nine conditions:

`360 forwards / pair`

For:

`N = 300`

exact total:

`108000 scientific model forwards`

Baseline model forwards:

`0`

Recommended fixed two-GPU execution split, if execution is later authorized:

- GPU 0: `xg1_fact_2101..xg1_fact_2250` — 150 pairs — `54000` forwards;
- GPU 1: `xg1_fact_2251..xg1_fact_2400` — 150 pairs — `54000` forwards.

No DDP.

No NCCL.

No extra scientific forward may be added merely to capture `h`, `(a_k,b_k)`,
or native coefficients.

## Static preparation requirements

Before implementation, deterministic CPU-only preparation must establish:

- exact fresh cohort `2101..2400`;
- zero overlap against all used XG1 `001..2100`;
- exact tokenizer revision and tokenizer file hashes;
- `PASS_300_OF_300` tokenizer/anchor eligibility;
- exact frozen P1/P2/P4/P5 vector identities;
- per-plane orthonormality;
- cross-plane orthogonality required by the frozen principal-plane basis;
- exact PP3 preservation under each P1/P2/P4/P5 intervention;
- non-target residual-plane coordinate preservation;
- `||c_k||_2 = ||r_k||_2`;
- `<c_k,r_k> = 0`;
- `B_k = h - c_k`;
- `R_k = B_k + c_k = h`;
- `C_k = B_k + r_k`;
- `||R_k-B_k||_2 = ||C_k-B_k||_2`;
- target-plane residual projection after neutralization is zero within frozen
  tolerance;
- quarter-turn replacement coordinates match `(-b_k,a_k)` within frozen
  tolerance.

Static preparation must use:

- scientific model forwards: `0`;
- checkpoint loads: `0`;
- GPU: `false`;
- scientific inference: `false`;
- p-values: `0`.

## Raw-run boundary

If execution is later authorized, the raw runner must record per item:

- `Q0`;
- for each P1/P2/P4/P5:
  - `Q_B,k`;
  - `Q_C,k`;
  - `S_k`;
  - `S_C,k`;
  - `D_SUF,k`;
  - native `(a_k,b_k)`;
  - native component norm;
  - quarter-turn component norm;
  - component dot product;
  - restoration-addition norm mismatch;
  - neutralized target-plane residual;
  - replacement target-plane coordinates;
  - PP3 coordinate drift;
  - non-target-plane coordinate drift;
- scientific forward counts.

The raw runner must not compute:

- t statistics;
- p-values;
- Holm decisions;
- supported-plane set;
- family-level scientific label.

Raw artifacts must state:

`primary_inference_executed = false`

`multiplicity_correction_executed = false`

`scientific_conclusion = null`

`training_executed = false`

`backward_executed = false`

The raw scientific run, if later authorized, is observation only.

## Interpretation if positive

A supported plane `Pk` may be described only as:

an individually detectable local restoration-sufficient contributor to the
frozen layer-17 / target-token native-Mamba susceptibility endpoint relative
to its own pre-specified equal-norm orthogonal within-plane quarter-turn
replacement from the same neutralized background.

This does not establish:

- that the plane alone is sufficient in an otherwise empty state;
- dominance or ranking among residual planes;
- equality of plane effects;
- additive decomposition of the aggregate residual mechanism;
- independence of residual-plane contributions;
- absence of interactions;
- causal status of the XG2-like residual-template orientation;
- behavioral or downstream-task sufficiency;
- benchmark improvement;
- checkpoint-, layer-, token-, generator-, dataset-, model-, or
  architecture-wide universality.

Multiple supported planes indicate distributed local restoration contributions.
They do not imply an additive decomposition.

## Interpretation if negative

For any plane failing the frozen gates, conclude only that individual
restoration sufficiency was not established for that plane under this design.

If no plane survives the familywise procedure, conclude only:

`PP3_EXCLUDED_INDIVIDUAL_RESIDUAL_PLANE_RESTORATION_SUFFICIENCY_LOCALIZATION_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

Do not reinterpret a negative restoration result as evidence against the
already validated individual necessity results.

No rescue analysis, alternative plane grouping, subgroup search, alternate
tail, effect-based selection, or additional p-value is authorized.

## Prohibited analyses

No:

- outcome-guided plane selection;
- plane ranking;
- dropping P4 or any other plane based on prior effect magnitude;
- plane-specific sample-size allocation;
- response-guided control tuning;
- XG2-like template intervention;
- aggregate-template hidden-state correction;
- interaction mining;
- subgroup mining;
- alternative tail;
- alternative multiplicity correction;
- additional confirmatory p-value;
- epsilon sweep;
- endpoint sweep;
- layer sweep;
- token sweep;
- checkpoint sweep;
- hyperparameter tuning;
- rescue analysis;
- training;
- backward;
- task-head evaluation;
- logits-based downstream interpretation.

## Falsification logic

The design is intentionally capable of returning any unordered supported
subset of:

`{P1, P2, P4, P5}`

including the empty set.

No plane is supported merely because its prior necessity result was positive.

No plane is supported merely because its raw restoration mean is positive.

Support requires all frozen gates and Holm familywise rejection.

## Immediate authorization boundary

Prospective design freeze:

`YES`

Deterministic static preparation after design freeze:

`YES`

Implementation:

`NO`

Scientific model execution:

`NO`

Kaggle scientific run:

`NO`

Training/backward:

`NO`

Primary inference:

`NO`

Commit/push:

manual only.