# Gen4 PP3-Excluded Residual Aggregate Restoration Sufficiency — Prospective Design

## Status

`DESIGN_ONLY_STATIC_PREPARATION_ALLOWED_NO_IMPLEMENTATION_NO_SCIENTIFIC_EXECUTION`

This document freezes the next causal question after completion of:

- PP3 transport, specificity, local necessity, and local restoration sufficiency;
- PP3-excluded XG2-like residual-template transport;
- aggregate local necessity of the complete PP3-excluded residual subspace;
- individual local necessity of P1, P2, P4, and P5;
- individual local restoration sufficiency of P1, P2, P4, and P5.

It authorizes deterministic static preparation only.

It does not authorize scientific model execution, checkpoint loading, GPU
execution, Kaggle scientific execution, statistical inference, implementation,
training, backward, task-head evaluation, logits analysis, or post-hoc endpoint
modification.

## Prior validated synthesis

Current synthesis commit:

`d254419b3314ae519210f79a936e79026931ab2e`

Current bounded mechanism picture:

`shared PP3 causal core + structured XG2-like PP3-excluded residual geometry + aggregate residual local necessity + individually localized local necessity and local restoration sufficiency across P1/P2/P4/P5`

Aggregate residual restoration sufficiency remains unestablished.

The positive individual-plane restoration results do not imply aggregate
restoration sufficiency because the frozen endpoint need not combine additively
across simultaneous P1/P2/P4/P5 intervention and cross-plane interactions
remain unresolved.

## Scientific question

Starting from the exact branch-local state in which the complete native
PP3-excluded residual component has been removed, does restoring that exact
complete native residual component recover the frozen susceptibility endpoint
more strongly than adding a prospectively fixed equal-norm orthogonal
quarter-turn replacement in the same complete residual subspace?

The complete residual subspace is fixed prospectively as:

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`

No plane selection, response-guided weighting, residual-template weighting, or
outcome-guided control construction is permitted.

This experiment tests aggregate local restoration sufficiency of `R`.

It does not test individual-plane effects again.

## Fresh prospective population

Use the next deterministic non-overlapping XG1 range:

`xg1_fact_2401..xg1_fact_2700`

Pair count:

`N = 300`

Expected six-cell rows:

`1800`

Before any scientific model execution, static preparation must prove:

1. exact pair range `2401..2700`;
2. exact cardinality `300 / 1800`;
3. zero pair-ID overlap with XG1 `001..2400`;
4. zero exact claim overlap with all previously used XG1 cohorts;
5. zero exact evidence overlap with all previously used XG1 cohorts;
6. zero exact `(claim, evidence)` overlap with all previously used XG1 cohorts;
7. preservation of deterministic XG1 generator semantics;
8. tokenizer/anchor eligibility for all 300 pairs;
9. exact frozen residual-plane vector identities;
10. exact frozen PP3 identities;
11. exact frozen XG2/XG4 endpoint basis identities;
12. zero scientific model forwards, zero checkpoint loads, and zero GPU use.

No outcome-dependent exclusion, replacement, regeneration, or substitution is
permitted.

Failure of any static gate blocks implementation.

## Frozen residual geometry

Residual plane order:

`[P1, P2, P4, P5]`

For each frozen residual plane `Pk`:

`Pk = span(p_k+, p_k-)`

For the branch-local native target-token strong-channel state `h`, define:

`a_k = <h, p_k+>`

`b_k = <h, p_k->`

Native component in plane `Pk`:

`c_k = a_k p_k+ + b_k p_k-`

Complete native PP3-excluded residual component:

`c_R = sum_{k in {1,2,4,5}} c_k`

For every plane, freeze the same deterministic quarter-turn:

`r_k = -b_k p_k+ + a_k p_k-`

Complete quarter-turn replacement component:

`r_R = sum_{k in {1,2,4,5}} r_k`

Because all frozen principal planes are mutually orthogonal:

`||c_R||_2 = ||r_R||_2`

and:

`<c_R, r_R> = 0`

subject only to the already frozen numerical tolerance.

No alternative rotation angle, plane weighting, sign choice, or response-guided
orientation is allowed.

PP3 remains untouched.

## Frozen conditions

Exactly three scientific conditions are frozen.

### Condition 0 — native

`T0(h) = h`

Endpoint:

`Q0 = Q(native)`

### Condition B — complete residual neutralization

Neutralized background:

`B = h - c_R`

Direct final-state correction:

`delta_B = -c_R`

Endpoint:

`Q_B = Q(B)`

Static and runtime validation must verify that all P1/P2/P4/P5 coordinates are
zero after neutralization within frozen numerical tolerance and that PP3
coordinates remain unchanged within frozen cast tolerance.

### Condition C — aggregate quarter-turn replacement

Matched replacement from the same neutralized background:

`C = B + r_R`

Equivalently:

`C = h - c_R + r_R`

Direct final-state correction relative to native:

`delta_C = -c_R + r_R`

Endpoint:

`Q_C = Q(C)`

Relative to the common neutralized background:

`R_native - B = c_R`

where:

`R_native = B + c_R = h`

and:

`C - B = r_R`

Therefore:

`||R_native - B||_2 = ||C - B||_2`

and:

`<R_native - B, C - B> = 0`

within frozen numerical tolerance.

The replacement must reproduce the per-plane quarter-turn coordinates:

`(-b_k, a_k)`

for every `Pk` simultaneously.

No separate scientific restoration condition is required because exact native
restoration is the shared native condition.

## Frozen susceptibility endpoint

Retain exactly the established broad endpoint.

For condition `c`:

`E_XG2(c) = (1/5) * sum_j J(v_XG2,j; c)^2`

`E_XG4(c) = (1/5) * sum_j J(v_XG4,j; c)^2`

`Q(c) = E_XG2(c) - E_XG4(c)`

Use exactly the already frozen five XG2 directions followed by the already
frozen five XG4 directions.

Finite-difference epsilon remains:

`epsilon = 0.025`

No endpoint refitting, basis refitting, sign search, direction selection, or
alternative epsilon is permitted.

## Aggregate restoration quantities

Exact aggregate native restoration gain:

`S_R = Q0 - Q_B`

Matched aggregate replacement gain:

`S_C = Q_C - Q_B`

Primary aggregate restoration-specific contrast:

`D_RES_SUF = S_R - S_C`

Therefore:

`D_RES_SUF = Q0 - Q_C`

The common neutralized-background term cancels algebraically from the primary
contrast but remains a required observed quantity for the restoration
interpretation.

The canonical stored primary contrast must be:

`D_RES_SUF = Q0 - Q_C`

The raw validator must also verify:

`D_RES_SUF = (Q0 - Q_B) - (Q_C - Q_B)`

within the frozen numerical tolerance.

## Primary confirmatory inference

Exactly one confirmatory hypothesis is frozen:

`H0: mean(D_RES_SUF) <= 0`

`H1: mean(D_RES_SUF) > 0`

Test:

- one-sample Student t-test;
- one-sided alternative: greater;
- `N = 300`;
- `df = 299`;
- `alpha = 0.05`;
- multiplicity correction: none;
- confirmatory p-value count: exactly `1`.

No separate inferential test is authorized for `Q0`, `Q_B`, `Q_C`, `S_R`,
`S_C`, any individual plane, or any interaction quantity.

## Positive-label gates

The positive aggregate-restoration label requires all provenance, runtime,
geometry, completeness, and artifact-validation gates plus all of:

1. `mean(Q0) > 0`;
2. `mean(S_R) > 0`;
3. `mean(D_RES_SUF) > 0`;
4. one-sided primary `p < 0.05`.

Positive label:

`PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_SUFFICIENCY_OVER_QUARTER_TURN_REPLACEMENT_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Otherwise:

`PP3_EXCLUDED_RESIDUAL_AGGREGATE_RESTORATION_SUFFICIENCY_OVER_QUARTER_TURN_REPLACEMENT_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

A negative result does not invalidate the already established aggregate
necessity or individual necessity/restoration results.

## Exact scientific forward budget

Each condition uses:

- 5 frozen XG2 susceptibility directions;
- 5 frozen XG4 susceptibility directions;
- 4 scientific model forwards per direction.

Therefore:

`40 forwards / condition / pair`

Three conditions:

`120 forwards / pair`

For:

`N = 300`

exact total:

`36000 scientific model forwards`

Baseline model forwards:

`0`

If execution is later authorized, the fixed two-GPU split is:

- GPU 0: `xg1_fact_2401..xg1_fact_2550`, 150 pairs, `18000` forwards;
- GPU 1: `xg1_fact_2551..xg1_fact_2700`, 150 pairs, `18000` forwards.

No DDP.

No NCCL.

No extra scientific forward may be added to capture native coefficients or
hidden states.

If implementation requires additional scientific forwards, it must stop and
return to design.

## Static preparation requirements

Before implementation, deterministic CPU-only preparation must establish:

- exact fresh cohort `2401..2700`;
- zero overlap against used XG1 `001..2400`;
- exact tokenizer revision and tokenizer-file identities;
- tokenizer/anchor eligibility `PASS_300_OF_300`;
- exact P1/P2/P4/P5 plus/minus vector identities;
- exact PP3 plus/minus identities;
- exact XG2/XG4 endpoint basis identities;
- unit norm and within-plane orthogonality for every residual basis pair;
- cross-plane orthogonality across P1/P2/P4/P5;
- residual-plane orthogonality to PP3;
- `c_R = sum c_k`;
- `r_R = sum r_k`;
- `||c_R||_2 = ||r_R||_2`;
- `<c_R,r_R> = 0`;
- `B = h - c_R`;
- `B + c_R = h`;
- `C = B + r_R`;
- equality of restoration and replacement addition norms;
- aggregate residual projection after neutralization is zero within tolerance;
- all four replacement coefficient pairs equal `(-b_k,a_k)` within tolerance;
- PP3 coordinate preservation under both aggregate interventions.

Static preparation must use:

- scientific model forwards: `0`;
- checkpoint loads: `0`;
- GPU: `false`;
- statistical inference: `false`;
- p-values: `0`.

## Raw-run boundary

If scientific execution is later authorized, raw artifacts must record per
item:

- `Q0`;
- `Q_B`;
- `Q_C`;
- `S_R`;
- `S_C`;
- `D_RES_SUF`;
- all native P1/P2/P4/P5 coefficient pairs;
- aggregate native residual-component norm;
- aggregate quarter-turn replacement norm;
- restoration/replacement addition-norm mismatch;
- aggregate native/replacement dot product;
- maximum residual-coordinate magnitude after neutralization;
- replacement-coordinate residual for every residual plane;
- PP3 coordinate drift;
- scientific forward counts.

The raw runner must not compute:

- t statistics;
- p-values;
- a scientific positive/negative label;
- individual-plane tests;
- interaction tests;
- additive decomposition;
- ranking.

Raw artifacts must state:

`primary_inference_executed = false`

`multiplicity_correction_executed = false`

`scientific_conclusion = null`

`training_executed = false`

`backward_executed = false`

`task_heads_executed = false`

`logits_read = false`

The raw scientific run is observation only.

## Interpretation if positive

A positive result supports only the bounded claim that the complete
pre-specified PP3-excluded residual principal-plane subspace has an aggregate
local restoration-sufficiency contribution to the frozen layer-17 /
target-token native-Mamba susceptibility endpoint relative to the
pre-specified equal-norm orthogonal within-R quarter-turn replacement from the
same complete residual-neutralized background.

It would complement the already validated aggregate necessity result.

It would not establish:

- additive equality between aggregate and individual restoration effects;
- independence or absence of interactions among P1/P2/P4/P5;
- ranking or dominance among residual planes;
- causal status of the XG2-like residual-template orientation;
- that `R` is sufficient in an otherwise empty hidden state;
- that PP3 plus `R` forms a complete additive decomposition;
- behavioral or downstream-task sufficiency;
- benchmark improvement;
- universality across checkpoints, layers, tokens, generators, datasets,
  architectures, or Mamba models generally.

## Interpretation if negative

A negative result means only that aggregate residual restoration sufficiency
was not established under this prospectively frozen simultaneous intervention.

It does not contradict the already validated individual-plane restoration
results, because simultaneous aggregate intervention can exhibit non-additive
cross-plane effects.

No rescue analysis follows.

## Prohibited analyses

No individual-plane p-values.

No plane ranking.

No interaction mining.

No additive-decomposition test.

No response-guided weighting or plane selection.

No XG2-like-template hidden-state intervention.

No alternative quarter-turn angle.

No alternative tail.

No rescue control.

No subgroup mining.

No epsilon, layer, token, checkpoint, or basis sweep.

No training.

No backward.

No task-head evaluation.

No logits-based downstream interpretation.

No additional confirmatory p-value.

## Falsification logic

The design is intentionally capable of producing a negative aggregate result
even though all four individual-plane restoration tests were positive.

The prior individual results are not used as a positive prior, weighting rule,
selection mechanism, or inferential shortcut.

Support requires the single prospectively frozen `D_RES_SUF` test and all
positive gates above.

## Immediate authorization boundary

Prospective design freeze:

`YES`

Deterministic static preparation after design freeze:

`YES`

Implementation:

`NO`

Scientific model execution:

`NO`

Kaggle scientific execution:

`NO`

Primary inference:

`NO`

Training/backward:

`NO`

Commit/push:

manual only.