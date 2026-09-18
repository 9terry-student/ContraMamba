# Gen4 PP3-Excluded Residual Aggregate Necessity — Prospective Design

## Status

`DESIGN_ONLY_NO_IMPLEMENTATION_NO_SCIENTIFIC_MODEL_EXECUTION`

This document prospectively defines the next causal question after the completed
PP3 local-causal chain and the completed PP3-excluded residual-template transport
result.

No scientific model execution, checkpoint loading, training, backward pass,
task-head evaluation, logits read, or scientific inference is authorized by
this document.

Tokenizer execution is authorized only for the static tokenizer/anchor
eligibility gate explicitly listed in `Next authorized phase`. That gate must
perform zero model forwards, zero checkpoint loads, and zero GPU execution.

## Prior evidence boundary

Current synthesis commit:

`7d269a3bb7dda97978316ebe95748b0520e6b21e`

Validated residual-template result commit:

`a32ae557b7b0e1797cb632e23ce023c78938b018`

The completed evidence supports:

- PP3 as a transportable, geometry-specific, locally necessary, and locally
  restoration-sufficient contributor to the frozen local susceptibility
  mechanism;
- a prospectively transported XG2-like aggregate orientation in the response
  remaining after PP3 is excluded.

The residual-template result is geometric, not causal.

It does not establish causal necessity, sufficiency, or ownership for
P1/P2/P4/P5.

The next question is therefore:

Does selective neutralization of the complete pre-specified PP3-excluded
principal-plane subspace attenuate the broader frozen cross-family
susceptibility endpoint more than an exact-norm, coefficient-preserving,
within-plane quarter-turn control?

## Why the XG2-like template is not itself an intervention direction

The frozen residual template lives in four-dimensional response-contribution
coordinates:

`[q_P1, q_P2, q_P4, q_P5]`

It is not a 395-dimensional hidden-state vector.

No arbitrary inverse map from this effect-space template into hidden-state space
is authorized.

No template-weighted hidden-state intervention is allowed.

The causal intervention instead uses the complete, prospectively fixed
PP3-excluded principal-plane subspace:

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`

This set is fixed as the complement of PP3 inside the already frozen five
principal planes and is not selected using fresh outcomes.

## Frozen broad endpoint

For any condition `c` and source pair `i`:

`E_XG2_i(c) = (1/5) * sum_{j=1..5} J_i(v_XG2,j ; c)^2`

`E_XG4_i(c) = (1/5) * sum_{j=1..5} J_i(v_XG4,j ; c)^2`

and:

`Q_i(c) = E_XG2_i(c) - E_XG4_i(c)`

The XG2 and XG4 five-dimensional bases must be reconstructed from their already
frozen identities.

No basis refitting, response-guided rotation, response-guided sign change,
k-selection, or plane selection is allowed.

Finite-difference epsilon remains:

`epsilon = 0.025`

## Prospective population

Use a new, non-overlapping XG1 holdout:

`xg1_fact_1501..xg1_fact_1800`

Expected population:

- source pairs: 300
- six-cell rows: 1800
- generator family: XG1

Before any scientific model execution, static preparation must prove:

1. exact pair range `1501..1800`;
2. exact cardinality 300 / 1800;
3. no pair-ID overlap with XG1 `001..1500`;
4. no exact claim-text overlap with XG1 `001..1500`;
5. no exact evidence-text overlap with XG1 `001..1500`;
6. no exact `(claim, evidence)` row-identity overlap with XG1 `001..1500`;
7. preservation of the original deterministic XG1 generator semantics;
8. tokenizer/anchor eligibility for all 300 pairs;
9. zero scientific model forwards during preparation.

Any failure blocks the experiment.

## Frozen principal-plane set

Causal treatment subspace:

`R = {P1, P2, P4, P5}`

PP3 is explicitly excluded from both the treatment and matched-control
corrections.

For each residual plane `Pk`, use the already frozen orthonormal pair:

`Pk = span(pk_plus, pk_minus)`

Static preparation must freeze exact SHA256 identities for all eight residual
vectors:

- P1+
- P1-
- P2+
- P2-
- P4+
- P4-
- P5+
- P5-

and verify the already frozen PP3+ / PP3- identities.

Static preparation must verify, within the frozen numerical tolerance:

- every principal vector has unit norm;
- plus/minus vectors are orthogonal within each plane;
- all vectors from distinct principal planes are mutually orthogonal;
- every residual vector is orthogonal to both PP3 vectors.

No alternative residual plane set is allowed.

## Intervention coordinate

Let `h_b` be the native 395-dimensional strong-channel state at the already
frozen layer-17 target intervention token for branch `b`.

The condition correction must be computed inside the intervention hook from
that branch-local native state before the small signed susceptibility probe is
applied.

No other token, gate channel, weak channel, layer, checkpoint, or intervention
location may change.

## Native residual coefficients

For each residual plane `Pk` and branch-local native state `h`, define:

`a_k = <h, pk_plus>`

`b_k = <h, pk_minus>`

The native residual-plane component is:

`c_k(h) = a_k * pk_plus + b_k * pk_minus`

The complete PP3-excluded residual component is:

`c_R(h) = sum_{k in {1,2,4,5}} c_k(h)`

No coefficient is reweighted using fresh outcomes, residual-template values, or
per-plane response magnitudes.

## Condition 0 — native state

`T0(h) = h`

This is the untouched local state.

## Condition R — complete residual neutralization

Define:

`delta_R(h) = -c_R(h)`

and:

`T_R(h) = h + delta_R(h)`

This is orthogonal removal of the complete frozen PP3-excluded principal-plane
component.

The runtime must verify that the projections of `T_R(h)` onto all eight
residual basis vectors are within frozen numerical tolerance of zero.

Because the residual planes are orthogonal to PP3, the runtime must also verify
that PP3 coefficients are unchanged by this correction within frozen cast
tolerance.

## Condition C — coefficient-preserving quarter-turn matched control

For every residual plane `Pk`, define the deterministic 90-degree rotation of
the native coefficient pair:

`(a_k, b_k) -> (-b_k, a_k)`

The corresponding rotated component is:

`r_k(h) = -b_k * pk_plus + a_k * pk_minus`

and the control correction is:

`delta_C(h) = - sum_{k in {1,2,4,5}} r_k(h)`

Therefore:

`T_C(h) = h + delta_C(h)`

This control is response-blind and uses:

- the same four prospectively fixed planes;
- the same branch-local native coefficient magnitudes;
- no fresh response values;
- no residual-template weighting;
- no per-plane selection.

For every plane:

`||c_k(h)||_2 = ||r_k(h)||_2`

Because the principal planes are mutually orthogonal:

`||delta_R(h)||_2 = ||delta_C(h)||_2`

for every branch and pair, subject only to frozen runtime cast tolerance.

The treatment and control corrections are also exactly orthogonal in the ideal
frozen geometry:

`<delta_R(h), delta_C(h)> = 0`

subject only to frozen runtime cast tolerance.

The runtime must verify both correction-norm equality and correction
orthogonality.

Because both corrections lie entirely in `R`, PP3 coefficients must remain
unchanged under both conditions within frozen cast tolerance.

## Signed susceptibility probe under each condition

For each condition `c in {0,R,C}`, each frozen XG2/XG4 basis direction `v`,
and orientation `o in {+1,-1}`, evaluate the same paired intervention semantics
used by the validated Gen4 runtime.

After applying `T_c` to the branch-local native state:

- target-plus branch receives `+ o * epsilon * v`;
- target-minus branch receives `- o * epsilon * v`.

Define:

`F_i(+epsilon; v, c)`

and:

`F_i(-epsilon; v, c)`

using the existing plus-path-efficiency minus minus-path-efficiency response.

Then:

`J_i(v;c) = [F_i(+epsilon;v,c) - F_i(-epsilon;v,c)] / (2*epsilon)`

No change to the frozen path-efficiency definition is allowed.

## Aggregate residual necessity quantities

For every pair:

`Q0_i = Q_i(0)`

`QR_i = Q_i(R)`

`QC_i = Q_i(C)`

Aggregate residual attenuation:

`A_R_i = Q0_i - QR_i`

Matched-control attenuation:

`A_C_i = Q0_i - QC_i`

Primary necessity contrast:

`D_RES_NEC_i = A_R_i - A_C_i`

Algebraically:

`D_RES_NEC_i = QC_i - QR_i`

The native condition is retained because a necessity claim requires actual
attenuation of the established broader phenomenon, not merely a difference
between two perturbed states.

## Forward budget

Per condition:

- XG2 basis directions: 5
- XG4 basis directions: 5
- total directions: 10
- forwards per direction: 4
- forwards per pair: 40

Across three conditions:

- forwards per pair: 120
- source pairs: 300
- exact scientific forward budget: 36000

No extra scientific model forward may be used merely to obtain branch-local
native coefficients.

The residual neutralization and quarter-turn control corrections must be
computed inside the intervention hook from the native state available in that
same forward.

If implementation requires extra baseline/capture forwards, implementation
must stop and return to design rather than silently changing the forward
budget.

## Primary confirmatory inference

Exactly one confirmatory hypothesis is authorized after validated import:

`H0: mean(D_RES_NEC) <= 0`

`H1: mean(D_RES_NEC) > 0`

Test:

- one-sample Student t-test
- one-sided alternative: greater
- N = 300
- df = 299
- alpha = 0.05
- multiplicity correction: none
- confirmatory p-value count: exactly 1

No separate inferential test is authorized for `Q0`, `QR`, `QC`, `A_R`, or
`A_C`.

## Positive-label gates

The positive aggregate-residual necessity label requires all
provenance/runtime/completeness gates plus all of:

1. `mean(Q0) > 0`
2. `mean(A_R) > 0`
3. `mean(D_RES_NEC) > 0`
4. one-sided primary `p < 0.05`

Positive label:

`PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_OVER_QUARTER_TURN_MATCHED_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Otherwise:

`PP3_EXCLUDED_RESIDUAL_AGGREGATE_NECESSITY_OVER_QUARTER_TURN_MATCHED_CONTROL_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

## Allowed descriptive statistics

Without additional p-values, the result report may include:

- mean and SD of Q0
- mean and SD of QR
- mean and SD of QC
- mean and SD of A_R
- mean and SD of A_C
- mean and SD of D_RES_NEC
- fraction D_RES_NEC > 0
- residual-neutralization correction L2 mean/SD
- quarter-turn-control correction L2 mean/SD
- maximum absolute per-branch correction-L2 mismatch
- maximum absolute treatment-control correction dot product
- maximum residual-plane projection after residual neutralization
- maximum absolute PP3 coefficient drift under residual neutralization
- maximum absolute PP3 coefficient drift under quarter-turn control

These remain descriptive.

## Prohibited analyses

Do not perform:

- P1/P2/P4/P5 individual necessity tests;
- per-plane p-values;
- response-guided plane selection;
- response-guided plane weighting;
- residual-template-weighted hidden-state intervention;
- alternative-control rescue;
- response-guided control rotation;
- epsilon sweep;
- k sweep;
- checkpoint sweep;
- layer sweep;
- token sweep;
- subgroup analysis;
- tail analysis;
- pair dropping or replacement after model responses;
- additional inferential p-values;
- training or backward passes;
- task-head evaluation;
- logits analysis.

## Interpretation boundary

A positive result would support a local causal necessity contribution of the
complete pre-specified PP3-excluded principal-plane subspace to the frozen
cross-family susceptibility contrast, beyond an exact-norm,
coefficient-preserving quarter-turn matched control.

It would not establish:

- causal necessity of any individual residual plane;
- sufficiency of the aggregate residual subspace;
- causality of the XG2-like residual-template orientation itself;
- that the residual subspace is the sole mechanism outside PP3;
- global model-behavior necessity;
- downstream task necessity;
- universal necessity across arbitrary generators, layers, tokens,
  checkpoints, or architectures.

A negative result would mean that the prospectively transported XG2-like
residual geometry has not been shown to make an aggregate local causal
necessity contribution under this matched-control intervention.

No rescue analysis follows.

## Next authorized phase

After this design is frozen, the next phase is static preparation only:

1. materialize XG1 `1501..1800`;
2. prove structural non-overlap and deterministic identity against XG1
   `001..1500`;
3. freeze exact P1/P2/P4/P5 plus/minus vector identities and re-verify frozen
   PP3 identities;
4. verify full principal-vector orthonormality;
5. verify quarter-turn control norm equality and treatment-control
   orthogonality algebra;
6. establish tokenizer/anchor eligibility without scientific model execution.

Implementation and scientific execution require later explicit freezes.
