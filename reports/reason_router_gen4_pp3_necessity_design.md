# Gen4 PP3 Necessity — Prospective Design

## Status

`DESIGN_ONLY_NO_IMPLEMENTATION_NO_SCIENTIFIC_MODEL_EXECUTION`

This document prospectively defines the next causal question after the completed
PP3 specificity branch.

No scientific model execution, checkpoint loading, training, backward pass,
task-head evaluation, logits read, or scientific inference is authorized by
this document.

Tokenizer execution is authorized only for the static tokenizer/anchor
eligibility gate explicitly listed in `Next authorized phase`. That gate must
perform zero model forwards, zero checkpoint loads, and zero GPU execution.

## Prior evidence boundary

Frozen synthesis commit:

`cdbffa37f6ac0cf78a7cd76cc5ad96cb3e971068`

The completed evidence supports PP3 as a specific distributed component of the
cross-generator local susceptibility geometry.

It does not establish necessity.

The present question is therefore:

Does selective neutralization of the frozen PP3 plane attenuate the broader
frozen cross-family susceptibility endpoint more than an intervention-magnitude
matched PP5 control?

## Why C_PP3 is not the necessity endpoint

The PP3-specific endpoint

`C_PP3 = (s3/5) * (J_PP3_PLUS^2 - J_PP3_MINUS^2)`

must not be used as the primary necessity outcome.

Removing PP3 and then testing whether PP3's own directional response decreases
would risk a structurally circular result.

The necessity outcome must instead be the previously established broader
five-dimensional cross-family susceptibility contrast.

## Frozen broad endpoint

For any condition `c` and source pair `i`:

`E_XG2_i(c) = (1/5) * sum_{j=1..5} J_i(v_XG2,j ; c)^2`

`E_XG4_i(c) = (1/5) * sum_{j=1..5} J_i(v_XG4,j ; c)^2`

and:

`Q_i(c) = E_XG2_i(c) - E_XG4_i(c)`

The XG2 and XG4 five-dimensional bases must be reconstructed from the already
frozen Phase-1 plans with their existing identities.

No basis refitting, rotation, response-guided sign change, or k-selection is
allowed.

Finite-difference epsilon remains:

`epsilon = 0.025`

## Prospective population

Use a new, non-overlapping XG1 holdout:

`xg1_fact_601..xg1_fact_900`

Expected population:

- source pairs: 300
- six-cell rows: 1800
- generator family: XG1

This cohort must be generated only from the already deterministic XG1 generator
logic.

Before any model execution, static preparation must prove:

1. exact pair range `601..900`;
2. exact cardinality 300 / 1800;
3. no pair-ID overlap with XG1 `001..600`;
4. no exact claim/evidence-row overlap with XG1 `001..600`;
5. preservation of the original deterministic generator semantics;
6. tokenizer/anchor eligibility for all 300 pairs;
7. zero scientific model forwards during preparation.

Any failure blocks the experiment.

## Frozen planes

Treatment plane:

`PP3 = span(pp3_plus, pp3_minus)`

Matched-control plane:

`PP5 = span(pp5_plus, pp5_minus)`

Use exactly the already frozen vectors:

PP3+ SHA256:

`66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`

PP3- SHA256:

`ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`

PP5+ SHA256:

`7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2`

PP5- SHA256:

`311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855`

Static preparation must verify:

- each vector is unit norm;
- PP3+ and PP3- are orthogonal;
- PP5+ and PP5- are orthogonal;
- all PP3-vs-PP5 cross-plane dot products are zero within frozen numerical
  tolerance.

No alternative principal pair is allowed.

## Intervention coordinate

Let `h_b` be the native 395-dimensional strong-channel state at the already
frozen layer-17 target intervention token for branch `b`.

The intervention must operate on that native state before the small signed
susceptibility probe is applied.

No other token, gate channel, weak channel, layer, or checkpoint may change.

## Condition 0 — native state

`T0(h) = h`

This is the untouched local state.

## Condition 3 — PP3 neutralization

For each branch-local native state `h`, define:

`a = <h, pp3_plus>`

`b = <h, pp3_minus>`

and:

`delta_PP3(h) = -a * pp3_plus - b * pp3_minus`

Therefore:

`T3(h) = h + delta_PP3(h)`

which is the orthogonal removal of the complete frozen PP3 component.

The runtime must verify that the residual projection of `T3(h)` onto both PP3
basis vectors is within the frozen numerical tolerance.

## Condition 5 — magnitude-matched PP5 coefficient-transfer control

Use the same PP3-derived coefficients `a` and `b`, but apply them in the frozen
PP5 plane:

`delta_PP5CTRL(h) = -a * pp5_plus - b * pp5_minus`

and:

`T5(h) = h + delta_PP5CTRL(h)`

This is not claimed to neutralize the native PP5 component.

It is a response-blind matched control intervention.

Because PP3 and PP5 each use an orthonormal two-vector basis and the same
coefficients are transferred, the intervention correction norms must satisfy:

`||delta_PP3(h)||_2 = ||delta_PP5CTRL(h)||_2`

for every branch and pair, subject only to frozen runtime cast tolerance.

This equality is a mandatory execution gate.

## Signed susceptibility probe under each condition

For each condition `c in {0,3,5}`, each frozen XG2/XG4 basis direction `v`,
and orientation `o in {+1,-1}`, evaluate the same paired intervention semantics
as the existing validated runtime.

After applying `Tc` to the branch-local native state:

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

## Necessity quantities

For every pair:

`Q0_i = Q_i(0)`

`Q3_i = Q_i(3)`

`Q5_i = Q_i(5)`

PP3 attenuation:

`A3_i = Q0_i - Q3_i`

matched-control attenuation:

`A5_i = Q0_i - Q5_i`

primary necessity contrast:

`D_NEC_i = A3_i - A5_i`

Algebraically:

`D_NEC_i = Q5_i - Q3_i`

The baseline is retained because a necessity claim requires demonstrating actual
attenuation of the phenomenon, not merely a difference between two perturbed
states.

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

No extra model forward may be used merely to obtain the branch-local native
state.

The branch-local neutralization/control correction must be computed inside the
intervention hook from the native state available in that same forward.

If implementation requires extra baseline/capture forwards, implementation must
stop and return to design rather than silently changing the forward budget.

## Primary confirmatory inference

Exactly one confirmatory hypothesis is authorized after validated import:

`H1: mean(D_NEC) > 0`

Test:

- one-sample Student t-test
- one-sided
- N = 300
- alpha = 0.05
- multiplicity correction: none
- confirmatory p-value count: exactly 1

No separate inferential test is authorized for `Q0`, `A3`, `A5`, `Q3`, or `Q5`.

## Positive-label gates

The positive necessity label requires all provenance/runtime/completeness gates
plus all of:

1. `mean(Q0) > 0`
2. `mean(A3) > 0`
3. `mean(D_NEC) > 0`
4. one-sided primary `p < 0.05`

Positive label:

`PP3_NECESSITY_OVER_MATCHED_PP5_CONTROL_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

Otherwise:

`PP3_NECESSITY_OVER_MATCHED_PP5_CONTROL_NOT_ESTABLISHED_ON_FRESH_XG1_HOLDOUT`

## Allowed descriptive statistics

Without additional p-values, the result report may include:

- mean and SD of Q0
- mean and SD of Q3
- mean and SD of Q5
- mean and SD of A3
- mean and SD of A5
- mean and SD of D_NEC
- fraction D_NEC > 0
- PP3-neutralization correction L2 mean/SD
- matched PP5-control correction L2 mean/SD
- maximum absolute per-branch correction-L2 mismatch
- maximum residual PP3 projection after PP3 neutralization

These remain descriptive.

## Prohibited analyses

Do not perform:

- PP1/PP2/PP4 rescue;
- alternative-control rescue;
- response-guided control selection;
- response-guided plane rotation;
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

A positive result would support a local causal necessity contribution of PP3 to
the frozen cross-family susceptibility contrast under this specific
layer-17/token intervention.

It would not establish:

- that PP3 is the sole mechanism;
- global model-behavior necessity;
- downstream task necessity;
- sufficiency of PP3;
- universal necessity across arbitrary generators, layers, tokens, checkpoints,
  or architectures.

The distributed-mechanism interpretation remains in force.

## Next authorized phase

After this design is frozen, the next phase is static preparation only:

1. materialize XG1 `601..900`;
2. prove structural non-overlap and deterministic identity;
3. freeze exact PP3/PP5 plane identities;
4. verify plane orthogonality and correction-norm matching algebra;
5. establish tokenizer/anchor eligibility without model execution.

Implementation and scientific execution require later explicit freezes.
