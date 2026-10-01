# Gen5 Optimization-Path Bypass Stage B Forward Execution Authority

## Authority status

SCIENTIFIC_EXECUTION_ALLOWED=YES_FORWARD_ONLY_STAGE_B_MATRIX

IMPLEMENTATION_FREEZE_COMMIT=0107c853b8b943e9171b2df68a869a7d6127f6dc

TRAINING_ALLOWED=NO

BACKWARD_ALLOWED=NO

CONFIRMATORY_9601_9900_ALLOWED=NO

## Scientific objective

Execute the already-frozen Gen5 Stage B functional decomposition against the
already-trained Phase3A correction checkpoints.

The sole question is whether the task-loss benefit of each learned correction is
carried by its R22-projected output-write component, its R22-orthogonal
output-write component, or an interaction between them.

This execution does not test optimization-time gradients, does not test a new
ownership intervention, and does not authorize any new training.

## Frozen implementation

Authorized implementation freeze:

`0107c853b8b943e9171b2df68a869a7d6127f6dc`

Authorized implementation files:

- `scripts/reason_router_gen5_optimization_path_bypass_stage_b_functional_decomposition.py`
- `tests/test_reason_router_gen5_optimization_path_bypass_stage_b_functional_decomposition.py`

Those files must remain byte-identical from the implementation freeze through
the Stage B execution head.

## Frozen source evidence

Stage A geometry evidence is frozen at:

`3b1adc2bd433f1fc697cfb7a268b2772cd83c1b5`

Phase3A terminal evidence is frozen at:

`ae61d962443cb374086420c62b33dd86441b4d8f`

The nine Phase3A final corrections originate from execution commit:

`d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

No checkpoint may be retrained, selected, tuned, or replaced.

## Evaluation population

Use only the already-frozen Phase3A development domain:

- 120 source pairs
- 840 labeled rows
- split seed 16384
- each checkpoint evaluated under its own matched training pressure

The held-out pair IDs 9601 through 9900 remain untouched.

They are not authorized for Stage B loading, encoding, forward execution, or
interpretation.

## Functional decomposition

For each trained correction:

`M = BA`

A remains exactly equal to the trained `A_theta.weight`.

Only B is decomposed using the frozen orthonormal R22 basis:

- `FULL`: `B`
- `R22_ONLY`: `P_R B`
- `R22_REMOVED`: `(I - P_R) B`
- `ZERO`: `0`

The implementation must verify:

`R22_ONLY + R22_REMOVED = FULL`

within the frozen numerical tolerance and verify that the R22 projection of
`R22_REMOVED` is numerically zero within tolerance.

No re-factorization of BA is authorized.

## Matrix

Frozen checkpoints:

- seed 6201: P0, PR, PC
- seed 6202: P0, PR, PC
- seed 6203: P0, PR, PC

Conditions per checkpoint:

- FULL
- R22_ONLY
- R22_REMOVED
- ZERO

Total authorized condition evaluations:

`9 × 4 = 36`

## Runtime boundary

Authorized:

- exact parent/model runtime authentication
- exact tokenizer encoding of the frozen Phase3A dev set
- matched-pressure frozen stressor application
- loading the nine frozen correction checkpoints
- forward-only evaluation
- final 3-way cross-entropy
- per-row logits, labels, predictions, and CE
- deterministic provenance and artifact generation

Forbidden:

- optimizer construction for scientific execution
- optimizer step
- backward
- gradient accumulation
- parameter update
- checkpoint selection
- hyperparameter search
- pressure search
- seed search
- R22/C22 modification
- alternate layer/token/plane search
- additional training
- confirmatory 9601..9900 access
- scientific p-value production

The runtime model must be in evaluation mode.

## Primary descriptive quantities

For each seed-pressure checkpoint record:

- `L_FULL`
- `L_R22_ONLY`
- `L_R22_REMOVED`
- `L_ZERO`

Derived descriptive quantities include:

- `L_ZERO - L_FULL`
- `L_ZERO - L_R22_ONLY`
- `L_ZERO - L_R22_REMOVED`
- `L_R22_REMOVED - L_FULL`
- `L_R22_ONLY - L_ZERO`

When `L_ZERO - L_FULL` is nonzero, descriptive retained-gain fractions may also
be recorded.

No equivalence margin, significance threshold, or promotion rule is introduced
by this authority.

## Interpretation boundary

A successful execution establishes only valid Stage B forward evidence.

Potential qualitative patterns may later be interpreted as:

- task benefit primarily outside R22,
- small-occupancy R22 with disproportionate functional leverage,
- or functional interaction between projected and orthogonal components.

The execution itself must not emit a scientific conclusion.

In particular it does not establish:

- gradient misalignment,
- optimizer preference at step 0,
- downstream Jacobian degeneracy,
- a non-substitutable optimization bottleneck,
- or a general causal-role invariant.

Those remain separate future questions.

## Stop conditions

Stop and do not interpret Stage B if any of the following occurs:

- implementation drift after the frozen implementation commit
- parent checkpoint identity mismatch
- Phase3A correction checkpoint identity mismatch
- dev encoding identity mismatch
- decomposition reconstruction failure
- R22-removal orthogonality failure
- non-finite logits or loss
- unexpected gradient creation
- parent parameter mutation
- confirmatory population access
- backward execution
- optimizer step
- training execution

## Output

The authorized run may create only Stage B result/provenance artifacts required
by the frozen implementation.

Successful runtime completion is not itself a scientific conclusion.

Stage B result interpretation must occur only after imported artifacts and
provenance are validated.
