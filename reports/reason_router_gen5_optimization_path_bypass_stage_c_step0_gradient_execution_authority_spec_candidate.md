# Gen5 Optimization-Path Bypass Stage C
## Step-0 Gradient Geometry Execution Authority

SCIENTIFIC_EXECUTION_ALLOWED=YES_BACKWARD_ONLY_STAGE_C_STEP0_GRADIENT_MATRIX

IMPLEMENTATION_FREEZE_COMMIT=93dfe8984bcb7210833cf8f6d8d021acdf139c9f

TRAINING_ALLOWED=NO

OPTIMIZER_ALLOWED=NO

OPTIMIZER_STEP_ALLOWED=NO

CONFIRMATORY_9601_9900_ALLOWED=NO

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP

## Objective

Measure the exact first loss-driven factorized correction geometry from the
frozen Phase3A setup.

Because B is exactly zero initialized, the primary optimization-pressure object
is the step-0 gradient:

`∇B L`

The execution asks whether the initial task-loss gradient already avoids R22 or
whether it initially points toward R22 and the later optimization trajectory
reroutes elsewhere.

## Frozen ancestry

Stage B evidence freeze:

`b49c1339231a1c5dbb7f870ddc81159109b08c1e`

Stage C implementation freeze:

`93dfe8984bcb7210833cf8f6d8d021acdf139c9f`

Source Phase3A execution:

`d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

No source Phase3A checkpoint, training split, pressure definition, R22, C22,
parent checkpoint, tokenizer identity, or runtime identity may be changed.

## Matrix

Exactly nine cells:

- seed6201 / P0
- seed6201 / PR
- seed6201 / PC
- seed6202 / P0
- seed6202 / PR
- seed6202 / PC
- seed6203 / P0
- seed6203 / PR
- seed6203 / PC

Each cell uses the frozen Phase3A training split of 3360 rows under its matched
pressure.

No dev-based selection and no confirmatory population are used.

## GPU topology

Use exactly two independent single-GPU workers.

GPU0 queue:

1. seed6201 / P0
2. seed6201 / PC
3. seed6202 / PR
4. seed6203 / P0
5. seed6203 / PC

GPU1 queue:

1. seed6201 / PR
2. seed6202 / P0
3. seed6202 / PC
4. seed6203 / PR

DDP is forbidden.

Models, correction modules, gradients, and runtime state are not shared between
workers.

Each cell starts from a fresh authenticated parent model and exact zero-B
correction initialization.

## Authorized computation per cell

Exactly:

- one matched-pressure train-mode forward
- one final 3-way CE computation
- one backward
- read `A_theta.weight.grad`
- read `B_theta.weight.grad`
- no gradient clipping
- no optimizer construction
- no optimizer step
- no parameter update
- no training loop

The reproduced step-0 loss must match the frozen Phase3A step-0 loss within the
implementation tolerance.

The zero-initialized factorization must satisfy:

`∇A = 0`

for the task-loss-driven step-0 geometry.

## Primary measurements

For each cell record:

- effective rank of `∇B`
- singular values of `∇B`
- R22-projected `∇B` energy fraction
- C22-projected `∇B` energy fraction
- principal angles between `span(∇B)` and R22
- principal angles between `span(∇B)` and C22
- effective rank of final learned B
- principal angles between `span(∇B_step0)` and `span(B_final)`

All subspace comparisons are basis-invariant.

If either compared object collapses below rank two, the corresponding rank-two
principal-angle interpretation must be marked invalid rather than forced.

The descriptive rank-two random-subspace affinity reference remains:

`2 / 24576`

No p-values are authorized.

## Interpretation boundary

The execution itself emits no scientific conclusion.

After validated import:

- step-0 ∇B far from R22 and close to B_final supports an
  initial-gradient bypass interpretation;
- step-0 ∇B near R22 but final B far from R22 supports trajectory rerouting;
- rank collapse or mixed geometry must be reported directly rather than mapped
  onto either simplified interpretation.

Stage C does not establish downstream functional equivalence. That remains a
separate Stage D question.

## Scientific firewall

Forbidden:

- training
- optimizer construction
- optimizer step
- gradient clipping
- checkpoint mutation
- parameter update
- seed expansion
- pressure expansion
- stressor search
- layer search
- token search
- plane search
- rank search
- confirmatory 9601..9900 access
- scientific p-values
- scientific conclusion during execution

## Stop conditions

Stop without interpretation if any of the following occurs:

- implementation drift
- authority drift
- parent checkpoint mismatch
- runtime/kernel identity mismatch
- train encoding mismatch
- step-0 loss replay mismatch
- non-finite loss or gradient
- nonzero step-0 A gradient
- parent gradient creation
- parent parameter mutation
- unexpected optimizer construction
- optimizer step
- training execution
- confirmatory population access
- GPU topology mismatch

A successful run establishes Stage C measurement evidence only.
