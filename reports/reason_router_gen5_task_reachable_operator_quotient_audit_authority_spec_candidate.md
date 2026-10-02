# Gen5 Task-Reachable Operator Quotient Audit Authority

SOURCE_FUNCTIONAL_EQUIVALENCE_EVIDENCE_COMMIT=5f079a66f7b0eb0caea30a8d5bc9a0fe757cc449
SOURCE_FUNCTIONAL_FINGERPRINT_EXECUTION_COMMIT=17a783c61e310d6298d4105233ea3f1b71ec8add
SOURCE_CAUSAL_GEOMETRY_EVIDENCE_COMMIT=22f465803f592e87c7ffc3a0fe0e0dac47a7ebc4
SOURCE_CAUSAL_EXECUTION_COMMIT=87f82551c721f953f710cd5dc102aca23161c4e7

STATUS=READY_FOR_TASK_REACHABLE_OPERATOR_QUOTIENT_AUDIT

TRAINING_ALLOWED=NO
BACKWARD_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
CUDA_EVALUATION_ALLOWED=YES_ONE_FROZEN_PHASE3A_DEV_FORWARD
MODEL_FORWARD_ALLOWED=YES_ONE_FROZEN_PHASE3A_DEV_FORWARD
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## Scientific question

The frozen causal intervention established that A initialization strongly selects
the final ambient read-side geometry.

The frozen functional-fingerprint evaluation then established that the resulting
geometrically distinct 3x3 solutions are nevertheless nearly functionally
equivalent on the same frozen Phase3A P0 dev distribution.

The present audit asks:

> Are the large ambient differences suppressed when the learned read/write
> operators are restricted to the task-reachable layer-22 input-state
> distribution?

## Exact task-reachable state definition

Use only the exact frozen Phase3A P0 dev set:

DEV_ROWS=840
SPLIT_SEED=16384
ARM=G5-C0
PRESSURE=P0

Use the exact hidden-state tensor passed as `hidden_states` into
`Phase2Layer22MixerWrapper.forward`.

This is the tensor read directly by:

`latent = A_theta(mixer_input)`

followed by:

`raw = B_theta(latent)`

Include only valid tokens selected by the wrapper's effective attention mask.
Padding-token states are excluded.

No downstream logits are required for the primary quotient calculation.

## State capture

Load the exact frozen parent model and install the frozen layer-22 wrapper with
B exactly zero.

Run exactly one no-grad, eval-mode forward over the frozen 840-row dev set.

Capture only the sufficient statistics of the valid-token layer-22 wrapper
inputs:

`G = X_task^T X_task`

and valid-token count.

Do not persist the full hidden-state tensor unless technically required.

The exact frozen dev encoding and row-order hashes must match the previously
validated identities.

## Exact operator grid

Use the already frozen 3x3 checkpoint grid:

A6201-R6201
A6201-R6202
A6201-R6203
A6202-R6201
A6202-R6202
A6202-R6203
A6203-R6201
A6203-R6202
A6203-R6203

Authenticate all source checkpoint/provenance identities before analysis.

No checkpoint may be modified.

## Primary endpoint family A: ambient versus task read-signal geometry

For each cell pair, report the frozen ambient `row(A)` subspace affinity.

Then define the task read-signal matrix:

`Z_i = X_task A_i^T`

Do not compare raw rank-2 coordinates directly because they are basis-sensitive.

Instead compute the two canonical correlations between the column spaces of
`Z_i` and `Z_j`, using the exact Gram matrix `G`.

Report:

- both canonical correlations;
- mean squared canonical correlation.

This measures whether different ambient read planes induce the same rank-2
signal subspace on task-reachable states.

## Primary endpoint family B: ambient versus task correction-operator action

Define the ambient correction operator:

`O_i = B_i A_i`

Do not materialize the 24576 x 768 operator if unnecessary.

For each pair compute the exact ambient Frobenius quantities using rank-2
identities:

- operator cosine;
- normalized residual

`R_ambient(i,j) =
 ||O_i - O_j||_F /
 sqrt(0.5 * (||O_i||_F^2 + ||O_j||_F^2))`

Then define the task-restricted action:

`Y_i = X_task A_i^T B_i^T`

Again use the exact Gram matrix `G` and low-rank identities rather than
materializing all 24576-dimensional writes.

For each pair report:

- task-action cosine;
- task normalized residual

`R_task(i,j) =
 ||Y_i - Y_j||_F /
 sqrt(0.5 * (||Y_i||_F^2 + ||Y_j||_F^2))`

- quotient suppression ratio

`Q(i,j) = R_task(i,j) / R_ambient(i,j)`

A ratio substantially below 1 indicates that ambient operator differences are
suppressed on task-reachable states.

No post-hoc threshold is authorized.

## Factor-grouped descriptive contrasts

As in the frozen causal design, report raw pairwise values plus grouped means
and ranges for:

1. same A-init, different training RNG;
2. same training RNG, different A-init.

No scientific p-values are authorized.

## Secondary task-state spectrum diagnostics

From `G`, report descriptive task-state anisotropy without assigning a
confirmatory threshold:

- trace;
- top-1, top-2, top-4, top-8, top-16, top-32, top-64, and top-128 cumulative
  energy fractions;
- participation-ratio effective dimension:
  `(tr G)^2 / tr(G^2)`.

These diagnostics may contextualize quotient behavior but are not themselves a
claim of a universal intrinsic dimension.

## Prospective interpretation cases

### Case 1: read-signal and operator-action quotient collapse

If ambient A/operator differences are large, while task read-signal canonical
correlations are near 1 and task-action residuals are strongly suppressed,
support the bounded mechanism:

`GEN5_A_INIT_SELECTS_DISTINCT_AMBIENT_REPRESENTATIVES_WHOSE_ACTION_COLLAPSES_ON_THE_TASK_REACHABLE_LAYER22_STATE_DISTRIBUTION`

This would connect:

`A initialization -> ambient representative selection -> task-reachable quotient equivalence -> near-identical dev function`

### Case 2: read-signal collapse but correction-action does not

Interpret the quotient as occurring at the read-signal level but being
re-expanded by B, with final functional equivalence emerging later.

### Case 3: correction-action does not collapse

If task-restricted `B_i A_i X` remains materially different despite
near-identical final logits, the functional equivalence must emerge downstream
of the layer-22 correction write.

The next localization target would then be recurrence/readout propagation, not
another training experiment.

## Execution form

No repository implementation change is required.

The audit may be performed by one exact registered Kaggle command whose command
hash is preserved by `cm run save`.

Required run artifacts:

- `task_reachable_operator_quotient_summary.json`
- `task_state_gram.pt`
- `run_provenance.json`

Write only under:

`reports/reason_router_gen5_task_reachable_operator_quotient_runs/<run-name>/`

No existing artifact may be overwritten.

## Runtime constraints

Use the same frozen model snapshot, parent checkpoint, tokenizer/runtime, and
validated two-T4 Kaggle environment used by the source runs.

Only GPU 0 is required.

The audit must:

- use `model.eval()`;
- use `torch.no_grad()`;
- perform exactly one frozen-dev model forward for state capture;
- construct no optimizer;
- call no backward;
- perform no training step;
- mutate no checkpoint;
- preserve parent parameter identity;
- load no confirmatory data.

## Stop conditions

Stop if:

- source evidence commit identity mismatches;
- any source checkpoint/provenance identity mismatches;
- the 3x3 grid is incomplete;
- parent identity mismatches;
- frozen dev encoding/order identity mismatches;
- layer-22 state-capture semantics differ from the frozen wrapper input;
- more than one model forward is required without a new authority decision;
- training/backward/optimizer construction would occur;
- confirmatory data would be accessed;
- output collision exists.

## Result boundary

This audit can establish only task-distribution-restricted operator/read-signal
equivalence for the frozen dev state distribution.

It cannot establish exact global operator equivalence, a formal gauge symmetry,
a universal quotient space, or generalization beyond the tested contract.
