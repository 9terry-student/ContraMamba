# Gen5 A-init RNG Functional Fingerprint Evaluation Authority

SOURCE_EVIDENCE_FREEZE_COMMIT=22f465803f592e87c7ffc3a0fe0e0dac47a7ebc4
SOURCE_EXECUTION_COMMIT=87f82551c721f953f710cd5dc102aca23161c4e7
SOURCE_IMPLEMENTATION_FREEZE_COMMIT=06a245f24e4474e2b039b954abd7b6cbf436e822

STATUS=READY_FOR_BOUNDED_FUNCTIONAL_FINGERPRINT_EVALUATION

TRAINING_ALLOWED=NO
BACKWARD_ALLOWED=NO
OPTIMIZER_ALLOWED=NO
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
CUDA_EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV_ONLY
MODEL_FORWARD_ALLOWED=YES_FROZEN_PHASE3A_DEV_ONLY
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

## Scientific question

The frozen 3x3 A-init x training-RNG intervention established that final read-side
geometry and the dominant right/input operator direction track A-initialization
identity far more strongly than the remaining training RNG.

The next question is:

Does the A-init-selected solution identity also appear in functional output
behavior on the same frozen Phase3A P0 dev set?

## Exact evaluation population

Use only the frozen Phase3A dev set:

DEV_ROWS=840
SPLIT_SEED=16384
ARM=G5-C0
PRESSURE=P0

Do not access confirmatory seeds 9601-9900 or any other new population.

## Exact checkpoint grid

Evaluate exactly the frozen 3x3 grid:

A6201-R6201
A6201-R6202
A6201-R6203
A6202-R6201
A6202-R6202
A6202-R6203
A6203-R6201
A6203-R6202
A6203-R6203

The six off-diagonal checkpoints come from the imported run:

`reports/reason_router_gen5_ainit_rng_causal_intervention_runs/gen5-ainit-rng-causal-six-offdiag-87f8255-r1/`

The three diagonal checkpoints are the frozen Phase3A P0 controls.

All checkpoint and provenance hashes must be authenticated before evaluation.

## Parent baseline

Evaluate the exact frozen parent model on the identical dev rows with the
correction contribution disabled exactly.

Define for every cell and every dev item:

`delta_logits(cell) = logits(cell) - logits(parent)`

The parent baseline is shared across all nine cells.

## Primary functional endpoints

For every unordered pair of cells, compute descriptive similarity of the
correction-induced functional fingerprint.

Primary endpoint 1:

Flattened centered `delta_logits` cosine.

Center each 3-class delta-logit vector across classes per example before
flattening, so the metric is invariant to a class-common scalar shift.

Primary endpoint 2:

Per-example pairwise margin-delta fingerprint cosine, using the three unique
class-pair logit differences per example.

Primary grouped contrasts:

1. same A-init, different training RNG;
2. same training RNG, different A-init.

Report all raw pairwise values, grouped means, ranges, and the descriptive
difference:

`mean(same_A_init) - mean(same_training_rng)`

No post-hoc threshold and no scientific p-value are authorized.

## Secondary endpoints

Report:

- prediction agreement between cell pairs;
- agreement of each cell's prediction-change mask relative to parent;
- dev 3-way CE and accuracy for every cell;
- mean and norm of `delta_logits`;
- classwise mean correction-induced logit residuals.

These are secondary diagnostics and guardrails.

## Prospective interpretation cases

Case 1: functional fingerprints are much more similar for same A-init than for
same training RNG.

Bounded interpretation:

`GEN5_CORRECTION_INDUCED_FUNCTIONAL_BEHAVIOR_TRACKS_A_INITIALIZATION_UNDER_FIXED_PHASE3A_P0_EVALUATION`

This would extend the causal chain to:

`A initialization -> final read-side geometry -> reproducible functional fingerprint`

Case 2: geometry follows A-init but functional fingerprints do not.

Interpretation:

distinct A-selected read-side geometries are substantially functionally
equivalent downstream on the frozen dev set.

Case 3: both factors materially affect functional fingerprints.

Interpretation:

A initialization and remaining training stochasticity jointly determine
functional realization despite the strong geometric A-init effect.

## Execution form

No repository implementation change is required.

The evaluation may be performed by one exact registered Kaggle command whose
command hash is preserved by `cm run save`.

The command may use existing frozen ContraMamba modules plus an embedded
read-only evaluator.

The evaluator may write only under:

`reports/reason_router_gen5_ainit_rng_functional_fingerprint_runs/<run-name>/`

Required output artifacts:

- `functional_fingerprint_summary.json`
- `cell_fingerprints.pt`
- `run_provenance.json`

No existing artifact may be overwritten.

## Runtime constraints

Use the same frozen Mamba model snapshot, parent checkpoint, tokenizer/runtime,
and two-T4 Kaggle environment already validated for the source causal run.

Only one GPU is required for this evaluation. No DDP or multi-GPU coordination
is required.

The evaluation must:

- run in `model.eval()`;
- use `torch.no_grad()`;
- construct no optimizer;
- call no backward;
- perform no training step;
- modify no checkpoint;
- preserve parent parameter identity;
- load no confirmatory data.

## Stop conditions

Stop if:

- source checkpoint/provenance identity mismatches;
- the 3x3 grid is incomplete;
- parent identity mismatches;
- frozen dev identity mismatches;
- evaluation would require training/backward/optimizer construction;
- confirmatory data would be accessed;
- an output path collision exists.

## Result boundary

A successful evaluation run establishes only functional fingerprint evidence.

It does not by itself establish a universal initialization law, generalization
outside the tested contract, or a statistical population-level claim.
