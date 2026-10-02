# Gen5 A-Initialization × Training-RNG Causal Intervention
# Execution Authority

## Authority identities

DESIGN_IMPLEMENTATION_AUTHORITY_COMMIT=d0c86e8725e5df9def6acad323f3cce155b9daea

IMPLEMENTATION_FREEZE_COMMIT=ba75e0879f1ea953ff3595fc8abc2f740d130df0

MECHANISM_ANALYSIS_FREEZE_COMMIT=155c1898f9f2d6dd0765fc93cc156190d8a08708

SCIENTIFIC_EXECUTION_ALLOWED=YES_GEN5_AINIT_RNG_CAUSAL_SIX_OFFDIAGONAL_MATRIX

CUDA_PREFLIGHT_ALLOWED=YES_OPTIONAL_OFFDIAGONAL

TRAINING_ALLOWED=YES_EXACT_GEN5_AINIT_RNG_SIX_OFFDIAGONAL

EVALUATION_ALLOWED=YES_FROZEN_PHASE3A_DEV

BACKWARD_ALLOWED=YES_TRAINING_ONLY

OPTIMIZER_ALLOWED=YES_ADAMW_20_STEPS

CONFIRMATORY_9601_9900_ALLOWED=NO

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP

## Status

`READY_FOR_EXACT_COMMIT_BOUND_EXECUTION`

This authority permits exactly one bounded Gen5 causal intervention:

separate the seed controlling `A_theta` initialization from the seed controlling
the remaining training RNG under the frozen Phase3A P0 contract.

No scientific delta beyond this seed-factor separation is authorized.

## Scientific purpose

Frozen static analysis established:

- final `row(A)` is strongly anchored to its same-seed initialization;
- final read-side geometry is much more seed-specific than output/write geometry;
- the dominant output/write direction is comparatively shared across seeds;
- SCALEMATCH near-full task recovery tracks preservation of the same-seed
  dominant rank-1 functional mode;
- the dominant right/input direction remains strongly seed-specific.

The remaining causal question is:

> Does final read-side geometry and the dominant right/input functional
> direction follow the `A_theta` initialization seed more strongly than the
> remaining training RNG seed when those two seed identities are separated?

This experiment is designed to answer only that question.

## Exact frozen implementation

Execution must use implementation frozen at:

`ba75e0879f1ea953ff3595fc8abc2f740d130df0`

Exact implementation files:

1. `scripts/train_reason_router_gen5_ainit_rng_causal_intervention.py`
2. `tests/test_reason_router_gen5_ainit_rng_causal_intervention.py`

Implementation validation before freeze:

- pytest: `27 passed`
- static terminal marker:
  `GEN5_AINIT_RNG_CAUSAL_INTERVENTION_STATIC_VERIFY_PASS`

Frozen downloaded implementation SHA256 identities used for validation:

Runner:

`7eb897be9ff70e46ae796989b5787066a16f28f785a785135c6c62411336bd44`

Test:

`eb32019ece853c7965ac1640f56b794a8a539b5b6016c6309a4837db4384bf6e`

No implementation edit is authorized during execution.

If an implementation defect is exposed, stop rather than patching the frozen
implementation in place.

## Exact factorial

Factor levels:

`A_INIT_SEED in {6201,6202,6203}`

`TRAINING_RNG_SEED in {6201,6202,6203}`

The full scientific interpretation uses a 3 × 3 factorial.

The three diagonal cells already exist as frozen Phase3A P0 evidence:

- `A6201-R6201`
- `A6202-R6202`
- `A6203-R6203`

They MUST NOT be rerun.

Exactly six new off-diagonal cells are authorized:

- `A6201-R6202`
- `A6201-R6203`
- `A6202-R6201`
- `A6202-R6203`
- `A6203-R6201`
- `A6203-R6202`

No seventh new training cell is authorized.

## Exact two-GPU topology

Exactly two visible Tesla T4 GPUs are required.

Compute capability:

`(7,5)`

Topology:

`TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`

No DDP, model parallelism, distributed gradient synchronization, shared model,
shared optimizer, or cross-worker training state is allowed.

Exact frozen worker assignment:

### Worker 0 / GPU 0

- `A6201-R6202`
- `A6202-R6203`
- `A6203-R6201`

### Worker 1 / GPU 1

- `A6201-R6203`
- `A6202-R6201`
- `A6203-R6202`

Each worker contains every A-init seed exactly once and every training-RNG seed
exactly once.

Worker assignment must not be changed after outcome inspection.

## A-initialization identity

The A-init seed controls only:

`A_theta.weight`

Exact initialization rule:

`CPU float32 kaiming_uniform_(shape=(2,768), a=sqrt(5), generator=manual_seed(A_INIT_SEED))`

Frozen canonical A-init SHA256 values:

### A-init seed 6201

`6a76f8e0690b8f9c3dab6bebe4ccffab5d61270870fafd2b0cb14a85eba206c0`

### A-init seed 6202

`7be31582805fdd6b04b67b827e320386808973346f28225bcc9bebadf32e4cab`

### A-init seed 6203

`05cc558563b68ff706a45bb7a3603bd5b1077b3dc603a2074ed2f6bc63665dca`

Before every cell:

- live `A_theta.weight` must authenticate against the requested A-init seed;
- `B_theta.weight` must be exact zero;
- parent parameters must remain frozen.

The training RNG seed must not affect reconstruction of `A_theta.weight`.

## Training-RNG identity

The training RNG seed controls the remaining stochastic runtime/training path.

Immediately before step-0 and training stochastic operations:

`torch.manual_seed(TRAINING_RNG_SEED)`

`torch.cuda.manual_seed_all(TRAINING_RNG_SEED)`

No later reseeding from `A_INIT_SEED` is authorized.

The runtime parent construction may be seeded from `TRAINING_RNG_SEED`; the
parent checkpoint itself remains frozen and authenticated.

## Frozen diagonal evidence

The factorial diagonal is reused from frozen Phase3A P0 corrections.

### seed6201

Path:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed6201/P0/final_correction.pt`

SHA256:

`157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf`

Interpretation:

`A_INIT_SEED=6201`

`TRAINING_RNG_SEED=6201`

### seed6202

Path:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed6202/P0/final_correction.pt`

SHA256:

`1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214`

Interpretation:

`A_INIT_SEED=6202`

`TRAINING_RNG_SEED=6202`

### seed6203

Path:

`reports/reason_router_gen5_phase3a_training_runs/gen5-phase3a-contention-qualification-9cell-d58e894-retry3/cells/seed6203/P0/final_correction.pt`

SHA256:

`c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770`

Interpretation:

`A_INIT_SEED=6203`

`TRAINING_RNG_SEED=6203`

These diagonal cells are comparison evidence only and are not execution targets.

## Frozen Phase3A scientific contract

Every new cell must preserve exactly:

- arm: `G5-C0`;
- pressure: `P0`;
- rank: `2`;
- `A_theta` shape: `[2,768]`;
- `B_theta` shape: `[24576,2]`;
- B exact zero initialization;
- trainable tensors: `A_theta.weight`, `B_theta.weight`;
- trainable numel: `50688`;
- train rows: `3360`;
- dev rows: `840`;
- split seed: `16384`;
- frozen parent checkpoint;
- frozen parent parameters;
- exact frozen data and row order;
- exact tokenizer/model/runtime identities;
- final 3-way cross entropy only;
- AdamW;
- learning rate: `0.001`;
- weight decay: `0.0001`;
- gradient clip norm: `5.0`;
- scheduler: none;
- exactly `20` optimizer steps;
- final fixed-step checkpoint;
- no early stopping;
- no checkpoint selection;
- frozen Phase3A dev evaluation only;
- no confirmatory population;
- no scientific p-values.

No other pressure is authorized.

## Parent checkpoint

Exact parent SHA256:

`1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`

Parent parameters must remain frozen.

Parent fingerprint must be identical before and after every cell.

## Data identity

Training rows:

`3360`

Dev rows:

`840`

Split seed:

`16384`

Pressure:

`P0`

Frozen train-order SHA256:

`0be453c8d5d78397e1387f1ec3aac7cc5e83983a9ec0b49107f5b4770d38a71b`

Frozen dev-order SHA256:

`b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25`

Frozen train-encoding SHA256:

`d845a53923db045ed58ad2514e2dbf397b4869c856700db06390f86c6c646f10`

Frozen dev-encoding SHA256:

`e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51`

Confirmatory population `9601..9900` is forbidden.

## Optimization contract

Exactly:

- optimizer: `torch.optim.AdamW`;
- trainable parameters: `A_theta.weight`, `B_theta.weight`;
- trainable numel: `50688`;
- learning rate: `0.001`;
- weight decay: `0.0001`;
- gradient clip norm: `5.0`;
- optimizer steps per new cell: `20`;
- total optimizer steps across six new cells: `120`;
- no scheduler;
- final 3-way CE only;
- no early stopping;
- no task-based checkpoint selection;
- final fixed step only.

No optimizer or hyperparameter sweep is authorized.

## Runtime surface

Execution inherits the frozen Phase3A/Stage-E runtime:

- Python `3.12.13`
- NumPy `2.0.2`
- PyTorch `2.10.0+cu128`
- CUDA runtime `12.8`
- Transformers `5.0.0`
- tokenizers `0.22.2`
- kernels `0.10.2`
- exactly two Tesla T4 GPUs
- compute capability `7.5`
- float32
- autocast disabled

Model snapshot:

`state-spaces/mamba-130m-hf`

Revision:

`40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`

Mamba scientific revision:

`c8ffc584c147878a6eb978ae0e8db4d116c93a8c`

Mamba binary SHA256:

`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

causal-conv1d scientific revision:

`f2651e776f66069cdcf842840db637583def1223`

causal-conv1d binary SHA256:

`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

Runtime or provisioning mismatch is a blocker.

## CUDA preflight

Exactly one preflight is authorized and required before the six-cell matrix:

`A6201-R6202`

The preflight may:

- authenticate exact execution HEAD;
- authenticate implementation freeze;
- authenticate runtime/package/kernel identities;
- authenticate parent checkpoint;
- instantiate the frozen parent model;
- reconstruct and authenticate A-init seed 6201;
- verify B exact-zero initialization;
- set training RNG seed 6202;
- execute one full training-domain forward/backward plumbing pass;
- verify finite A/B correction gradients;
- verify no parent gradients;
- verify no parent mutation;
- verify no optimizer construction or optimizer step;
- verify no task evaluation;
- verify confirmatory data are not loaded.

Required preflight state:

- backward executed: true;
- optimizer constructed: false;
- optimizer step count: 0;
- training executed: false;
- task evaluation executed: false;
- confirmatory 9601..9900 loaded: false;
- scientific p-value count: 0;
- scientific conclusion: null.

Expected terminal marker:

`GEN5_AINIT_RNG_CAUSAL_CUDA_PREFLIGHT_PASS`

The preflight artifact is runtime validation only.

It MUST NOT be collected/imported as scientific evidence.

If preflight fails, stop.

Do not run the scientific matrix and do not collect the failed preflight.

A retry after a genuine implementation/runtime defect requires an explicit
recovery amendment.

## Exact scientific matrix

After preflight PASS, run exactly the six off-diagonal cells through the frozen
two-worker matrix orchestration.

Expected terminal marker:

`GEN5_AINIT_RNG_CAUSAL_SIX_OFFDIAGONAL_MATRIX_PASS`

Required matrix properties:

- new cells: `6`;
- reused diagonal cells: `3`;
- GPU workers: `2`;
- GPU topology: `TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`;
- optimizer steps per cell: `20`;
- total new optimizer steps: `120`;
- pressure: `P0`;
- training executed: true;
- backward executed: true;
- frozen dev task evaluation executed: true;
- confirmatory 9601..9900 loaded: false;
- scientific p-value count: 0;
- scientific conclusion: null.

If any worker or cell fails:

- the run is failed;
- do not collect it;
- do not import it;
- stop for failure diagnosis.

Only a full matrix PASS is eligible for collection.

## Required per-cell artifacts

Each new off-diagonal cell must produce:

- `final_correction.pt`
- `training_report.json`
- `run_provenance.json`

Each must preserve at minimum:

- `a_init_seed`;
- `training_rng_seed`;
- unambiguous cell name;
- A-init SHA256;
- arm `G5-C0`;
- pressure `P0`;
- parent checkpoint identity;
- train/dev identities;
- training losses;
- step-0 loss;
- post-step20 matched-RNG loss;
- gradient norms;
- optimizer identity;
- learning rate;
- weight decay;
- gradient clipping;
- exactly 20 optimizer steps;
- dev final 3-way CE;
- dev accuracy;
- final A hash;
- final B hash;
- parent fingerprint before/after;
- runtime provenance;
- final-correction file hash;
- no confirmatory access;
- no scientific p-value;
- `scientific_conclusion = null`.

Top-level matrix artifacts must include:

- `matrix_summary.json`
- `run_provenance.json`
- worker logs;
- exact two-worker assignment;
- six-cell count;
- total optimizer-step count.

## Prospective analysis endpoints

Scientific interpretation occurs only after successful collect/import and
artifact validation.

### Primary endpoint A — final read-side inheritance

For every off-diagonal cell compare:

`row(A_final)`

against all three canonical A-init planes.

Primary descriptive statistic:

`AFFINITY_TO_MATCHED_A_INIT - MEAN_AFFINITY_TO_OTHER_A_INITS`

### Primary endpoint B — dominant right/input inheritance

For every cell compute the dominant right singular direction of:

`B_final A_final`

and its captured energy in all three A-init planes.

Primary descriptive statistic:

`CAPTURE_IN_MATCHED_A_INIT - MEAN_CAPTURE_IN_OTHER_A_INITS`

### Factor attribution

Assemble the full 3 × 3 factorial from:

- three frozen diagonal cells;
- six new off-diagonal cells.

Compare whether geometry groups more strongly by:

- A-init seed;
- training-RNG seed.

Report raw pairwise geometry and grouped descriptive means.

No post-hoc threshold and no scientific p-value are authorized.

## Prospective interpretation cases

### Case 1 — A-init seed dominates

If final read-side geometry and dominant right/input direction track A-init
seed across changed training RNG seeds:

Support the bounded statement:

`GEN5_READ_SIDE_GEOMETRY_CAUSALLY_TRACKS_A_INITIALIZATION_UNDER_FIXED_PHASE3A_P0_TRAINING`

### Case 2 — training RNG dominates

If geometry tracks training RNG seed more strongly:

The prior static initialization anchoring does not survive causal decoupling.

Shift interpretation toward stochastic optimization-path selection.

### Case 3 — both factors matter

If both produce substantial structured effects:

Interpret final geometry as jointly determined by initialization geometry and
stochastic training trajectory.

### Case 4 — neither dominates cleanly

If off-diagonal solutions reorganize idiosyncratically:

Reject the simple one-factor explanation and retain a more general nonlinear
path-dependence interpretation.

## Collection boundary

Only a full scientific matrix PASS may proceed to collection.

A successful preflight is not collected.

A failed preflight is not collected.

A failed or partial matrix is not collected.

For a successful matrix:

1. preserve exact run directory;
2. `cm run save <run-name>`;
3. `cm run <run-name>`;
4. only after matrix PASS, use `cm collect <run-name>`;
5. run the collector in Kaggle;
6. download the handoff ZIP;
7. import locally with `cm import <handoff.zip>`;
8. validate hashes/provenance before interpretation.

A successful run alone does not establish the scientific conclusion.

## Prohibited expansion

Not authorized:

- rerunning any diagonal cell;
- more than six new cells;
- new seed;
- pressure expansion;
- new layer;
- rank sweep;
- learning-rate sweep;
- weight-decay sweep;
- optimizer sweep;
- horizon/step-count sweep;
- architecture change;
- alternative A initialization distribution;
- nonzero B initialization;
- data or split changes;
- tokenizer changes;
- model changes;
- confirmatory population access;
- scientific p-values;
- post-hoc success threshold;
- implementation edits during execution;
- DDP or cross-GPU synchronization.

## Stop conditions

Stop without scientific interpretation if:

- execution HEAD mismatch;
- implementation-freeze mismatch;
- worktree dirty;
- runtime/package/kernel mismatch;
- parent-checkpoint mismatch;
- A-init hash mismatch;
- B is not exact zero at initialization;
- training-RNG seed is not isolated from A-init construction;
- unsupported cell is scheduled;
- a diagonal cell is scheduled;
- GPU count is not exactly two;
- either GPU is not Tesla T4 capability 7.5;
- DDP or cross-worker synchronization occurs;
- NaN/Inf occurs;
- parent gradient appears;
- parent parameter mutates;
- optimizer-step count differs from 20 per new cell;
- confirmatory data are accessed;
- scientific p-value is computed.

No automatic scientific fallback is authorized.

## Interpretation boundary

Keep separate:

1. code correctness;
2. runtime/execution success;
3. artifact/provenance validity;
4. scientific interpretation.

Scientific interpretation is authorized only after successful collection,
local import, and independent artifact validation.

## Final execution disposition

AUTHORIZED_NEW_CELL_COUNT=6

REUSED_DIAGONAL_CELL_COUNT=3

GPU_COUNT=2

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP

PREFLIGHT_CELL=A6201-R6202

PREFLIGHT_COLLECTION=FORBIDDEN

FAILED_RUN_COLLECTION=FORBIDDEN

PRESSURE=P0

OPTIMIZER=AdamW

LEARNING_RATE=0.001

WEIGHT_DECAY=0.0001

GRADIENT_CLIP_NORM=5.0

OPTIMIZER_STEPS_PER_NEW_CELL=20

TOTAL_NEW_OPTIMIZER_STEPS=120

CONFIRMATORY_ASSAY_ACCESS=FORBIDDEN

SCIENTIFIC_P_VALUE_COUNT=0

STATUS=READY_FOR_EXACT_COMMIT_BOUND_EXECUTION
