# Gen5 A-init Representation-Freedom Temporal Birth Audit Authority

CURRENT_EVIDENCE_FREEZE_COMMIT=d53c33b5a64e02b4f439a1f6b283b07990296bf8
SOURCE_CONFIRMATORY_EVIDENCE_FREEZE_COMMIT=1468938af9753fa9f4a511d4e7f740dea0110bba
SOURCE_FORWARD_JACOBIAN_RECOVERY_EVIDENCE_FREEZE_COMMIT=a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e
SOURCE_INTERNAL_PRECURSOR_AUTHORITY_COMMIT=e7eba19b102016131e4990f724c825cbec49ec5c
SOURCE_INTERNAL_PRECURSOR_CORRECTION_COMMIT=e2c563188e9534e898c6ea944c5a05f4056b2fe3
SOURCE_INTERNAL_PRECURSOR_EVIDENCE_FREEZE_COMMIT=d53c33b5a64e02b4f439a1f6b283b07990296bf8

STATUS=READY_FOR_TEMPORAL_BIRTH_IMPLEMENTATION_AND_CONDITIONAL_EXECUTION

COMBINED_IMPLEMENTATION_AND_EXECUTION_AUTHORITY=YES_CONDITIONAL
IMPLEMENTATION_ALLOWED=YES_EXACT_TEMPORAL_REPLAY_AND_RAW_WRITE_ANALYSIS
SCIENTIFIC_EXECUTION_ALLOWED=YES_ONLY_AFTER_IMPLEMENTATION_VALIDATION_AND_FREEZE
IMPLEMENTATION_FREEZE_POLICY=RUNTIME_EXPECTED_HEAD_MUST_EQUAL_IMPLEMENTATION_FREEZE_COMMIT

TRAINING_ALLOWED=YES_EXACT_HISTORICAL_20_STEP_REPLAY_ONLY
OBJECTIVE_CHANGE_ALLOWED=NO
LEARNING_RATE_CHANGE_ALLOWED=NO
WEIGHT_DECAY_CHANGE_ALLOWED=NO
GRADIENT_CLIP_CHANGE_ALLOWED=NO
STEP_COUNT_CHANGE_ALLOWED=NO
DATA_OR_SPLIT_CHANGE_ALLOWED=NO
LABEL_CHANGE_ALLOWED=NO
CHECKPOINT_SELECTION_CHANGE_ALLOWED=NO
PARENT_PARAMETER_UPDATE_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO

CUDA_PREFLIGHT_ALLOWED=YES_NARROW_SEMANTIC_PREFLIGHT
PREFLIGHT_COLLECTION=FORBIDDEN
FAILED_RUN_COLLECTION=FORBIDDEN
SUCCESSFUL_SCIENTIFIC_RUN_COLLECTION=REQUIRED
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP
SCIENTIFIC_GPU_COUNT=2
SCIENTIFIC_GPU_MODEL=Tesla_T4
CROSS_GPU_SCIENTIFIC_TENSOR_REDUCTION=FORBIDDEN

## 1. Scientific question

The current frozen evidence establishes that A-init-specific representational
non-identifiability is already present at the learned layer-22 raw-write
boundary:

`raw_write = B_theta A_theta x`

with a very small task-visible component carrying most measurable endpoint
functional difference and a much larger component being downstream low-gain.

The correction initialization is structurally special:

- `A_theta: 768 -> 2`, deterministic CPU Kaiming initialization;
- `B_theta: 2 -> 24576`, exact zero initialization;
- no bias;
- rank 2;
- `G5-C0` uses identity projection, so `effective_write == raw_write`.

Therefore at optimizer state `t=0`:

`B_0 = 0`

and raw write is exactly zero for every A-init seed.

This audit asks:

1. at which optimizer update does A-init-specific raw-write geometry first
   become nonzero;
2. whether that early separation is A-init-driven rather than training-RNG
   driven;
3. at which optimizer update the raw-write residual first satisfies the
   already-frozen task-visible / low-gain finite causal decomposition
   criterion.

No performance tuning question is authorized.

## 2. Single-authority structure

This one authority covers two ordered subphases.

### Phase A — exact temporal trajectory replay

Replay the historical 20-step optimization trajectory for the full frozen 3x3
A-init x training-RNG grid while recording only compact A/B/gradient trajectory
artifacts.

Phase A performs training because exact optimizer replay is scientifically
necessary.

### Phase B — frozen raw-write functional-birth scan

After Phase A has been successfully collected, imported, and authenticated,
analyze the frozen Phase A snapshots at the raw-write boundary only.

Phase B performs no optimizer update and no parameter training. It may use
analysis autograd only under the recovered true-forward `joint` semantics and
only on detached analysis leaves, matching the previously frozen internal
precursor contract.

Phase B MUST NOT start if Phase A final-checkpoint authentication fails.

## 3. Authorized implementation scope

The implementation may touch only:

- `scripts/train_reason_router_gen5_ainit_rng_causal_intervention.py`
- `tests/test_reason_router_gen5_ainit_rng_causal_intervention.py`
- `scripts/audit_reason_router_gen5_ainit_temporal_birth.py`
- `tests/test_reason_router_gen5_ainit_temporal_birth.py`

The existing causal-intervention runner may be changed only enough to expose a
low-level full-factorial cell validation/runtime-preparation primitive that
permits the three diagonal cells for the new temporal audit.

Existing historical modes must preserve their exact scientific contract:

- `--cuda-preflight-only` remains off-diagonal only;
- `--run-cell` remains off-diagonal only;
- `--run-matrix` remains the six off-diagonal-cell matrix;
- existing execution-authority validation remains unchanged for those modes.

Do not refactor unrelated code.

Do not modify:

- model architecture;
- dataset files;
- split logic;
- labels;
- frozen checkpoints;
- existing historical run artifacts;
- existing validated evidence reports;
- existing authority semantics.

## 4. Frozen factorial grid

Use exactly:

`A_init in {6201,6202,6203}`

`training_RNG in {6201,6202,6203}`

All nine cells are replayed:

- `A6201-R6201`
- `A6201-R6202`
- `A6201-R6203`
- `A6202-R6201`
- `A6202-R6202`
- `A6202-R6203`
- `A6203-R6201`
- `A6203-R6202`
- `A6203-R6203`

No other seed is authorized.

## 5. Exact 2-GPU scientific sharding

Use both Tesla T4 GPUs.

Use row-major 3x3 cell order and deterministic parity sharding:

GPU0 / worker0:

- `A6201-R6201`
- `A6201-R6203`
- `A6202-R6202`
- `A6203-R6201`
- `A6203-R6203`

GPU1 / worker1:

- `A6201-R6202`
- `A6202-R6201`
- `A6202-R6203`
- `A6203-R6202`

A complete cell trajectory must remain on one GPU.

Do not split one trajectory across GPUs.

Do not use DDP.

Do not perform cross-GPU scientific tensor reduction.

Only compact worker artifacts/sufficient statistics may be merged on CPU after
both workers finish.

## 6. Frozen training recipe

Replay exactly the historical Phase3A P0 recipe:

- train rows: `3360`
- dev rows identity retained but Phase A need not perform task evaluation
- split seed: `16384`
- arm: `G5-C0`
- pressure: `P0`
- optimizer: `torch.optim.AdamW`
- learning rate: `0.001`
- weight decay: `0.0001`
- gradient clip norm: `5.0`
- scheduler: none
- objective: final 3-way cross entropy only
- optimizer steps: exactly `20`
- checkpoint selection: final fixed step only
- train loop order:
  `zero_grad(set_to_none=True) -> forward -> CE -> backward -> clip_grad_norm_ -> optimizer.step()`

The frozen parent parameters must never mutate.

## 7. Fixed temporal axis

Capture all optimizer states:

`t = 0,1,2,...,20`

Definitions:

- `t=0`: immediately before the first training forward/backward/update;
- `t=k`: immediately after optimizer update `k`.

No time point may be added or removed after seeing results.

## 8. Required Phase A capture

For every cell, record at `t=0..20`:

- exact `A_theta.weight` snapshot;
- exact `B_theta.weight` snapshot;
- SHA256 of each tensor;
- parameter norms;
- A displacement from its exact initialized tensor;
- B norm;
- compact rank-2 `BA` operator sufficient statistics;
- nonzero singular values of `BA` or an algebraically equivalent exact
  rank-2 computation;
- training loss for each pre-update step.

For every backward step, record:

- `grad_A` norm before clipping;
- `grad_B` norm before clipping;
- total gradient norm before clipping.

Additionally preserve exact step-0 tensors:

- `grad_A_0`;
- `grad_B_0`.

These tensors are small enough to persist and are required to distinguish the
direct chain-rule prediction from the observed PyTorch/AdamW update behavior.

Do not save full hidden-state trajectories.

Do not save full raw-write tensors.

## 9. Step-0 semantic authentication

The narrow CUDA semantic preflight must authenticate before the main scientific
matrix:

- `B_0` is exact zero;
- `A_0` exactly matches same-process deterministic reconstruction;
- `grad_A_0` is finite and exactly zero under the frozen direct write path;
- `grad_B_0` is finite and nonzero;
- parent parameters receive no gradients;
- clipping executes with the frozen norm;
- one AdamW optimizer step can be observed without nonfinite parameters;
- the observed `A_1` displacement and `B_1` update are recorded.

The preflight is implementation/runtime validation, not scientific evidence.

Do not collect it.

If it fails, fix the implementation/runtime issue before any scientific run.
Do not collect the failed preflight.

## 10. Mandatory final-checkpoint replay authentication

Phase A is scientifically invalid unless every replayed step-20 A/B tensor is
exactly equal to the corresponding frozen historical checkpoint A/B tensor.

Authenticate both:

- `torch.equal` on `A_theta.weight` and `B_theta.weight`;
- exact tensor SHA256 equality.

Also authenticate each frozen checkpoint file before loading it.

Frozen checkpoint file SHA256 values:

- `A6201-R6201`
  `157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf`
- `A6201-R6202`
  `ef03fbbedf3fab8efb255f2ccb6ec33cdb65ee6881a92e6b4d43fe1e719e40d4`
- `A6201-R6203`
  `15582eda034befb1c8d202f04c494f7fd5882bf9fd60ddc963232761057b3df7`
- `A6202-R6201`
  `3c217b43eb164583980bee91c39d53a7a3dc20d181e0341db35ffe8ece162ed3`
- `A6202-R6202`
  `1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214`
- `A6202-R6203`
  `d1c478b448c5f53e2a97552196a454f63f9d9081de614a6d9a5cd4e389e30ee2`
- `A6203-R6201`
  `d9e1062baf554867b212da57c9d30fe8efeccafc77255ba869affac691606359`
- `A6203-R6202`
  `32943df3558ef72eb7f7b7bfb70c6a1a03185a1ea6ca4dd83b9677cd59d13698`
- `A6203-R6203`
  `c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770`

If any replayed final tensor differs, report:

`TEMPORAL_REPLAY_AUTHENTICATION_FAILED`

and stop.

Do not interpret trajectory science.

## 11. Phase A trajectory comparisons

For every `t=0..20`, compute and report compact parameter/operator trajectory
comparisons for:

Primary factor class:

- same training RNG;
- different A-init.

Natural control factor class:

- same A-init;
- different training RNG.

Report at minimum:

- grouped `BA` operator-distance trajectory;
- grouped normalized operator residual trajectory;
- A-main / R-main / A-by-R interaction fractions using the fixed 3x3 design;
- step-0 `grad_B` factor comparisons;
- stepwise A/B displacement trajectories.

These are trajectory diagnostics.

They do not replace Phase B task-space raw-write analysis.

## 12. Phase B population

Use only the already frozen Phase3A P0 development population:

- rows: `840`
- split seed: `16384`
- arm: `G5-C0`
- pressure: `P0`.

The consumed `xg1_fact_9601..xg1_fact_9900` confirmatory population is
permanently forbidden.

No new population may be opened.

## 13. Phase B raw-write geometric birth

Analyze only:

`raw_write = B_theta A_theta x`

No recurrence/C-readout/gate/out-projection temporal repetition is authorized;
the spatial precursor has already been frozen at `raw_write`.

For every `t=0..20`, compute actual frozen-dev raw-write residual statistics for
the complete 3x3 grid.

The raw-write input is common across cells for a fixed dev example because the
parent and upstream computation are frozen.

The implementation may exploit this algebraically with compact input covariance
or equivalent sufficient statistics instead of persisting full raw-write
tensors.

Define geometric birth as the earliest post-update `t` for which the grouped
same-RNG/different-A raw-write squared residual is strictly positive under the
fixed deterministic numerical computation.

Hard guard:

- `t=0` raw-write residual must be exactly zero.

Report the full `t=0..20` residual trajectory regardless of the birth step.

## 14. Phase B factor birth

At every `t=0..20`, compare:

- same RNG / different A;
- same A / different RNG.

Report:

- grouped normalized raw-write residuals;
- A/R residual ratio when defined;
- factorial A, R, and A-by-R energy fractions.

Do not tune a factor-dominance threshold after observing the trajectory.

## 15. Phase B true-forward task-visible analysis

Use the recovered true-forward `joint` analysis-gradient semantics established
by:

`a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e`

Historical edge-specific backward semantics are prohibited for scientific
Jacobian interpretation.

At the selected raw-write snapshot:

1. preserve the exact frozen forward value;
2. detach raw write;
3. reintroduce it as an equal-valued analysis leaf;
4. resume exact frozen downstream computation;
5. compute the two nontrivial class-margin gradient rows using
   `torch.autograd.grad`;
6. never request parameter gradients;
7. never call `.backward()` in Phase B.

Margins:

`m_refute = logit_refute - logit_not_entitled`

`m_support = logit_support - logit_not_entitled`

Use the exact local two-row Euclidean projector:

`P d = J^T (J J^T)^+ J d`

with deterministic float64 2x2 Gram/pseudoinverse calculation.

## 16. Fixed raw-write controls and finite intervention gate

Reuse the exact raw-write signed-permutation control family from the frozen
internal precursor audit:

`SHA256("GEN5_INTERNAL_PRECURSOR_V1|raw_write|<control_index>")`

for exactly eight control indices.

Do not create a new control family.

For actual same-RNG/different-A residuals, compute:

- local task-row-space energy fraction;
- actual/control enrichment;
- visible-only finite intervention;
- complement-only finite intervention;
- interaction residual;
- centered-logit effect ratios;
- two-margin effect ratios.

The fixed functional-freedom gate is unchanged:

- `0.60 <= R_visible <= 1.40`
- `R_complement <= 0.05`
- `R_interaction <= 0.05`
- `E_task_actual / E_task_control_mean >= 5`

No threshold may change after execution.

## 17. Phase B endpoint authentication

Before interpreting an earlier temporal step, the Phase B implementation must
reproduce the already frozen step-20 raw-write result within predeclared
float32/aggregation tolerances.

Frozen step-20 raw-write targets include:

- grouped normalized residual: `0.45922708704`
- local task-row-space squared-energy fraction:
  `0.000323695863574`
- signed-permutation control mean:
  `0.0000263172556629`
- enrichment:
  `12.2997575325`

Centered logits:

- `R_visible = 0.89483541376`
- `R_complement = 0.00438080799597`
- `R_interaction = 0.00840346338907`

Two margins:

- `R_visible = 0.894958000002`
- `R_complement = 0.00440660190005`
- `R_interaction = 0.00843951843385`

If step-20 authentication fails, stop before temporal functional interpretation.

## 18. Functional-freedom birth scan

`t=0` is structurally no-residual and is not eligible for the finite-ratio gate.

Starting at the already computed geometric-birth step, evaluate temporal
snapshots in strictly increasing step order.

The functional-freedom birth step is the first step satisfying the complete
fixed raw-write finite causal gate in both centered-logit and two-margin
coordinates.

A predeclared sequential stopping rule is authorized:

- test `t = geometric_birth_step`;
- if it fails, test the next integer step;
- continue in order;
- stop at the first passing step;
- if no step through `20` passes, report
  `NO_FUNCTIONAL_FREEDOM_BIRTH_LOCALIZED`.

No step may be skipped.

No earlier failure may be discarded.

No alternative threshold, pair subset, orientation, or control family may be
introduced as rescue.

## 19. Interpretation cases

### IMMEDIATE_FIRST_UPDATE_BIRTH

If:

- `t=0` raw write is exactly zero;
- step-0 `grad_A` is exactly zero;
- step-0 `grad_B` is nonzero and A-init dependent;
- geometric birth occurs at `t=1`;
- the fixed functional gate also first passes at `t=1`;

then the bounded mechanism claim may state that A-init representational freedom
is born at the first optimizer update and is already functionally
non-identifiable at its first observed birth.

### GEOMETRY_FIRST_FUNCTION_LATER

If geometry is born at `t=1` but the functional gate first passes later, the
interpretation is that A-init-specific variation begins immediately while
optimization subsequently organizes it into the frozen visible/low-gain
decomposition.

### DELAYED_GEOMETRIC_BIRTH

If geometric birth occurs after `t=1`, the first optimizer update is not
sufficient under the exact replay.

### EARLY_RNG_DOMINANCE

If training-RNG variation dominates the early trajectory, endpoint A-init
dominance is a later-emergent phenomenon.

### REPLAY_AUTHENTICATION_FAILURE

Any step-20 frozen-checkpoint mismatch blocks all temporal scientific
interpretation.

## 20. Required outputs

Phase A outputs only under:

`reports/reason_router_gen5_ainit_temporal_birth_replay_runs/<run-name>/`

Required top-level artifacts:

- `temporal_birth_replay_summary.json`
- `temporal_birth_trajectory.pt`
- `run_provenance.json`
- deterministic worker logs/manifest as needed for auditability.

The trajectory artifact may contain exact A/B snapshots and exact step-0
gradient tensors because their total size is bounded.

Do not persist full hidden-state or raw-write trajectories.

Phase B outputs only under:

`reports/reason_router_gen5_ainit_temporal_birth_analysis_runs/<run-name>/`

Required artifacts:

- `temporal_birth_analysis_summary.json`
- `temporal_birth_metrics.pt`
- `run_provenance.json`.

## 21. Collection policy

Preflight:

`DO_NOT_COLLECT`

Failed or partial scientific run:

`DO_NOT_COLLECT`

Successful Phase A scientific run:

`COLLECT_AND_IMPORT_REQUIRED`

Successful Phase B scientific run:

`COLLECT_AND_IMPORT_REQUIRED`

`cm` being technically capable of packaging a nonzero exit code does not
authorize collecting it in this stage.

## 22. Provenance requirements

Every scientific run must bind:

- full 40-character git commit;
- exact command SHA256;
- exact authority blob;
- exact implementation freeze commit;
- frozen parent checkpoint;
- model/tokenizer snapshot;
- runtime versions;
- 2x Tesla T4 identity for scientific runs;
- deterministic cell/pair sharding;
- artifact SHA256;
- run log SHA256;
- run meta SHA256;
- handoff ZIP SHA256 after collection/import.

Any mismatch fails closed.

## 23. Stop conditions

Stop before scientific interpretation if any of the following occurs:

- authority identity mismatch;
- implementation freeze is not the exact execution HEAD;
- worktree is dirty at execution;
- parent checkpoint or frozen snapshot mismatch;
- train/dev identity mismatch;
- confirmatory 9601..9900 is accessed;
- GPU topology differs from the authorized scientific topology;
- a cell trajectory is split across GPUs;
- parent parameters mutate;
- optimizer/objective/LR/weight decay/clip/step count differs;
- step-0 semantic preflight fails;
- any step-20 replay A/B tensor differs from its frozen checkpoint;
- Phase B step-20 raw-write authentication fails;
- parameter gradients are accumulated during Phase B;
- `.backward()` is called during Phase B;
- output collision occurs.

## 24. Scientific boundary

This authority can establish a bounded optimization-trajectory mechanism for
the frozen Gen5 layer-22 rank-2 correction under the frozen Phase3A P0 task.

It does not establish:

- exact gauge symmetry;
- a global gauge group;
- a universal Mamba law;
- arbitrary perturbation invariance;
- a global null manifold;
- behavior for other models/tasks/domains;
- Mamba superiority;
- Transformer inferiority.

No manuscript priority claim is authorized by this stage.
