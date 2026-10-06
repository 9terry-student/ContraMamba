# Gen5 A-init Temporal Mechanism Scan — Combined Implementation and Conditional Execution Authority

SOURCE_CURRENT_EVIDENCE_FREEZE_COMMIT=07cf87b90fc94bd090692e414995eb9923d8b9ab
SOURCE_TEMPORAL_BIRTH_AUTHORITY_PATH=reports/reason_router_gen5_ainit_representation_freedom_temporal_birth_audit_authority_spec_candidate.md
SOURCE_PHASE_A_TRAJECTORY_PATH=reports/reason_router_gen5_ainit_temporal_birth_replay_runs/gen5-ainit-temporal-birth-phase-a-numerical-auth-d940e19-r1/temporal_birth_trajectory.pt
SOURCE_PHASE_A_TRAJECTORY_SHA256=0f7cd4248faa92223597e0816597b59e426f08829dadd366f9603f56a8de809e
SOURCE_BEHAVIORAL_FREEZE_COMMIT=76dbf99883cd6dd99d50270b2ddbc1540d8c4c5e
SOURCE_FIRST_UPDATE_FREEZE_COMMIT=23c4f5c4d4cd35cabbe286d8897d5360aca73adf
SOURCE_SHARED_VULNERABILITY_FREEZE_COMMIT=07cf87b90fc94bd090692e414995eb9923d8b9ab

STATUS=READY_FOR_TEMPORAL_MECHANISM_SCAN_IMPLEMENTATION_AND_CONDITIONAL_EXECUTION

COMBINED_IMPLEMENTATION_AND_EXECUTION_AUTHORITY=YES_CONDITIONAL
IMPLEMENTATION_ALLOWED=YES_BOUNDED_TEMPORAL_MECHANISM_SCAN_ONLY
SCIENTIFIC_EXECUTION_ALLOWED=YES_ONLY_AFTER_IMPLEMENTATION_VALIDATION_COMMIT_AND_PUSH
TRAINING_ALLOWED=NO
OPTIMIZER_CONSTRUCTION_ALLOWED=NO
OPTIMIZER_STEP_ALLOWED=NO
PARAMETER_UPDATE_ALLOWED=NO
BACKWARD_ALLOWED=NO
ANALYSIS_AUTOGRAD_ALLOWED=YES_DETACHED_INTERNAL_STAGE_LEAVES_ONLY
PARAMETER_GRADIENTS_ALLOWED=NO
CHECKPOINT_MUTATION_ALLOWED=NO
CONFIRMATORY_9601_9900_ALLOWED=NO
NEW_SEEDS_ALLOWED=NO
DATA_OR_SPLIT_CHANGE_ALLOWED=NO
LABEL_CHANGE_ALLOWED=NO
MODEL_ARCHITECTURE_CHANGE_ALLOWED=NO
COMMIT_PUSH_ALLOWED=MANUAL_ONLY

GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP
SCIENTIFIC_GPU_COUNT=2
SCIENTIFIC_GPU_MODEL=Tesla_T4
CROSS_GPU_SCIENTIFIC_TENSOR_REDUCTION=FORBIDDEN

## 1. Purpose

The frozen Gen5 trajectory now establishes all of the following under the exact
Phase3A P0 3x3 A-init x training-RNG grid:

- raw-write/operator geometry is born at the first optimizer update;
- the first update is exactly reconstructible from frozen t=0 gradients and
  AdamW semantics;
- prediction divergence is already present at t=1 on the A-init axis;
- training-RNG behavioral divergence first appears later;
- a common set of 120 rows follows the same discrete
  `SUPPORT -> REFUTE -> NOT_ENTITLED` transient corridor;
- the 120-row onset timing is overwhelmingly A-init controlled;
- prediction behavior permanently reconverges across all factors from t=17
  through t=20 while internal/operator A-init separation remains substantial;
- endpoint evidence already establishes a strong task-reachable quotient,
  downstream residual suppression, a compact task-visible component, a much
  larger downstream-low-gain complement, and internal localization beginning
  at the learned raw-write boundary.

The missing scientific object is the temporal bridge between these facts.

This stage asks four linked questions:

1. **t=1 micro-origin:** inside the first post-update forward computation, at
   which internal boundary does the A-init-dependent write first become
   decision-visible?
2. **transient decision geometry:** what continuous three-logit/two-margin path
   do the frozen 120 shared rows follow through their
   `SUPPORT -> REFUTE -> NOT_ENTITLED` excursion?
3. **reconvergence:** does t=17 prediction reconvergence also correspond to
   continuous logit/margin reconvergence, or only to common argmax decisions?
4. **critical-time propagation:** how does A-init-specific residual magnitude
   propagate through the exact correction stages at the scientifically frozen
   temporal landmarks?

No performance-tuning question is authorized.

## 2. Existing evidence must be reused, not repeated

This stage MUST reuse the already frozen endpoint evidence. It MUST NOT rerun
or recreate endpoint experiments whose questions are already answered,
including:

- task-reachable operator quotient;
- endpoint correction-residual propagation localization;
- endpoint task-sensitive/null-alignment audits;
- endpoint visible-vs-complement causal intervention;
- endpoint internal task-visible precursor localization;
- confirmatory 9601..9900 evaluation.

The new execution is authorized only for temporal linkage of the already frozen
t=0..20 trajectory.

## 3. Frozen source population and factorial grid

Use only the frozen Phase3A P0 dev population:

- dev rows: 840
- split seed: 16384
- arm: G5-C0
- pressure: P0

Use exactly:

`A_INIT_SEED in {6201,6202,6203}`

`TRAINING_RNG_SEED in {6201,6202,6203}`

All nine cells are required.

No new seed, row, checkpoint, split, population, or confirmatory data is
authorized.

The exact t=0..20 A/B snapshots MUST come from the frozen Phase A trajectory
artifact identified above and MUST authenticate its SHA256 before use.

## 4. Frozen shared-vulnerability row set

The 120-row shared vulnerable set is immutable and MUST be read from:

`reports/reason_router_gen5_ainit_shared_vulnerability_discrete_static_23c4f5c_v1.json`

Do not redefine the set from the new logits.

The scan may evaluate all 840 frozen dev rows for authentication and global
trajectory summaries, but every row-specific vulnerability claim MUST refer to
this already frozen 120-row identity.

## 5. Full temporal behavioral coordinates

For every cell and every optimizer snapshot `t=0..20`, evaluate the frozen dev
set and retain compact row-level task coordinates:

- three raw logits;
- three class-centered logits;
- predictions;
- gold labels;
- two independent margins:
  - `m_refute = logit_refute - logit_not_entitled`
  - `m_support = logit_support - logit_not_entitled`.

The row-level tensor artifact is authorized because its bounded size is small.
Do not persist hidden-state trajectories for all rows/times.

For the frozen 120-row set, derive without threshold tuning:

- per-row margin trajectory;
- exact first SUPPORT/REFUTE boundary crossing;
- exact REFUTE/NOT_ENTITLED exit crossing;
- minimum/maximum margin values over the episode;
- duration;
- A-init and training-RNG timing contrasts;
- whether rowwise trajectories are merely time-shifted or also differ in
  continuous path shape.

No post-hoc margin threshold, rank, or row subset may be introduced.

## 6. t=1 within-update micro-decomposition

Use the already frozen exact tensors `A0`, `B0`, `A1`, `B1`.

Evaluate exactly these four parameter states on the frozen dev set:

1. `T0 = (A0, B0)`
2. `A_DECAY_ONLY = (A1, B0)`
3. `B_UPDATE_ONLY = (A0, B1)`
4. `FULL_T1 = (A1, B1)`

These are analysis counterfactuals over frozen tensors. They perform no
optimizer step and no training.

Mandatory authentication:

- `B0 == 0` exactly;
- `A_DECAY_ONLY` correction output must equal `T0` exactly under the G5-C0
  direct-write structure;
- `FULL_T1` must reproduce the frozen t=1 behavioral scan within the existing
  float32/output authentication tolerance.

Report for all four states:

- all-840 logits/margins/predictions;
- frozen-120 row logits/margins/predictions;
- A-axis and R-axis prediction disagreement summaries where defined.

Primary t=1 micro-origin question:

> Is the first behavioral birth already reproduced by the B1-driven write with
> A held at A0, while A1's decay-only change contributes negligibly or not at
> all to the t=1 decision change?

This stage may answer that question only from the explicit frozen
counterfactual evaluations above.

## 7. Critical temporal landmarks

The detailed internal scan is prospectively fixed to:

`CRITICAL_TIMES = {1,2,4,10,11,16,17,20}`

Rationale is frozen from evidence available before this authority:

- t1: first A-init behavioral divergence and geometric birth;
- t2: first training-RNG behavioral divergence;
- t4: first new transient decisive-error entry;
- t10: peak A-init behavioral disagreement;
- t11: peak accuracy spread / near-peak active excursion;
- t16: final pre-reconvergence step;
- t17: start of permanent all-factor prediction reconvergence;
- t20: authenticated endpoint.

Do not add or remove critical times after inspecting new outputs.

`t=0` remains the structural zero-write baseline and is authenticated through
the t=1 micro-decomposition rather than treated as a nonzero propagation stage.

## 8. Critical-time internal stage chain

At each critical time and for each of the nine cells, replay the exact frozen
layer-22 correction chain:

1. `raw_write = B A x`
2. `recurrent_state`
3. `c_readout_pre_gate`
4. `gated_scan`
5. `layer22_out_proj`
6. final logits/margins

Use the same exact frozen implementation semantics as the validated endpoint
correction-residual localization.

The implementation MUST authenticate at t=20 against the already frozen
endpoint residual-localization evidence before interpreting earlier times.

For the first five internal stages, accumulate compact sufficient statistics
only; do not persist full internal tensors.

For every critical time and stage, report:

- same-RNG / different-A cosine;
- same-RNG / different-A normalized residual;
- same-A / different-RNG cosine;
- same-A / different-RNG normalized residual;
- grouped A/R contrast;
- stage-to-stage survival ratio;
- cumulative survival from raw_write.

This is a no-grad geometric propagation analysis.

## 9. Task-visible decomposition at birth and reconvergence landmarks

To directly compare the first behavioral birth with later reconvergence, perform
the already validated detached-leaf true-forward task-visible analysis only at:

`TASK_VISIBLE_TIMES = {1,17,20}`

and at the same five internal stages:

- raw_write
- recurrent_state
- c_readout_pre_gate
- gated_scan
- layer22_out_proj

Use the recovered true-forward `joint` semantics already validated for the
existing internal precursor evidence:

- preserve the exact frozen forward value;
- detach only the selected internal stage value;
- reintroduce it as an equal-valued analysis leaf;
- resume exact frozen downstream computation;
- use `torch.autograd.grad` only for the two margin coordinates;
- never call `.backward()`;
- never request parameter gradients.

Reuse the existing signed-permutation control family and visible/complement
finite-intervention definitions. Do not invent a new control family.

For each task-visible time/stage report, at minimum:

- actual local two-margin row-space energy fraction;
- matched control mean;
- actual/control enrichment;
- finite `R_visible`;
- finite `R_complement`;
- finite `R_interaction`;
- prediction disagreement under visible-only and complement-only
  interventions.

Mandatory t=20 authentication MUST reproduce the already frozen endpoint
internal-precursor values within predeclared tolerances before t1/t17
interpretation.

## 10. Reconvergence analysis

The known discrete fact is:

`PERMANENT_ALL_FACTOR_PREDICTION_RECONVERGENCE_STEP = 17`

This stage MUST NOT redefine that event.

Instead, determine what remains different at and after that event.

For every pair class at t16, t17, and t20 report:

- prediction disagreement;
- centered-logit cosine and normalized residual;
- two-margin cosine and normalized residual;
- rowwise margin-distance distribution for the frozen 120 rows;
- critical-stage residual chain.

Interpretation must distinguish:

### ARGMAX_ONLY_RECONVERGENCE

Predictions agree from t17 but continuous logit/margin separation remains
substantial.

### FUNCTIONAL_RECONVERGENCE_WITH_AMBIENT_RESIDUAL

Continuous task coordinates strongly reconverge while large internal/ambient
A-init residual remains.

### DOWNSTREAM_CANCELLATION_RECONVERGENCE

Substantial task-visible/internal difference remains through layer22_out_proj
but is strongly reduced only in final logits/margins.

### MIXED_RECONVERGENCE

Multiple internal/downstream reductions are required.

No post-hoc threshold is authorized to force one label. If the trajectory does
not cleanly fit one case, report the mixed quantitative chain.

## 11. t=1 precursor interpretation boundary

A t=1 precursor claim is authorized only in this bounded sense:

- the first-update optimizer origin has already been frozen;
- this stage may localize the earliest tested internal forward boundary at which
  the frozen first-update difference becomes task-visible;
- the `A_DECAY_ONLY` / `B_UPDATE_ONLY` counterfactual may identify whether the
  first behavioral birth is attributable to the new B write rather than the
  tiny decay-only A change.

This does NOT authorize a claim about a precursor before the t=0 training
gradient computation, nor about frozen parent layers 0..21.

## 12. Implementation scope

Only these files may be modified:

- `scripts/audit_reason_router_gen5_ainit_temporal_birth.py`
- `tests/test_reason_router_gen5_ainit_temporal_birth.py`

No new production module is authorized unless implementation becomes
impossible without one; if so, stop and report the exact blocker before
creating it.

Existing temporal-birth modes and historical semantics MUST remain unchanged.

Implement the new scan as a new explicit mode/CLI path, not by changing the
meaning of an existing mode.

## 13. Validation before execution

Before scientific execution:

1. `git diff --check`
2. Python compile of the two authorized files
3. the exact temporal-birth test file
4. narrow CPU/static dry-run or schema validation for the new mode with zero
   model forward if implemented
5. `cm ship`
6. manual commit and push of the exact authorized implementation files

Execution MUST be pinned to that implementation-freeze commit.

A validation PASS does not itself authorize any uncommitted execution.

## 14. Scientific execution constraints

Scientific execution may occur only after implementation validation, commit,
and push.

Use two independent Tesla T4 workers with deterministic complete-cell or
complete-pair sharding. Do not split one scientific unit across GPUs. Do not
use DDP. Do not perform cross-GPU scientific tensor reduction.

Required execution behavior:

- `model.eval()`;
- no training;
- no optimizer construction;
- no optimizer step;
- no `.backward()`;
- no parameter gradient accumulation;
- no checkpoint mutation;
- no confirmatory population access;
- no change to row order or frozen encoding;
- no output overwrite;
- fail closed on provenance/authentication mismatch.

## 15. Required run artifacts

Write only under:

`reports/reason_router_gen5_ainit_temporal_mechanism_runs/<run-name>/`

Required files:

- `temporal_mechanism_summary.json`
- `temporal_behavioral_coordinates.pt`
- `temporal_internal_stage_metrics.pt`
- `run_provenance.json`
- deterministic worker logs/manifest as needed

Do not persist full hidden-state trajectories.

The behavioral-coordinate artifact may contain the bounded 21 x 9 x 840 x 3
logit tensor and derived margin/prediction tensors.

## 16. Required source authentication

Before scientific interpretation, authenticate:

- current implementation-freeze commit;
- this authority artifact identity;
- frozen Phase A trajectory SHA256;
- frozen parent checkpoint;
- frozen 3x3 historical checkpoint identities where required for t20 checks;
- frozen dev encoding SHA256;
- frozen dev row-order SHA256;
- frozen 120-row identity;
- t=1 frozen behavioral endpoint;
- t=20 endpoint propagation/internal-precursor targets.

Any mismatch blocks interpretation.

## 17. Stop conditions

Stop without scientific interpretation if:

- authority or implementation identity mismatches;
- Phase A trajectory hash mismatches;
- the 3x3 grid or t=0..20 axis is incomplete;
- frozen dev or 120-row identity mismatches;
- t=1 FULL_T1 authentication fails;
- t=20 endpoint authentication fails;
- an existing historical mode changes semantics;
- training or optimizer construction occurs;
- `.backward()` is called;
- parameter gradients are requested or accumulated;
- checkpoint mutation occurs;
- confirmatory 9601..9900 data is accessed;
- a post-hoc critical time, row subset, margin threshold, basis rank, or control
  family is introduced;
- output collision occurs.

## 18. Scientific result boundary

This scan can establish a bounded temporal mechanism for how the frozen Gen5
A-init-specific first-update difference becomes decision-visible, produces the
shared transient decision excursion, and later becomes behaviorally
task-equivalent despite persistent internal representation differences.

It cannot establish:

- exact global gauge symmetry;
- a universal null manifold;
- a universal Mamba mechanism;
- behavior on natural language or other domains;
- behavior outside the frozen Phase3A P0 contract;
- a precursor before the t=0 optimization event;
- arbitrary latent controllability;
- manuscript novelty or priority.

A later finite factor-swap intervention remains required before claiming that a
specific compact latent variable causally controls excursion timing or
reconvergence.
