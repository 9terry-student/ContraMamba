# Stage196-B2-B6P9-P1 Teacher State Observability Design Spec

This is a static, source-backed design stage.  It does not select a teacher, implement a loss, choose a coefficient, authorize direction/order sharing, or authorize a training intervention.

## Upstream Authority

Use only:

`reports/stage196b2b6p9p0_stability_teacher_target_authority_20260724_100036`

Required CLI inputs:

- `--stage196b2b6p9p0-analysis-json`
- `--stage196b2b6p9p0-teacher-candidate-audit-csv`
- `--stage196b2b6p9p0-state-lifecycle-audit-csv`
- `--repo-root`
- `--output-dir`

The analyzer requires P9-P0 to have decision `STAGE196B2B6P9P0_NO_JUSTIFIED_TEACHER`, recommended next stage `STAGE196B2B6P9P1_TEACHER_STATE_OBSERVABILITY_DESIGN`, no blockers, `failure = null`, passing contracts, and the exact six-candidate audit set.

Terminal exclusions are preserved without reinterpretation:

- `CURRENT_NATIVE_STOP_GRAD = INVALID_ALGEBRAIC_IDENTITY`
- `FRAME_LOCAL_ONLY_DONOR_STOP_GRAD = BLOCKED_DONOR_ARM_IMITATION_RISK`
- `FIXED_RECONSTRUCTED_CHECKPOINT = DIAGNOSTIC_ONLY_NOT_PORTABLE`

## Design Candidates

P9-P1 audits only:

- `PREVIOUS_STEP_FROZEN_SNAPSHOT`
- `PREVIOUS_EPOCH_FROZEN_SNAPSHOT`
- `EMA_STUDENT_TEACHER`

Each candidate is evaluated separately for `direction` and `candidate_order`.  Shared read-only observer infrastructure is allowed only when direction metrics, order metrics, target counts, and gate control remain separate, losses remain absent, and no shared state is treated as teacher approval.

## Source Insertion Points

The source-backed lifecycle anchors are:

- `scripts/train_controlled_v6b_minimal.py::main::<locals>.run_training_v6b`
- `scripts/train_controlled_v6b_minimal.py::_make_cuda_grad_scaler`
- `scripts/train_controlled_v6b_minimal.py::_save_model_checkpoint`
- `scripts/train_controlled_v6b_minimal.py::_save_stage160_checkpoint`
- `scripts/train_controlled_v6b_minimal.py::_save_stage176a0_selected_checkpoint`
- `scripts/train_controlled_v5.py::build_optimizer`
- `src/contramamba/modeling_v6b_minimal.py::ContraMambaV6BMinimal.forward`

No current trainer hook owns teacher state.  Future instrumentation therefore needs explicit default-off trainer-local observer hooks.

## Candidate Boundaries

`PREVIOUS_STEP_FROZEN_SNAPSHOT` is initialized before the first batch from the initialized student.  Its first snapshot equals the initialized student.  It reads as the immediately preceding successful optimizer-step snapshot and updates only after a successful optimizer update.  Skipped or overflowed steps do not advance it.

`PREVIOUS_EPOCH_FROZEN_SNAPSHOT` uses the authoritative boundary `start-of-current-epoch`.  Epoch 1 starts from the initialized student.  For later epochs, the snapshot equals the end of the previous epoch.  It does not use best-so-far, selected checkpoint, or clean-dev performance as teacher authority.

`EMA_STUDENT_TEACHER` is trainer-owned, initialized from the student, updated after successful optimizer updates, and audited without choosing a decay.  Decay ownership, representation, update formula, warm-up, effective age, buffer handling, integer buffers, serialization, resume, and drift metrics are required in future instrumentation.

## Baseline Invariance

Future observability must be default-off.  Disabled mode requires no extra model copy, no teacher allocation, no extra forward, no altered RNG or dropout sequence, no checkpoint schema change, no optimizer or scheduler change, no logit or prediction change, no model mutation, and no trainer control-flow change except inert argument parsing.

## Required Outputs

The analyzer writes exactly:

1. `stage196b2b6p9p1_analysis.json`
2. `stage196b2b6p9p1_report.md`
3. `stage196b2b6p9p1_candidate_lifecycle_matrix.csv`
4. `stage196b2b6p9p1_update_timing_design.csv`
5. `stage196b2b6p9p1_serialization_resume_audit.csv`
6. `stage196b2b6p9p1_determinism_stochastic_audit.csv`
7. `stage196b2b6p9p1_observability_schema.csv`
8. `stage196b2b6p9p1_decision_gate.csv`
9. `stage196b2b6p9p1_contract.csv`

No additional output files are allowed.

## Decision Semantics

The decision hierarchy is:

- `STAGE196B2B6P9P1_BLOCKED_UPSTREAM_AUTHORITY`
- `STAGE196B2B6P9P1_NO_OBSERVABILITY_DESIGN_READY`
- `STAGE196B2B6P9P1_PREVIOUS_STEP_ONLY_READY`
- `STAGE196B2B6P9P1_PREVIOUS_EPOCH_ONLY_READY`
- `STAGE196B2B6P9P1_EMA_ONLY_READY`
- `STAGE196B2B6P9P1_MULTIPLE_CANDIDATES_READY`

`MULTIPLE_CANDIDATES_READY` means multiple lifecycle designs are explicit enough for separate instrumentation experiments.  It does not authorize combining them.

## Scientific Interpretation

P9-P1 may grant lifecycle design readiness.  It does not grant instrumentation implementation readiness, teacher suitability, teacher authorization, or intervention authorization.
