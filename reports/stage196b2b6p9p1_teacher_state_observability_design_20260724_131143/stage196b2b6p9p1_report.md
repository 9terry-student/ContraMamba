# Stage196-B2-B6P9-P1: Teacher State Observability Design

## Decision

`STAGE196B2B6P9P1_MULTIPLE_CANDIDATES_READY`

Recommended next stage: `STAGE196B2B6P9P2_SEPARATE_OBSERVABILITY_INSTRUMENTATIONS`.

This grants lifecycle design readiness only.  It does not select a teacher, authorize teacher targets, implement a loss, choose coefficients, combine direction/order interventions, or approve any training intervention.

## Upstream Authority

Used only the Git-preserved final P9-P0 authority directory:

`reports/stage196b2b6p9p0_stability_teacher_target_authority_20260724_100036`

P9-P0 decision, zero blockers, zero failed contracts, exact six-candidate audit set, and terminal exclusions were required before any P9-P1 design row could be ready.

## Source-Backed Insertion Points

- `scripts/train_controlled_v6b_minimal.py::main::<locals>.run_training_v6b`: trainer-local owner, initialization, epoch loop, optimizer-step boundary, eval/no_grad read location, and future checkpoint metadata integration point.
- `scripts/train_controlled_v6b_minimal.py::_save_model_checkpoint`, `_save_stage160_checkpoint`, `_save_stage176a0_selected_checkpoint`: current checkpoint payloads save `model_state_dict` and metadata; no teacher state key exists today.
- `scripts/train_controlled_v5.py::build_optimizer`: optimizer is `AdamW`; P9-P1 found no scheduler directly used by `run_training_v6b`.
- `src/contramamba/modeling_v6b_minimal.py::ContraMambaV6BMinimal.forward`: `output["logits"]` is final logits; teacher state is not model-owned today.

## Candidate Readiness

{"EMA_STUDENT_TEACHER":{"authorizes_loss":false,"authorizes_teacher":false,"candidate_order":"OBSERVABILITY_DESIGN_READY","direction":"OBSERVABILITY_DESIGN_READY"},"PREVIOUS_EPOCH_FROZEN_SNAPSHOT":{"authorizes_loss":false,"authorizes_teacher":false,"candidate_order":"OBSERVABILITY_DESIGN_READY","direction":"OBSERVABILITY_DESIGN_READY"},"PREVIOUS_STEP_FROZEN_SNAPSHOT":{"authorizes_loss":false,"authorizes_teacher":false,"candidate_order":"OBSERVABILITY_DESIGN_READY","direction":"OBSERVABILITY_DESIGN_READY"}}

Each candidate was audited separately for `direction` and `candidate_order`.  Shared read-only observer infrastructure is allowed only if metrics, target counts, and gate control remain separate and no family can modify the other.

## Lifecycle Designs

`PREVIOUS_STEP_FROZEN_SNAPSHOT`: initialized before the first batch from the initialized student; read as the immediately previous successful optimizer-step snapshot; updated only after successful optimizer updates.  Skipped/overflowed steps do not advance it.  This is near-identity-prone, so target-count degeneracy and drift metrics are mandatory.

`PREVIOUS_EPOCH_FROZEN_SNAPSHOT`: boundary is explicitly `start-of-current-epoch`.  Epoch 1 uses the initialized student snapshot.  For epoch N>1, the start-of-current-epoch snapshot equals the end-of-previous-epoch student state.  It must not use best-so-far, selected checkpoint, or clean-dev performance as teacher authority.

`EMA_STUDENT_TEACHER`: trainer-owned EMA state initialized from the student.  Decay ownership, representation, update formula, warm-up, age, drift, buffers, integer buffers, and resume metadata are observable design requirements, but P9-P1 does not select a decay and does not infer EMA superiority.

## Serialization And Resume

Enabled future instrumentation requires a `teacher_state.*` subtree in checkpoints and fail-closed resume validation when teacher state is missing.  Disabled baseline requires no checkpoint schema change.  The current trainer has no general optimizer/scheduler/teacher resume lifecycle, so future implementation needs explicit hooks rather than implicit trainer hooks.

## Determinism

Disabled instrumentation must consume no RNG and leave dropout sequence unchanged.  Enabled teacher reads must run in eval/no_grad mode and must not perturb student RNG state.  GradScaler overflow/skipped optimizer updates must not advance previous-step or EMA state.

## Observability Metrics

The schema defines common state metrics, direction metrics, and candidate-order metrics for each candidate-family row.  All metrics are observational only; P9-P1 implements no loss.

## Scientific Interpretation

Lifecycle design readiness means the state owner, boundaries, copying, gradient isolation, mode policy, serialization, resume, determinism, warm-up, exact ties, drift, target counts, baseline guard, and source insertion points are explicit enough for a future instrumentation-only stage.

Instrumentation implementation readiness is narrower and belongs to P9-P2.

Teacher suitability remains unproven.

Teacher authorization remains absent.

Intervention authorization remains absent.
