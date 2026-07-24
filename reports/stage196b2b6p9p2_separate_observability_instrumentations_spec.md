# Stage196-B2-B6P9-P2 Separate Teacher-State Observability Instrumentations

## Upstream Authority

This implementation uses the Git-preserved final P9-P1 authority at `reports/stage196b2b6p9p1_teacher_state_observability_design_20260724_131143` and the final P9-P0 target definitions:

- `stage196b2b6p9p0_direction_target_design.csv`
- `stage196b2b6p9p0_order_target_design.csv`

P9-P1 must remain exactly:

- `decision = STAGE196B2B6P9P1_MULTIPLE_CANDIDATES_READY`
- `recommended_next_stage = STAGE196B2B6P9P2_SEPARATE_OBSERVABILITY_INSTRUMENTATIONS`
- `blocking_reasons = []`
- `failure = null`
- `failed_contract_count = 0`

Runtime-only P4-P8 artifacts are not required. P8/P9-P0 candidate identities and semantics are consumed through the existing replay API and the exact opaque candidate order; no lexical candidate order is derived.

## Public CLI

The trainer exposes three default-off observer arguments:

- `--teacher-observer-mode {off,previous_step,previous_epoch,ema}`, default `off`
- `--teacher-observer-target-family {none,direction,candidate_order}`, default `none`
- `--teacher-observer-ema-decay`, no default, required only for `ema`, forbidden otherwise, and validated as `0 < decay < 1`

Validation is fail-closed:

- `mode=off` requires `target_family=none`
- enabled modes require exactly one target family in `{direction,candidate_order}`
- there is no `both` target-family mode
- enabled observer modes require `architecture=v6b_minimal` because the authorized exact counterfactual replay path is v6B/P8

## Observer Ownership

The observer is trainer-owned and lives in `src/contramamba/teacher_state_observer.py`. It is not a model component and not an optimizer component. When disabled, `build_teacher_observer(...)` returns `None` before allocating a model copy or teacher state.

When enabled, the observer owns exactly one frozen teacher module for the selected mode. Its parameters have `requires_grad=false`, it is never passed to `v5.build_optimizer`, and all reads are performed under `torch.no_grad()` with eval-mode teacher reads. Student train/eval mode and CPU/CUDA RNG states are captured and restored around observational reads and observational student replay.

## Candidate And Target Semantics

Direction runs observe teacher counterfactual deltas. Exact zero teacher deltas are counted as ties and excluded from active direction targets.

Candidate-order runs observe teacher pair gaps over the exact opaque candidates. Exact pair ties are counted and excluded from active order targets.

Teacher signs/pairs and teacher outputs are stop-gradient. Student quantities come from `LIVE_FORWARD_REUSE` when the exact P8 replay state is present on the live forward; otherwise the observer performs `OBSERVATIONAL_STUDENT_REPLAY` under no-grad with RNG and mode restoration. The source and student/teacher forward counts are recorded.

## Lifecycle

`previous_step` initializes from the initialized/restored student before the first batch. Before observation, it reads the snapshot from the immediately preceding successful optimizer step. It updates only after an optimizer step is applied. AMP GradScaler overflow skips are detected from the scale transition, skipped steps increment `skipped_step_count`, and teacher state does not advance.

`previous_epoch` uses `start_of_current_epoch`. Epoch 1 starts from the initialized student. During epoch N, reads use the snapshot captured at the start of epoch N. After epoch N train/eval/report work and immediately before epoch N+1, the observer copies the actual current student state. It never uses best, selected, or best-so-far checkpoint state.

`ema` initializes equal to the initialized/restored student. The decay is supplied only by explicit CLI; the implementation does not choose, prefer, or recommend a decay. After successful optimizer steps, floating tensors update as `ema = decay * ema + (1 - decay) * student`; integer and non-floating buffers copy the student exactly. Stable state-dict key ordering is used.

## Sidecars

Enabled runs write exactly five files under one observer directory:

1. `teacher_observer_manifest.json`
2. `teacher_observer_batch_metrics.jsonl`
3. `teacher_observer_epoch_metrics.csv`
4. `teacher_observer_run_summary.json`
5. `teacher_observer_state_audit.json`

Disabled runs write none of these files.

## Checkpoints And Resume

When enabled, checkpoint payloads may add exactly one namespaced subtree: `teacher_observer_state`. It contains schema version, mode, target family, teacher state, counters, boundary metadata, and EMA decay when applicable. When disabled, no `teacher_observer_state` key is added and the existing checkpoint schema is unchanged.

The observer exposes fail-closed restore validation for enabled resume: schema, mode, target family, key coverage, shapes, dtypes, decay, boundary metadata, and non-negative update counters must match. Missing enabled observer state is not silently reinitialized.

## Loss And Gradient Policy

The stage is observational only. It adds no loss, no coefficient, no teacher selection, no EMA decay selection, no clean-dev labels, no recovery/harm labels, no checkpoint-selection change, no prediction change, no total-loss change, and no backward change.

Nonzero loss and nonzero gradient fields are emitted as unavailable with:

```json
{"available": false, "reason": "NO_LOSS_OR_BACKWARD_IN_OBSERVATIONAL_STAGE"}
```

Numeric zero is not used to fabricate absent loss or gradient counts.

## Analyzer Contract

`scripts/analyze_stage196b2b6p9p2_separate_observability_instrumentations.py` is a static source analyzer. It does not import or execute the trainer and writes exactly nine outputs:

1. `stage196b2b6p9p2_analysis.json`
2. `stage196b2b6p9p2_report.md`
3. `stage196b2b6p9p2_implementation_surface.csv`
4. `stage196b2b6p9p2_cli_contract.csv`
5. `stage196b2b6p9p2_lifecycle_hook_audit.csv`
6. `stage196b2b6p9p2_baseline_invariance_audit.csv`
7. `stage196b2b6p9p2_checkpoint_schema_audit.csv`
8. `stage196b2b6p9p2_decision_gate.csv`
9. `stage196b2b6p9p2_contract.csv`

The final decision hierarchy and recommended next stages are exactly those specified for P9-P2. `STAGE196B2B6P9P2_SEPARATE_IMPLEMENTATIONS_READY` requires all three candidates to be implemented, mutually exclusive, and independently runnable for direction and candidate-order. It does not authorize teacher use in a loss.
