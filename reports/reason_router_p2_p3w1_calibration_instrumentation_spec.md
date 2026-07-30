# Reason Router P3-W1 Calibration-Only Instrumentation Spec

## Authority Commits

- P3W0_SPEC_COMMIT: `efb18e0660e9c886e2c64c0557282313431383d0`
- P3_A0_EXECUTION_COMMIT: `b118a984c749ca70bad63919f95aad918881a29d`
- P3-W0 final decision: `P3W0_BLOCKED_BY_INSUFFICIENT_SCALE_AUTHORITY`

## Files Changed

- `scripts/train_controlled_v6b_minimal.py`
- `scripts/aggregate_reason_router_p3w1_calibration.py`
- `tests/test_reason_router_p3w1_calibration.py`
- `reports/reason_router_p2_p3w1_calibration_instrumentation_spec.md`
- `reports/reason_router_p2_p3w1_calibration_instrumentation_manifest.json`

## CLI

Trainer opt-in flags:

- `--reason-router-weight-calibration-export PATH`
- `--reason-router-weight-calibration-execution-commit FULL_SHA`
- `--reason-router-weight-calibration-forward-batch-size 8`

Aggregator CLI:

- exactly three `--unit-json PATH` arguments
- `--output-json PATH`
- `--expected-execution-commit FULL_SHA`
- `--expected-dataset-sha256 SHA256`
- `--expected-sidecar-semantic-sha256 SHA256`
- `--expected-split-seed 174`
- `--expected-ordered-train-row-count INTEGER`
- `--expected-ordered-train-row-identity-hash SHA256`
- `--expected-dev-ratio 0.2`

## Calibration-Only Gate

The trainer validates calibration-only invocation with A3, resolved `conditional_first_blocker`, resolved `explicit_local`, `reason_loss_weight=0.0`, and the frozen P3 factorial configuration: `architecture=v6b_minimal`, `backbone=mamba`, `model_name=state-spaces/mamba-130m-hf`, `max_length=128`, `device=cuda`, `flag_source=controlled_heuristic`, frozen encoder, `reason_router_epsilon=1e-8`, no train batch size, no balanced sampler, no weighted label loss, no class weighting, no normal report/prediction/checkpoint output, no checkpoint load path, no external/OOD evaluation, no bridge data, no temporal/predicate comparator, and P2 objective neutralization preserved.

The calibration export path must not already exist. The declared commit is compared with observed Git HEAD before artifact write, and both values must be exactly matching 40-hexadecimal full SHAs.

## Positive-Weight Exception Scope

The existing A1/A3 positive reason weight requirement is bypassed only when all of the following are true:

- calibration export flag is present
- arm is A3
- `reason_loss_weight` is exactly `0.0`

All other P2 fail-fast and objective-neutralization gates remain active. A3 with `reason_loss_weight=0.0` and no calibration export remains rejected.

## No-Step Control Flow

The trainer branches to `_p3w1_run_reason_weight_calibration_unit(...)` after data, split, sidecar supervision, model, inputs, and local ownership setup are prepared, and before checkpoint loading, optimizer creation, training, dev forward, reports, predictions, external/OOD evaluation, or checkpoint writes.

The calibration function uses `model.train()` with `torch.no_grad()`. The complete train split is one logical calibration unit computed through deterministic microbatches of 8; it computes scalar means over the complete authoritative train split, writes the unit artifact, and returns immediately.

## Unit Artifact Schema

Schema: `reason_router_p3w1_calibration_unit_v1`.

The unit artifact records seed, unit identity, ordered train identity, fixed measurement configuration, frozen P3 model/input configuration, train-only calibration data scope, train supervision built status, dev/A0 isolation booleans, no-step booleans, zero parameter update counters, final 3-way CE mean/count/reconstructed sum, primary reason CE mean/count/reconstructed sum, dataset and sidecar identity, split seed/dev ratio, declared and observed execution commit verification, and decision `P3W1_CALIBRATION_UNIT_PASS`.

## Aggregate Schema

Schema: `reason_router_p3w1_calibration_aggregate_v1`.

The aggregate validator enforces three PASS unit artifacts, exact seeds `[180, 181, 182]`, no duplicate seed, frozen P3 model/input configuration in every unit, train-only/dev-absent/A0-unused isolation fields in every unit, expected split seed independently fixed to `174`, matching execution/data/sidecar/split identity, matching ordered train row identity, finite positive losses, count validity, and reconstructed-sum tolerance.

## Pooled Estimator

The aggregate computes pooled count-weighted estimates:

- `mu_final = sum(final_loss_sum_reconstructed) / sum(final_applicable_count)`
- `mu_reason = sum(reason_loss_sum_reconstructed) / sum(reason_eligible_count)`
- `resolved_reason_loss_weight = mu_final / mu_reason`

It does not use seed mean averages, seed ratio averages, mean-of-means, or seed-specific weights.

## Tests Added

`tests/test_reason_router_p3w1_calibration.py` adds pure unit tests for positive-weight gate behavior, calibration-only input gate failures, ordered train hash determinism, unit validation, aggregate seed/identity validation, pooled estimator behavior, and overwrite rejection.

## Remaining Blockers

- `MISSING_CALIBRATION_ONLY_INSTRUMENTATION` is addressed by this static implementation and remains pending static review.
- `UNRESOLVED_REASON_LOSS_WEIGHT` remains unresolved until actual calibration unit artifacts and aggregate review are completed.
- `UNRESOLVED_P2_IMPLEMENTATION_TESTED_COMMIT` remains unresolved because this step did not execute validation.

## Non-Executed Validation

Per instruction, no Python, pytest, py_compile, help, model load, training, calibration, aggregation, evaluation, Kaggle, git add, commit, push, or pull was executed.

## Static Review Repair Notes

P3-W1 initial static review result `P3W1_FINAL_REVIEW_BLOCKED` was addressed with the following static-only repairs:

- fatal function-boundary syntax defect corrected at the P3-W1 helper to P2 A0 reference validator boundary
- existing P2 objective gate preserved by moving the `use_intervention_loss=true` calibration test expectation to `_p2_resolve_arm_contract()` rejection
- unit schema validator made strict with required-field checks, exact int checks, exact bool checks, non-empty strings, and finite `dev_ratio` validation
- atomic no-overwrite publication hardened in trainer and aggregate writers using temporary file plus `os.link()` instead of `os.replace()`
- calibration CLI flags coupled so export-only and execution-commit-only invocations fail fast
## Static Review V2 Repair Notes

P3-W1 static review v2 result `P3W1_FINAL_REVIEW_V2_BLOCKED` was addressed with static-only repairs:

- sidecar semantic SHA changed from declared copy to observed verification using the production semantic sidecar SHA computation already performed by the P2 sidecar loader
- execution commit changed from declared copy to observed HEAD verification through `_p3w1_observed_git_head(ROOT)` before artifact write
- calibration seed/split gate completed with seed in `{180,181,182}`, resolved split seed `174`, and `dev_ratio == 0.2`
- logical unit separated from deterministic computational microbatching: `calibration_forward_batch_size = 8`, `logical_units_per_seed = 1`, and `logical_unit_scope = COMPLETE_AUTHORITATIVE_TRAIN_SPLIT`
- aggregate now checks external authoritative train identity using required expected row count, row identity hash, and expected dev ratio CLI values

Identity authority is recorded as declared identity, observed identity, and verified equality. CLI expected values are not described as observed values. Cross-seed equality remains a consistency check only; it is not treated as authoritative train-universe authority.

Computational microbatching does not create multiple calibration units. The dropout surface is defined by fixed batch size 8, fixed row order, and fixed seed. Mean/count reconstruction remains over the complete authoritative train split.
## Static Review V3 Repair Notes

P3-W1 static review v3 result `P3W1_FINAL_REVIEW_V3_BLOCKED` was addressed with static-only repairs:

- P3 factorial model/input configuration frozen in trainer gate, unit artifacts, aggregate unit validator, and aggregate artifact
- aggregate expected split seed independently fixed to `174` before unit split-seed comparison

## Static Review V4 Repair Notes

P3-W1 static review v4 result `P3W1_FINAL_REVIEW_V4_BLOCKED` was addressed with static-only repairs:

- calibration-only path no longer requires or reads A0 reference predictions
- calibration-only reason supervision is train-only through a dedicated train-only helper and A0-disabled audit record
- dev targets, dev counts, dev minimum gates, dev cohort-degeneracy gates, and dev metrics are excluded from calibration authority
- normal P2 A1/A2/A3 A0 reference requirements and train+dev reason-supervision contracts remain unchanged

## Decision

`P3W1_IMPLEMENTATION_READY_FOR_STATIC_REVIEW`

This report does not claim `P3W1_IMPLEMENTATION_PASS`, `P3W1_CALIBRATION_EXECUTED`, `P3W0_REASON_LOSS_WEIGHT_RESOLVED`, `A1_READY`, `A2_READY`, `A3_READY`, or `P3_PASS`.