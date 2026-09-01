# P3-W7 Current-Lineage Reason-Weight Calibration Result-Acceptance Authority Spec Candidate

## Status

READY: accept the consumed calibration run as valid current-lineage reason-loss-weight calibration evidence, while preserving the wrapper-level execution status as `EXIT_CODE=1`.

This document is a static provenance/result-acceptance authority candidate only. It does not authorize training, evaluation, calibration rerun, source-code modification, dataset regeneration, checkpoint mutation, A0 evidence changes, run-registry changes, or git-history changes.

## Authority Inputs

- Frozen calibration execution authority commit: `ba3fd1a82cba029dac05ba38a86d518252ab858f`
- Exact execution commit: `34a5df306a00a28492b08f666bf3f7ac06c26944`
- Imported execution artifact directory: `reports/reason_router_p3w7_current_lineage_reason_weight_calibration_execution_34a5df306a00a28492b08f666bf3f7ac06c26944/`
- Frozen aggregator inspected: `scripts/aggregate_reason_router_p3w1_calibration.py`
- Phase: static provenance/result-acceptance authority drafting only

## Run Provenance

- Run name: `p3w7-current-lineage-reason-weight-calibration-34a5df3`
- Execution commit: `34a5df306a00a28492b08f666bf3f7ac06c26944`
- Authority freeze: `ba3fd1a82cba029dac05ba38a86d518252ab858f`
- Command SHA256: `52cf31c7389bfc5d0da98b4066178708455b6fda8abb7e3a5c063f24f07d4efb`
- Run started: `2026-09-01T17:43:27Z`
- Run finished: `2026-09-01T17:49:43Z`
- Wrapper exit code: `1`
- Run log SHA256: `16452698df01e3ef55812c45558c97059bcba5dea70d7481c80f57a8bcb49283`
- Run meta SHA256: `319acca6ae5cdead0afd4a01e59d922863c0d1bc876ac6a63f6db158cad20a20`
- Handoff ZIP SHA256: `2051fb2f2c83fd72dff4f666618a8952b38cf7c39ccae568a3bc6470ba0fa2a9`
- `cm` import: `PASS`, `VALIDATED=4`, `COPIED=4`

The four imported JSON artifacts were directly inspected and independently hash/size checked in the local repository state used to draft this candidate.

## Artifact Identity Checks

| Artifact | SHA256 | Size | Result |
| --- | --- | ---: | --- |
| `seed180/calibration_unit.json` | `7999a7e48891ed718d65a5741bb8c830aaae902b06ca46daca6ce02a69806b61` | 4202 | PASS |
| `seed181/calibration_unit.json` | `e92799dc34a5ed20bafe977602da72862e5e534864d9484f8c1bff71f04fe098` | 4200 | PASS |
| `seed182/calibration_unit.json` | `ca649fa47edd07473d03b84c7f12c407e283e4db3a4cf2ed80aab0b294e5ff30` | 4202 | PASS |
| `calibration_aggregate.json` | `efc07d259a9ebaf8b0f1de5b6ea574beb2ccffdd44f58f6558d9f16713107e51` | 4804 | PASS |

## Aggregate Revalidation

The frozen aggregator implementation was inspected directly. It validates the three unit artifacts and returns aggregate schema `reason_router_p3w1_calibration_aggregate_v1`, status `PASS`, and decision `P3W1_CALIBRATION_AGGREGATE_PASS_PENDING_REVIEW`.

Independent revalidation against the frozen aggregator confirmed:

- `AGGREGATE_NORMALIZED_RECOMPUTE_MATCH=PASS` after semantic normalization of JSON-integer object keys and path separator representation.
- `status=PASS`
- `decision=P3W1_CALIBRATION_AGGREGATE_PASS_PENDING_REVIEW`
- `execution_commit=34a5df306a00a28492b08f666bf3f7ac06c26944`
- `dataset_sha256=eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- `sidecar_semantic_sha256=0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`
- `split_seed=174`
- `ordered_train_row_count=2880`
- `ordered_train_row_identity_hash=cbce1775ddc73f2fbad024ded6a314d15e2eb1988ef107fa72a5eacbdd836784`
- `all_three_seeds_present=true`
- `all_unit_gates_pass=true`
- `all_execution_commits_verified=true`
- `all_sidecar_hashes_verified=true`
- `all_primary_reason_min_count_gates_pass=true`
- `all_local_binary_cohorts_training_ready=true`
- `all_polarity_local_training_ready=true`
- `normal_a1_a3_training_ready=true`
- `A1_A3_common_weight=true`
- `A1_A3_released=false`
- `nonfinite_count=0`

The aggregate schema contains top-level `all_polarity_local_training_ready`. It does not contain top-level `polarity_local_training_ready`.

## Pooled Estimator

Stored aggregate totals:

- `total_final_count=8640`
- `total_reason_count=4350`
- `total_final_loss_sum=7519.760513305664`
- `total_reason_loss_sum=6104.04389500618`

Recomputed means:

- `mu_final=7519.760513305664 / 8640 = 0.8703426520029703`
- `mu_reason=6104.04389500618 / 4350 = 1.403228481610616`

Recomputed pooled weight:

- `resolved_reason_loss_weight = mu_final / mu_reason = 0.6202430063306562`

The recomputed value agrees with the stored aggregate value exactly at the recorded precision.

## Per-Seed Validated Summaries

| Seed | Final Count | Reason Count | Final Sum | Reason Sum | Polarity Ready | Normal Ready |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 180 | 2880 | 1450 | 2370.5655097961426 | 1928.9840519428253 | true | true |
| 181 | 2880 | 1450 | 2535.5868530273438 | 2145.028591156006 | true | true |
| 182 | 2880 | 1450 | 2613.6081504821777 | 2030.0312519073486 | true | true |

Each unit independently records:

- `fresh_initialization=true`
- `parameter_update_count=0`
- `gradient_tracking_enabled=false`
- `before_backward=true`
- `before_optimizer_step=true`
- `optimizer_step_executed=false`
- `scheduler_step_executed=false`
- `dev_forward_executed=false`
- `checkpoint_loaded=false`

These fields rule out parameter update, backward execution, optimizer step, scheduler step, dev forward, checkpoint load, and gradient-tracked calibration measurement within the consumed unit artifacts.

## EXIT_CODE=1 Localization

The frozen aggregator completed successfully and wrote `calibration_aggregate.json`; the imported aggregate artifact is present, hash-valid, size-valid, and semantically reproducible from the three unit artifacts.

The frozen aggregate schema emits `all_polarity_local_training_ready` as the aggregate-level polarity readiness key. The three unit artifacts contain unit-level `polarity_local_training_ready`, but `calibration_aggregate.json` does not contain a top-level `polarity_local_training_ready` key. Therefore the observed `KeyError` on `polarity_local_training_ready` is localized to a controller-added post-aggregate observer/assertion that accessed a nonexistent aggregate key after aggregate construction and write.

This failure localization means the wrapper-level execution status remains `EXIT_CODE=1` and must not be rewritten as execution PASS. It also means the calibration computational completion is accepted separately: the three calibration units and aggregate completed and validated before the post-aggregate observer assertion failed. No evidence indicates that the observer failure altered the already written four JSON artifacts.

No trainer/calibration rerun is permitted or needed merely to repair this observer error.

## Acceptance Decision

The four imported JSON artifacts are accepted as provenance-valid current-lineage reason-loss-weight calibration evidence.

The frozen common A1/A3 reason-loss-weight hyperparameter candidate is:

`0.6202430063306562`

Decision semantics:

1. Wrapper-level execution status remains `EXIT_CODE=1`; this document does not rewrite the execution as PASS.
2. Calibration computational completion is accepted separately from wrapper exit status.
3. The four imported artifacts are provenance-valid calibration evidence.
4. `resolved_reason_loss_weight` is frozen as `0.6202430063306562`.
5. This value is one common A1/A3 hyperparameter candidate for subsequent factorial authority.
6. No seed-specific weight is permitted.
7. Historical P3-W2 weight `0.6518018402446165` remains non-transferable.
8. `A1_A3_released` remains false.
9. A2 is also not independently released.
10. No scientific claim about router effectiveness follows from this calibration.
11. No calibration rerun is authorized by this document.
12. Subsequent A1/A2/A3 execution requires a separate independently verified frozen factorial execution authority.

## Explicit Non-Changes

This candidate does not change, and does not authorize changing:

- imported calibration JSONs
- trainer
- aggregator
- tests
- any previous authority/report
- A0 evidence
- dataset or sidecar
- run registry
- git history

## Release Boundary

A1, A2, and A3 remain unreleased. The accepted calibration evidence only resolves one common A1/A3 reason-loss-weight candidate for possible later factorial authority. It does not release A1/A3, does not independently release A2, and does not authorize any A1/A2/A3 execution.

## No Rerun Statement

No model loading, trainer invocation, calibration rerun, Kaggle execution, training, or evaluation was performed or authorized by this document.
