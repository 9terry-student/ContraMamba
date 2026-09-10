# ContraMamba Seed8192 reason-loss calibration-v2 validated aggregate evidence freeze authority — candidate

## 1. Verdict and governing authorities

`PASS_READY_FOR_INDEPENDENT_CALIBRATION_V2_VALIDATED_AGGREGATE_EVIDENCE_FREEZE_VERIFICATION`

This is a REPORT-ONLY VALIDATED AGGREGATE EVIDENCE FREEZE AUTHORITY candidate. It freezes completed, provenance-valid Seed8192 calibration-v2 evidence and its scientific boundary. It does not authorize normal A1/A2/A3 training, A1/A3 release, factorial execution, or any alteration of the frozen weight.

Authority worktree and governing operational authority HEAD:

```text
21403f5e6cff6ca813c6df127c7ee0295998c597
```

Governing authorities:

```text
510d29a9267832dcae521e0c419c2ae0a95c575  calibration-v2 execution authority
850b9e38ce64698885e0f24f132a3ab0f20bd42a  Seed180 GPU recovery / common execution commit
53144c36ca629294157d37c677e6cceed1f261b7  runtime-restart collection recovery
8dbab7c83eb4bee07e98776007586a69e5287fab  salvage-path correction
21403f5e6cff6ca813c6df127c7ee0295998c597  aggregate-input-provisioning operational authority
```

```text
COMMON_CALIBRATION_EXECUTION_COMMIT=850b9e38ce64698885e0f24f132a3ab0f20bd42a
COMMON_CALIBRATION_EXECUTION_COMMIT_PRESERVED=PASS
```

The later operational-authority commits do not replace this common calibration execution identity.

## 2. Accepted unit evidence

The following accepted unit artifacts were independently read and hashed at their specified local source paths. Every byte count and SHA256 matches the frozen expected identity.

| Seed | Source | Bytes | SHA256 | Final loss mean / count | Reason loss mean / count |
| --- | --- | ---: | --- | --- | --- |
| 180 | `C:\\c850\\reports\\reason_router_p3w7_seed8192_reason_loss_calibration_v2\\seed180\\calibration_unit.json` | 4359 | `9354c900e9625c86989ac51948e1f095303c70b29061f11706aefafe4d1a2326` | `0.8248528242111206` / `2880` | `1.3173856735229492` / `1409` |
| 181 | `C:\\s181\\reports\\reason_router_p3w7_seed8192_reason_loss_calibration_v2\\seed181\\calibration_unit.json` | 4358 | `7dc30dddb748f0e778f84efeebf9eb18cf35ca4f26f50b4264baf0949afe9f51` | `0.8834292888641357` / `2880` | `1.4673876762390137` / `1409` |
| 182 | `C:\\s182\\reports\\reason_router_p3w7_seed8192_reason_loss_calibration_v2\\seed182\\calibration_unit.json` | 4358 | `aee5877141a13549faf9f9bf2fe95696315a0a3cab1e6c3c80ae569a6481cec8` | `0.909580647945404` / `2880` | `1.3883105516433716` / `1409` |

Every accepted unit preserves all of the following:

```text
schema_version=reason_router_p3w1_calibration_unit_v2
status=PASS
decision=P3W1_CALIBRATION_UNIT_PASS
execution_commit=850b9e38ce64698885e0f24f132a3ab0f20bd42a
split_seed=8192
dev_ratio=0.2
ordered_train_row_count=2880
p4x_ordered_train_row_sha256=478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
p3w1_ordered_train_row_label_sha256=4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
measurement_arm=conditional_first_blocker
measurement_gradient_ownership=explicit_local
reason_loss_weight_placeholder=0.0
calibration_data_scope=TRAIN_ONLY
weight_resolution_measurement_valid=true
normal_a1_a3_training_ready=true
```

## 3. Aggregate execution and collection provenance

The completed aggregate run provenance is frozen as follows:

```text
RUN=p3w7-seed8192-reason-calibration-v2-aggregate
EXPECTED_EXECUTION_COMMIT=850b9e38ce64698885e0f24f132a3ab0f20bd42a
AGGREGATE_COMMAND_SHA256=b45d87d1410c4d55876ff8781fb4b28e1cf7aac1a01ede31b3279f75fd3cde4b
STARTED_UTC=2026-09-10T03:19:00Z
FINISHED_UTC=2026-09-10T03:19:00Z
EXIT_CODE=0
RUN_LOG_SHA256=c43b7b9c13e446d79d8819f8ccf4833c6c58c235fad5efbd863111fd439de222
RUN_META_SHA256=57ad07f2e5ae645101e3a9cf5a49cd83edf694934bf603383a278bb8946d6434
FILES_COLLECTED=1
AGGREGATE_ARTIFACT_INCLUDED=true
PROVISIONED_SEED180_181_182_UNITS_INCLUDED=false
AGGREGATE_MANIFEST_GATE=PASS
IMPORTED_ZIP_SHA256=a393dc87655a29dffdfb1778d8fceac706e69fb0db08ed9bd6d9fac0cb5ec268
NORMAL_CM_IMPORT_VALIDATED=1
NORMAL_CM_IMPORT_COPIED=1
NORMAL_CM_IMPORT_IDENTICAL=0
NORMAL_CM_IMPORT=PASS
```

## 4. Validated aggregate artifact

The aggregate was independently inspected at `C:\\agg850\\reports\\reason_router_p3w7_seed8192_reason_loss_calibration_v2\\calibration_aggregate.json`.

```text
BYTES=4687
SHA256=505cf9c8ea4304a1bce83c1d0eaf331d5b49c9e07a0cbed2daf16fce0a320ac1
schema_version=reason_router_p3w1_calibration_aggregate_v2
status=PASS
decision=P3W1_CALIBRATION_AGGREGATE_PASS_PENDING_REVIEW
calibration_seeds=[180,181,182]
execution_commit=850b9e38ce64698885e0f24f132a3ab0f20bd42a
dataset_sha256=eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3
sidecar_semantic_sha256=2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9
split_seed=8192
ordered_train_row_count=2880
p4x_ordered_train_row_sha256=478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
p3w1_ordered_train_row_label_sha256=4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
total_final_count=8640
total_reason_count=4227
total_final_loss_sum=7539.444751739502
total_reason_loss_sum=5879.875217080116
mu_final=0.8726209203402201
mu_reason=1.3910279671351116
resolved_reason_loss_weight=0.6273209029272248
all_three_seeds_present=true
all_unit_gates_pass=true
all_sidecar_hashes_verified=true
all_execution_commits_verified=true
all_weight_resolution_measurements_valid=true
A1_A3_common_weight=true
A1_A3_released=false
```

## 5. Independent pooled-estimator recomputation

The three exact accepted unit files were independently recomputed using the required pooled estimator, with production floating-point arithmetic:

```text
mu_final = sum_s(n_final[s] * ell_final[s]) / sum_s(n_final[s])
mu_reason = sum_s(n_reason[s] * ell_reason[s]) / sum_s(n_reason[s])
resolved_reason_loss_weight = mu_final / mu_reason
```

The recomputed totals and values exactly reproduce the aggregate within production numerical tolerance:

```text
total_final_count=8640
total_reason_count=4227
total_final_loss_sum=7539.444751739502
total_reason_loss_sum=5879.875217080116
mu_final=0.8726209203402201
mu_reason=1.3910279671351116
resolved_reason_loss_weight=0.6273209029272248
POOLED_ESTIMATOR_RECOMPUTATION=PASS
```

No mean-of-means substitution was used.

## 6. Final validation evidence and diagnostic-only false negatives

Completed final validator results:

```text
ALL_ACCEPTED_INPUT_UNITS_REVALIDATED=PASS
IMPORTED_POSIX_PATH_PROVENANCE=PASS
PRODUCTION_JSON_SERIALIZATION_ROUNDTRIP=PASS
SERIALIZATION_AWARE_PRODUCTION_REBUILD_MATCH=PASS
FROZEN_AGGREGATE_CONTRACT=PASS
POOLED_ESTIMATOR_RECOMPUTATION=PASS
CALIBRATION_V2_AGGREGATE_VALIDATION=PASS
CALIBRATION_V2_CHAIN=PASS
CALIBRATION_V2_ARTIFACT_PROVENANCE_VALIDITY=PASS
```

The following two validator failures are diagnostic-only false negatives:

```text
VALIDATOR_FAILURE_1=CROSS_PLATFORM_PATH_SERIALIZATION_FALSE_NEGATIVE
Cause: Production stores str(Path); the Linux artifact uses "/" while a Windows in-memory production rebuild uses "\\".

VALIDATOR_FAILURE_2=PRE_JSON_INTEGER_KEY_SERIALIZATION_FALSE_NEGATIVE
Cause: In-memory local_binary_cohort_counts uses integer 0/1 keys; JSON serialization converts object keys to strings "0"/"1".
```

Neither failure modified the aggregate or invalidated execution, import, or provenance. The aggregate SHA256 remained unchanged. No aggregate reexecution or reimport was required. The final serialization-aware production rebuild passed.

## 7. Accepted calibration conclusion and exact scientific boundary

Under the frozen Seed8192 calibration measurement contract, the provenance-valid common A1/A3 reason-loss calibration weight is:

```text
0.6273209029272248
```

This is a calibration result only. It establishes no claim of improved model quality, A1 superiority, A3 superiority, A1/A3 superiority over A0/A2, causal mechanism, promotion, dev/test/OOD performance, or factorial success. It does not itself authorize normal training.

```text
CALIBRATION_RESULT_ACCEPTED=YES
COMMON_REASON_LOSS_WEIGHT=0.6273209029272248
A1_A3_RELEASED=NO
NORMAL_FACTORIAL_EXECUTION_AUTHORIZED=NO
```

## 8. Next-authority boundary

Any future normal A1/A3 execution requires a NEW authority that consumes the frozen common weight exactly, separately decides release and execution scope, preserves the frozen reason-router semantics, and does not reinterpret this calibration result as model-performance evidence. This report does not make that release decision.

## 9. Repository hygiene and explicit non-actions

The expected primary-authority-worktree delta is exactly one new report candidate: this file. The pre-existing untracked A0 roots remain untouched:

```text
reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/
reports/reason_router_p3w7_seed8192_revised_split_a0_runs/
```

The required independent hygiene verification is `git diff --check`, `git diff --name-only`, `git diff --cached --name-only`, and `git status --short`, followed by candidate byte count, SHA256, Git blob, UTF-8/BOM/CR/terminal-LF/trailing-whitespace checks. No existing repository file is modified.

```text
TRAINING_ALLOWED=NO
EVALUATION_ALLOWED=NO
AGGREGATE_REEXECUTION=NO
AGGREGATE_REIMPORT=NO
KAGGLE_CUDA_MODEL_EVALUATION_TRAINING_EXECUTED=NO
CONTROLLER_MODIFIED=NO
REGISTRY_MODIFIED=NO
STAGED=NO
COMMITTED=NO
PUSHED=NO
```
