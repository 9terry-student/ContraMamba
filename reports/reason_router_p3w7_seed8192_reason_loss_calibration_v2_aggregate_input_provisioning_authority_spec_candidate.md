# ContraMamba Seed8192 calibration-v2 aggregate input-provisioning authority — candidate

## 1. Verdict and authority

`PASS_READY_FOR_INDEPENDENT_AGGREGATE_INPUT_PROVISIONING_AUTHORITY_VERIFICATION`

This is a REPORT-ONLY AGGREGATE INPUT-PROVISIONING AUTHORITY candidate. It closes the operational input-provisioning gap for the already-authorized pure-JSON aggregate without changing scientific semantics.

Authority worktree HEAD at creation:

```text
8dbab7c83eb4bee07e98776007586a69e5287fab
```

Frozen governing authorities:

```text
850b9e38ce64698885e0f24f132a3ab0f20bd42a  Seed180 GPU activation recovery execution authority
53144c36ca629294157d37c677e6cceed1f261b7  runtime-restart collection recovery authority
8dbab7c83eb4bee07e98776007586a69e5287fab  salvage-path correction authority
```

The common calibration execution and aggregate/import commit is preserved exactly:

```text
COMMON_CALIBRATION_EXECUTION_COMMIT_PRESERVED=850b9e38ce64698885e0f24f132a3ab0f20bd42a
```

This candidate authorizes no training, evaluation, Kaggle action, staging, commit, push, controller change, registry change, tracked `.gitignore` change, or modification of an existing repository file.

## 2. Accepted input units

All three units below are already ACCEPTED. Their exact artifact and provenance validity gates are PASS; they are frozen input bytes, not artifacts to regenerate or modify.

| Seed | Frozen relative path | Bytes | SHA256 |
| --- | --- | ---: | --- |
| 180 | `reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json` | 4359 | `9354c900e9625c86989ac51948e1f095303c70b29061f11706aefafe4d1a2326` |
| 181 | `reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed181/calibration_unit.json` | 4358 | `7dc30dddb748f0e778f84efeebf9eb18cf35ca4f26f50b4264baf0949afe9f51` |
| 182 | `reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed182/calibration_unit.json` | 4358 | `aee5877141a13549faf9f9bf2fe95696315a0a3cab1e6c3c80ae569a6481cec8` |

For each accepted unit: `schema=reason_router_p3w1_calibration_unit_v2`; `status=PASS`; `decision=P3W1_CALIBRATION_UNIT_PASS`; `seed` is the listed seed; `execution_commit=850b9e38ce64698885e0f24f132a3ab0f20bd42a`; unit validator PASS; artifact/provenance validity PASS.

The common frozen contents are:

```text
split_seed=8192
dev_ratio=0.2
ordered_train_row_count=2880
P4-X=478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
P3-W1=4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
measurement_arm=conditional_first_blocker
measurement_gradient_ownership=explicit_local
reason_loss_weight_placeholder=0.0
calibration_data_scope=TRAIN_ONLY
weight_resolution_measurement_valid=true
```

The accepted input files are absent from this authority worktree because they were accepted through separate clean execution worktrees/Kaggle runtimes. That absence neither revises their accepted identities nor authorizes a substitute. Before a provisioning cell is generated, a local source process must read only these three accepted artifacts and independently verify their exact byte counts and SHA256 values above. A mismatch, missing source artifact, or any non-byte-identical source is fail-closed.

## 3. Sole operational gap and prohibited responses

The frozen aggregate command requires all three relative unit paths in one execution repository. The units were accepted through separate clean execution worktrees/Kaggle runtimes. The `cm run` wrapper rejects non-clean Git status before execution. Therefore blindly copying the three untracked units into the aggregate runtime would fail the wrapper clean-worktree gate.

Do not commit the units merely to satisfy clean status; change the common execution commit; reset or clean existing user worktrees; alter `.gitignore`; alter the aggregate command; bypass `cm run`; or weaken the wrapper clean gate.

## 4. Authorized provisioning mechanism

Only the following mechanism is authorized.

1. Start a fresh Kaggle repository checkout pinned exactly to `850b9e38ce64698885e0f24f132a3ab0f20bd42a`, with run name exactly `p3w7-seed8192-reason-calibration-v2-aggregate` and GPU OFF.
2. Before the aggregate `cm run` cell and before its start marker is created, write exact byte copies of the three accepted units to their frozen relative paths. The local generator must use only the three already accepted artifacts after the local byte/SHA checks in Section 2.
3. A generated Kaggle provisioning cell may contain the verified accepted bytes encoded as base64. Base64 is transport only: Kaggle must decode bytes, then verify the exact byte count and SHA256 for each unit before continuing.
4. Add **only** the three exact unit paths to the execution checkout's local `.git/info/exclude`. Do not modify tracked `.gitignore` and do not modify the Git index.
5. After provisioning, require all of the following before `cm run`:

```text
git rev-parse HEAD
= 850b9e38ce64698885e0f24f132a3ab0f20bd42a

git diff --name-only
= empty

git diff --cached --name-only
= empty

git status --porcelain
= empty

seed180 exact bytes/SHA PASS
seed181 exact bytes/SHA PASS
seed182 exact bytes/SHA PASS
```

`git status` honors `.git/info/exclude` for these untracked inputs while the files remain present and directly readable by the frozen aggregate command. The local exclude is solely an execution-checkout cleanliness mechanism; it neither stages, tracks, nor changes input bytes.

The three provisioned files must exist before `cm run` creates the aggregate start marker. The normal collector's `-newer "$START_MARKER"` discovery therefore must not collect input units; only newly generated post-marker artifacts are eligible. There must be no runtime restart or session change between aggregate execution and collector.

## 5. Static controller audit

Controller audited: `C:\Users\Home1\.contramamba\cm.ps1`, bytes `91954`, SHA256 `d619329478197bee866b91ca95bf52d26dcb8500f350449e3f27e60f6f40800e`.

### Clean-gate source audit

The generated run wrapper resolves and compares `ACTUAL_COMMIT` to `EXPECTED_COMMIT`, then uses ordinary `git status --porcelain` at controller lines 1027–1039. A non-empty result emits `RUN BLOCKED: repository is DIRTY before execution.` and exits 43. It does not enumerate untracked files independently or override Git ignore behavior. Thus the gate is compatible with local `.git/info/exclude` for precisely the three provisioned untracked paths, provided all Section 4 checks pass. No gate bypass or weakening is authorized.

`GIT_INFO_EXCLUDE_COMPATIBILITY_VERDICT=PASS`

### Collector/start-marker source audit

The run wrapper performs clean preflight before marker creation. At lines 1042–1075 it prepares provenance paths and reconstructs/hash-checks the authorized command. At lines 1077–1085 it creates `START_MARKER` immediately before the wrapped command begins. The normal generated collector requires that marker, then at lines 1455–1465 uses:

```bash
find . \
    -type f \
    -newer "$START_MARKER" \
    ! -path './.git/*' \
    ! -path './.venv/*' \
    ! -path './__pycache__/*' \
    ! -path '*/__pycache__/*' \
    ! -path './.pytest_cache/*' \
    ! -name '*.pyc'
```

It copies only that discovered list. It has no deliberate rule to collect ignored pre-existing inputs. Because provisioning is completed before marker creation, the three input units are older than the marker and are not normal collector targets. This conclusion depends on preserving the authorized sequence and uninterrupted runtime.

```text
AGGREGATE_INPUT_PROVISIONING_STATIC_VALIDATION=PASS
```

## 6. Exact aggregate command and collection

The command is preserved exactly, with only the already-frozen common execution commit substituted:

```bash
python scripts/aggregate_reason_router_p3w1_calibration.py --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed181/calibration_unit.json --unit-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed182/calibration_unit.json --output-json reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/calibration_aggregate.json --expected-execution-commit 850b9e38ce64698885e0f24f132a3ab0f20bd42a --expected-dataset-sha256 eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3 --expected-sidecar-semantic-sha256 2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9 --expected-split-seed 8192 --expected-ordered-train-row-count 2880 --expected-p4x-ordered-train-row-sha256 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8 --expected-p3w1-ordered-train-row-label-sha256 4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b --expected-dev-ratio 0.2
```

After aggregate process exit 0, run the normal cm-generated collector immediately in the same uninterrupted runtime. Require `FILES_COLLECTED >= 1`. Its manifest must contain:

```text
reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/calibration_aggregate.json
```

The manifest must explicitly be checked to prove that none of these three provisioned input paths appears in it:

```text
reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed181/calibration_unit.json
reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed182/calibration_unit.json
```

Only then download the ZIP. No runtime/session change is allowed before collector and ZIP construction.

## 7. Import and aggregate validation contract

Use normal local `cm import` only, from a clean dedicated worktree pinned exactly to `850b9e38ce64698885e0f24f132a3ab0f20bd42a`. Do not use authority HEAD `8dbab7c83eb4bee07e98776007586a69e5287fab` as the execution/import HEAD.

After import, validate the aggregate under the current production aggregator contract. Require `schema=reason_router_p3w1_calibration_aggregate_v2`; seeds exactly `[180,181,182]`; common execution commit exactly `850b9e38ce64698885e0f24f132a3ab0f20bd42a`; the exact dataset SHA and sidecar semantic SHA from Section 6; `split_seed=8192`; `ordered_train_row_count=2880`; the exact P4-X and P3-W1 identities; and a finite positive `resolved_reason_loss_weight`.

The estimator remains exclusively:

```text
mu_final = sum_s(n_final[s] * ell_final[s]) / sum_s(n_final[s])
mu_reason = sum_s(n_reason[s] * ell_reason[s]) / sum_s(n_reason[s])
resolved_reason_loss_weight = mu_final / mu_reason
```

No mean-of-means substitution is authorized. Current production `scripts/aggregate_reason_router_p3w1_calibration.py` implements this pooled calculation through total loss sums and total counts, then requires the resolved weight to be finite and greater than zero.

## 8. Scientific boundary

A successful aggregate establishes only a provenance-valid common Seed8192 reason-loss calibration weight for A1/A3 under the frozen calibration measurement contract. It does **not** establish improved model quality, A1/A3 superiority, a causal mechanism, promotion, factorial execution authority, or dev/test/OOD performance. No normal A1/A2/A3 training is authorized.

## 9. Final hygiene and explicit non-actions

Expected repository delta: exactly this one new report. Required independent verification must run `git diff --check`, `git diff --name-only`, `git diff --cached --name-only`, and `git status --short`, then compute this candidate's byte count, SHA256, Git blob ID, UTF-8/BOM status, CR status, terminal-LF status, and trailing-whitespace status.

```text
TRAINING_ALLOWED=NO
EVALUATION_ALLOWED=NO
AGGREGATE_EXECUTED=NO
KAGGLE_ACTION=NO
STAGED=NO
COMMITTED=NO
PUSHED=NO
CONTROLLER_MODIFIED=NO
REGISTRY_MODIFIED=NO
```
