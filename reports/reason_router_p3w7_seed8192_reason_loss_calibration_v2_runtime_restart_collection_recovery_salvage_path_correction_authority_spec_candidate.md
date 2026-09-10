# Seed8192 Calibration-V2 Runtime-Restart Recovery Salvage-Path Correction Authority Specification Candidate

## Verdict, scope, and narrow supersession

```text
verdict = PASS_READY_FOR_INDEPENDENT_SALVAGE_PATH_CORRECTION_AUTHORITY_VERIFICATION
phase = REPORT-ONLY OPERATIONAL RECOVERY AUTHORITY CORRECTION
candidate = reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2_runtime_restart_collection_recovery_salvage_path_correction_authority_spec_candidate.md
frozen_parent_authority_commit = 53144c36ca629294157d37c677e6cceed1f261b7
common_calibration_execution_commit = 850b9e38ce64698885e0f24f132a3ab0f20bd42a
```

This report supersedes **ONLY** the exact lowercase repository-path literal in the retry4 salvage cell frozen by commit `53144c36ca629294157d37c677e6cceed1f261b7`:

```bash
# defective
REPO="/kaggle/working/contramamba"

# corrected
REPO="/kaggle/working/ContraMamba"
```

The verified Kaggle case-sensitive path audit is immutable operational evidence:

```text
PATH=/kaggle/working/ContraMamba
EXISTS=true
REALPATH=/kaggle/working/ContraMamba
IS_GIT_REPO=true
HEAD=850b9e38ce64698885e0f24f132a3ab0f20bd42a

PATH=/kaggle/working/contramamba
EXISTS=false

PATHS_RESOLVE_SAME=false
```

No other salvage-cell semantic, provenance identity, manifest value, artifact identity, recovery ZIP name, import command, collection ordering, or scientific contract changes.  The frozen parent report is not modified.

## Preserved retry4 evidence and salvage classification

The following evidence remains exactly frozen:

```text
run_name = p3w7-seed8192-reason-calibration-v2-seed180-retry4
calibration_execution_commit = 850b9e38ce64698885e0f24f132a3ab0f20bd42a
command_sha256 = d1a2de8ef3b244b70dd2b0bf9e313ce038468bcd0ba6a547ab8c6955737180cc
process_exit = 0
STARTED_UTC = 2026-09-10T02:18:58Z
FINISHED_UTC = 2026-09-10T02:21:40Z
run_log_sha256 = 18b11dc5c8ffa762c52dd5bd56b5375fd538fa17d00255025895a23242407577
run_meta_sha256 = 64fe98de8d1771553abd12c60946a67872af2c36c4ea34ded2bfc87b9514dc09
artifact_path = reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json
artifact_bytes = 4359
artifact_sha256 = 9354c900e9625c86989ac51948e1f095303c70b29061f11706aefafe4d1a2326
schema = reason_router_p3w1_calibration_unit_v2
seed = 180
split_seed = 8192
dev_ratio = 0.2
ordered_train_row_count = 2880
P4-X = 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
P3-W1 = 4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
artifact_execution_commit = 850b9e38ce64698885e0f24f132a3ab0f20bd42a

POST_EXECUTION_RUNTIME_RESTART_START_MARKER_MTIME_INVALIDATION
RETRY4_ARTIFACT_NONPREEXISTENCE_ARGUMENT=PASS
IMPORTER_ARTIFACT_DISCOVERY_IS_NONBINDING=true
SALVAGE_MANIFEST_IMPORTER_COMPATIBLE=PASS
```

The existing `FILE_COUNT=0` handoff is invalid and forbidden to import or modify.  No retry5 and no Seed180 rerun are authorized.

## Exact corrected retry4 salvage cell

Only the `REPO` literal below differs from the frozen parent authority.  In particular, `LOG_ROOT` and `HANDOFF_ROOT` retain their lowercase directory names.  This is the exact authorized future recovery cell; it is not authorization to execute it now.  It contains no trainer invocation and no normal collector invocation.

```bash
%%bash
set -euo pipefail

REPO="/kaggle/working/ContraMamba"
LOG_ROOT="/kaggle/working/contramamba_run_logs"
HANDOFF_ROOT="/kaggle/working/contramamba_handoffs"
RUN_NAME="p3w7-seed8192-reason-calibration-v2-seed180-retry4"
COMMIT="850b9e38ce64698885e0f24f132a3ab0f20bd42a"
SHORT_COMMIT="850b9e38ce64"
COMMAND_SHA256="d1a2de8ef3b244b70dd2b0bf9e313ce038468bcd0ba6a547ab8c6955737180cc"
RUN_LOG_SHA256="18b11dc5c8ffa762c52dd5bd56b5375fd538fa17d00255025895a23242407577"
RUN_META_SHA256="64fe98de8d1771553abd12c60946a67872af2c36c4ea34ded2bfc87b9514dc09"
ARTIFACT_REL="reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json"
ARTIFACT_SHA256="9354c900e9625c86989ac51948e1f095303c70b29061f11706aefafe4d1a2326"
ARTIFACT_BYTES="4359"
RUN_LOG="$LOG_ROOT/${RUN_NAME}_${SHORT_COMMIT}.log"
RUN_META="$LOG_ROOT/${RUN_NAME}_${SHORT_COMMIT}.meta"
COMMAND_FILE="$LOG_ROOT/${RUN_NAME}_${SHORT_COMMIT}.command.sh"
ARTIFACT="$REPO/$ARTIFACT_REL"
HANDOFF_DIR="$HANDOFF_ROOT/${RUN_NAME}_${SHORT_COMMIT}_runtime_restart_recovery"
ZIP_PATH="$HANDOFF_ROOT/${RUN_NAME}_${SHORT_COMMIT}_runtime_restart_recovery.zip"

test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"
test -f "$RUN_LOG" && test -f "$RUN_META" && test -f "$COMMAND_FILE" && test -f "$ARTIFACT"
test "$(sha256sum "$RUN_LOG" | awk '{print $1}')" = "$RUN_LOG_SHA256"
test "$(sha256sum "$RUN_META" | awk '{print $1}')" = "$RUN_META_SHA256"
test "$(sha256sum "$COMMAND_FILE" | awk '{print $1}')" = "$COMMAND_SHA256"
test "$(sha256sum "$ARTIFACT" | awk '{print $1}')" = "$ARTIFACT_SHA256"
test "$(stat -c '%s' "$ARTIFACT")" = "$ARTIFACT_BYTES"
grep -Fx "RUN_NAME=$RUN_NAME" "$RUN_META"
grep -Fx "EXPECTED_COMMIT=$COMMIT" "$RUN_META"
grep -Fx "ACTUAL_COMMIT=$COMMIT" "$RUN_META"
grep -Fx "COMMAND_SHA256=$COMMAND_SHA256" "$RUN_META"
grep -Fx 'STARTED_UTC=2026-09-10T02:18:58Z' "$RUN_META"
grep -Fx 'FINISHED_UTC=2026-09-10T02:21:40Z' "$RUN_META"
grep -Fx 'EXIT_CODE=0' "$RUN_META"

if [ -e "$HANDOFF_DIR" ] || [ -e "$ZIP_PATH" ]; then
  echo 'RECOVERY BLOCKED: recovery handoff destination already exists.' >&2
  exit 70
fi
mkdir -p "$HANDOFF_DIR/files/$(dirname "$ARTIFACT_REL")"
cp -p "$RUN_LOG" "$HANDOFF_DIR/run.log"
cp -p "$RUN_META" "$HANDOFF_DIR/run.meta"
cp -p "$COMMAND_FILE" "$HANDOFF_DIR/command.sh"
cp -p "$ARTIFACT" "$HANDOFF_DIR/files/$ARTIFACT_REL"

export HANDOFF_DIR RUN_NAME COMMIT COMMAND_SHA256 RUN_LOG_SHA256 RUN_META_SHA256 ARTIFACT_REL ARTIFACT_SHA256 ARTIFACT_BYTES
python - <<'PY'
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path

handoff = Path(os.environ['HANDOFF_DIR'])
manifest = {
    'schema': 'contramamba-handoff-v3',
    'run_name': os.environ['RUN_NAME'],
    'expected_commit': os.environ['COMMIT'],
    'actual_commit': os.environ['COMMIT'],
    'command_file': 'command.sh',
    'command_sha256': os.environ['COMMAND_SHA256'],
    'exit_code': 0,
    'started_utc': '2026-09-10T02:18:58Z',
    'finished_utc': '2026-09-10T02:21:40Z',
    'run_log_sha256': os.environ['RUN_LOG_SHA256'],
    'run_meta_sha256': os.environ['RUN_META_SHA256'],
    'artifact_discovery': 'explicit_recovery_path_post_execution_runtime_restart_start_marker_mtime_invalidation',
    'collected_utc': datetime.now(timezone.utc).isoformat(),
    'file_count': 1,
    'files': [{
        'path': os.environ['ARTIFACT_REL'],
        'size_bytes': int(os.environ['ARTIFACT_BYTES']),
        'sha256': os.environ['ARTIFACT_SHA256'],
    }],
}
(handoff / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
PY

test "$(sha256sum "$HANDOFF_DIR/run.log" | awk '{print $1}')" = "$RUN_LOG_SHA256"
test "$(sha256sum "$HANDOFF_DIR/run.meta" | awk '{print $1}')" = "$RUN_META_SHA256"
test "$(sha256sum "$HANDOFF_DIR/command.sh" | awk '{print $1}')" = "$COMMAND_SHA256"
test "$(sha256sum "$HANDOFF_DIR/files/$ARTIFACT_REL" | awk '{print $1}')" = "$ARTIFACT_SHA256"
test "$(stat -c '%s' "$HANDOFF_DIR/files/$ARTIFACT_REL")" = "$ARTIFACT_BYTES"
(cd "$HANDOFF_DIR" && zip -qr "$ZIP_PATH" manifest.json run.log run.meta command.sh files)
echo "RECOVERY_HANDOFF_ZIP=$ZIP_PATH"
```

The recovery destination remains exactly `p3w7-seed8192-reason-calibration-v2-seed180-retry4_850b9e38ce64_runtime_restart_recovery`; the ZIP remains exactly `p3w7-seed8192-reason-calibration-v2-seed180-retry4_850b9e38ce64_runtime_restart_recovery.zip`.  The manifest remains `file_count = 1` with the sole exact artifact entry shown above and:

```text
artifact_discovery = explicit_recovery_path_post_execution_runtime_restart_start_marker_mtime_invalidation
```

The cell checks the repository HEAD is exactly `850b9e38ce64698885e0f24f132a3ab0f20bd42a`; preserves every original provenance hash and artifact byte/hash check; does not modify original retry4 `run.log`, `run.meta`, `command.sh`, `start.marker`, or `calibration_unit.json`; and only creates a new recovery handoff/ZIP.

```text
CORRECTED_SALVAGE_PATH_STATIC_VALIDATION=PASS
```

Static validation basis: literal inspection confirms the corrected `REPO` path, immutable HEAD expectation, absence of trainer and normal-collector invocations, source files used only as copy inputs, separate new-destination guard, frozen hashes/sizes, and shell/Python syntax structure.  No cell execution occurred.

## Import contract unchanged

Import must run from a clean dedicated local execution worktree whose HEAD is exactly `850b9e38ce64698885e0f24f132a3ab0f20bd42a`, because `cm import` validates local HEAD against the registered retry4 execution commit.  Do not reset or clean the user's current authority worktree.  The exact command remains:

```powershell
& 'C:\Users\Home1\.contramamba\cm.ps1' import 'C:\Users\Home1\Downloads\p3w7-seed8192-reason-calibration-v2-seed180-retry4_850b9e38ce64_runtime_restart_recovery.zip'
```

## Execution-commit and later-seed order unchanged

This correction is operational only.  Its eventual freeze commit MUST NOT become `CALIBRATION_EXECUTION_COMMIT`.  The common calibration execution commit remains `850b9e38ce64698885e0f24f132a3ab0f20bd42a` for Seed180 retry4, Seed181, Seed182, and aggregate.

```text
COMMON_CALIBRATION_EXECUTION_COMMIT_PRESERVED=PASS
```

For Seed181 and Seed182, the required ordering remains exactly:

```text
GPU ON
→ CUDA availability precheck
→ pinned run at 850b9e3...
→ exit 0
→ SAME runtime, no accelerator/session change
→ immediate collector
→ FILE_COUNT >=1
→ exact unit artifact in manifest
→ download
→ only then GPU OFF/runtime change
→ local import
→ unit-v2 validation
```

No later seed may begin before previous unit acceptance.

```text
report_created = 1
existing_files_modified = 0
kaggle_action = false
salvage_zip_created = false
import_executed = false
training_executed = false
evaluation_executed = false
calibration_executed = false
staged = false
commit = false
push = false
controller_modified = false
run_registry_modified = false
```
