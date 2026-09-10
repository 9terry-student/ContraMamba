# Seed8192 Calibration-V2 Runtime-Restart Collection Recovery Authority Specification Candidate

## Verdict, scope, and authority binding

```text
verdict = PASS_READY_FOR_INDEPENDENT_RUNTIME_RESTART_COLLECTION_RECOVERY_AUTHORITY_VERIFICATION
phase = REPORT-ONLY PROVENANCE / COLLECTION RECOVERY AUTHORITY
candidate = reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2_runtime_restart_collection_recovery_authority_spec_candidate.md
current calibration execution commit = 850b9e38ce64698885e0f24f132a3ab0f20bd42a
```

This is an operational/provenance recovery authority only.  It neither changes
the calibration mechanism nor creates a new calibration execution commit.  If
this candidate is later independently verified and frozen, that freeze commit
is the identity of this recovery authority only; it MUST NOT replace, rebind,
or be substituted for the common calibration execution commit.

Every accepted calibration-v2 unit and the aggregate remain bound exactly to:

```text
850b9e38ce64698885e0f24f132a3ab0f20bd42a
```

This applies without exception to Seed180 retry4, Seed181, Seed182, and the
aggregate.  Future commands must retain that exact value in both the pinned
repository checkout and the calibration execution-commit argument.  If the
authority branch advances after this report is frozen and `cm run`/`cm save`
requires a matching local `HEAD`, use a separate clean execution worktree
pinned exactly to `850b9e38ce64698885e0f24f132a3ab0f20bd42a`.  Do not reset,
clean, or otherwise alter the user's existing worktree, and do not rebind the
scientific execution provenance to the recovery-authority commit.

No scientific semantics change: no new Seed180 execution, no retry5, and no
loss, gradient, data, split, label, or command change is authorized.  A
successful recovery establishes only provenance validity for the already
successful retry4 measurement.  Scientific interpretation remains prohibited
until all three units and the aggregate validate.

## Frozen retry4 evidence and failure classification

The following retry4 evidence is immutable:

```text
run_name = p3w7-seed8192-reason-calibration-v2-seed180-retry4
execution_commit = 850b9e38ce64698885e0f24f132a3ab0f20bd42a
command_sha256 = d1a2de8ef3b244b70dd2b0bf9e313ce038468bcd0ba6a547ab8c6955737180cc
process_exit = 0
STARTED_UTC = 2026-09-10T02:18:58Z
FINISHED_UTC = 2026-09-10T02:21:40Z
run_log_sha256 = 18b11dc5c8ffa762c52dd5bd56b5375fd538fa17d00255025895a23242407577
run_meta_sha256 = 64fe98de8d1771553abd12c60946a67872af2c36c4ea34ded2bfc87b9514dc09
artifact_path = reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json
artifact_bytes = 4359
artifact_sha256 = 9354c900e9625c86989ac51948e1f095303c70b29061f11706aefafe4d1a2326
artifact_schema_version = reason_router_p3w1_calibration_unit_v2
seed = 180
split_seed = 8192
dev_ratio = 0.2
ordered_train_row_count = 2880
p4x_ordered_train_row_sha256 = 478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8
p3w1_ordered_train_row_label_sha256 = 4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
artifact_execution_commit = 850b9e38ce64698885e0f24f132a3ab0f20bd42a
```

After the operator disabled the Kaggle GPU and observed a runtime restart, the
following mtimes were observed:

```text
run log mtime = 2026-09-10 02:22:58.398614874 +0000
run meta mtime = 2026-09-10 02:22:58.398614874 +0000
command file mtime = 2026-09-10 02:22:58.398614874 +0000
start marker mtime = 2026-09-10 02:22:58.398614874 +0000
artifact mtime = 2026-09-10 02:22:57.624605006 +0000
artifact newer-than-marker = false
collector result = FILES_COLLECTED=0
```

The exact classification is:

```text
POST_EXECUTION_RUNTIME_RESTART_START_MARKER_MTIME_INVALIDATION
```

The current collector discovers artifacts using `find ... -newer
"$START_MARKER"`.  The observed rebased marker is later than the already
written artifact, which explains the zero-file result.  Source audit found no
collector operation that mutates the original `RUN_LOG`, `RUN_META`,
`COMMAND_FILE`, or `START_MARKER`; it copies provenance to a handoff and does
not rewrite those originals.  The mtime rebasing is temporally associated with
the operator-observed GPU-disable runtime restart.  This report does **not**
claim that the exact low-level Kaggle restore mechanism has been independently
proven.

The existing zero-file handoff/ZIP is invalid.  It MUST NOT be imported or
promoted, and its manifest and ZIP MUST NOT be changed to convert
`FILE_COUNT=0` into `1`.  Any salvage is a new recovery handoff with the
explicit recovery semantics below.

## Why retry4's artifact was not preexisting

The provenance argument is sufficient for one explicit-path salvage under the
current importer contract:

1. The wrapper requires a clean Git status before execution.
2. Retry4 passed that gate and entered execution.
3. The calibration-v2 namespace was fresh and is currently untracked.
4. Therefore the retry4 artifact could not have existed at the pre-execution
   clean-worktree gate: an untracked artifact at that exact path would have
   failed the gate.
5. The exact authorized command names that exact path as its calibration
   export destination, and the registered retry4 command SHA256 exactly equals
   `d1a2de8ef3b244b70dd2b0bf9e313ce038468bcd0ba6a547ab8c6955737180cc`.
6. The wrapper records process exit `0`; the immutable run-log and run-meta
   hashes above identify that completed invocation.
7. The observed artifact's schema, seed, execution commit, P4-X, P3-W1,
   split seed, dev ratio, and train-row count exactly match that authorized
   run.

This is a narrow provenance conclusion, not an inference from a zero exit
code alone.  It permits collection of precisely the one already-produced file
after source-hash and source-size rechecks.  It does not authorize collection
of any other post-marker file, mutation of original provenance, a new run, or
scientific interpretation.

## Current importer contract and recovery compatibility

The inspected current controller is
`C:\\Users\\Home1\\.contramamba\\cm.ps1` (91,954 bytes;
SHA256 `d619329478197bee866b91ca95bf52d26dcb8500f350449e3f27e60f6f40800e`).
Its `cm import` requires:

- ZIP root entries `manifest.json`, `run.log`, `run.meta`, `command.sh`, and a
  `files/` directory;
- manifest schema `contramamba-handoff-v3`; a registered syntactically safe
  run name; `command_file` exactly `command.sh`; canonical raw-string UTC
  `started_utc` and `finished_utc`; non-negative integer `exit_code`; and
  `file_count` exactly equal to `files` entry count;
- the registered run's HEAD and UTF-8 command SHA256 to equal manifest
  `expected_commit` and `command_sha256`; `actual_commit` to equal the same
  40-hex commit; and the local import worktree `HEAD` to equal it;
- SHA256 validation of `run.log`, `run.meta`, and `command.sh`, plus semantic
  agreement of run-meta `RUN_NAME`, commits, command SHA, timestamps, and exit
  code with the manifest;
- each artifact path to be unique, relative, contained below both ZIP
  `files/` and the repository destination, present, exact-size, exact-hash,
  and non-colliding (or already byte-identical) before any copy.

The importer has no validation branch for `artifact_discovery`: that field is
emitted by normal collection but is not consumed as an acceptance condition.
Consequently an honest recovery value is importer-compatible.  Required
manifest semantics are:

```text
schema = contramamba-handoff-v3
run_name = p3w7-seed8192-reason-calibration-v2-seed180-retry4
expected_commit = 850b9e38ce64698885e0f24f132a3ab0f20bd42a
actual_commit = 850b9e38ce64698885e0f24f132a3ab0f20bd42a
command_file = command.sh
command_sha256 = d1a2de8ef3b244b70dd2b0bf9e313ce038468bcd0ba6a547ab8c6955737180cc
exit_code = 0
started_utc = 2026-09-10T02:18:58Z
finished_utc = 2026-09-10T02:21:40Z
run_log_sha256 = 18b11dc5c8ffa762c52dd5bd56b5375fd538fa17d00255025895a23242407577
run_meta_sha256 = 64fe98de8d1771553abd12c60946a67872af2c36c4ea34ded2bfc87b9514dc09
artifact_discovery = explicit_recovery_path_post_execution_runtime_restart_start_marker_mtime_invalidation
file_count = 1
files[0].path = reports/reason_router_p3w7_seed8192_reason_loss_calibration_v2/seed180/calibration_unit.json
files[0].size_bytes = 4359
files[0].sha256 = 9354c900e9625c86989ac51948e1f095303c70b29061f11706aefafe4d1a2326
```

`collected_utc` may record the actual recovery construction time.  It is
operational handoff metadata, not a claim that normal start-marker discovery
found the artifact.  The recovery ZIP contains immutable byte copies of the
original retry4 `run.log`, `run.meta`, and `command.sh`; it contains no start
marker because import neither requires nor validates one.

## Exact future Kaggle salvage cell

Run this cell only after independent verification/freeze of this authority and
only in the runtime holding the listed retry4 source files.  It performs no
training, evaluation, or artifact/provenance mutation.  It refuses any hash,
size, or metadata mismatch; creates a **new** recovery handoff path; and
copies only the exact authorized artifact.  Do not run the normal collector
for this recovery.

```bash
%%bash
set -euo pipefail

REPO="/kaggle/working/contramamba"
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

The local import must run from the dedicated clean worktree whose `HEAD` is
the common execution commit, after the downloaded ZIP is available.  The exact
command is:

```powershell
& 'C:\\Users\\Home1\\.contramamba\\cm.ps1' import 'C:\\Users\\Home1\\Downloads\\p3w7-seed8192-reason-calibration-v2-seed180-retry4_850b9e38ce64_runtime_restart_recovery.zip'
```

Import is accepted only if normal `cm import` validation passes.  After it,
run the already-authorized unit-v2 validation; do not treat the import alone as
scientific interpretation.

## Corrected remaining collection order

This authority supersedes only the prior operational ordering that required
GPU OFF before collection.  For each of Seed181 and Seed182, and only after
the preceding imported unit-v2 validation is PASS, require this uninterrupted
order:

```text
GPU ON
→ CUDA availability precheck PASS
→ calibration run pinned to 850b9e38ce64698885e0f24f132a3ab0f20bd42a
→ process exit 0
→ without accelerator change, runtime restart, or session termination:
  immediately run the cm-generated collector cell in the same Kaggle runtime
→ require FILE_COUNT >= 1 and exact expected unit path in manifest
→ download the handoff ZIP
→ only then GPU OFF / accelerator change / runtime termination
→ local cm import
→ unit-v2 validation PASS
```

The collector remains CPU-only; the GPU remains allocated briefly only to
preserve same-runtime filesystem timestamp continuity.  Do not execute a
later seed until the prior seed's import and unit validation pass.  The
aggregate remains after all three accepted units and retains the same common
execution commit.

## Deferred generic hardening and prohibitions

Generic controller hardening from filesystem-mtime discovery to a
restart-stable discovery mechanism is a separate future controller task after
the current calibration chain.  It is not a blocker here because same-runtime
collect-before-restart corrects Seed181/182 operationally and the explicit,
hash-validated recovery handoff salvages retry4.  This authority modifies no
controller and no run registry.

```text
report_created = 1
existing_files_modified = 0
cm.ps1_modified = false
run_registry_modified = false
training_executed = false
evaluation_executed = false
calibration_executed = false
aggregation_executed = false
kaggle_action = false
staged = false
commit = false
push = false
```
