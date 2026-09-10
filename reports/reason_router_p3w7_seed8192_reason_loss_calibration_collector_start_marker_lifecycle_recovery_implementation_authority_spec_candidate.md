# Seed8192 Calibration Collector Start-Marker Lifecycle-Recovery Implementation Authority Candidate

## Verdict, authority, and phase boundary

```text
verdict = PASS_READY_FOR_INDEPENDENT_COLLECTOR_START_MARKER_RECOVERY_AUTHORITY_VERIFICATION
phase = REPORT-ONLY COLLECTOR PROVENANCE RECOVERY AUTHORITY DESIGN
current_HEAD = ee0ddd3154f1dceba79683ded152a1c23610ea3d
frozen_dual_identity_implementation_commit = ee0ddd3154f1dceba79683ded152a1c23610ea3d
```

This authority candidate is narrowly limited to a prospective correction of
the external controller's mutable deterministic run-start-marker lifecycle.
It authorizes no controller implementation in this report-only phase, no
repository source or test modification, no run-registry modification, and no
Kaggle, training, evaluation, calibration, aggregation, model/tokenizer, CUDA,
commit, or push activity.

The frozen external controller identity is:

```text
current_cm.ps1 = C:\Users\Home1\\.contramamba\cm.ps1
bytes = 91498
SHA256 = 09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae
```

## Frozen forensic lifecycle findings

The generated Kaggle run wrapper has the following relevant order:

```text
1036-1040 = dirty worktree guard
1045      = LOG_FILE deterministic from RUN_NAME + SHORT_COMMIT
1046      = META_FILE deterministic from RUN_NAME + SHORT_COMMIT
1047      = COMMAND_FILE deterministic from RUN_NAME + SHORT_COMMIT
1048      = START_MARKER deterministic from RUN_NAME + SHORT_COMMIT
1051      = COMMAND_FILE is written
1053-1061 = command hash validation
1066      = touch "$START_MARKER"
1067      = STARTED_UTC recorded
1096      = bash -x "$COMMAND_FILE" executes authorized command
1102      = FINISHED_UTC
```

Therefore, the marker is after the dirty/commit guards and before command
execution.  The collector does not touch the marker.  The marker path is
deterministic, no pre-existing provenance-path reuse guard exists, and `touch`
mutates the mtime of an existing marker.

The collector remains:

```text
1439 = find . -type f -newer "$START_MARKER" ...
1464 = FILES_COLLECTED
1520 = artifact_discovery = filesystem_start_marker
```

## Defect classification and historical retry2 disposition

The exact defect classification is:

```text
MUTABLE_DETERMINISTIC_RUN_START_MARKER_REUSE_DEFECT
```

The safety defect is architectural.  A registered run identity is `RUN_NAME +
commit + command hash`, but its run-provenance files use deterministic paths
and are mutable on another invocation of the same generated run cell.  A
repeated invocation can overwrite/reuse `COMMAND_FILE`, `META_FILE`, and
`LOG_FILE`, and can retouch `START_MARKER`.  Since the collector uses the
`START_MARKER` mtime as its discovery lower bound, marker mutation after an
earlier successful artifact was written can make that artifact invisible to:

```bash
find -newer "$START_MARKER"
```

This hazard is sufficient to block future scientific collection even though
the exact historical retry2 mutation actor is not proven.

```text
retry2 run = p3w7-seed8192-reason-calibration-seed180-retry2
retry2 head = a44c6394323da14b423654a88a11a9d0ed3507f6
retry2 command SHA256 = 5cbba8dba815fc50aef822099d0a678f37ebe1f456cca7dfb6b7b4b676c2fe06
retry2 registry command recomputation exact match = true
retry2 artifact mtime = 2026-09-09 15:34:05.195412335 +0000
retry2 observed start-marker mtime = 2026-09-09 15:34:05.872464357 +0000
```

The observed marker boundary is later than the artifact and cannot serve as
valid discovery provenance for retry2.  Exact historical causation remains
**NOT PROVEN**: this authority does not claim that a specific rerun or actor
retouched the historical retry2 marker.

Preserved historical disposition:

```text
retry2 process execution = historical PASS
retry2 artifact provenance = INVALID for corrected evidence
retry2 measurement = NOT ACCEPTED
retry2 artifact mutation = FORBIDDEN
retry2 marker recreation = FORBIDDEN
retry2 v1 -> v2 promotion = FORBIDDEN
FILES_COLLECTED = 0
retry2 ZIP import input = FORBIDDEN
```

The correction is prospective.  Fresh seed180 retry3 remains required under a
future execution authority.

## Future implementation whitelist and correction contract

After this candidate is independently verified and a later implementation
phase is explicitly authorized, the exact and only implementation whitelist
is:

```text
C:\Users\Home1\\.contramamba\cm.ps1
```

No repository source/test file is authorized.  No `run-registry.json`
modification, cleanup, reset, or other file change is authorized.

Immediately after defining `LOG_FILE`, `META_FILE`, `COMMAND_FILE`, and
`START_MARKER`, and before writing `COMMAND_FILE`, the generated Kaggle run
wrapper must fail closed if any of those four paths already exists.  The
semantic contract is:

```bash
for path in \
   "$LOG_FILE" \
   "$META_FILE" \
   "$COMMAND_FILE" \
   "$START_MARKER"
do
   if [ -e "$path" ]; then
       echo "RUN BLOCKED: run provenance path already exists."
       echo "$path"
       echo "Use a new run name; run identities are single-use."
       exit <dedicated-nonzero-code>
   fi
done
```

Equivalent shell formatting is permitted.  The dedicated exit code must not
collide with existing run-wrapper codes `41`, `42`, `43`, or `44`; use `45`
unless inspection finds it already reserved in this run-wrapper contract.
The single-use check must precede:

```bash
printf ... > "$COMMAND_FILE"
```

Consequently, a repeated invocation must not mutate the command file, metadata
file, run log, or start marker.  The run identity namespace is single-use.

Replace mutable:

```bash
touch "$START_MARKER"
```

with create-only semantics.  Creation must succeed only when the marker does
not already exist; an existing marker's mtime must never change; and failure
to create the marker must fail closed before command execution.  A preferred
equivalent form is:

```bash
if ! ( set -o noclobber; : > "$START_MARKER" ) 2>/dev/null; then
   echo "RUN BLOCKED: failed to create immutable run start marker."
   exit <dedicated-nonzero-code>
fi
```

Use a separate non-colliding run-wrapper exit code, preferably `46`.
Equivalent robust create-exclusive implementation is permitted.  Do not
implement `rm -f "$START_MARKER"` before creation, touching an existing
marker, automatic recovery/recreation, marker timestamp rewriting, a
`STARTED_UTC` fallback, or run-name reuse.  Existing run provenance is
immutable once created.

Preserve current decoded-command SHA256 verification.  A command-hash mismatch
before marker creation may continue deleting only the newly written
`COMMAND_FILE` as current behavior does.  Do not weaken the repository
existence gate, commit gate, dirty worktree gate, or command SHA gate.

## Preserved collector and compatibility contracts

This correction must not redesign generic artifact discovery.  For newly
generated single-use runs, preserve:

```bash
find . -type f -newer "$START_MARKER"
```

The correction makes the marker provenance boundary immutable.  Do not add a
`STARTED_UTC` fallback, invent post-hoc marker reconstruction, or authorize
retry2 recovery by replacing its marker.

Do not globally change generic collector zero-file semantics in this authority
unless repository/controller evidence independently proves every legitimate
`cm collect` must contain a repository artifact.  For the future calibration
execution authority, require `FILE_COUNT >= 1`, explicit presence of the
expected calibration unit artifact in the manifest, and validation of the
expected artifact hash/provenance before import.

The new wrapper is prospective.  Existing historical run provenance may not
satisfy the new single-use assumptions and must not be silently promoted.  No
compatibility fallback may retouch or regenerate an old marker.

## Required later implementation validation

After a separately authorized `cm.ps1` implementation, an independent
verifier must perform and record all of the following:

1. Confirm only `C:\Users\Home1\\.contramamba\cm.ps1` changed.
2. Record old and new controller bytes and SHA256.
3. Statically audit the generated cell to prove this order:

   ```text
   repository/commit/dirty gates
   < provenance-path single-use check
   < COMMAND_FILE write/hash validation
   < create-only START_MARKER
   < STARTED_UTC/meta/log
   < bash -x command execution
   ```

4. Prove that no `touch "$START_MARKER"` remains in the run wrapper.
5. Prove that the collector contains zero marker-mutation operations.
6. Use a CPU-only synthetic shell harness or equivalent isolated local test to
   prove: first provenance namespace creation succeeds; a second invocation
   with the same paths fails before mutation; marker mtime before/after the
   second invocation is identical; existing command/meta/log bytes are
   unchanged; and the authorized command is not executed by the second
   invocation.  No Kaggle is required for this synthetic validation.
7. Verify the current collector still uses `-newer "$START_MARKER"`.
8. Verify the run registry is not modified.
9. Verify the repository is not modified by controller implementation
   validation.
10. Obtain independent verification, because this changes provenance/authority
    transport semantics.

Successful controller correction alone does not authorize calibration
execution.  After implementation and independent verification, a separate new
calibration execution-authority candidate is still required at the frozen
repository implementation commit.

## Report-only completion declaration

```text
created_report = reports/reason_router_p3w7_seed8192_reason_loss_calibration_collector_start_marker_lifecycle_recovery_implementation_authority_spec_candidate.md
files_created = 1
existing_files_modified = 0
cm.ps1_modified = false
production_code_modified = false
tests_modified = false
run_registry_modified = false
training_executed = false
evaluation_executed = false
calibration_executed = false
aggregation_executed = false
model_loaded = false
tokenizer_loaded = false
cuda_used = false
kaggle_used = false
staged = false
commit = false
push = false
```

Final `git status --short`:

```text
?? reports/reason_router_p3w7_seed8192_reason_loss_calibration_collector_start_marker_lifecycle_recovery_implementation_authority_spec_candidate.md
?? reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/
?? reports/reason_router_p3w7_seed8192_revised_split_a0_runs/
```

The latter two untracked directories pre-existed this report creation and were
not inspected, changed, staged, or otherwise acted upon.
