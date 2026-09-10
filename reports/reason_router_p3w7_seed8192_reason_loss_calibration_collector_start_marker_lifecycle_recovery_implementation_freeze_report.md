# Collector Start-Marker Lifecycle Recovery Implementation Freeze Report

## Final verdict

```text
PASS_COLLECTOR_START_MARKER_RECOVERY_IMPLEMENTATION_FROZEN_READY_FOR_MANUAL_COMMIT
```

## Governing authority

```text
repository_HEAD = 4d0b55e5258e7b6adf6f39c82aeb3c7db3c72ed7
authority_blob = c66973c83bebde9869b3f648c319cba4a33c27ec
defect = MUTABLE_DETERMINISTIC_RUN_START_MARKER_REUSE_DEFECT
```

Frozen implementation authority report:

```text
reports/reason_router_p3w7_seed8192_reason_loss_calibration_collector_start_marker_lifecycle_recovery_implementation_authority_spec_candidate.md
```

## External controller identities

```text
pre:
bytes=91498
SHA256=09097e460ce9f05d5ead09ad3ee9499ac6b9da4298d0a69e8abde6006e5facae

post / frozen:
bytes=91954
SHA256=d619329478197bee866b91ca95bf52d26dcb8500f350449e3f27e60f6f40800e
```

## Frozen implemented semantics

A single-use pre-existence guard now covers `LOG_FILE`, `META_FILE`, `COMMAND_FILE`, and `START_MARKER` before `COMMAND_FILE` creation. An existing namespace fails with exit 45. The former `touch "$START_MARKER"` is removed. Marker creation is create-exclusive/create-only using noclobber semantics. Create failure exits 46.

Required resulting order:

```text
repository gate
<
commit gate
<
dirty gate
<
provenance path definitions
<
single-use path guard
<
command file creation
<
command hash validation
<
exclusive start-marker creation
<
STARTED_UTC/meta/log
<
command execution
<
FINISHED_UTC/EXIT_CODE
```

## Preserved contracts

```text
41 = repository missing
42 = commit mismatch
43 = dirty worktree
44 = decoded command SHA mismatch
45 = existing provenance namespace
46 = marker exclusive-creation failure
```

The collector still uses `find . -type f -newer "$START_MARKER"`. The collector does not mutate the marker. Generic `FILE_COUNT=0` collector behavior was not changed. `run-registry.json` was not modified.

## Static independent-verifier evidence

```text
POWER_SHELL_PARSE_ERRORS=0
BASH_RUN_WRAPPER_SYNTAX=PASS

Static lifecycle audit = PASS
exit mapping = PASS
touch removal = PASS
create-only semantics = PASS
collector invariance = PASS
run-registry unchanged = PASS
```

The independent verifier initially returned BLOCKED only because its orchestration environment could not execute the required Git Bash disposable harness. No implementation defect was found by that verifier. That original BLOCKED invocation is not relabeled as PASS.

## Completed local dynamic closure evidence

```text
PREEXIST_LOG_REJECT=PASS
PREEXIST_META_REJECT=PASS
PREEXIST_COMMAND_REJECT=PASS
PREEXIST_MARKER_REJECT=PASS

SYNTH_FIRST_EXIT=0
SYNTH_SECOND_EXIT=45
SYNTH_EXEC_COUNT=1
COMMAND_BYTES_UNCHANGED=true
META_BYTES_UNCHANGED=true
LOG_BYTES_UNCHANGED=true
MARKER_BYTES_UNCHANGED=true
MARKER_MTIME_UNCHANGED=true
SECOND_COMMAND_EXECUTED=false
SYNTH_SINGLE_USE_PROVENANCE=PASS

CREATE_ONLY_FIRST=PASS
CREATE_ONLY_SECOND_REJECTED=PASS
CREATE_ONLY_MTIME_UNCHANGED=PASS

RACE_CREATOR_SUCCESS_COUNT=1
RACE_CREATOR_FAILURE_COUNT=1
RACE_MARKER_EXISTS=true
RACE_LOSER_COMMAND_EXECUTED=false
CREATE_ONLY_RACE_FAIL_CLOSED=PASS

BAD_HASH_EXIT=44
BAD_HASH_COMMAND_REMOVED=true
BAD_HASH_MARKER_ABSENT=true
BAD_HASH_META_ABSENT=true
BAD_HASH_LOG_ABSENT=true
BAD_HASH_COMMAND_NOT_EXECUTED=true
BAD_HASH_FAIL_CLOSED=PASS

COLLECTOR_DYNAMIC_HARNESS=PASS
DYNAMIC_HARNESS_EXIT=0
ALL_DYNAMIC_REQUIREMENTS=PASS
SYNTH_TEMP_CLEANUP=PASS
```

## Post-validation invariance

```text
cm.ps1 after validation:
bytes=91954
SHA256=d619329478197bee866b91ca95bf52d26dcb8500f350449e3f27e60f6f40800e

run-registry pre/post:
1d1f61ab70b44b0e7760bad1eee9d9728313c0575b82ecd4e59169fc99ef1496

RUN_REGISTRY_UNCHANGED=true
CM_CONTROLLER_UNCHANGED_DURING_VERIFY=true

repository HEAD after validation:
4d0b55e5258e7b6adf6f39c82aeb3c7db3c72ed7

tracked repository delta:
none
```

## Historical retry2 disposition

```text
run = p3w7-seed8192-reason-calibration-seed180-retry2

process execution = historical PASS
artifact provenance = INVALID
measurement = NOT ACCEPTED
artifact mutation = FORBIDDEN
marker recreation = FORBIDDEN
v1 -> v2 promotion = FORBIDDEN
FILES=0 ZIP = FORBIDDEN
historical marker mutation actor = NOT PROVEN
```

The controller correction is prospective.

## Scientific and execution boundary

This freeze does NOT authorize retry3; seed181/182 calibration; aggregation; training/evaluation; model/tokenizer loading; CUDA; Kaggle; or scientific interpretation.

Fresh seed180 retry3 requires a separate execution authority at the frozen dual-identity repository implementation and frozen corrected controller.

## Final state

```text
git diff --check = PASS
git diff --name-only = empty
git diff --cached --name-only = empty
```

Final git status may contain only:

```text
?? reports/reason_router_p3w7_seed8192_reason_loss_calibration_collector_start_marker_lifecycle_recovery_implementation_freeze_report.md
?? reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/
?? reports/reason_router_p3w7_seed8192_revised_split_a0_runs/
```

## Freeze report identity

The final validated identity is recorded after this report's content is written:

```text
UTF-8 validity = true
BOM = absent
CR count = 0
terminal LF count = 1
trailing whitespace = none
```

Expected delta: exactly one new freeze report.
