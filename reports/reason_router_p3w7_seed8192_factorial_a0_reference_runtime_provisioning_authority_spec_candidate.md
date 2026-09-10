# Seed8192 factorial A0-reference runtime provisioning authority — candidate

## 1. Verdict, phase, and authority binding

**Verdict: candidate ready for independent high-risk verification, subject to
the fail-closed gates below.** This is a REPORT-ONLY / STATIC OPERATIONAL
AUTHORITY AUTHORING artifact. It creates no runtime input, executes no
provisioning, and authorizes neither factorial execution nor training,
evaluation, Kaggle, staging, commit, push, controller modification, scientific
input change, or Git-index change.

```text
AT_AUTHORING_TIME = CANDIDATE_ONLY
RUNTIME_PROVISIONING_ALLOWED = NO
FACTORIAL_EXECUTION_ALLOWED = NO
TRAINING_EVALUATION_ALLOWED = NO
KAGGLE_ALLOWED = NO
SCIENTIFIC_CONCLUSION_FROM_THIS_REPORT = NONE

A0_REFERENCE_IDENTITY_AUTHORITY_COMMIT = aa0270a8d974a81928aa2025047b778ae641d7a2
A0_REFERENCE_IDENTITY_AUTHORITY_BLOB = d65ec5a90b5bedefd8424b58fbf594149722ce02
REFERENCE_ARTIFACT_TYPE = training_report_predictions.jsonl

FACTORIAL_EXECUTION_AUTHORITY_COMMIT = <EXACT_FULL_40_HEX_COMMIT_FROZEN_BY_FINAL_FACTORIAL_AUTHORITY>
AA0270A_IS_FACTORIAL_EXECUTION_COMMIT = FALSE
PROVISIONING_AUTHORITY_FREEZE_ALONE_AUTHORIZES_FACTORIAL = FALSE
FINAL_FACTORIAL_AUTHORITY_REQUIRED_BEFORE_RUNTIME_PROVISIONING = TRUE
```

The frozen identity/admissibility authority above is the exclusive authority
for the three paths, bytes, hashes, and admissibility in this report. The
calibration aggregate provisioning authority at
`21403f5e6cff6ca813c6df127c7ee0295998c597` is precedent only, not direct
authority for these prediction references.

No runtime provisioning may occur until a later final factorial authority is
independently verified, frozen, pushed, and remotely authenticated; it must
consume both this authority and the aa0270a identity authority, resolve the
placeholder to its own full 40-hex execution-authority commit, and require
runtime `HEAD` to equal it.

## 2. Frozen exact A0 inputs and same-seed matrix

| Future runs | Sole admitted runtime input (exact relative path) | Bytes | SHA256 |
| --- | --- | ---: | --- |
| seed180 A1/A2/A3 | `reports/reason_router_p3w7_seed8192_revised_split_a0_replacement_runs/seed180/replacement_r1/A0/training_report_predictions.jsonl` | 3937018 | `80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334` |
| seed181 A1/A2/A3 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed181/A0/training_report_predictions.jsonl` | 3935282 | `d7a2d79091e2b076610d58b6e3539a347c29706b796c9364b6998c5995b42472` |
| seed182 A1/A2/A3 | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed182/A0/training_report_predictions.jsonl` | 3938383 | `ae1a1dfd9a844437e032a506716abb68544b16ea1fcdff7eaba14a040a48e650` |

```text
FACTORIAL_A0_REFERENCE_PROVISIONING_MODE = PER_RUN_SINGLE_SAME_SEED_PREMARKER_EXACT_BYTE
AUTHORIZED_REFERENCE_IDENTITIES = 3
FUTURE_FACTORIAL_RUN_COUNT_COVERED = 9
PROVISIONED_A0_REFERENCE_COUNT_PER_RUN = 1
SAME_SEED_REFERENCE_REQUIRED = TRUE
CROSS_SEED_REFERENCE_PROVISIONING_ALLOWED = FALSE
```

Each individual run provisions only its matrix row; it does not provision the
other two. Seed180 historical r2 and every historical split174 reference are
forbidden, regardless of local existence or byte equality. These identities
are authoritative only because aa0270a freezes them; local presence is only a
future source-availability check, never an identity inference.

## 3. Fresh, isolated runtime prerequisite

Every A1/A2/A3 run begins in a fresh or independently isolated checkout pinned
to the resolved final factorial execution-authority commit. A prior run
checkout, particularly one containing result artifacts, is never cleaned,
reset, deleted, or reused to manufacture a clean `cm run` state. This authority
does not set or alter final factorial run order.

```text
FRESH_OR_ISOLATED_CHECKOUT_PER_FACTORIAL_RUN = TRUE
GIT_CLEAN_RESET_FOR_RUN_REUSE_AUTHORIZED = FALSE
PREVIOUS_RUN_OUTPUT_DELETION_AUTHORIZED = FALSE
```

## 4. Current controller static audit

Read-only audit of `C:\Users\Home1\.contramamba\cm.ps1` found:

```text
CONTROLLER_BYTES = 91954
CONTROLLER_SHA256 = d619329478197bee866b91ca95bf52d26dcb8500f350449e3f27e60f6f40800e
CURRENT_CONTROLLER_IDENTITY_VERIFIED = PASS
GIT_INFO_EXCLUDE_CLEAN_GATE_COMPATIBILITY = PASS
START_MARKER_ORDERING_AUDIT = PASS
```

The current `cm run` generated cell checks `git status --porcelain` and fails
closed before it defines/creates the provenance files. Its immutable start
marker is created immediately before wrapped-command execution. Git's ordinary
status semantics omit an untracked file matched by the checkout-local
`.git/info/exclude`; consequently an exactly excluded provisioned input can
pass the unweakened clean preflight. The controller collector uses `find .
-type f -newer "$START_MARKER"` (excluding its normal internal namespaces),
copies discovered files, and records them in its manifest. A provisioned input
written before marker creation is therefore outside normal discovery, provided
its timestamp check below passes. If any future controller hash or these
semantics differ materially, this protocol is BLOCKED; no bypass is allowed.

## 5. Local source verification and payload generation

At actual execution time, a local source process must select exactly the one
matrix row for the intended seed, read the admitted local file, and verify
before encoding: aa0270a report commit/blob binding, exact path, regular-file
existence, byte count, SHA256, and seed mapping. It must prove no substitution.
Missing or nonmatching source is a hard stop: do not regenerate, reconstruct,
reserialize, transform, or substitute it.

The process may generate one future Kaggle provisioning cell embedding the
verified bytes as base64. Base64 is transport only, not identity; this report
contains no payload. The cell must generate only the selected seed's payload,
decode it byte-for-byte at the specified relative target, and verify its exact
decoded count and SHA256 before continuing.

```text
BASE64_TRANSPORT_ALLOWED = TRUE
BASE64_IS_ARTIFACT_IDENTITY = FALSE
JSONL_RESERIALIZATION_ALLOWED = FALSE
BYTE_TRANSFORMATION_ALLOWED = FALSE
```

## 6. Exact per-run pre-marker provisioning protocol

For exactly one future factorial run, in this order:

1. Start the fresh/isolated checkout at the already-resolved full execution
   commit. Before provisioning, require `git rev-parse HEAD` exactly equals it;
   `git diff --name-only`, `git diff --cached --name-only`, and
   `git status --porcelain` are each empty.
2. Require the one exact target reference path is absent. Existing target bytes
   are an unknown source and cause STOP, even if they hash correctly.
3. Decode/write exactly one verified same-seed byte stream to that target.
   Require a regular file, exact expected byte count and SHA256, untracked
   status, and absence of either other authorized A0 reference path.
4. Preserve all existing `.git/info/exclude` bytes/content and append only one
   exact relative target-path line. Do not change tracked `.gitignore`, Git
   index, staging area, global excludes, either complete A0 root, all
   `reports/`, a wildcard directory, or the factorial output namespace.
5. Require `git check-ignore -v -- <target>` to identify that exact target via
   the checkout-local `.git/info/exclude`, and require no broader rule is used.
6. Re-run empty `git diff --name-only`, empty `git diff --cached --name-only`,
   and empty `git status --porcelain`; then rehash/recount the target. Only
   then may the future final authority permit its normal `cm run` cell.

```text
PROVISIONING_OCCURS_BEFORE_CM_RUN_START_MARKER = TRUE
TRACKED_GITIGNORE_MODIFICATION_ALLOWED = FALSE
GIT_INDEX_MODIFICATION_ALLOWED = FALSE
CM_CLEAN_GATE_BYPASS_ALLOWED = FALSE
LOCAL_EXCLUDE_SCOPE = EXACT_SINGLE_REFERENCE_PATH
FACTORIAL_OUTPUT_NAMESPACE_IGNORED = FALSE
A0_ROOT_WILDCARD_IGNORE_ALLOWED = FALSE
```

## 7. Input immutability, marker ordering, and collection

The provisioned JSONL is read-only scientific input. Revalidate exact bytes
and SHA256 immediately before `cm run`, and immediately after exit 0 before
collection. The trainer/run must not rewrite, replace, touch intentionally, or
otherwise mutate it. Also require the provisioned input is not newer than the
controller start marker (for example, `test <target> -nt <marker>` must be
false); if unavailable or false, STOP. This is feasible under the audited
marker layout and proves the normal post-marker collector must not discover
the pre-provisioned input.

After success, without GPU shutdown, restart, or runtime termination,
immediately run the final authority's normal authenticated collector in the
same uninterrupted runtime. Audit its manifest before accepting/downloading
handoff evidence: expected factorial outputs must be present as required by
the final authority; the provisioned target and any other A0 reference must be
absent. Manifest failure or input presence invalidates the handoff and stops
acceptance/download as valid evidence. GPU/session shutdown is allowed only
after run exit 0, post-run rehash PASS, collector PASS, ZIP creation, manifest
audit, and ZIP download.

```text
A0_REFERENCE_MUTATION_AUTHORIZED = FALSE
POST_RUN_A0_REFERENCE_IDENTITY_REVALIDATION_REQUIRED = TRUE
PROVISIONED_A0_REFERENCE_IN_COLLECTOR_MANIFEST = FALSE
SAME_RUNTIME_RUN_TO_COLLECTION = TRUE
```

## 8. Import boundary and failure disposition

The provisioned input is not a generated factorial output and must not enter
the imported factorial result set. Normal `cm import` remains required when
the final execution authority permits it, but local import/audit must never
overwrite the frozen local source based on the matching relative path. A
handoff containing an A0 input reference is contaminated and invalid, not an
import candidate.

```text
A0_REFERENCE_IS_FACTORIAL_OUTPUT = FALSE
A0_REFERENCE_IMPORT_FROM_FACTORIAL_HANDOFF_ALLOWED = FALSE
```

Fail closed, requiring controller disposition, on any missing source, hash or
count mismatch, seed mismatch, target preexistence, decode failure, tracking,
exclude mismatch, dirty state, unresolved/wrong execution commit, controller
mismatch, input mutation, collector contamination, runtime restart before
collection, or other provenance mismatch. Never substitute a seed, use r2 or
split174, reconstruct predictions, rerun A0, change trainer, weaken the clean
gate, run `git add`, `git clean`, or `git reset`, delete user artifacts, or
automatically construct a recovery stack.

## 9. Relationship to blocked factorial candidate and lifecycle

The read-only blocked candidate remains
`reports/reason_router_p3w7_seed8192_a1_a2_a3_factorial_execution_authority_spec_candidate.md`,
with SHA256
`f1a37e1a63dc46bb0290d239dc9c2b383c221b33b4d02bfa904235252dabd5c5`.

```text
FACTORIAL_CANDIDATE_STATUS = BLOCKED_PENDING_A0_REFERENCE_PROVENANCE_AND_RUNTIME_PROVISIONING
```

This candidate does not unblock it merely by authoring or freeze. Only after
independent verification, exact candidate byte/SHA/Git-blob freeze, explicit
single-file staging, dedicated user commit, push, and remote commit/blob
authentication may that blocked candidate receive the smallest bounded
correction to consume aa0270a and the eventual frozen provisioning authority.
It must then be independently reverified and frozen. Neither authority proves
A1/A2/A3 success, superiority, improvement, mechanism, causal evidence,
promotion, significance, or dev/test/OOD performance.

PASS_READY_FOR_INDEPENDENT_SEED8192_FACTORIAL_A0_REFERENCE_RUNTIME_PROVISIONING_AUTHORITY_VERIFICATION
