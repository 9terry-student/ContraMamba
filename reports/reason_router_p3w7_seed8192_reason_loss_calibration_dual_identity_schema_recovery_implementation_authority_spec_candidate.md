# Seed8192 Calibration Dual-Identity Schema-Recovery Implementation Authority Candidate

## Verdict, authority, and phase boundary

```text
verdict = PASS_READY_FOR_INDEPENDENT_DUAL_IDENTITY_RECOVERY_AUTHORITY_VERIFICATION
phase = REPORT-ONLY / PROVENANCE RECOVERY AUTHORITY DESIGN
current_HEAD = a44c6394323da14b423654a88a11a9d0ed3507f6
```

This candidate addresses only the `DUAL_IDENTITY_SCHEMA_DEFECT` identified by
the independent provenance audit.  It authorizes no implementation now and no
training, calibration execution, evaluation, model/tokenizer loading, CUDA,
Kaggle, commit, or push.  It changes no scientific claim or calibration
estimator.

Current frozen CLI-correction execution authority, applicable only to the
historical retry2 context:
`reports/reason_router_p3w7_seed8192_reason_loss_calibration_execution_authority_cli_correction_spec_candidate.md`,
blob `aa253ee04199528d71e3ebc610824accea0bbe4f`.

Parent defective execution authority:
`reports/reason_router_p3w7_seed8192_reason_loss_calibration_execution_authority_spec_candidate.md`,
blob `dae1a848d709f8cf113a212fde6fdf79f95058bb`.

Governing calibration authority:
`reports/reason_router_p3w7_seed8192_reason_loss_calibration_authority_spec_candidate.md`,
commit `4a5494df4f5e049c1673cf337d3a763064a37751`, blob
`05ef59aa6a7f94c92cc1e795d26eb3c5a83cade9`.

The verified implementation identity remains commit
`47ff8d16a28a17cb3dca2104c51b4d63c67d6109` with blobs:

```text
scripts/train_controlled_v6b_minimal.py = bb1639525916d99cbc4d458ba4771c277cd4d46b
scripts/aggregate_reason_router_p3w1_calibration.py = 418f747df0e4224bc25124d0053e235fa8bd95b9
tests/test_reason_router_p3w1_calibration.py = 723c63297338aa4d412b39cfcafb19bd7d7e798a
```

## Deterministic dual identities

The completed audit and direct committed-dataset/split recomputation establish
that the same 2,880 Seed8192 train records, in the same order, have two valid
but semantically distinct serializations.  Dataset Git blob is
`2b6829bf04a1333446aac6f7c603d9178b339f36`; canonical/execution SHA256 is
`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`.

```text
split_seed = 8192
dev_ratio = 0.2
train_rows = 2880
dev_rows = 720

p4x_ordered_train_row_sha256 =
478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8

p3w1_ordered_train_row_label_sha256 =
4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
```

The frozen byte-level algorithms are exactly:

```text
p4x_ordered_train_row_sha256:
SHA256 over, for each actual ordered train_record:
str(record["id"]) + "\t" + str(record["pair_id"]) + "\n"

p3w1_ordered_train_row_label_sha256:
SHA256 over, for each actual ordered train_record:
_p2_row_identity(record) + "\t" + _p2_reference_pair_id(record) + "\t"
+ _s28e_normalize_label(record["final_label"]) + "\n"
```

For this dataset the row and pair resolvers yield the P4-X `id`/`pair_id`
values, and labels are canonical; the semantic difference is therefore the
additional TAB plus normalized final label in every P3-W1 line.  Equality is
not expected or permitted as an inference: both values must be separately
computed from the actual ordered `train_records` used by the calibration
forward, and the two expected values are intentionally unequal.

## Frozen v2 schema contract

Later implementation must replace the v1 artifact schema names with exactly:

```text
unit schema_version = reason_router_p3w1_calibration_unit_v2
aggregate schema_version = reason_router_p3w1_calibration_aggregate_v2
```

Every v2 unit and aggregate must contain the following two distinct,
authoritative fields with the exact meanings above:

```text
p4x_ordered_train_row_sha256
p3w1_ordered_train_row_label_sha256
```

The v1 `ordered_train_row_identity_hash` is overloaded and **must not remain
authoritative in v2**.  It must not be silently reinterpreted, accepted as an
alias, or used as a v2 fallback.  This is fail-closed schema evolution: a v1
artifact is not a v2 artifact merely because other fields match.  The v2
aggregate must retain both observed identities and both corresponding expected
identity values explicitly; no ambiguous single expected-identity contract is
allowed.

## Later implementation authorization only

This candidate authorizes a later bounded implementation change only in:

```text
scripts/train_controlled_v6b_minimal.py
scripts/aggregate_reason_router_p3w1_calibration.py
tests/test_reason_router_p3w1_calibration.py
```

No additional production file is justified.  The current trainer maps the two
serializations to `_p2_row_identity_hash` and `_p3w1_ordered_train_identity`;
the exporter is the point where the latter is currently overloaded.  The
current pure-JSON aggregator owns v1 schema/required-field/CLI validation.

### Trainer v2 requirements

The exporter must independently compute P4-X over the actual ordered
`train_records`, exactly using `id<TAB>pair_id<LF>`, and independently preserve
the P3-W1 row+pair+normalized-label hash.  It must verify P4-X against the
frozen Seed8192 P4-X expected value before artifact write, preserve the 2,880
row-count gate and all valid dataset/sidecar/provenance/split gates, write both
v2 fields, and fail closed before publication on every mismatch.  Copying the
frozen P4-X constant to the artifact is forbidden.

It must not change split construction, membership, row ordering, labels,
normalized-label semantics, reason/final loss, gradient ownership,
first-blocker semantics, architecture, estimator, seed list, forward batch
size, or CUDA/model behavior except provenance export/validation.

### Aggregator v2 requirements

The pure-JSON v2 aggregator must require both fields from every unit, validate
each independently against its expected hash, require all three units to agree
on each identity separately, and write both actual and expected identities to
the v2 aggregate.  It must use these exact required CLI arguments:

```text
--expected-p4x-ordered-train-row-sha256
478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8

--expected-p3w1-ordered-train-row-label-sha256
4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
```

The old `--expected-ordered-train-row-identity-hash` contract must not be
authoritative for v2.

## Required later CPU-only tests

The later test change must use the real committed dataset and Seed8192 split,
without model, tokenizer, or CUDA, and prove all of the following:

1. exact recomputation: 2,880 train rows, P4-X `478013...`, and P3-W1
   `4a66cc...`;
2. the hashes are intentionally unequal while covering the same train rows in
   the same order under distinct serializations;
3. trainer/export v2 units contain both correct fields;
4. the aggregator accepts only when both expected identities match and rejects
   correct-P4-X/wrong-P3-W1, wrong-P4-X/correct-P3-W1, swapped hashes, either
   missing field, and a v1 overloaded-only artifact;
5. reversing ordered train records changes both identities;
6. a label mutation with unchanged row/order leaves P4-X unchanged and changes
   P3-W1; and
7. a row-id or pair-id mutation affects both as defined.

Synthetic placeholder hashes alone (for example `"d" * 64`) cannot establish
the end-to-end authority-compatibility proof.

## Historical retry2 disposition and execution transition

```text
retry2 run_name = p3w7-seed8192-reason-calibration-seed180-retry2
retry2 execution_commit = a44c6394323da14b423654a88a11a9d0ed3507f6
retry2 command_sha256 = 5cbba8dba815fc50aef822099d0a678f37ebe1f456cca7dfb6b7b4b676c2fe06
retry2 artifact = reports/reason_router_p3w7_seed8192_reason_loss_calibration/seed180/calibration_unit.json
retry2 artifact_sha256 = c879780d8049c978e4d951c83632cca11946c4dd48c694b80788da68ccbb9bbc
retry2 artifact_bytes = 4253
retry2 observed_overloaded_identity = 4a66ccbdc8e13758e2fcebce50a15bf93cd7a01a7750a5ab71bfcb76f188071b
```

Retry2 process execution is a historical process-level PASS.  Its artifact is
provenance-INVALID for corrected calibration evidence, and its scientific
measurement is NOT ACCEPTED.  Artifact mutation, post-hoc identity-field
replacement, and deriving a v2 accepted unit from the v1 artifact are
FORBIDDEN.

`a44c6394323da14b423654a88a11a9d0ed3507f6` remains historical authority for
retry2 only and cannot authorize v2 execution.  After the bounded correction
is completed, independently verified, and manually frozen, a new
execution-authority candidate is required; only that future freeze commit may
be the calibration execution commit.  A fresh seed180 calibration forward is
required from a clean output namespace.  The reserved/recommended future name
is `p3w7-seed8192-reason-calibration-seed180-retry3`; its exact command is not
authorized here.  Seeds 181/182 remain unauthorized until the immediately
preceding v2 unit passes provenance validation; aggregation remains
unauthorized until all three v2 units validate.

The retry2 artifact may not be overwritten.  A future clean exact-commit
Kaggle clone has no imported retry2 artifact in its worktree, so reuse of the
canonical namespace is possible only if the future execution authority proves
no collision/no overwrite during preflight.  This candidate invents no
migration or overwrite procedure.

## Separate collector blocker

`COLLECTOR_START_MARKER_LIFECYCLE / PROVENANCE_DISCOVERY_DEFECT` is separate
from this repository schema defect.  The artifact mtime was
`2026-09-09 15:34:05.195412335 +0000`; the collection marker mtime was
`2026-09-09 15:34:05.872464357 +0000`; `find -newer` consequently produced
`FILES_COLLECTED=0`.  The resulting empty handoff ZIP is forbidden as import
input.

This candidate authorizes no cm/collector change.  Collector recovery requires
a separate authority and audit because cm is external to this repository and
ownership for exact marker recreation is unestablished.  Even after v2 is
implemented, new Kaggle execution remains blocked until the controller
separately establishes a valid collection/provenance path.  The two defects
must not be conflated.

## Preserved scientific boundaries and verification

Unchanged: `conditional_first_blocker`, `explicit_local`, A3 measurement
semantics, reason-loss placeholder `0.0`, stage174c clean-polarity-preservation
weight `0.0`, seeds 180/181/182, Seed8192 split and 0.2 dev ratio, train-only
complete logical unit, no backward/optimizer/scheduler, no dev calibration
access, no external/OOD evaluation, no A0 prediction/logit/metric/checkpoint
reuse, pooled `mu_final / mu_reason`, and no historical split174-weight reuse.
No normal A1/A2/A3 training or new calibration execution is authorized, and no
scientific conclusion may be drawn from seed180 loss values.

Independent verification of this candidate must inspect the listed authorities
and the three whitelisted files, verify both source algorithms and hashes,
verify the exact minimal whitelist, run `git diff --check`, and confirm exactly
one intended new report with zero existing-file modifications.

```text
production_code_modified = false
tests_modified = false
existing_authority_modified = false
training_executed = false
evaluation_executed = false
calibration_executed = false
model_loaded = false
tokenizer_loaded = false
cuda_used = false
kaggle_used = false
staged = false
commit = false
push = false
```
