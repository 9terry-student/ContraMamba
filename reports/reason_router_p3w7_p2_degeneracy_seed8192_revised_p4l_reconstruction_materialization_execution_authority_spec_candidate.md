# Seed8192 Revised P4-L Phase-II Reconstruction/Materialization Execution Authority Spec Candidate

## 1. Verdict and Lifecycle Status

`PHASE_II_RECONSTRUCTION_MATERIALIZATION_AUTHORITY_CONTENT`

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

`MATERIALIZATION_AUTHORITY_ADMISSIBLE`

`ACTIVE_SEED8192_REVISED_P4L_RECONSTRUCTION_MATERIALIZATION_AUTHORITY = NONE_YET`

`MATERIALIZATION = NOT_AUTHORIZED_BY_THIS_CANDIDATE`

`TRAINING/EVALUATION/CUDA/KAGGLE = NOT_AUTHORIZED`

This is a report-only, fail-closed candidate for one exact local CPU filesystem reconstruction/materialization after its separate activation lifecycle completes. It does not itself activate authority; it does not invoke the producer, `--materialize`, training, evaluation, CUDA, Kaggle, network access, a trainer, a model, or a checkpoint; and it creates no P4-L artifact.

The candidate independently answers the Phase-II question as follows:

`CAN_THE_FROZEN_149adf32_IMPLEMENTATION_BE_SAFELY_AUTHORIZED_FOR_ONE_EXACT_PHASE_II_MATERIALIZATION = YES`

That conclusion is limited to the exact producer, source identities, revised lineage mode, output namespace, preflight, and one-shot policy specified here. It is not a Phase-III consumer-rebind, A0, calibration, training, evaluation, or promotion authorization.

## 2. Authority Chain Authentication

Authority precedence and roles authenticated for this candidate:

| Role | Immutable identity | Determination |
|---|---|---|
| Frozen Phase-I implementation | `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b` | Exact frozen producer/test implementation to be considered for Phase II |
| Phase-I implementation-delta authority | `1f05ae3aca63138c482101633412690be213b36d` | Parent of frozen implementation |
| Revised P4-L reconstruction/rebinding/provenance authority | `ff181f565cefa0a28280c084246862286daf1f2d` | Defines revised design and explicitly withholds reconstruction/materialization execution |
| Revised split authority | `b4fbb5666d796161f95ae23612ce2448c25063ee` | Binds seed8192 and its exact split identities |
| Split-contract remedy | `c82a164ac460599c68318a3b29180303f12cbc1a` | Supports active split-contract remedy lineage |
| P2 root-cause authority | `eea0714904ea1f95c42da48e85cd1af4bad23123` | Root-cause lineage authority |
| Authority-lineage reconciliation | `1bb08179adb38637e9391491ba72cfd7e9bff3b3` | Confirms execution requires separate explicit authority |
| Unauthorized-execution correction | `0f6e00642fb6126ec86d7b7dde4b84626befca67` | Preserves the explicit-execution-authority boundary |
| Repository contract | `AGENTS.md` | Report-only, provenance-preserving, fail-closed governing rules |

The active revised P4-L authority says that it does not authorize P4-L reconstruction or provenance generation, identifies the future requirement as “a separately authorized reconstruction/materialization step,” and classifies `artifact_materialization_authorized_by_p4l` as a preserved false value. The reconciliation and incident-correction authorities independently preserve the rule that execution is not acquired merely from candidate existence, a commit subject, or a historical action. Thus this candidate is an external lineage authority candidate, not a reinterpretation of P4-L authority.

## 3. Opening Repository State

Observed before candidate creation:

| Check | Required / observed | Result |
|---|---|---|
| branch | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | PASS |
| HEAD | `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b` | PASS |
| upstream | `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` at the same SHA | PASS |
| ahead / behind | `0 / 0` | PASS |
| `git status --short` | empty | PASS |
| `git diff --name-status` | empty | PASS |
| `git diff --cached --name-status` | empty | PASS |
| `git diff --check` | no output, exit 0 | PASS |

No reset, clean, restore, staging, commit, push, training, evaluation, or materialization was performed.

## 4. Frozen Implementation Authentication and Verification History

The implementation commit is present locally and authenticated as:

| Item | Required / observed |
|---|---|
| commit | `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b` |
| parent | `1f05ae3aca63138c482101633412690be213b36d` |
| subject | `Implement seed8192 revised P4-L producer enablement` |
| exact changed files | producer and focused builder tests only |
| producer delta | `+278/-56` |
| test delta | `+166/-0` |
| total delta | `+444/-56` |

The completed independent implementation verification concluded `PASS_READY_FOR_IMPLEMENTATION_FREEZE`. Its relevant verified behavior was: historical seed174 remains the default; revised seed8192 is explicit; revised split is seed8192 with dev ratio 0.2 and exact identities; revised P4-B bridge inputs use canonical HEAD blobs and fail closed for dirty staged/unstaged paths; the primary dataset has Git/LF and semantic validation; a real revised in-memory build passed; materialization was not performed; no future final artifact hash was frozen; and no canonical revised output directory was created. Those in-memory observations are diagnostic precedent only, not Phase-II artifacts or frozen output hashes.

### Exact producer and test identities

| Path | Git blob | canonical Git/LF SHA256 | canonical bytes | canonical LF |
|---|---|---|---:|---:|
| `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py` | `a6f774c4ff79c3d047600d3e559c90c04768088c` | `58bbda4c136323037868fc3b0b4a6d99d651932383bf6147624d34b0da207e6f` | 58799 | 1242 |
| `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py` | `c7b87f7a675023748a7f7c6313dc30f7aeba8a55` | `074b1b5f27aaecb64db9fd512237c83678ab1158285a5f56fddfbcf2f8cc8a89` | 22944 | 541 |

## 5. Phase-II Admissibility and Provenance Semantics

The source audit found revised mode `revised-seed8192`, revised fixed authority and split commits, the exact fixed output naming rule, exact expected split identities, canonical HEAD-blob consumption for revised P4-B bridge inputs, no-overwrite atomic publishing, and the requested `--created-at`, `--lineage-mode`, and `--materialize` interface.

`artifact_materialization_authorized_by_p4l = False` is semantically compatible with a separately activated Phase-II materialization. Its name and the revised authority’s explicit classification mean only that the P4-L reconstruction/rebinding/provenance authority did not itself authorize materialization. It must remain false in generated provenance. It neither asserts that no external Phase-II authority exists nor purports to encode this later authority in a field with a different meaning.

The same audit found `implementation_authorized = True`, while `training_admission_released`, `a0_execution_authorized`, `training_authorized`, `evaluation_authorized`, `kaggle_authorized`, and `gpu_authorized` are all false. For the eventual artifact, these values remain truthful: the producer implementation is frozen/authorized in its distinct Phase-I lineage; P4-L itself did not authorize materialization; and no training-related authorization is conferred. The externally activated Phase-II authority is evidenced by its own immutable activation lineage and execution record, not falsely encoded into `artifact_materialization_authorized_by_p4l`.

Therefore:

`MATERIALIZATION_AUTHORITY_ADMISSIBLE`

No schema or producer change is needed before safe Phase-II execution. If this semantic interpretation cannot be independently reproduced from the cited authority bodies, the required verdict is `BLOCKED_PHASE_II_MATERIALIZATION_AUTHORITY_PROVENANCE_SEMANTICS` and no materialization may occur.

## 6. Builder Source Identity Representation

The producer binds `builder_source_commit` from the supplied `--builder-commit` but computes `builder_source_sha256` by hashing its physical working-tree source file. The authenticated canonical Git representation is the producer blob and Git/LF SHA identity in Section 4. The verified Windows physical observation before freeze was:

| representation | SHA256 | bytes | LF | CR / CRLF |
|---|---|---:|---:|---:|
| Git canonical LF | `58bbda4c136323037868fc3b0b4a6d99d651932383bf6147624d34b0da207e6f` | 58799 | 1242 | 0 / 0 |
| clean Windows working tree | `a24e2c33ac7104dfac7fc8464e1c19e2ac7a77c99ff162b7fdf946066c4f9134` | 59722 | 1242 | 923 / 923 |

This is admissible, rather than a provenance contradiction, because the two identifiers represent distinct facts: `builder_source_commit` plus canonical Git blob/SHA identifies the immutable reviewed implementation, while `builder_source_sha256` truthfully records the exact physical source bytes executed. The future preflight must (a) confirm both index and worktree cleanliness for the producer, (b) resolve the frozen HEAD blob to `a6f774c4ff79c3d047600d3e559c90c04768088c`, (c) confirm canonical Git/LF SHA `58bb...e6f`, and (d) measure and report the exact physical working-tree SHA immediately before execution. A differing physical SHA is permitted only when the path is Git-clean and content-equivalent to the frozen commit under Git’s checked-out representation; it is not a substitute for blob authentication. A dirty or non-equivalent path blocks materialization.

If an independent verifier finds that the producer’s physical SHA cannot be paired with the canonical blob identity under these conditions, the required verdict is `BLOCKED_PHASE_II_MATERIALIZATION_AUTHORITY_BUILDER_SOURCE_IDENTITY`.

## 7. Frozen Inputs and Exact Output Namespace

### Source dataset contract

| Item | Frozen value |
|---|---|
| path | `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` |
| Git/LF SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| semantic SHA256 | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| rows / pairs | `3600 / 300` |

No regeneration, edits, reorder, label change, or historical-artifact substitution is authorized.

### Exact split contract

| Item | Frozen value |
|---|---|
| seed / dev ratio | `8192 / 0.2` |
| train/dev pairs | `240 / 60` |
| train/dev rows | `2880 / 720` |
| pair leakage | `0` |
| pair universe | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` |
| dev/train pair SHA | `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` / `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` |
| shuffled sequence | `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| ordered train/dev rows | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` / `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |
| historical seed174 dev reference | `259bfce57e85121d6c1adccd20f3ac070108ff6310cfff546a2edd054835899d` |

### Exact future namespace

The future directory is exactly:

```text
reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/
```

It may contain exactly these two files:

```text
p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl
p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json
```

The directory must be absent before execution, including absence as a file or symlink. No latest/glob selection, alternate builder commit, overwrite, historical reuse, or alternate directory is authorized.

## 8. Authorized Execution Shape and Platform

The eventual execution is only a local CPU filesystem materialization: no Kaggle, GPU, CUDA, trainer, model, checkpoint, or network requirement.

After this candidate has completed its activation lifecycle, the controller may use only this command shape with an exact canonical repository root and the single timestamp captured under Section 9:

```powershell
python scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py `
  --repo-root <exact-root> `
  --builder-commit 149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b `
  --created-at <single-captured-UTC-value> `
  --lineage-mode revised-seed8192 `
  --materialize
```

This candidate supplies no executable authorization until activation; this is a future command contract, not a command to execute now. Historical mode is not authorized for Phase II.

## 9. Created-at Policy

Immediately before the one authorized producer invocation, capture exactly one UTC RFC3339 timestamp into one shell variable, echo/report that same value, and pass it unchanged through `--created-at`. Do not use the producer’s `datetime.now()` fallback, do not capture a second timestamp, and do not guess or hard-code a future timestamp. The captured value becomes execution evidence and is intentionally part of generated physical bytes; semantic sidecar hashing excludes only `created_at` under the frozen implementation’s existing rule.

## 10. Fail-Closed Pre-Execution Conditions

All conditions below must pass after activation and before timestamp capture. Any mismatch means `NO_MATERIALIZATION`.

1. This exact Phase-II authority is independently verified, dedicatedly frozen, pushed, and remote body/blob/tip verified under Section 14.
2. Branch is exactly `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`.
3. HEAD is exactly the verified Phase-II authority activation commit, whose implementation freeze binding is `149adf32...`.
4. The upstream is the named origin branch and ahead/behind are both zero.
5. No staged or unstaged changes exist.
6. No untracked files or artifacts relevant to the run exist; specifically none may shadow inputs, the output directory, or target-parent staging namespace.
7. The producer at HEAD resolves to Git blob `a6f774c4ff79c3d047600d3e559c90c04768088c`; its canonical Git/LF SHA, bytes, and LF count equal Section 4.
8. The focused test path at HEAD resolves to Git blob `c7b87f7a675023748a7f7c6313dc30f7aeba8a55` and its canonical identity equals Section 4.
9. `149adf32...`, its parent, subject, exact two-file diff, and implementation verification history are authenticated again.
10. The producer file is index-clean and worktree-clean, content-equivalent to the frozen commit, and its physical SHA256 is measured and recorded immediately before invocation.
11. Frozen source dataset Git/LF and semantic identities, counts, and source-order contract pass; revised P4-B bridge inputs pass the producer’s canonical HEAD-blob identity and clean-path checks.
12. The exact output directory is absent as directory, file, and symlink; no `.p4l-staging-*` collision for that target exists at its parent.
13. The exact one UTC RFC3339 `created_at` value is captured once, reported, and retained unchanged for the invocation.
14. No Phase-III consumer path/hash rebind or other consumer modification has happened before the run.

## 11. One-Shot and Failure Policy

This authority permits at most one successful materialization in the exact namespace. If the target exists before the run, block. If the producer fails before publication and no canonical output appears, stop and inspect the preserved evidence; do not automatically retry with altered parameters. If publication reports failure, stop. If a partial or unexpected canonical directory appears, stop for a failure-recovery audit; do not casually delete, overwrite, or invoke recovery4. No second successful materialization in another namespace is authorized by this authority.

## 12. Post-Materialization Validation Contract

Process success alone is not scientific evidence. An independently verifiable Phase-II success requires all of the following:

1. process success and canonical publication success;
2. the exact canonical directory and exactly the two expected regular files;
3. measured sidecar physical SHA256 and semantic SHA256;
4. measured provenance physical SHA256;
5. sidecar row count 3600, one-to-one source-row coverage, and unique row IDs;
6. every Section 7 dataset identity and Section 7 split identity/count, including zero pair leakage;
7. `builder_source_commit = 149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b` and the measured physical builder-source SHA paired with its canonical Git blob identity;
8. `p4l_authority_commit = ff181f565cefa0a28280c084246862286daf1f2d` and `split_authority_commit = b4fbb5666d796161f95ae23612ce2448c25063ee`;
9. exact output paths and structurally valid provenance JSON;
10. `provenance_physical_sha256_self_certified = false`;
11. `artifact_materialization_authorized_by_p4l = false`, `training_admission_released = false`, and all A0/training/evaluation/Kaggle/GPU authorization flags remain false;
12. sidecar canonical UTF-8, no BOM, LF-only, and final-LF representation; and
13. proof that no historical seed174 artifact was relabeled as revised.

## 13. Future-Hash Boundary and Phase-III Boundary

Before the authorized materialization, and inside this candidate, the only permitted values are:

```text
REVISED_P4L_SIDECAR_PHYSICAL_SHA256 = TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION
REVISED_P4L_SIDECAR_SEMANTIC_SHA256 = TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION
REVISED_P4L_PROVENANCE_PHYSICAL_SHA256 = TO_BE_COMPUTED_BY_AUTHORIZED_RECONSTRUCTION
```

The Phase-I in-memory observations `83e15b00c45b51f551916bef1ffa36a425d6398ec4d33cc39bc8537243067ff4` and `2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9`, and any prior provenance hash, remain diagnostic precedent only. None is a final artifact identity.

This authority does not authorize edits to `scripts/train_controlled_v6b_minimal.py`, edits to `tests/test_reason_router_p4x_trainer_rebind.py`, consumer path/hash rebinding, A0, calibration, A1/A2/A3, training, or evaluation. After independently validated Phase-II artifacts exist, a separate Phase-III consumer-rebind authority is required.

## 14. Candidate Activation Lifecycle

Candidate existence, independent verification, finalization, staging, local commit, and push alone are not activation. Activation is fail closed and requires all of the following:

1. fresh independent verification of this exact candidate body and its one-file scope;
2. staging only this candidate and verifying the staged raw Git blob/body;
3. a dedicated activation commit with parent exactly `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b`;
4. recording the full activation SHA;
5. push to `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`;
6. remote tip verification to that exact activation SHA;
7. remote file/blob verification against the independently verified staged body; and
8. independent remote body-level verification that this authority activates precisely one Phase-II materialization and preserves all listed prohibitions.

`COMMIT_MESSAGE_DOES_NOT_OVERRIDE_BODY_LEVEL_AUTHORITY_STATUS`

Until all conditions pass, `ACTIVE_SEED8192_REVISED_P4L_RECONSTRUCTION_MATERIALIZATION_AUTHORITY = NONE_YET` and materialization remains prohibited.

## 15. Candidate Path, Fileset, and Next Authorized Action

The sole permitted file from this report-only task is:

`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_reconstruction_materialization_execution_authority_spec_candidate.md`

No source, test, dataset, sidecar, provenance, or other report file is authorized. If a second tracked file is required, the result is `BLOCKED_PHASE_II_AUTHORITY_FILESET_INSUFFICIENT`.

Exact next authorized action on this PASS is:

`FRESH_INDEPENDENT_VERIFICATION_OF_THE_PHASE_II_AUTHORITY_CANDIDATE`

Materialization is not recommended or authorized yet.

## 16. Mandatory Phase-II Execution-Authority Record

The following requirements are normative and supplement (rather than relax or replace) every earlier section of this candidate.

`PHASE_II_EXECUTION_RECORD = REQUIRED`

`PHASE_II_EXECUTION_AUTHORITY_TRACEABILITY = MANDATORY_EXTERNAL_EXECUTION_RECORD`

`ARTIFACT_PROVENANCE_ALONE = INSUFFICIENT_TO_CLOSE_PHASE_II`

A successful producer process and two generated P4-L files are not sufficient to complete Phase II. Phase-II artifact/provenance validity requires a separate, durable execution record that binds the materialization to the activated Phase-II authority. The execution record is not one of the two canonical P4-L output files.

The canonical P4-L output directory remains exactly the Section 7 directory:

```text
reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_integrity_sidecar_ff181f565cefa0a28280c084246862286daf1f2d_149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b/
```

and it contains exactly these two files, and no execution record:

```text
p3w7_seed8192_revised_p4l_effective_integrity_sidecar.jsonl
p3w7_seed8192_revised_p4l_effective_integrity_sidecar_provenance.json
```

Exactly one execution record is required for the one-shot authority, outside that directory, at this deterministic path template:

```text
reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase2_execution_record_<PHASE_II_AUTHORITY_ACTIVATION_COMMIT>.json
```

`<PHASE_II_AUTHORITY_ACTIVATION_COMMIT>` is the full 40-character SHA of the eventually activated Phase-II authority commit. It is intentionally unknown while authoring this candidate and is resolved only after freeze -> dedicated authority commit -> push -> remote verification -> activation. This is not circular: the record is created only after activation. `latest`, `final`, `run1`, timestamp-only names, random suffixes, alternate namespaces, overwrites, and alternate-record retries are forbidden.

## 17. Fixed Execution-Record Schema and Values

The record is a JSON object with `schema_version = P3W7_SEED8192_REVISED_P4L_PHASE2_EXECUTION_RECORD_V1` and exactly the following required top-level fields (a nested `split_identity` object is permitted only to retain all split hashes listed below):

```text
schema_version
record_type
phase_ii_authority_commit
phase_ii_authority_path
phase_ii_authority_git_blob
phase_ii_authority_sha256
execution_head
branch
upstream
repo_root
frozen_builder_implementation_commit
producer_path
producer_git_blob
producer_canonical_sha256
producer_worktree_sha256
test_path
test_git_blob
created_at
command_argv
command_sha256
producer_exit_code
producer_stdout
producer_stdout_sha256
output_directory
sidecar_path
provenance_path
sidecar_physical_sha256
sidecar_semantic_sha256
provenance_physical_sha256
sidecar_byte_count
provenance_byte_count
row_count
train_pair_count
dev_pair_count
train_row_count
dev_row_count
pair_leakage
dataset_path
dataset_git_lf_sha256
dataset_semantic_sha256
revised_p4l_authority_commit
split_authority_commit
builder_source_commit
builder_source_sha256
artifact_materialization_authorized_by_p4l
training_admission_released
a0_execution_authorized
training_authorized
evaluation_authorized
kaggle_authorized
gpu_authorized
record_physical_sha256_self_certified
```

Fixed values are:

| Field | Required value |
|---|---|
| `record_type` | `seed8192_revised_p4l_phase2_materialization_execution` |
| `repo_root` | `C:\\p3w7-a0-n3-validated-evidence-analysis` |
| `frozen_builder_implementation_commit` / `builder_source_commit` | `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b` |
| `producer_path` | `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py` |
| `producer_git_blob` | `a6f774c4ff79c3d047600d3e559c90c04768088c` |
| `producer_canonical_sha256` | `58bbda4c136323037868fc3b0b4a6d99d651932383bf6147624d34b0da207e6f` |
| `test_path` / `test_git_blob` | `tests/test_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar_builder.py` / `c7b87f7a675023748a7f7c6313dc30f7aeba8a55` |
| `revised_p4l_authority_commit` | `ff181f565cefa0a28280c084246862286daf1f2d` |
| `split_authority_commit` | `b4fbb5666d796161f95ae23612ce2448c25063ee` |
| `row_count`, `train_pair_count`, `dev_pair_count` | `3600`, `240`, `60` |
| `train_row_count`, `dev_row_count`, `pair_leakage` | `2880`, `720`, `0` |
| `dataset_git_lf_sha256` | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| `dataset_semantic_sha256` | `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b` |
| `artifact_materialization_authorized_by_p4l`, `training_admission_released`, `a0_execution_authorized`, `training_authorized`, `evaluation_authorized`, `kaggle_authorized`, `gpu_authorized` | all `False` |
| `record_physical_sha256_self_certified` | `False` |

`record_physical_sha256_self_certified` must be false: the record must not recursively contain its own physical hash. Its physical SHA256 is measured externally only after the record has been written and reread. `producer_worktree_sha256`, authority identities, `created_at`, command identity, stdout, output identities, byte counts, and generated hashes are observations to be populated only at authorized execution.

The durable `split_identity` representation, if nested, must contain every Section 7 split identity: pair universe, dev/train pair SHA256s, shuffled sequence SHA256, ordered train/dev row SHA256s, and historical seed174 dev reference, in addition to the counts and zero leakage above. Omission is a record mismatch.

## 18. Activated Authority, Command, and Timestamp Bindings

Immediately before execution, independently authenticate all four authority bindings; a branch name alone is never an authority identity:

```text
phase_ii_authority_commit = full activated authority commit
phase_ii_authority_path = reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_reconstruction_materialization_execution_authority_spec_candidate.md
phase_ii_authority_git_blob = exact blob of that path at the activation commit
phase_ii_authority_sha256 = raw SHA256 of the committed authority bytes
execution_head = the same activated authority commit
```

The execution record's `branch` and `upstream` record the independently authenticated active branch and remote-tracking state. The producer path must be Git-clean in both index and worktree immediately before execution; its HEAD blob must equal `a6f774c4ff79c3d047600d3e559c90c04768088c`. Measure the exact execution-working-tree bytes as `producer_worktree_sha256`. After materialization, generated provenance must report `builder_source_commit = 149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b` and `builder_source_sha256` equal to that pre-execution measurement. Any mismatch blocks Phase-II completion. The historical Windows physical SHA `a24e2c33...` is precedent only and is not a universal frozen value.

`command_argv` is the ordered JSON array of the actual arguments and must equal exactly this semantic argv, with `<EXACT_SINGLE_CAPTURED_CREATED_AT>` replaced once by the captured value:

```text
python
scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py
--repo-root
C:\p3w7-a0-n3-validated-evidence-analysis
--builder-commit
149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b
--created-at
<EXACT_SINGLE_CAPTURED_CREATED_AT>
--lineage-mode
revised-seed8192
--materialize
```

Canonical command identity is the SHA256 of UTF-8 bytes from `json.dumps(command_argv, ensure_ascii=False, separators=(",", ":"))`, with no added whitespace. Shell quoting and pretty formatting are not command identity. The exact executed argv must equal the recorded argv.

Capture exactly one UTC RFC3339 timestamp immediately before producer execution. Store and report it once; use the identical string in `--created-at`, `created_at`, and the producer provenance where applicable. There is no internal `now()` fallback, second capture, or post-hoc reconstruction.

## 19. Producer, Artifact, and Path Validation Bindings

Capture the producer exit code and exact observed UTF-8 stdout bytes/text. Require `producer_exit_code = 0`; store that text in `producer_stdout` and its exact-byte SHA256 in `producer_stdout_sha256`. Parse `producer_stdout` as JSON and require its reported `status`, `row_count`, `sidecar_physical_sha256`, `sidecar_semantic_sha256`, and `provenance_physical_sha256` to match independently recomputed post-materialization values. Any mismatch is `PHASE_II_EXECUTION_RECORD_MISMATCH` and Phase II is incomplete.

After successful publication, independently compute from canonical filesystem files the sidecar physical SHA256, sidecar semantic SHA256, provenance physical SHA256, and both byte counts. Record those values only after validating that they equal: (1) the filesystem computation, (2) the applicable producer stdout value, and (3) provenance-contained values where applicable. No hash may be copied blindly from stdout.

`output_directory`, `sidecar_path`, and `provenance_path` must be exact paths under the frozen Section 7 namespace. Resolve them without glob discovery and reject file, directory, or symlink substitution; no alternate location is allowed. Dataset path and both dataset identities must equal Section 7, and all required split identities/counts must be represented as Section 17 requires. Historical relabeling is forbidden.

## 20. Execution-Record Serialization, Creation, and Failure Rules

Execution-record bytes must be UTF-8 without BOM, sorted object keys, compact separators, `ensure_ascii=False`, `allow_nan=False`, exactly one final LF, and no trailing whitespace. The required conceptual serialization is:

```python
json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
```

Creation is atomic and no-overwrite. If the exact record path already exists as a file, directory, or symlink, block and investigate. Do not edit it in place, overwrite it, create an alternate filename, or silently retry with changed parameters.

The mandatory order is:

1. verify activated Phase-II authority;
2. execute preflight;
3. capture `created_at`;
4. execute the exact producer argv;
5. verify producer exit success;
6. independently validate generated sidecar/provenance;
7. construct the record from independently verified observations;
8. atomically/no-overwrite write the exact record path;
9. reread the record;
10. independently verify every required binding; and
11. compute the execution-record physical SHA256 externally.

Phase-II artifact/provenance validity is not complete before step 10. If the producer fails, do not create a successful execution record. If artifacts exist but validation fails, do not create a PASS record. If record creation fails, Phase II remains incomplete even if canonical sidecar/provenance exist. No recovery4 is authorized.

## 21. Evidence Freeze, Traceability, and Phase-III Gate

`PHASE_II_MATERIALIZATION_EXECUTED` and `PHASE_II_EVIDENCE_FROZEN` are distinct statuses. A successful producer invocation can establish only `PHASE_II_MATERIALIZATION_EXECUTED`. `PHASE_II_EVIDENCE_FROZEN` requires independent validation of all three evidence files: the canonical revised sidecar, canonical revised provenance, and exact Phase-II execution record. Later controller review must freeze exact identities, explicitly stage only the evidence, perform `cm ship`, make a dedicated evidence/result commit, push, and remotely verify commit/file/blob identities. This candidate does not pre-authorize staging, commit, or push now.

The future evidence/result commit is Phase-II evidence preservation only, not Phase III, training, evaluation, or consumer rebind. Its SHA must not be invented now and is not required inside the record. The record binds execution to the activation commit; the later evidence commit binds record and artifacts into Git history.

The durable non-optional chain is:

```text
Phase-II activation commit
-> exact command SHA256 / argv
-> exact created_at
-> exact execution HEAD
-> frozen builder commit/blob
-> producer stdout
-> canonical sidecar/provenance paths
-> generated artifact hashes
-> execution record
-> execution-record SHA256
-> later evidence-freeze commit
```

No edge is optional. The generated P4-L provenance may omit the activation SHA because the mandatory separate execution record supplies that authority edge.

Phase III remains blocked until materialization success + artifact validation + execution-record validation + `PHASE_II_EVIDENCE_FROZEN` are externally established. No trainer/P4-X rebind, A0, calibration, training, evaluation, CUDA, GPU, or Kaggle action is authorized.

There is no circularity: the activation SHA is unknown now but known before execution; the record path resolves after activation; generated hashes are computed after materialization; record physical SHA is computed externally after write; and the later evidence/result commit SHA is unknown and deliberately absent from the record.
