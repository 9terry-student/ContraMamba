# Seed8192 Revised P4-L Phase-IV Prelaunch Static-Control Authority Candidate

## 1. Verdict

`PASS_READY_FOR_FRESH_INDEPENDENT_PHASE_IV_AUTHORITY_VERIFICATION`

```
PHASE_IV = PHASE_IV_PRELAUNCH_STATIC_CONTROL
PHASE_IV_PRELAUNCH_STATIC_CONTROL_IMPLEMENTATION_DELTA_REQUIRED = YES
ACTIVE_SEED8192_REVISED_P4L_PHASE4_PRELAUNCH_STATIC_CONTROL_AUTHORITY = NONE_YET
PHASE_IV_IMPLEMENTATION_OR_STATIC_EXECUTION = NOT_AUTHORIZED_BY_THIS_CANDIDATE
```

The frozen Phase-III trainer has substantial fail-closed *in-process* canonical
identity/provenance validation.  Its `_p2_prepare_reason_supervision` path
derives the exact four applicable binary cohorts for train and dev, checks
binary support, and rejects zero-side degeneracy with
`P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE`.  Those controls are reached
only within the trainer process/startup path; they do not provide an
independently invocable static gate that terminates before a trainer process is
launched.  Therefore option **B** is required: a bounded Phase-IV
implementation delta must be completed, validated under a separately activated
authority, and frozen before any later scientific execution authority may be
considered.

This report is a fail-closed candidate only.  It authorizes no implementation,
static-gate execution, trainer launch, materialization, training, evaluation,
or interpretation of a static result as scientific evidence.

## 2. Opening repository state

The required opening state was authenticated by read-only Git inspection:

| Check | Required / observed | Result |
| --- | --- | --- |
| branch | `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` | PASS |
| HEAD | `d1d9ea97eb50f0a61e798eb4193b7731a1ddc7ba` | PASS |
| upstream tip | same SHA | PASS |
| ahead / behind | `0 / 0` | PASS |
| tracked and staged state | clean | PASS |

Git emitted only pre-existing permission warnings while attempting to inspect
inaccessible pytest-cache directories; it reported no tracked, staged, or
untracked path entries before this candidate was created.

## 3. Authority-chain and Phase-III freeze authentication

All supplied authority identities resolve to commits and form this parent chain:

| Role | Commit |
| --- | --- |
| P2 root-cause authority | `eea0714904ea1f95c42da48e85cd1af4bad23123` |
| split-contract remedy | `c82a164ac460599c68318a3b29180303f12cbc1a` |
| revised split authority | `b4fbb5666d796161f95ae23612ce2448c25063ee` |
| revised P4-L reconstruction/rebinding authority | `ff181f565cefa0a28280c084246862286daf1f2d` |
| upstream implementation delta | `1f05ae3aca63138c482101633412690be213b36d` |
| Phase-II freeze | `ef26310f3532368b9de6cb96a19cb26e7626716d` |
| active Phase-III authority | `b0edd999df9246ad5c9f0940b7375738c5917197` |
| frozen Phase-III implementation | `d1d9ea97eb50f0a61e798eb4193b7731a1ddc7ba` |

`d1d9...` has parent `b0edd999df9246ad5c9f0940b7375738c5917197` and
exactly two changed paths:

```
scripts/train_controlled_v6b_minimal.py
tests/test_reason_router_p4x_trainer_rebind.py
```

Their required frozen blobs are respectively
`8252f5778944974e20c21acfe203e8f7bb5f3218` and
`5699ec92459aebda711fdb94470430cdf9349fce`.  No conflict in this chain was
found.

## 4. Existing Phase-IV artifact search and boundary

No tracked artifact matching this proposed Phase-IV prelaunch static-control
candidate path, nor another tracked Phase-IV/prelaunch static-control authority
artifact applicable to seed8192 revised P4-L, was found.  Historical P4-L,
Phase-II, and Phase-III candidate/evidence artifacts are not Phase-IV authority.

Phase IV is exclusively `PHASE_IV_PRELAUNCH_STATIC_CONTROL`: a deterministic,
fail-closed check of immutable inputs and readiness **before trainer process
launch**.  It preserves the primary failure exactly:

```
FROZEN_SEED174_SPLIT_CONTRACT_INCOMPATIBLE_WITH_A1_A3_DEV_POLARITY_BINARY_READINESS
```

Its mandatory preventive target is the secondary defect:

```
MISSING_PRELAUNCH_P2_APPLICABLE_COHORT_FEASIBILITY_CHECK
PRELAUNCH_P2_APPLICABLE_COHORT_FEASIBILITY_CHECK_REQUIRED = YES
```

This candidate neither changes nor reinterprets the scientific root cause.

## 5. Required identity controls

The future standalone static checker shall fail closed unless all of the
following are authenticated from clean `HEAD` bytes and actual parsed content:

1. exact expected repository root, branch, HEAD, upstream tip, zero ahead/behind,
   and clean tracked/index state;
2. trainer blob `8252...` and focused rebind-test blob `5699...`;
3. source dataset path, tracked Git blob `2b6829bf04a1333446aac6f7c603d9178b339f36`,
   Git/LF SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`,
   and consumed-record semantic SHA256
   `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`;
4. revised canonical sidecar path, blob
   `83d119e327acacda7cff6b4e24c6502898294e03`, Git/LF physical SHA256
   `9bbbb48a3ac0b52cf420c0bcc52019ee85f7528e274b85c60fd7077d347e1f4d`,
   semantic SHA256 `2528a05eb8ab6fa1b80abd86d4860beb36f38921f0bbc71e9a5b56b63ea832c9`;
5. revised provenance path, blob
   `6c970033fae82286452f6d635b94f441d0f3d048`, and Git/LF physical SHA256
   `170647d71d9c074c8bd7e87923b44d590b4159c693348cb335cd91a50ec777e8`;
6. exact schemas `P3W7_SEED8192_REVISED_P4L_EFFECTIVE_INTEGRITY_SIDECAR_V1`
   and `P3W7_SEED8192_REVISED_P4L_INTEGRITY_SIDECAR_PROVENANCE_V1`,
   `lineage_mode = revised-seed8192`, P4-L authority `ff181f...`, split
   authority `b4fbb...`, builder source `149adf32d9e8edbb0e7ea9294f7aeb330a71fc1b`,
   Phase-II activation `cb6f4482b463d5f85331e2a6ddfbbd34499c930a`, and
   Phase-II freeze `ef26310f3532368b9de6cb96a19cb26e7626716d`;
7. execution-record path and blob `0d07e52dd4240a84c60805a4495b18a727e289d3`
   as a static identity only, with an explicit assertion that it is not opened,
   parsed, or otherwise made a runtime dependency by the trainer.

The checker shall reject symlink/substituted paths, untracked input, staged or
unstaged dirt, malformed records, duplicate/misaligned IDs, and any historical
seed174 P4-L path/blob/schema/hash/authority as current revised evidence.
It must consume the authenticated Git blob bytes, not a separately reopened
working-tree copy.  Dataset semantic identity is computed over the records
actually parsed from those authenticated bytes.

## 6. Revised split controls

The checker shall recompute pair membership and ordered row identities from the
authenticated dataset/sidecar join and require all values below:

| Assertion | Required value |
| --- | --- |
| seed; pair count | `8192`; `300` |
| train / dev pairs | `240 / 60` |
| train / dev rows | `2880 / 720` |
| pair leakage | `0` |
| pair universe | `41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2` |
| shuffled | `ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55` |
| train pairs | `f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049` |
| dev pairs | `30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4` |
| ordered train rows | `478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8` |
| ordered dev rows | `7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4` |

## 7. Mandatory applicable-cohort feasibility control

For each parsed/joined row, the checker shall independently rederive the
first-blocker reason from the source exact-binary axes in the immutable order
`FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`; validate the sidecar eligibility,
split, canonical-row, integrity/intervention, and source-join contract; then
form the same actual P2 loss cohorts:

```
frame       = reason_supervision_eligible
predicate   = frame and frame_compatible_label == 1
sufficiency = predicate and predicate_covered_label == 1
polarity    = sufficiency and sufficiency_label == 1
              and final_label in {REFUTE, SUPPORT}
```

It shall count both binary sides (`0` and `1`) in each cohort, separately for
train and dev, and reject zero, absent, malformed, non-binary, or identity-mismatched
cohorts.  A persisted PASS text, provenance count, or sidecar label alone is
insufficient: the result must be produced from the authenticated records and
the above frozen semantics.

Independent read-only parsing during this authoring task produced:

| split | frame 0/1 | predicate 0/1 | sufficiency 0/1 | polarity 0/1 |
| --- | ---: | ---: | ---: | ---: |
| train | `714 / 695` | `119 / 576` | `238 / 338` | `100 / 238` |
| dev | `186 / 174` | `31 / 143` | `62 / 81` | `19 / 62` |

All sixteen required binary sides are non-zero.  The same derivation gave
eligible/ineligible `1769 / 1831` and primary-reason counts train
`FRAME/PREDICATE/SUFFICIENCY/AUTHORIZED = 714/119/238/338`, dev
`186/31/62/81`; this reconciles each cohort to actual frozen data/evidence,
not an asserted readiness string.

## 8. P4-X revised count and authorization-flag controls

Require derived sidecar totals:

```
P2 reason eligible / ineligible = 1769 / 1831
Integrity ELIGIBLE / INELIGIBLE / UNRESOLVED = 1769 / 1562 / 269
Positive margin eligible / ineligible = 695 / 2905
```

`724 / 2876` is historical seed174 expectation and must fail as a current
revised assertion.

Require exact, typed provenance flags:

```
implementation_authorized = True
artifact_materialization_authorized_by_p4l = False
training_admission_released = False
a0_execution_authorized = False
training_authorized = False
evaluation_authorized = False
kaggle_authorized = False
gpu_authorized = False
provenance_physical_sha256_self_certified = False
```

These false P4-L execution flags do not invalidate the separately authorized,
already frozen Phase-II materialization.  They do prohibit treating P4-L itself
as an execution authority.

## 9. Existing and missing control coverage

Existing Phase-III coverage in
`scripts/train_controlled_v6b_minimal.py`—notably
`_p4x_authenticated_head_bytes`, `_p4x_validate_provenance`,
`_p4x_validate_sidecar_rows`, `_p4x_validate_stable_join`, and
`_p4x_validate_canonical_integrity_binding`—authenticates many revised inputs,
schemas, split identities, aggregate P4-X counts, flags, and canonical bytes.
`tests/test_reason_router_p4x_trainer_rebind.py` covers that rebind and selected
failure modes.  In the same trainer process, `_p2_prepare_reason_supervision`
derives the exact four applicable binary cohorts for train and dev, checks both
binary sides, and rejects zero-side degeneracy with
`P2_APPLICABLE_COHORT_BINARY_CLASS_DEGENERATE`.

However, these controls execute only from the trainer’s input-loading/startup
path, after a trainer process has begun.  They do not expose a bounded,
independently invocable prelaunch command that authenticates the frozen trainer
and focused-test blobs as launch prerequisites and performs the required
cohort-feasibility checks before trainer-process launch.  No admissible
prelaunch checker terminates at that process boundary.  Thus
`MISSING_PRELAUNCH_P2_APPLICABLE_COHORT_FEASIBILITY_CHECK` means absence of an
independently invocable **PRELAUNCH** feasibility gate, not absence of cohort
degeneracy logic anywhere in the trainer.  Passing that focused Phase-III
pytest therefore cannot satisfy Phase IV.

## 10. Minimum future implementation fileset

The exact minimal future fileset is two new files only:

| File | Required scope |
| --- | --- |
| `scripts/validate_reason_router_p4x_prelaunch_static_control.py` | a standalone, no-model/no-trainer-launch static checker implementing sections 5–8, emitting a machine-readable PASS only after every check, and fail-closing otherwise |
| `tests/test_reason_router_p4x_prelaunch_static_control.py` | focused static fixtures/tests for clean success and each identity, historical-evidence, split, cohort-side, malformed-record, aggregate-count, flag, and execution-record-nondependency failure |

The checker may share immutable constants/algorithms only by explicitly
authenticated source or a narrow pure helper; it must not import or invoke
`main`, construct a model, load checkpoints, initialize CUDA, or start a
trainer.  No modification to trainer, existing tests, datasets, sidecar,
provenance, producer, split, labels, losses, gradient ownership, EMA,
calibration, ablations, evaluation, or promotion semantics is authorized by
this candidate.

## 11. Future validation command contract

Because the delta is YES, no static validation command is authorized now and
there is no Phase-IV-NO command set.  After a separate Phase-IV implementation
authority is activated and only after its bounded patch exists, the later
static-control execution authority shall permit at most:

```powershell
git diff --check
pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py
python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head d1d9ea97eb50f0a61e798eb4193b7731a1ddc7ba
```

The final command is a static checker, not a trainer CLI.  Its exact expected
HEAD must be rebound only by a later authority if the authorized implementation
commit changes it; no moving branch name is acceptable.

## 12. Scientific/execution boundary and lifecycle

```
TRAINER_PROCESS_LAUNCH_AUTHORIZED = NO
TRAINING_AUTHORIZED = NO
EVALUATION_AUTHORIZED = NO
A0_EXECUTION_AUTHORIZED = NO
CALIBRATION_EXECUTION_AUTHORIZED = NO
A1_A2_A3_EXECUTION_AUTHORIZED = NO
CUDA_GPU_AUTHORIZED = NO
KAGGLE_AUTHORIZED = NO
```

Phase-IV static control is not scientific evidence.  A future static PASS
establishes no scientific claim and does not release any later execution.
No later execution authority may be considered until Phase IV is separately
completed and frozen.

Candidate authoring or verification alone does not activate Phase IV.  Activation
requires: (1) independent verification; (2) exact candidate byte/blob freeze;
(3) exact one-file staging; (4) dedicated authority commit; (5) push; (6)
remote branch-tip verification; (7) parent verification; and (8) remote
candidate blob/body verification.  Only then may the next bounded Phase-IV
action identified here occur.

## 13. Authoring-task validation and final state

No pytest, trainer CLI, producer, materialization, Phase-IV gate, CUDA/GPU, or
Kaggle action was run.  Only read-only Git/file inspection and this one report
creation occurred.  Post-write measurement and final Git-state verification are
recorded in the authoring response, not guessed here.

Candidate path:

`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_prelaunch_static_control_authority_spec_candidate.md`

Remaining blocker: the exact two-file Phase-IV implementation delta has not
been separately authorized, implemented, statically validated, or frozen.

Exact next authorized action:

`FRESH_INDEPENDENT_VERIFICATION_OF_PHASE_IV_PRELAUNCH_STATIC_CONTROL_AUTHORITY_CANDIDATE`
