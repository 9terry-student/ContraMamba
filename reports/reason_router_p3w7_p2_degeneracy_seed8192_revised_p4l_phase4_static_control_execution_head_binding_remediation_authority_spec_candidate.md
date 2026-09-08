# Seed8192 Revised P4-L Phase-IV Static-Control Execution-Head Binding Remediation Authority Candidate

## 1. Verdict

`PASS_READY_FOR_FRESH_INDEPENDENT_PHASE_IV_EXECUTION_HEAD_BINDING_REMEDIATION_AUTHORITY_VERIFICATION`

This report is a report-only candidate. It authorizes neither an implementation
change nor execution. It records a bounded future remediation authority that
must itself receive fresh independent verification, byte/blob freeze, explicit
staging, dedicated activation commit/push, and remote verification before use.

## 2. Opening repo state

At authoring inspection, the current branch was
`p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` and `HEAD` was
`e9080366cd2de8c70ebe796b5c8b8a51475983ec`. `git status --porcelain=v1`
reported no repository changes. This report does not treat that observation as
execution authorization.

## 3. Defect statement

`ACTIVATION_COMMIT_INVALIDATES_STRICT_EXPECTED_HEAD_EXECUTION_CONTRACT`

The frozen implementation is `fd315f59c6592499a19aa0b0266f93676c199bb7`.
The execution-validation authority activation is
`e9080366cd2de8c70ebe796b5c8b8a51475983ec`, whose sole parent is the frozen
implementation commit. Activation therefore necessarily advances the exact
authorized branch/upstream tip from `fd315f59...` to `e908036...`.

This is an authority/provenance process-boundary defect. It is not scientific
evidence and is not a cohort, split, loss, dataset, or model defect.

## 4. Why e908036 activation and fd315 strict equality are incompatible

The frozen checker's `_validate_repository_identity` requires the exact branch,
current `HEAD == --expected-head`, `@{up} == --expected-head`, and
ahead/behind `0/0`. The activated authority requires:

```text
--expected-head fd315f59c6592499a19aa0b0266f93676c199bb7
```

Once the authority activation commit is in the authorized branch lineage,
clean synchronized `HEAD` and its upstream are `e908036...`, not `fd315...`.
Thus the same repository cannot simultaneously contain the activation in its
clean synchronized current tip and satisfy the frozen checker's two strict
equality checks against `fd315...`. No normal execution state can meet both
requirements.

## 5. Evidence distinguishing activation provenance validity from execution usability

The activation commit has valid recorded parentage: `e908036...^` resolves to
`fd315...`. That proves the observed authority placement in the branch lineage;
it does not prove the frozen checker is executable under the authority's own
strict current-head contract. The frozen source shows `HEAD` and `@{up}` are
each compared to the `--expected-head` argument. Activation provenance is
therefore valid while execution usability is structurally impossible. These are
separate claims and must remain separate.

## 6. Exact remediation fileset

The future implementation remediation is authorized in exactly these files:

```text
scripts/validate_reason_router_p4x_prelaunch_static_control.py
tests/test_reason_router_p4x_prelaunch_static_control.py
```

No third implementation file is authorized. In particular, no prior authority,
checker-adjacent producer, trainer, dataset, sidecar, provenance artifact,
execution record, split, or evidence artifact may be modified.

## 7. Proposed repository-identity semantics

The future remediation shall preserve the existing `--expected-head` CLI
spelling as the smallest safe delta, but explicitly redefine its contract:

```text
--expected-head = immutable implementation anchor, FULL_40_HEX_SHA only
```

The implementation must update all relevant help text, normalization/error
semantics, internal parameter naming where necessary for clarity, and focused
tests so that this is not a silent semantic change. It must not accept a
moving ref, short SHA, tag, or branch name as a substitute.

## 8. Immutable implementation-anchor semantics

`--expected-head` must be a full, lowercase-normalized 40-hex object ID that
resolves to an exact commit. It is immutable provenance for the frozen
implementation anchor, not an assertion that the current execution `HEAD` is
identical to that commit. For this defect's remediation, the anchor is
`fd315f59c6592499a19aa0b0266f93676c199bb7`.

Malformed or non-full SHA input fails closed. A ref name, abbreviated SHA, tag,
or any other moving or symbolic identifier fails closed even if it presently
resolves to the same object.

## 9. Exact upstream-ref semantics

The current checked-out branch must be exactly:

```text
p3w7-a1-a2-a3-factorial-execution-authority-n3-v2
```

Its exact upstream tracking ref must be exactly:

```text
origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2
```

The checker must validate that configured upstream ref identity, rather than
merely accepting any `@{up}` that happens to resolve to the same commit.

## 10. Required ancestry relation

The immutable implementation anchor must be an ancestor of the clean current
`HEAD` (including equality when independently applicable). A non-ancestor
anchor fails closed. Current `HEAD` is explicitly **not** required to equal the
immutable implementation anchor. This ancestry relation admits later authority
commits while retaining an immutable implementation provenance anchor.

## 11. Required clean/synchronized current-HEAD semantics

The current `HEAD` must resolve to the exact tip of the exact upstream ref
above, and `HEAD...@{up}` must report ahead/behind `0/0`. The repository must
be clean with no tracked working-tree change, index change, or untracked path.
Frozen artifact/blob validation remains against the actual clean current `HEAD`
bytes; it is not redirected to the anchor tree. Existing Phase-II
lineage/evidence checks remain unweakened.

## 12. Required tests

The future focused tests must cover at minimum:

- clean synchronized descendant `HEAD` PASS;
- implementation anchor equal to a parent/ancestor PASS;
- implementation anchor not ancestor FAIL;
- malformed or non-full SHA FAIL;
- wrong branch FAIL;
- wrong exact upstream tracking ref FAIL;
- `HEAD` not equal to the exact upstream tip FAIL;
- ahead/behind mismatch FAIL;
- dirty tracked, index, and untracked states FAIL;
- frozen artifact/blob mismatch FAIL;
- regression proving an authority commit after implementation no longer makes
  valid execution structurally impossible;
- no weakening of Phase-II lineage/evidence checks; and
- no trainer, model, checkpoint, or CUDA path introduced.

## 13. Explicitly forbidden recovery shortcuts

No reset, rebase, force-push, branch rewind, remote-tracking-ref falsification,
or forced placement of the branch tip back at `fd315...` is admissible. The
existing `e908036...` authority must not be edited or force-rewritten. Changing
the branch/upstream identity, using a moving ref as an anchor, or weakening
cleanliness, upstream, frozen-blob, Phase-II lineage, or evidence checks is not
remediation.

## 14. Scientific boundary

```text
TRAINER_PROCESS_LAUNCH = NOT_AUTHORIZED
TRAINING = NOT_AUTHORIZED
EVALUATION = NOT_AUTHORIZED
A0/A1/A2/A3 = NOT_AUTHORIZED
CALIBRATION = NOT_AUTHORIZED
CUDA/GPU = NOT_AUTHORIZED
KAGGLE = NOT_AUTHORIZED
PRODUCER/MATERIALIZATION = NOT_AUTHORIZED
```

This bounded remediation concerns repository identity and provenance only. It
does not authorize any trainer, model, checkpoint, CUDA, producer, dataset, or
scientific execution path.

## 15. Candidate raw SHA256/bytes/LF/CR/CRLF/BOM/final-LF/trailing whitespace

The exact raw SHA256 and Git blob of a file cannot be embedded as its own final
literal values: writing either value changes the bytes being identified. The
candidate therefore requires an independent final-byte measurement immediately
after the last write and before activation; the handoff records that measurement
externally. The immutable formatting contract for the final candidate is:

```text
RAW_SHA256 = independently recompute from final raw bytes
BYTES = independently recompute from final raw bytes
LF = LF-only count from final raw bytes
CR = 0
CRLF = 0
UTF8_BOM = false
FINAL_LF = true
TRAILING_WHITESPACE_LINES = 0
```

## 16. Predicted Git blob

```text
PREDICTED_GIT_BLOB = independently compute with git hash-object from final bytes
```

## 17. git diff --check

`git diff --check` completed with no reported whitespace errors after candidate
creation. It is a report-authoring static hygiene check, not pytest or checker
execution. The future independent verifier must repeat it after any change.

## 18. Final repo state

The intended final state is exactly one new untracked report file and no
modification of any existing file. The report remains unstaged. This candidate
does not activate or repair the execution-validation authority.

## 19. Files created/modified

Created:

```text
reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_execution_head_binding_remediation_authority_spec_candidate.md
```

Modified: none.

## 20. Confirmation no pytest/checker/trainer execution ran

No pytest, checker CLI, trainer, producer/materialization, training,
evaluation, CUDA/GPU, or Kaggle execution is authorized or run during this
report-only authoring phase.

## 21. Staging/commit/push

No staging, commit, push, reset, rebase, or force operation is authorized or
performed by this candidate-authoring task. Any activation must occur only
after the lifecycle gates below.

## 22. Remaining blockers

`e908036...` is unusable as execution authority because of the recorded
identity contradiction. It must not be reused as executable authority. The
future two-file remediation, independent high-risk provenance/authority review,
implementation freeze, and construction of a new corrected static-control
execution authority remain pending. Only then may focused pytest/checker be
considered under that new authority.

## 23. Exact next authorized action

`FRESH_INDEPENDENT_STATIC_VERIFICATION_OF_PHASE_IV_EXECUTION_HEAD_BINDING_REMEDIATION_AUTHORITY_CANDIDATE`

Required lifecycle after this candidate:

1. fresh independent verification;
2. byte/blob freeze;
3. explicit stage;
4. dedicated activation commit/push;
5. bounded checker/test implementation remediation;
6. independent high-risk provenance/authority verifier;
7. implementation freeze commit;
8. construct a NEW corrected static-control execution authority; and
9. only then run focused pytest/checker.
