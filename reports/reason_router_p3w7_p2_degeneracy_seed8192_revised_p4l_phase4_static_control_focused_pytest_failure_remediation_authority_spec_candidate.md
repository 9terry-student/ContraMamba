# Phase-IV Focused-Pytest Failure Remediation Authority Candidate

## 1. Purpose, phase, and decision

This is a report-only failure-analysis and remediation-authority candidate. It
does not authorize a code change or any execution. Its sole prospective purpose
is to authorize a later, separately activated, bounded remediation of the
focused static-control pytest failures observed under corrected execution
authority `111663990d98e8aa4badd09473e321fc8f236b26`.

Verdict: `PASS_READY_FOR_FRESH_INDEPENDENT_REVERIFICATION_OF_PHASE_IV_FOCUSED_PYTEST_FAILURE_REMEDIATION_AUTHORITY_CANDIDATE`.

The failed pytest is code-validation evidence only. It is not scientific,
training, evaluation, A1/A2/A3, calibration, or promotion evidence.

## 2. Opening state and failed-execution authority authentication

The opening-state attestation for this candidate is: branch
`p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`; HEAD
`111663990d98e8aa4badd09473e321fc8f236b26`; configured upstream
`origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`; upstream tip
`111663990d98e8aa4badd09473e321fc8f236b26`; ahead/behind `0/0`; and zero
tracked, staged, and untracked modifications before candidate creation.

The post-failure external attestation established full porcelain exit 0 with
empty stdout and stderr, zero unstaged/staged/untracked counts, unchanged
authority/checker/test HEAD blobs, and `git diff --check` exit 0. `.pytest_cache`
exists but is not Git dirt and is outside this authority's mutation scope.

Commit `111663990d98e8aa4badd09473e321fc8f236b26` authenticates as the
corrected execution-validation authority: its sole parent is
`ba20220a8a58d7c0615306fa90afb3206b05adc3`, it changes exactly
`reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_corrected_execution_validation_authority_spec_candidate.md`,
and that report's blob is `5ba8daef50c0a12b38b28d69a16e585510b25d10`.
That authority allowed `git diff --check`, exactly one focused pytest target,
and the standalone checker only after pytest PASS; it prohibited remediation
after failure. The failure therefore correctly returns control to this new,
separate remediation-authority lifecycle.

## 3. Frozen failed-execution inputs and observed result

The failed-execution input identities are frozen, not relabeled as passing:

| Input | Blob | Raw SHA256 |
| --- | --- | --- |
| `scripts/validate_reason_router_p4x_prelaunch_static_control.py` | `dcf62ef63b33c9fcf0f204bede926cad4d71cd9c` | `dae8f88f855d204890c599798d0d4d654f317cddc187c5a7483ec1a7b061ad84` |
| `tests/test_reason_router_p4x_prelaunch_static_control.py` | `22b566dc352882768f482c64b8c9252efd8ae2d7` | `817c48551f3a38eb82540fa4f1f15a141ec88426ebffe0c9fd185ccb82284043` |

The authorized sequence recorded preflight PASS, authority-report blob PASS,
checker/test current-HEAD blob PASS, implementation-anchor ancestry PASS, and
`git diff --check` PASS with exit 0. The exact focused command was:

```text
pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py
```

It exited 1: `6 failed, 70 passed`. The checker was NOT RUN because fail-closed
ordering requires pytest PASS first. Scientific execution, training, and
evaluation were NONE.

## 4. Exact observed failures and root-cause classification

The six observed failures are retained individually:

1. `test_real_git_identity_failure_modes[untracked-P4X_UNTRACKED_INPUT]`
   expected `P4X_UNTRACKED_INPUT`, but observed `P4X_GIT_COMMAND_FAILED` from
   `ls-files --error-unmatch -- other.txt`.
2. `test_real_git_identity_failure_modes[symlink-P4X_SYMLINK_SUBSTITUTION]`
   observed Windows `OSError` / WinError 1314 while attempting `os.symlink`,
   before the checker contract was reached.
3. `test_real_git_implementation_anchor_identity_contracts` observed
   `P4X_UPSTREAM_TIP_MISMATCH` before its intended anchor assertions.
4. `test_wrong_configured_upstream_ref_fails_even_at_same_sha` observed
   `git rev-parse @{up}` exit 128 in its temporary repository before the
   intended same-SHA wrong-upstream assertion.
5. `test_implementation_anchor_unavailable_and_non_commit_fail` expected
   `P4X_IMPLEMENTATION_ANCHOR_COMMIT_UNAVAILABLE`, but observed
   `P4X_UPSTREAM_TIP_MISMATCH` first.
6. `test_ahead_behind_contract_is_explicit_and_fail_closed` expected
   `P4X_AHEAD_BEHIND_MISMATCH`, but observed `P4X_UPSTREAM_TIP_MISMATCH` first.

### 4.1 Failure 1: `PRODUCTION_NAMED_ERROR_CONTRACT_DEFECT`

`authenticated_head_bytes(...)` calls `_reject_symlink(...)` and then evaluates
`_git(root, "ls-files", "--error-unmatch", "--", relative, text=True)` inside
the argument to `_require(..., "P4X_UNTRACKED_INPUT: ...")`. `_git` raises
`ContractError("P4X_GIT_COMMAND_FAILED: ...")` on the nonzero Git exit before
`_require` executes. This confirms a production error-contract defect: rejection
is fail-closed, but the required named untracked-input contract is not surfaced.

The future checker remediation may catch/normalize only that trackedness-command
failure to `P4X_UNTRACKED_INPUT`. It must retain rejection of untracked input;
it must not change the test to expect generic `P4X_GIT_COMMAND_FAILED`.

### 4.2 Failure 2: `TEST_FIXTURE_WINDOWS_SYMLINK_PORTABILITY_DEFECT`

The test gates only on `hasattr(os, "symlink")`. That establishes API presence,
not that the current Windows process has the privilege or policy permission to
create a symlink. WinError 1314 occurs before production code receives the
fixture. This is a test-fixture portability defect, not a production symlink
defense defect.

The future test remediation may exercise `P4X_SYMLINK_SUBSTITUTION` when
creation succeeds, and otherwise skip with an explicit Windows
platform/privilege-policy reason. It must not remove or bypass production
symlink rejection and must not change Windows policy, Developer Mode, ACLs, or
privileges.

### 4.3 Fresh independent-verifier blocker and production selector defect

Fresh independent verification BLOCKED the prior candidate because its
root-cause model was incomplete. It incorrectly stated or implied that no
additional production defect existed, that temporary-Git fixture incompleteness
alone explained the upstream failures, that future checker remediation needed
only the untracked-input named-contract correction, and that same-SHA
wrong-upstream testing could continue to use `@{up}`.

A second bounded production defect is confirmed:
`PRODUCTION_UPSTREAM_REVISION_SELECTOR_DEFECT`. The frozen checker invokes
`rev-parse "@{up}"` and `rev-list --left-right --count "HEAD...@{up}"`.
Git accepts `@{upstream}` and its shorthand `@{u}`, but `@{up}` is not a valid
upstream revision spelling and fails resolution. The one canonical production
spelling for remediation is `@{upstream}`; mixed spellings are not authorized.
Future checker remediation may replace `@{up}` with `@{upstream}` only where it
means the configured branch upstream. This selector correction must not weaken
exact configured-upstream-name validation, current HEAD equal to the exact
upstream tip, ahead/behind `0/0`, or same-SHA wrong-upstream-name rejection.

### 4.4 Temporary-Git remote-tracking fixture defect

The frozen `identity_repo()` establishes the expected branch, a
`refs/remotes/origin/<branch>` ref, and the `branch.<branch>.remote = origin`
and `.merge = refs/heads/<branch>` keys. It does not construct the complete
normal remote-tracking mapping sufficient to model Git upstream resolution.
This is a `TEST_FIXTURE_TEMP_GIT_REMOTE_TRACKING_CONFIGURATION_DEFECT`.

Future bounded test remediation may add the minimum local-only temporary-Git
`origin` remote/refspec configuration needed for `git rev-parse --abbrev-ref
"@{upstream}"` to resolve to the intended symbolic remote-tracking branch and
for `git rev-parse "@{upstream}"` to resolve to the intended commit SHA. It
must establish a genuine `origin` tracking mapping, including required remote
tracking/refspec configuration rather than merely synthesizing a
remote-tracking ref. No network access or real-repository Git configuration is
authorized.

Failure 3, `test_real_git_implementation_anchor_identity_contracts`, is a
CASCADED FAILURE WITH PRODUCTION SELECTOR DEFECT: the checker attempts invalid
`@{up}` and converts resolution failure to `P4X_UPSTREAM_TIP_MISMATCH`. The
incomplete fixture is also a defect that must be corrected so intended upstream
semantics are genuinely tested after the selector fix; this is not fixture-only.

Failure 4, `test_wrong_configured_upstream_ref_fails_even_at_same_sha`, is a
TEST SELECTOR DEFECT + TEMP-GIT FIXTURE DEFECT: the test itself runs invalid
`git rev-parse "@{up}"` before checker invocation, and its temporary fixture
lacks complete normal upstream tracking configuration. It is not evidence that
production same-SHA wrong-upstream rejection is wrong.

Failure 5, `test_implementation_anchor_unavailable_and_non_commit_fail`, is a
CASCADED PRODUCTION SELECTOR FAILURE. The checker reaches invalid upstream
resolution before the intended anchor-unavailable branch. After selector and
fixture remediation it must reach `P4X_IMPLEMENTATION_ANCHOR_COMMIT_UNAVAILABLE`
and separately `P4X_IMPLEMENTATION_ANCHOR_OBJECT_TYPE_MISMATCH`.

Failure 6, `test_ahead_behind_contract_is_explicit_and_fail_closed`, is a
CASCADED PRODUCTION SELECTOR FAILURE. The checker fails during invalid upstream
resolution before `rev-list`. Its future monkeypatch must target
`("rev-list", "--left-right", "--count", "HEAD...@{upstream}")` and demonstrate
that execution reaches `P4X_AHEAD_BEHIND_MISMATCH`.

The same-SHA wrong-upstream fixture must prove before checker invocation that
`git rev-parse --abbrev-ref "@{upstream}"` equals `origin/wrong`, differs from
`origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`, and that
`git rev-parse "@{upstream}"` equals current HEAD SHA. Only then may checker
invocation assert `P4X_UPSTREAM_REF_IDENTITY_MISMATCH`; it must not pass due to
an unresolved upstream, different SHA, wrong branch, missing mapping, or an
earlier contract.

### 4.5 Corrected six-failure classification register

1. `test_real_git_identity_failure_modes[untracked-P4X_UNTRACKED_INPUT]`:
   `PRODUCTION_NAMED_ERROR_CONTRACT_DEFECT`; generic
   `P4X_GIT_COMMAND_FAILED` escapes before `P4X_UNTRACKED_INPUT`.
2. `test_real_git_identity_failure_modes[symlink-P4X_SYMLINK_SUBSTITUTION]`:
   `TEST_FIXTURE_WINDOWS_SYMLINK_PORTABILITY_DEFECT`; WinError 1314 occurs
   during fixture symlink construction before the production defense.
3. `test_real_git_implementation_anchor_identity_contracts`: `CASCADED
   FAILURE WITH PRODUCTION SELECTOR DEFECT`; invalid `@{up}` is translated to
   `P4X_UPSTREAM_TIP_MISMATCH`, with the remote-tracking fixture defect also
   requiring correction.
4. `test_wrong_configured_upstream_ref_fails_even_at_same_sha`: `TEST SELECTOR
   DEFECT + TEMP-GIT FIXTURE DEFECT`; the test's invalid `@{up}` fails before
   checker invocation and the mapping is incomplete.
5. `test_implementation_anchor_unavailable_and_non_commit_fail`: `CASCADED
   PRODUCTION SELECTOR FAILURE`; invalid upstream resolution precedes the two
   intended anchor contracts.
6. `test_ahead_behind_contract_is_explicit_and_fail_closed`: `CASCADED
   PRODUCTION SELECTOR FAILURE`; invalid upstream resolution precedes the
   intended ahead/behind contract.

## 5. Bounded prospective implementation authority

After separate activation, the later implementation may modify exactly these
two files and no third implementation file:

```text
scripts/validate_reason_router_p4x_prelaunch_static_control.py
tests/test_reason_router_p4x_prelaunch_static_control.py
```

The two and only two known bounded production checker defects are
`PRODUCTION_NAMED_ERROR_CONTRACT_DEFECT` (translate the specific authenticated
untracked-input `ls-files --error-unmatch` failure to `P4X_UNTRACKED_INPUT`
without accepting untracked input) and `PRODUCTION_UPSTREAM_REVISION_SELECTOR_DEFECT`.
The latter permits replacement of `@{up}` with canonical `@{upstream}` in the
configured upstream-tip and ahead/behind checks only. Preserve the separate
direct remote-tracking identity lookup `refs/remotes/origin/<EXPECTED_BRANCH>`
unless source analysis proves a strictly necessary mechanical adaptation within
this exact contract. No other production defect has been established by these
six failures.

Future test remediation is limited to complete local temporary `origin`
remote/upstream tracking configuration; canonical `@{upstream}` in relevant
Git calls; monkeypatch expectation `HEAD...@{upstream}`; preservation of the
genuine same-SHA wrong-upstream adversary; explicit Windows symlink-permission
skip only; and assertion of the named `P4X_UNTRACKED_INPUT` contract. No
trainer, dataset, sidecar, provenance, split, model, checkpoint, or scientific-
semantics change is authorized.

The following contracts must remain fail-closed and unweakened: exact branch
and configured upstream; current HEAD equal to exact configured-upstream tip;
ahead/behind `0/0`; full 40-hex immutable ancestor implementation anchor;
wrong upstream name at same SHA; malformed, unavailable, non-commit, and
non-ancestor anchors; unstaged, staged, and untracked dirt; authenticated
untracked input; symlink substitution; Phase-II lineage/evidence; current-HEAD
blob authentication; and no cache, Codex, or platform exceptions in production.

## 6. Required future coverage and boundaries

The remediated focused tests must prove: tracked authenticated input passes with
the correct blob/SHA; untracked input gives exactly `P4X_UNTRACKED_INPUT`;
staged and unstaged input give `P4X_STAGED_DIRTY` and `P4X_UNSTAGED_DIRTY`;
permitted symlink substitution gives `P4X_SYMLINK_SUBSTITUTION`; denied Windows
symlink creation gives a documented skip; clean synchronized anchor and
descendant/authority-like repositories pass; non-ancestor anchor and wrong
branch fail; same-SHA wrong configured upstream gives
`P4X_UPSTREAM_REF_IDENTITY_MISMATCH`; upstream-tip mismatch gives
`P4X_UPSTREAM_TIP_MISMATCH`; explicit ahead/behind reaches
`P4X_AHEAD_BEHIND_MISMATCH`; unavailable and non-commit anchors reach their
respective named contracts; Phase-II success/failure coverage remains intact;
and no trainer/model/checkpoint/CUDA dependency is introduced. The normal
synchronized fixture must first assert that `git rev-parse --abbrev-ref
"@{upstream}"` is
`origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2` and that
`git rev-parse "@{upstream}"` is the exact expected remote-tracking tip before
checker invocation. A static assertion/search must verify that remediated
checker and focused test calls contain no semantically active `@{up}` selector;
historical explanatory text is not runtime behavior.

Required coverage remains A--R: A tracked authenticated success; B exact
`P4X_UNTRACKED_INPUT`; C staged `P4X_STAGED_DIRTY`; D unstaged
`P4X_UNSTAGED_DIRTY`; E successful symlink creation gives
`P4X_SYMLINK_SUBSTITUTION`; F denied Windows symlink creation skips explicitly;
G normal synchronized HEAD==anchor pass after symbolic and resolved
`@{upstream}` assertions; H synchronized descendant with older anchor pass; I
authority-like descendant pass; J non-ancestor anchor failure; K wrong-branch
failure; L genuine same-SHA wrong configured upstream-name failure; M current
HEAD != upstream-tip failure; N explicit `P4X_AHEAD_BEHIND_MISMATCH` using
`HEAD...@{upstream}`; O unavailable anchor; P non-commit anchor; Q retained
Phase-II success/failure coverage; and R no trainer/model/checkpoint/CUDA
dependency. All named failures retain the exact contracts stated above.

During this candidate phase, pytest, checker, trainer, producer,
materialization, training, evaluation, A0/A1/A2/A3, calibration, CUDA, GPU,
and Kaggle are NOT AUTHORIZED. No staging, commit, or push is authorized.

Required lifecycle: fresh independent re-verification; candidate raw-byte
and blob freeze; explicit one-report staging; `cm ship`; dedicated authority
activation commit; push and remote verification; bounded two-file
implementation; independent high-risk verification; implementation byte/diff
freeze; dedicated implementation-freeze commit/push; and a new execution-
validation authority. Only after that new authority activates may focused pytest
and then, after pytest PASS, the checker run. Commit
`111663990d98e8aa4badd09473e321fc8f236b26` must not be reused as execution
authority for changed implementation bytes.

## 7. Candidate closeout

This candidate authorizes neither implementation nor execution. It preserves
Phase-II lineage/evidence as fail-closed and has no scientific scope.

Exact next authorized action:

```text
FRESH_INDEPENDENT_STATIC_REVERIFICATION_OF_PHASE_IV_FOCUSED_PYTEST_FAILURE_REMEDIATION_AUTHORITY_CANDIDATE
```
