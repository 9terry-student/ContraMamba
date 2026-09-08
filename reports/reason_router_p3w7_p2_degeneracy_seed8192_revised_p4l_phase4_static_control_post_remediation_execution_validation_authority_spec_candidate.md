# Post-Remediation Phase-IV Static-Control Execution-Validation Authority Candidate

## Status and scope

This is a non-self-referential, report-only candidate for a new Phase-IV static-control execution-validation authority. It is bound to the remediated implementation freeze below. It authorizes no execution while this candidate is authored, independently verified, byte/blob frozen, staged, committed, pushed, and remotely verified.

If activated, its only prospective authorization is the fail-closed sequence in **Required execution order**: `git diff --check`, the one exact focused pytest command, and, only after that pytest passes, the one exact checker command. It does not authorize trainer process, training, evaluation, scientific execution, CUDA/GPU/Kaggle, producer/materialization, checkpoint/model load, or A0-A3.

## Authenticated opening state for candidate authoring

- Branch: `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`
- HEAD: `25569c0234086fb05d1120a7b0b5490aa751c182`
- Configured upstream: `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`
- Upstream tip: `25569c0234086fb05d1120a7b0b5490aa751c182`
- Ahead/behind: `0/0`
- Tracked modifications: `0`; staged: `0`; untracked: `0` before this candidate is created.

No fetch, ref/config, cache, ACL, or ignore mutation is authorized. Candidate creation is the sole repository mutation.

## Remediation authority authentication

Activated focused-pytest failure remediation authority: `f45341e60c3fa634ffbd2805ae14b3d540441afc`.

- Its required parent is `111663990d98e8aa4badd09473e321fc8f236b26`.
- Its exact one changed report path is `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_focused_pytest_failure_remediation_authority_spec_candidate.md`.
- Its authority blob is `1b647a30123d0e6efd0eccf060ec0aa70dd1a296`.

That authority authorized exactly the bounded two-file remediation and required a new execution-validation authority after the implementation freeze. It did not authorize this candidate to execute anything.

## Remediated implementation freeze authentication

Remediated implementation freeze: `25569c0234086fb05d1120a7b0b5490aa751c182`.

- Sole parent: `f45341e60c3fa634ffbd2805ae14b3d540441afc`.
- Exact changed paths, with no third path:
  - `scripts/validate_reason_router_p4x_prelaunch_static_control.py`
  - `tests/test_reason_router_p4x_prelaunch_static_control.py`
- Committed checker blob: `8b21843b0f78356262fdc843c0aacee9ab419b75`.
- Committed focused-test blob: `8d56db1a00963cb188aac9c0927da5c572dc20b1`.
- Checker raw SHA256: `04ec6811777e674910ea0aef35509691b499d1c462a274a46b8092cf04233852`; bytes: `24902`.
- Focused-test raw SHA256: `d3c6d0384b2c3fc58e86396d061bd22e8c3e1b613e32bb2e78e138adf22cffaa`; bytes: `30564`.
- Canonical implementation diff from `f45341e60c3fa634ffbd2805ae14b3d540441afc`: checker `+7/-3`; test `+27/-6`; total `+34/-9`.
- Canonical full-index diff SHA256: `5e24f6e865e4d7e05d31e572e6b1f568bbb00ad47026396351c7894284e7dab6`.

## Remediation semantics bound

Static inspection of the frozen checker binds these two production corrections only:

1. An authenticated untracked-input membership failure is normalized narrowly to `P4X_UNTRACKED_INPUT`, solely around `git ls-files --error-unmatch -- <relative>`. The normalization does not swallow later Git failures.
2. The configured-upstream selector is exactly canonical `@{upstream}` in both `rev-parse "@{upstream}"` and `rev-list --left-right --count "HEAD...@{upstream}"`. No active `@{up}` or `@{u}` alternative belongs to the frozen production contract.

The focused-test remediation statically includes local `origin` with `remote.origin.fetch` mapping; symbolic and resolved `@{upstream}` precondition assertions; a genuine same-SHA wrong-upstream adversary; the ahead/behind target `HEAD...@{upstream}`; Windows symlink skipping only when creation is impossible because the API is unavailable or Windows `WinError 1314`; and AST/static canonical-selector coverage. These are implementation/test-fixture corrections, not scientific changes.

## Immutable implementation anchor

`IMPLEMENTATION_ANCHOR = 25569c0234086fb05d1120a7b0b5490aa751c182`.

The future checker invocation is exactly:

```text
python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head 25569c0234086fb05d1120a7b0b5490aa751c182
```

`--expected-head` means only the immutable implementation anchor. It is not current execution HEAD, authority activation commit, branch tip, upstream tip, or a moving ref. `HEAD`, branch names, tags, short SHAs, refs, and environment-derived alternatives are rejected. The implementation anchor must be a commit with exactly this full identity and must be an ancestor of the execution HEAD.

## Future activation and later descendants

`POST_REMEDIATION_EXECUTION_AUTHORITY_ACTIVATION_COMMIT` means the future dedicated activation commit containing this independently verified and byte/blob-frozen authority report. Its exact SHA is deliberately unknown during candidate authoring; this report neither guesses it nor self-references it.

After dedicated activation, push, and remote verification, the observed activation SHA becomes the exact required execution-time current HEAD and configured upstream tip. It need not equal the implementation anchor; `25569c0234086fb05d1120a7b0b5490aa751c182` must instead be its ancestor. A later descendant is not automatically authorized: execution requires the exact activated authority HEAD, and a descendant needs separately authorized successor/binding action.

## Future exact branch/upstream and authority-report binding

Future execution requires all of the following:

- Branch `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`.
- Configured upstream `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`.
- Current HEAD exactly the observed post-remediation execution-authority activation SHA.
- Exact configured upstream tip the same activation SHA.
- Ahead/behind exactly `0/0`.

A wrong configured upstream name remains invalid even if it resolves to the same SHA. Final candidate bytes and blob are independently measured only after authoring; no final report blob is guessed here. After independent verification and byte/blob freeze, that exact frozen report blob is committed in the dedicated activation. At execution, current HEAD must contain this report at that exact blob; mismatch blocks before checker or test execution.

At current HEAD, not merely in the implementation-anchor tree, execution must authenticate checker blob `8b21843b0f78356262fdc843c0aacee9ab419b75` and focused-test blob `8d56db1a00963cb188aac9c0927da5c572dc20b1`. Anchor ancestry and current-HEAD executable/test-byte authentication are distinct mandatory controls.

## Required fail-closed execution order

Every failure stops all later steps. The semantic order is exactly:

1. Authenticate the exact post-remediation execution-authority activation commit.
2. Authenticate exact branch, configured upstream ref, current HEAD, and exact upstream tip.
3. Authenticate ahead/behind equals `0/0`.
4. Authenticate tracked clean, index clean, and no unexpected untracked paths.
5. Authenticate this post-remediation execution-validation authority report at current HEAD against its independently frozen Git blob.
6. Authenticate at current HEAD the checker blob `8b21843b0f78356262fdc843c0aacee9ab419b75` and focused-test blob `8d56db1a00963cb188aac9c0927da5c572dc20b1`.
7. Authenticate immutable anchor `25569c0234086fb05d1120a7b0b5490aa751c182` as an ancestor of current execution HEAD.
8. Run `git diff --check`.
9. Run exactly `pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py`.
10. Only if focused pytest passes, run exactly `python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head 25569c0234086fb05d1120a7b0b5490aa751c182`.
11. Capture exact commands, exit codes, stdout, and stderr.
12. Re-authenticate post-execution repository identity, exact blobs, and cleanliness.

## Focused pytest and checker contracts

Focused pytest success means exit code `0`, only `tests/test_reason_router_p4x_prelaunch_static_control.py` was run, no persistent repository mutation occurred, and no unexpected untracked artifacts exist. A Windows symlink fixture skip is admissible only when the focused suite itself treats unavailable API or `WinError 1314` creation failure as a pytest skip and the overall command exits `0`; it is not a production symlink PASS. Its success establishes only `CODE_CORRECTNESS_EVIDENCE`.

Static inspection, without checker execution, establishes the checker success JSON schema exactly as top-level `status`, `split`, `cohorts`, and `execution_record_opened`. On success `status` is `"PASS"` and `execution_record_opened` is `false`. `split` is the recomputed audit object with keys `pair_count`, `train_pair_count`, `dev_pair_count`, `train_row_count`, `dev_row_count`, `pair_universe_sha256`, `shuffled_pair_sha256`, `train_pair_sha256`, `dev_pair_sha256`, `ordered_train_row_sha256`, and `ordered_dev_row_sha256`. `cohorts` has `train` and `dev`, each with `frame`, `predicate`, `sufficiency`, and `polarity` binary `{0,1}` count maps. Failure is machine-readable stderr JSON exactly shaped with `status: "FAIL"` and `contract: <named contract>`; no fields are invented.

Checker success requires exit code `0`, parseable machine-readable PASS with that exact schema, and passing Phase-II/data/split/cohort/provenance contracts. Malformed output, nonzero exit, or FAIL blocks.

## Historical failed execution and preservation

Prior execution authority: `111663990d98e8aa4badd09473e321fc8f236b26`. Under the old implementation, focused pytest observed `6 failed, 70 passed`, exit `1`; checker was `NOT RUN`. This is historical code-validation evidence only. It is not reactivated for changed bytes and is neither erased nor reinterpreted.

The following remain unchanged: `PHASE_II_ACTIVATION_COMMIT = cb6f4482b463d5f85331e2a6ddfbbd34499c930a`; `PHASE_II_EVIDENCE_FREEZE_COMMIT = ef26310f3532368b9de6cb96a19cb26e7626716d`; frozen dataset; seed8192 split; sidecar/provenance; execution record; trainer/rebind identities; cohort counts; aggregate identities; and historical seed174 rejection. No schema, label, loss, gradient, EMA, calibration, or scientific semantics changes.

## Failure and scientific boundary

Validation blocks at least on wrong activation SHA, branch, configured upstream ref, current HEAD, upstream tip, nonzero ahead/behind, tracked/index/untracked dirt, authority-report blob mismatch, checker/test blob mismatch, unavailable/wrong/non-commit/non-ancestor anchor, Phase-II lineage/evidence failure, nonzero pytest, malformed checker output, nonzero checker, checker FAIL, and frozen dataset/split/provenance/cohort/aggregate mismatch. This authority does not authorize remediation after failure.

All of trainer process, training, evaluation, A0, A1, A2, A3, calibration, producer, materialization, CUDA, GPU, Kaggle, checkpoint load, and model load remain not authorized. Successful future validation establishes only `CODE_CORRECTNESS_EVIDENCE` and `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS`; it does not establish a scientific result, training/evaluation success, A1/A2/A3 readiness, or promotion eligibility.

## Lifecycle and authoring controls

Required lifecycle: author this candidate; fresh independent static verification; candidate byte/blob freeze; explicit one-report staging; `cm ship`; dedicated execution-authority activation commit; push; remote verification; bind exact activation SHA; only then execute the authorized diff-check/focused-pytest/checker sequence. No execution occurs during authoring or verification.

Allowed authoring validation is read-only Git/source/report inspection, candidate writing, raw SHA/blob calculations, AST/static inspection, and `git diff --check`. Pytest, checker CLI, trainer, producer/materialization, training/evaluation, CUDA/GPU, Kaggle, and model/checkpoint load are forbidden. Do not stage, commit, or push.

## Candidate-format and stop conditions

This candidate must be UTF-8 without BOM, LF-only with a final LF, and have zero trailing-whitespace lines. After final writing, externally report raw SHA256, byte count, LF/CR/CRLF counts, BOM, final-LF status, trailing-whitespace count, predicted Git blob, and manual raw Git blob SHA-1. Do not self-embed the final report SHA or blob.

Stop blocked if the implementation freeze fails authentication; a third implementation path appears; checker schema materially differs; anchor and current-HEAD semantics cannot remain distinct; broader pytest/checker execution is needed; scientific scope expands; or an unknown activation SHA/blob would need self-reference.

## Final authoring state and next action

After authoring, required state is HEAD `25569c0234086fb05d1120a7b0b5490aa751c182`, no tracked modifications, no staged changes, and exactly this untracked report path: `reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_post_remediation_execution_validation_authority_spec_candidate.md`. No existing file may be modified.

Exact next authorized action: `FRESH_INDEPENDENT_STATIC_VERIFICATION_OF_POST_REMEDIATION_PHASE_IV_STATIC_CONTROL_EXECUTION_VALIDATION_AUTHORITY_CANDIDATE`.
