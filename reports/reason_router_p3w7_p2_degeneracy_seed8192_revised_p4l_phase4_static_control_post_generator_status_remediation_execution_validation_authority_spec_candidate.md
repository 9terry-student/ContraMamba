# Phase-IV Post-Generator-Status-Remediation Execution-Validation Authority Candidate

## Status, scope, and opening authentication

This is a report-only authority candidate. It authorizes no execution while it is authored, independently verified, byte/blob frozen, staged, shipped with `cm ship`, committed as a dedicated activation commit, pushed, and remotely verified. This candidate authorizes neither pytest nor the standalone checker during authoring.

Candidate authoring opening state is authenticated as follows:

- Worktree: `C:\p3w7-a0-n3-validated-evidence-analysis`
- Branch: `p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`
- HEAD: `0968012d61aeb81b098a3ea41c0adf7cec1132ed`
- Configured upstream: `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`
- Upstream tip: `0968012d61aeb81b098a3ea41c0adf7cec1132ed`
- Ahead/behind: `0/0`; tracked unstaged: `0`; staged: `0`; nonignored untracked: `0` before this report is created.

Any material mismatch stops authoring. Candidate creation is the only repository mutation. Do not modify checker/tests, execute builder/trainer/training/evaluation, use model/checkpoint loading, GPU/CUDA/Kaggle, regenerate artifacts, stage, commit, or push.

## Immutable remediation implementation authentication

The complete execution-checker implementation anchor is commit `0968012d61aeb81b098a3ea41c0adf7cec1132ed`, with sole parent `5d01c85210890e6fdd6ed7896696433c89d95cc3`. Its exact delta contains only:

- `scripts/validate_reason_router_p4x_prelaunch_static_control.py`
- `tests/test_reason_router_p4x_prelaunch_static_control.py`

At that commit, authenticate:

| Artifact | Git blob | Raw SHA256 | Bytes |
| --- | --- | --- | ---: |
| Checker | `fe820f745068af1478140b27231e9b2b3340eaa4` | `4a2f70a7153341e68b00047c5b4d75ee5959c2a5368283794c4b1556de0249ef` | 26152 |
| Focused test | `e9e1d8bedd0cccb6bc0e3deaf43e320ea5480849` | `a69b065b318aac31f19f13c9f132ce51cdbdc837da5c7a518398077b4181ccbb` | 39863 |

This anchor is immutable execution-checker implementation identity. Historical `8e637bdd439d62d429e1c019281efe398ee8c368` remains split/hash implementation evidence only; it is not the complete checker anchor and must not be supplied to the future checker. The exact future argument is:

```text
--expected-head 0968012d61aeb81b098a3ea41c0adf7cec1132ed
```

## Preserved evidence and non-inference

Preserve, without reinterpretation: historical checker failure `P4X_SPLIT_IDENTITY_MISMATCH`; later checker failure `P4X_GENERATOR_STATUS_DEFECT: orion_approval__polarity_flip`; root-cause finding `CHECKER_CONTRACT_MISMATCH = ESTABLISHED`, `DEFECT_LAYER = PHASE_IV_CHECKER`, `DEFECT_SCOPE = MULTI_ROW_SYSTEMATIC`; generator-status remediation independent implementation verification PASS; implementer targeted and focused validation PASS; independent targeted and focused verification PASS. These are implementation/code-correctness evidence only.

Before a future passing execution: `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = NOT_ESTABLISHED` and `SCIENTIFIC_CONCLUSION = NONE`. Prior failures are never converted retroactively into PASS.

## Required lifecycle before any execution

1. Author this authority candidate.
2. Perform fresh independent high-risk authority verification.
3. Freeze exact report bytes and Git blob.
4. Complete the report-only stage.
5. `cm ship`.
6. Create a dedicated authority-activation commit.
7. Push it.
8. Remotely verify parent, blob, and branch tip.
9. Authenticate repository and implementation immediately before execution.
10. Run fresh focused pytest under the external temp/cache contract.
11. Only if pytest passes, run the standalone checker.
12. Re-authenticate repository state after execution.
13. Classify evidence.
14. Controller chooses the next authorized research action.

No execution is authorized before step 8 completes. The later activation commit SHA is intentionally not guessed here; at execution, current HEAD must be exactly that remotely authenticated activation SHA and `0968012d61aeb81b098a3ea41c0adf7cec1132ed` must be its ancestor.

## Future pre-execution authentication

Require the exact activation HEAD; expected branch and configured upstream `origin/p3w7-a1-a2-a3-factorial-execution-authority-n3-v2`; HEAD equal to upstream tip; ahead/behind `0/0`; tracked unstaged `0`; staged `0`; and nonignored untracked `0`. Require the immutable implementation anchor to be an ancestor of activation HEAD.

At activation HEAD authenticate checker blob `fe820f745068af1478140b27231e9b2b3340eaa4` and focused-test blob `e9e1d8bedd0cccb6bc0e3deaf43e320ea5480849`, plus these frozen inputs, with no mutation/regeneration:

| Frozen artifact | Blob |
| --- | --- |
| Dataset | `2b6829bf04a1333446aac6f7c603d9178b339f36` |
| Sidecar | `83d119e327acacda7cff6b4e24c6502898294e03` |
| Provenance | `6c970033fae82286452f6d635b94f441d0f3d048` |

## Frozen checker authentication, split serialization, and Phase-II lineage binding

Future execution is bound to the frozen checker at `0968012d61aeb81b098a3ea41c0adf7cec1132ed`, in which the following fail-closed authentication controls must remain operative: Git-canonical blob-byte authentication for frozen tracked inputs; tracked-file identity binding; staged/unstaged cleanliness checks for authenticated inputs; repository cleanliness checks; configured upstream, branch, and ancestor validation; Phase-II activation lineage authentication; Phase-II evidence-freeze lineage/binding; and frozen Phase-II execution-record blob identity binding. This execution authority does not independently reimplement those controls; it binds future execution to the frozen checker where they are already operative. No Phase-II commit or blob identity is modified by this authority.

The successor execution preserves the already-corrected split-identity serialization exactly. Pair-list identity serialization is each value followed by LF, equivalently `value + "\n"`. Ordered row identity serialization is `row_id + TAB + pair_id + LF`, equivalently `row_id<TAB>pair_id<LF>`. This corrected LF-terminated serialization is frozen and must remain operative in the checker anchored at `0968012d61aeb81b098a3ea41c0adf7cec1132ed`. The historical non-final-LF / row-id-only serialization bug must not reappear. The frozen `SPLIT_IDENTITIES` below, including every recorded split hash, remain exact and unchanged.

## Fresh external focused-pytest gate

Create one fresh unique execution root outside the repository, preferably beneath `$HOME\.contramamba\pytest-exec\`, with distinct `temp` and `cache` children. For the pytest child process set only as necessary:

```text
TEMP=<external temp>
TMP=<external temp>
TMPDIR=<external temp>
PYTHONDONTWRITEBYTECODE=1
```

Execute exactly:

```text
pytest -q -o cache_dir=<EXTERNAL_CACHE_DIR> tests/test_reason_router_p4x_prelaunch_static_control.py
```

Do not use `--basetemp`, `--cache-clear`, `-p no:cacheprovider`, node selectors, filters, skip/xfail injection, or a broad suite. The already authenticated Windows symlink-fixture skip is permissible only when it is the same environment-dependent skip. Capture complete stdout/stderr and exact exit code.

Focused pytest is first. PASS requires exit code `0`. On any failure, stop: do not run checker, retry, remediate, or clean repository artifacts. Classify `FRESH_EXECUTION_VALIDATION_FOCUSED_TEST_CORRECTNESS = NOT_ESTABLISHED`, `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = NOT_ESTABLISHED`, `SCIENTIFIC_CONCLUSION = NONE`, and `A0_A1_A2_A3_AUTHORIZED = FALSE`. A separate controller authority is then required.

Only after pytest PASS, re-authenticate unchanged HEAD, upstream tip, ahead/behind `0/0`, tracked/index/untracked cleanliness, checker blob, and focused-test blob. Any mismatch stops without checker, cleanup, or remediation.

## Conditional standalone checker

Only after both prior gates pass, execute exactly:

```text
python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head 0968012d61aeb81b098a3ea41c0adf7cec1132ed
```

Set `PYTHONDONTWRITEBYTECODE=1`. Do not alter arguments. No trainer, model, GPU, or other environment is required. Capture complete stdout/stderr and exact exit code.

Checker PASS requires exit `0`; empty stderr; stdout exactly one nonempty line; valid JSON; and exact top-level output schema with no invented PASS fields:

```text
status, split, cohorts, execution_record_opened
```

Require `status == "PASS"`, `execution_record_opened == false`, and exact equality of `split` and `cohorts` to the frozen values below. These are all fields in the frozen successful checker output schema.

`execution_record_opened = false` has a precise boundary meaning: the Phase-II execution record remains identity-bound by frozen Git/blob authentication, but the checker does not open or parse its contents as execution evidence during Phase-IV static validation. Thus artifact identity binding is retained without consumption of prior execution evidence. Explanatory authority statements are: `PHASE_II_EXECUTION_RECORD_IDENTITY_BOUND = TRUE` and `PHASE_II_EXECUTION_RECORD_OPENED_AS_EXECUTION_EVIDENCE = FALSE`. They are not checker stdout fields; the actual checker stdout schema remains exactly `status`, `split`, `cohorts`, and `execution_record_opened`.

```json
{"pair_count":300,"train_pair_count":240,"dev_pair_count":60,"train_row_count":2880,"dev_row_count":720,"pair_universe_sha256":"41f7a2cc533b9026a49d2b2587dd34894fadb908deab9f0a79133345569758f2","shuffled_pair_sha256":"ef15a6c3dc0f45ccad0f4e4e203eab9ff5dbfe8d64dde96ae14df3811bbd2d55","train_pair_sha256":"f6fffb94b6c33112bcfc8afb6da9f3aa76ae6e1327b8c38e69724fa4c2641049","dev_pair_sha256":"30951a7c637b10a5693289be40911ec5bf32de6eca3efd37a81f3fa268cd25a4","ordered_train_row_sha256":"478013207699462a9434ce8f44991ce75b33650593b9aa942fff0f2be659c2a8","ordered_dev_row_sha256":"7870c83fe1f6e3a65311311ab05122736a007e6a92f4f04c28b2c72584ddfaa4"}
```

Required `cohorts` are `train: frame {0:714,1:695}, predicate {0:119,1:576}, sufficiency {0:238,1:338}, polarity {0:100,1:238}` and `dev: frame {0:186,1:174}, predicate {0:31,1:143}, sufficiency {0:62,1:81}, polarity {0:19,1:62}`.

If exit is nonzero, stderr is unexpectedly nonempty, stdout is malformed, JSON/status/schema is invalid, or any identity differs: stop; do not retry pytest/checker, remediate, clean, or regenerate. Preserve exact failure output and classify `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = NOT_ESTABLISHED`, `SCIENTIFIC_CONCLUSION = NONE`, and `A0_A1_A2_A3_AUTHORIZED = FALSE`.

## Post-execution authentication and success boundary

After checker PASS or FAIL where practical without state change, read-only re-authenticate unchanged HEAD/upstream/ahead-behind/cleanliness, checker/focused-test blobs, dataset/sidecar/provenance blobs, and anchor ancestry. No cleanup is authorized.

Only when every gate passes classify:

```text
IMPLEMENTATION_DELTA_CORRECTNESS = ESTABLISHED
IMPLEMENTER_TARGETED_VALIDATION = PASS
IMPLEMENTER_FOCUSED_VALIDATION = PASS
INDEPENDENT_TARGETED_VERIFICATION = PASS
INDEPENDENT_FOCUSED_VERIFICATION = PASS
FRESH_EXECUTION_VALIDATION_FOCUSED_TEST_CORRECTNESS = ESTABLISHED
PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS = ESTABLISHED
ARTIFACT_PROVENANCE_VALIDITY = AUTHENTICATED_BY_PASSING_FROZEN_STATIC_CONTROL
SCIENTIFIC_CONCLUSION = NONE
A0_A1_A2_A3_AUTHORIZED = FALSE
TRAINING_EVALUATION_EXECUTED = FALSE
GPU_CUDA_KAGGLE_USED = FALSE
REPOSITORY_POST_EXECUTION_CLEAN = TRUE
```

A passing static-control execution does not authorize A0-A3. Trainer execution, A0-A3, training, evaluation, model/checkpoint loading, GPU, CUDA, Kaggle, regeneration, builder execution, code/test modification, cleanup, and remediation remain prohibited. The controller must choose the next authorized research action; branch name is not scientific authority.

## Candidate closure

After authoring require HEAD unchanged at `0968012d61aeb81b098a3ea41c0adf7cec1132ed`, zero tracked unstaged and staged paths, and exactly this one nonignored untracked candidate. Run only `git diff --check` as authoring validation. Candidate status, if independently verified and identity-frozen, is `PASS_READY_FOR_INDEPENDENT_PHASE_IV_POST_GENERATOR_STATUS_REMEDIATION_EXECUTION_VALIDATION_AUTHORITY_VERIFICATION`.

Exact next action: `FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_POST_GENERATOR_STATUS_REMEDIATION_EXECUTION_VALIDATION_AUTHORITY_VERIFICATION`.
