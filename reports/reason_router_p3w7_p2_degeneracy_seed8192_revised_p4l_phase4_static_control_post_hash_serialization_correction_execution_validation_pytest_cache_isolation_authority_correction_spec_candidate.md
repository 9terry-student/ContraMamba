# Phase-IV Post-Hash-Serialization Execution-Validation Pytest Cache-Isolation Authority Correction Specification Candidate

## Status and authority boundary

**Candidate status:** `PASS_READY_FOR_INDEPENDENT_PHASE_IV_POST_HASH_SERIALIZATION_EXECUTION_VALIDATION_PYTEST_CACHE_ISOLATION_AUTHORITY_CORRECTION_VERIFICATION`

**Exact next action:** `FRESH_INDEPENDENT_HIGH_RISK_PHASE_IV_POST_HASH_SERIALIZATION_EXECUTION_VALIDATION_PYTEST_CACHE_ISOLATION_AUTHORITY_CORRECTION_VERIFICATION`

This report-only candidate prospectively corrects only the pytest persistent-cache-location contradiction in activated execution-validation authority `148ed168e8c099e32cc86925befc1522c06a32dc`. It authorizes no execution, pytest, standalone checker, implementation change, dataset or provenance regeneration, staging, commit, push, training, evaluation, or scientific conclusion. It preserves every activated-authority semantic except the authorized location of pytest's persistent cache directory.

## 1. Opening state and activated-authority authentication

Candidate authoring is valid only from this exact opening state:

```text
worktree = C:\p3w7-a0-n3-validated-evidence-analysis
branch = p3w7-a1-a2-a3-factorial-execution-authority-n3-v2
HEAD = 148ed168e8c099e32cc86925befc1522c06a32dc
upstream tip = 148ed168e8c099e32cc86925befc1522c06a32dc
ahead / behind = 0 / 0
tracked unstaged = 0
staged = 0
nonignored untracked = 0
```

The activated authority must authenticate exactly as follows:

```text
activated authority commit = 148ed168e8c099e32cc86925befc1522c06a32dc
sole parent = 8e637bdd439d62d429e1c019281efe398ee8c368
sole changed path = reports/reason_router_p3w7_p2_degeneracy_seed8192_revised_p4l_phase4_static_control_post_hash_serialization_correction_execution_validation_authority_spec_candidate.md
activated authority report blob = 10afe4777434ccc3599de2309f19d870751fb79d
```

Any mismatch blocks authoring and requires a separate controller decision. The activated authority's mechanism, evidence, checker, test, provenance, and frozen implementation contracts remain preserved exactly unless this report expressly addresses pytest cache location.

## 2. Frozen defect classification and contradiction evidence

```text
PRIMARY = PYTEST_CACHE_ISOLATION_AUTHORITY_CONTRADICTION
SCOPE = execution environment / pytest cache location only
```

This is not a checker defect, test defect, implementation defect, split defect, provenance defect, or scientific failure.

The activated authority simultaneously requires: (1) the focused command `pytest -q tests/test_reason_router_p4x_prelaunch_static_control.py` with no pytest-option change; (2) no pytest temp/cache directory inside the repository; and (3) post-execution nonignored untracked `= 0`. The repository `.gitignore` does not ignore `.pytest_cache/`, while the host has demonstrated that the focused suite can create nonignored repository-root pytest cache/temp paths. Pytest's cache provider defaults to `.pytest_cache` under `rootdir` unless `cache_dir` is changed. Consequently, the activated authority is not safely executable as written.

`TEMP`, `TMP`, and `TMPDIR` isolate temporary directories used by `tmp_path` and base-temp behavior, but do not by themselves relocate pytest's persistent `cache_dir`. Therefore setting those environment variables alone cannot satisfy all three activated requirements.

## 3. Corrected future external execution environment

Only a dedicated, later activated successor authority may execute. Before execution, it must resolve, create, and print one fresh external writable root outside the repository, dedicated to that execution. Conceptually:

```text
EXEC_TMP_ROOT = C:\Users\Home1\.contramamba\pytest-exec\<fresh-run-id>
EXTERNAL_TEMP_DIR = <EXEC_TMP_ROOT>\temp
EXTERNAL_CACHE_DIR = <EXEC_TMP_ROOT>\cache
```

`<fresh-run-id>` must identify a fresh run. The successor authority must create distinct `temp` and `cache` directories beneath that root, resolve their concrete absolute paths, and print both paths before execution. The external root, its temp directory, and its cache directory are not repository artifacts and must never be imported, copied, staged, or treated as evidence artifacts.

For the focused pytest process only, the successor authority must set:

```text
TEMP=<EXTERNAL_TEMP_DIR>
TMP=<EXTERNAL_TEMP_DIR>
TMPDIR=<EXTERNAL_TEMP_DIR>
PYTHONDONTWRITEBYTECODE=1
```

No ACL modification, repository cleanup, deletion of an existing pytest root, or broader environment mutation is authorized.

## 4. Corrected focused pytest contract

The successor authority explicitly authorizes relocation of pytest `cache_dir` outside the repository. After all required pre-execution authentication succeeds, the exact focused command shape is:

```text
pytest -q -o cache_dir=<EXTERNAL_CACHE_DIR> tests/test_reason_router_p4x_prelaunch_static_control.py
```

Here `EXTERNAL_CACHE_DIR` is exactly `<EXEC_TMP_ROOT>\cache`, and its concrete absolute external path must be resolved and printed before execution. The focused file remains exactly `tests/test_reason_router_p4x_prelaunch_static_control.py`; no selector, test filtering, repository-wide suite, xfail/skip injection, fixture alteration, or source alteration is authorized.

The sole pytest-option change is `-o cache_dir=<EXTERNAL_CACHE_DIR>`. No `--cache-clear`, `-p no:cacheprovider`, selector, broad suite, or any other pytest-option change is authorized. Focused pytest success still requires exit code `0`.

`cache_dir` controls environmental persistent-cache storage only. This correction does not change test collection, test selection, assertion behavior, fixtures, test source, checker, implementation, split, provenance, or the expected success criterion. It is therefore not test weakening and does not alter scientific or test semantics.

## 5. Conditional checker contract and stop rules

Only if the corrected fresh focused pytest passes, the successor authority must run this unchanged standalone checker command exactly:

```text
python scripts/validate_reason_router_p4x_prelaunch_static_control.py --repo-root . --expected-head 8e637bdd439d62d429e1c019281efe398ee8c368
```

The checker command and its immutable expected-head anchor do not change. If the focused pytest fails, stop: no checker, retry, remediation, implementation change, or artifact regeneration is authorized. If the checker fails, stop: no retry or remediation is authorized.

## 6. Cleanliness and frozen implementation boundary

Before execution, and after focused pytest and any reached checker execution, the successor authority must require:

```text
tracked unstaged = 0
staged = 0
nonignored untracked = 0
```

The external temp/cache root prevents pytest persistent cache from becoming a repository path. No post-run repository cleanup is authorized or necessary.

The frozen implementation and evidence identities remain:

| Role | Immutable identity |
| --- | --- |
| Implementation anchor | `8e637bdd439d62d429e1c019281efe398ee8c368` |
| Checker blob | `90b96a329cd6e92d3c5c79b76d02dc2d31233574` |
| Focused test blob | `e310dcdd72f9dc39ca281626ea20dd6bd19a7132` |
| `IMPLEMENTATION_DELTA_CORRECTNESS` | `ESTABLISHED` |
| Historical implementation-phase `FOCUSED_TEST_CORRECTNESS` | `ESTABLISHED` |
| `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS` | `NOT_ESTABLISHED` |
| `SCIENTIFIC_CONCLUSION` | `NONE` |

No A0/A1/A2/A3, trainer, training, evaluation, GPU/CUDA/Kaggle, dataset/sidecar/provenance regeneration, or label/loss/gradient/EMA/calibration change is authorized.

## 7. Evidence preservation and non-retroactivity

The successor authority must preserve distinct evidence layers for implementation-delta correctness, historical implementation-phase focused-test correctness, new fresh focused-pytest execution result, prelaunch static-control execution success, artifact/provenance validity, and scientific conclusion. A focused pytest pass does not establish checker success; only a passing checker under its unchanged contract can establish `PRELAUNCH_STATIC_CONTROL_EXECUTION_SUCCESS`.

This correction is prospective only. It must not reinterpret activated authority `148ed168e8c099e32cc86925befc1522c06a32dc` as safely executed: no execution occurred under it. It must not alter, weaken, or reinterpret the historical `dd34cd00336d04d384767fd533c33253d2c9c6ac` failure evidence. No scientific conclusion follows from this environmental correction.

## 8. Mandatory lifecycle

The required lifecycle is:

1. Author this cache-isolation correction candidate.
2. Perform fresh independent high-risk verification.
3. Freeze exact bytes and blob.
4. Perform the report-only stage.
5. Run `cm ship`.
6. Create a dedicated successor execution-authority correction activation commit.
7. Push.
8. Verify remotely.
9. Perform pre-execution authentication.
10. Run corrected fresh focused pytest with external temp and external cache.
11. If and only if pytest passes, run the unchanged standalone checker.
12. Perform post-execution re-authentication.
13. Classify evidence layers.
14. Controller chooses the next authorized research action.

There is no shortcut. Candidate authoring itself authorizes no execution.

## 9. Candidate disposition

At candidate authoring completion, required state is:

```text
HEAD = 148ed168e8c099e32cc86925befc1522c06a32dc
tracked unstaged = 0
staged = 0
nonignored untracked = exactly one candidate report
```

No pytest, checker, implementation change, staging, commit, or push is authorized. Any failure of later authentication, focused pytest, checker, or post-execution cleanliness requires a separate report-only authority decision.
