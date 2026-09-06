# Longterm O0c Transformers Distribution-Root Derivation Correction Implementation Authority Spec Candidate

## 1. Verdict

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

This is a report-only implementation-authority candidate for a minimal, test-backed correction of the confirmed O0c Transformers distribution-root derivation defect.

This authoring task does not modify implementation files, tests, existing reports, packages, checkpoints, datasets, run registries, or scientific artifacts. It does not run Kaggle, rerun diagnostics, rerun preflight v2, create preflight v3, load models/tokenizers/datasets, train, evaluate, stage, commit, or push.

## 2. Authority Chain

Authority order used:

1. Current workflow-controller instruction for `REPORT_ONLY_IMPLEMENTATION_AUTHORITY_AUTHORING`.
2. Frozen diagnostic interpretation: commit `3fc6fbbe9aed00594677c6f35029200ab055e883`, `reports/longterm_o0c_transformers_distribution_root_ambiguity_diagnostic_interpretation_report_candidate.md`.
3. Frozen diagnostic execution authority: commit `ba5d605654f20d0f022e9a6d2d57ea9685b68b76`.
4. Frozen preflight implementation: commit `de874a22df4f60adbdc5efbcf294961c7b3a48a5`.
5. Imported diagnostic provenance: `longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1`.
6. Repository `AGENTS.md` and applicable workflow/runbooks.

Canonical repository:

`C:\Users\Home1\Desktop\ContraMamba`

Expected authoring HEAD was verified before writing:

`3fc6fbbe9aed00594677c6f35029200ab055e883`

## 3. Evidence Inspected

Direct inspection completed before authoring:

- frozen interpretation report at `3fc6fbbe9aed00594677c6f35029200ab055e883`;
- frozen implementation file at `de874a22df4f60adbdc5efbcf294961c7b3a48a5`;
- frozen test file at `de874a22df4f60adbdc5efbcf294961c7b3a48a5`;
- existing genuine ambiguity regression `test_ambiguous_distribution_roots_block`;
- imported diagnostic provenance under `C:\Users\Home1\.contramamba\imports\longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1_de874a22df4f_20260907_035035`;
- current repository state, current branch, current HEAD, tracked diff, staged diff, and untracked files.

The imported diagnostic provenance reports run name `longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1`, execution commit `de874a22df4f60adbdc5efbcf294961c7b3a48a5`, command SHA256 `5d4fd534e1d0dd96ea02a1393a43aa4f0379219301e550632d549040b39e2fc1`, exit code `0`, started UTC `2026-09-06T18:45:03Z`, finished UTC `2026-09-06T18:45:10Z`, and import schema `contramamba-local-import-v2`.

## 4. Confirmed Defect

At frozen implementation commit `de874a22df4f60adbdc5efbcf294961c7b3a48a5`, `scripts/preflight_longterm_o0c_runtime_source_provenance.py::_distribution_root()` uses materially:

```python
root = located.parent if located.name == "__init__.py" else located.parents[len(parts) - 2]
```

The frozen diagnostic interpretation classified the observed blocker as:

`VALIDATOR_DISTRIBUTION_ROOT_DERIVATION_FALSE_AMBIGUITY`

The diagnostic found `2062` qualifying Transformers distribution metadata entries, `428` frozen-derived roots, and `427` false extra roots caused by nested `__init__.py` entries being assigned their immediate subpackage directory as root. Uniform path-depth derivation reconciled all qualifying entries to one package root.

This classification is an infrastructure correctness finding about the validator/preflight source-provenance code. It is not an O0c scientific result.

## 5. Future Implementation Scope

The later implementer may change exactly these implementation/test files:

1. `scripts/preflight_longterm_o0c_runtime_source_provenance.py`
2. `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

No other implementation or test file is authorized by this candidate. If the correction proves impossible within those two files, the implementer must block and report the concrete dependency rather than expanding scope.

No report modification is authorized during the later implementation.

## 6. Required Correction Semantics

For every qualifying distribution metadata entry where `parts` is non-empty and `parts[0] == "transformers"`, derive the Transformers package root uniformly from the full relative distribution path depth:

```python
root = located.parents[len(parts) - 2]
```

This must apply to all qualifying entries, including top-level and nested `__init__.py` files.

The future patch must preserve:

- `canonical_path()` resolution before root derivation;
- filtering to first path component exactly `"transformers"`;
- fallback behavior when no qualifying distribution files exist;
- case-normalized distinct-root comparison;
- fail-closed `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS` behavior when genuinely independent roots remain;
- import-root reconciliation;
- source-shadowing checks;
- runtime version checks;
- output/artifact schema and unrelated preflight behavior;
- CPU-only semantics;
- no scientific instrumentation changes.

## 7. Mechanical Path-Depth Proof

The uniform formula is correct because the number of parents to climb is determined by the relative distribution path depth after the top-level `transformers` component.

Examples:

- `transformers/__init__.py`: `len(parts) == 2`, so `located.parents[0]` is the common `transformers` package directory.
- `transformers/models/__init__.py`: `len(parts) == 3`, so `located.parents[1]` climbs from `.../transformers/models/__init__.py` to the common `transformers` package directory.
- `transformers/models/mamba/__init__.py`: `len(parts) == 4`, so `located.parents[2]` climbs from `.../transformers/models/mamba/__init__.py` to the common `transformers` package directory.
- `transformers/models/mamba/modeling_mamba.py`: `len(parts) == 4`, so `located.parents[2]` climbs from `.../transformers/models/mamba/modeling_mamba.py` to the common `transformers` package directory.

Top-level `__init__.py` remains correct under the uniform formula because `located.parents[0]` equals `located.parent`.

Nested `__init__.py` becomes correct because it no longer stops at the immediate subpackage directory.

Ordinary nested files preserve existing intended behavior because they already used the path-depth formula in the frozen implementation.

Genuinely independent distribution roots remain detectable because each qualifying entry is still canonicalized and assigned the package root implied by its own resolved installed tree. If one qualifying entry resolves under another installed `transformers` tree, the case-normalized root set still contains more than one root and must fail closed with `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS`.

## 8. Required Regression Tests

The later implementation must add or update focused tests in `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`.

A. Nested initializer false-ambiguity regression:

Construct a `FakeDist.files` population containing at minimum:

- `transformers/__init__.py`
- `transformers/models/__init__.py`
- `transformers/models/mamba/__init__.py`
- `transformers/models/mamba/modeling_mamba.py`
- `transformers/cache_utils.py`

All entries must resolve inside one package tree. Require `_distribution_root()` or `resolve_transformers_sources()` to reconcile to exactly the common Transformers package root and not raise `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS`. This test must fail against frozen `de874a22` behavior and pass after the authorized correction.

B. Deep nested initializer regression:

Include at least one deeper package initializer, for example `transformers/data/datasets/__init__.py`, with another ordinary file in that subtree if useful. Require reconciliation to the same top-level Transformers package root.

C. Genuine ambiguity preservation:

Preserve or strengthen the existing genuine multi-root regression equivalent to `test_ambiguous_distribution_roots_block`. At least one qualifying entry must resolve into a genuinely separate installed tree after uniform path-depth derivation. Require `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS`.

D. Top-level initializer behavior:

Explicitly verify `transformers/__init__.py` still derives the package root.

E. Existing suite:

Require the complete existing `tests/test_preflight_longterm_o0c_runtime_source_provenance.py` suite to remain passing.

## 9. Static Validation Requirement

The later verifier must confirm at source level that `_distribution_root()` no longer contains the special `__init__.py` branch:

```python
located.parent if located.name == "__init__.py"
```

Equivalent conditional special handling for `located.name == "__init__.py"` inside `_distribution_root()` is not authorized. The corrected logic must use the uniform path-depth formula for every qualifying `transformers` entry.

## 10. Validation Commands And Gate

Minimum validation required for the later implementer:

```text
git diff --check
python -m pytest -q tests/test_preflight_longterm_o0c_runtime_source_provenance.py
```

Authoring inspection found no repository-local `.cm` or `cm` gate directory and no specific registered O0c preflight implementation gate to require. Do not invent a gate. If an independent verifier identifies a registered gate that specifically covers O0c preflight implementation, that gate must also be required before freezing the corrected implementation.

## 11. Execution And Scientific Boundary

Successful local implementation tests establish code correctness only.

They do not authorize:

- Kaggle preflight execution;
- preflight v3;
- scientific O0c execution;
- model or tokenizer loading;
- dataset loading;
- training or evaluation.

A separate execution authority is required after corrected implementation is independently verified and frozen.

Consumed runs must be preserved and not rerun or overwritten:

- `longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`
- `longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1`

No corrected preflight run name is reserved by this candidate.

## 12. Expected Future Implementation Delta

The expected later implementation delta is exactly:

```text
M scripts/preflight_longterm_o0c_runtime_source_provenance.py
M tests/test_preflight_longterm_o0c_runtime_source_provenance.py
```

No report modifications are authorized during implementation.

## 13. Expected Delta For This Authoring Task

This authoring task creates exactly one task-attributable untracked report candidate:

`reports/longterm_o0c_transformers_distribution_root_derivation_correction_implementation_authority_spec_candidate.md`

No existing file modification is authorized.

## 14. Candidate Identity To Verify

Candidate path:

`reports/longterm_o0c_transformers_distribution_root_derivation_correction_implementation_authority_spec_candidate.md`

Raw SHA256 / bytes / LF / CR / final-LF must be computed after writing this report.

## 15. Required Independent Verification Report Fields

An independent verifier should report:

1. Verdict: `PASS_READY_FOR_INDEPENDENT_VERIFICATION` or `BLOCKED`.
2. Authority chain.
3. Confirmed defect.
4. Exact future implementation scope.
5. Exact correction semantics.
6. Mechanical path-depth proof/examples.
7. Required regression tests.
8. Genuine ambiguity preservation.
9. Validation commands/gate.
10. Execution/scientific boundary.
11. Expected future implementation delta.
12. Candidate path and raw identity.
13. Git state.
14. Explicit no-implementation/no-execution/no-commit/no-push confirmation.

## 16. Non-Authorization Confirmation

This candidate does not authorize or perform:

- implementation changes during authoring;
- test changes during authoring;
- existing report modification;
- Kaggle execution;
- diagnostic rerun;
- preflight v2 rerun;
- preflight v3 creation;
- scientific O0c execution;
- model, tokenizer, or dataset loading;
- package or environment mutation;
- training or evaluation;
- staging, committing, pushing, resetting, or cleaning user work.

Next authorized action:

Independent verification of this report candidate.
