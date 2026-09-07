# ContraMamba O0c backend-path validator correction implementation authority — candidate

## 1. Verdict

`PASS_READY_FOR_INDEPENDENT_BACKEND_PATH_VALIDATOR_CORRECTION_IMPLEMENTATION_AUTHORITY_VERIFICATION`

This is a report-only implementation-authority candidate. It authorizes neither implementation nor execution until independent authority verification freezes it. Training/evaluation, Kaggle, run registration/execution, staging, commit, and push are `NO`.

## 2. Starting HEAD/state

```text
ROOT=C:\o0c-preflight-exec-auth-686b745
HEAD=65ebe123832a4a15c2f5e11f81daff878c13469f
TRACKED_MODIFICATIONS=none
STAGED=none
UNTRACKED=none
CANDIDATE_PATH=absent
TASK_TEMPORARY_FILES=none
```

These conditions were independently checked before authoring. A mismatch would have blocked authoring without changes.

## 3. Frozen authority chain

```text
INTERPRETATION_ROOT_CAUSE_AUTHORITY=65ebe123832a4a15c2f5e11f81daff878c13469f
RECOVERY_EXECUTION_AUTHORITY=c67122e0b2f629d82ba1025f01dad3fd204f9b6f
PARENT_BACKEND_DIAGNOSTIC_AUTHORITY=4ced96fa8f8f2d71492fba3c8a5a2dec1a7c6c0c
VALIDATED_DIAGNOSTIC_EXECUTION_COMMIT=686b7457ed220e7d74ecdf41eb7ec500d24bb23f
```

## 4. Frozen root cause

```text
ROOT_CAUSE_STATUS=ESTABLISHED_FOR_FROZEN_BACKEND_VALIDATOR_BLOCKER
PRIMARY_ROOT_CAUSE=VALIDATOR_OPTIONAL_BACKEND_REACHABILITY_OVERAPPROXIMATION
SECONDARY_MASKED_VALIDATOR_DEFECT=VALIDATOR_BACKEND_PATH_PATTERN_FALSE_NEGATIVE
SCIENTIFIC_CONCLUSION=NONE
```

The primary defect is not an assertion that optional symbols are absent. It is the invalid inference from module/global symbol occurrence to possible intervention on the inspected CPU/non-CUDA `MambaMixer.forward` selection path. The secondary defect is independent: even after reachability is correctly scoped, the frozen proof helper does not recognize the validated CUDA-fast-return plus direct slow-fallthrough control-flow shape.

## 5. Baseline production/test identities

At authority `HEAD=65ebe123832a4a15c2f5e11f81daff878c13469f`, the required Git blob identities are:

```text
scripts/preflight_longterm_o0c_runtime_source_provenance.py
2dc72aaed31a1932e77ec93273918b83835a12e1

tests/test_preflight_longterm_o0c_runtime_source_provenance.py
a009e375cef8de0a3f3d93c400c09523fd24a513
```

The future implementation must block if either identity does not match before editing.

## 6. Exact future implementation scope

The future implementation may modify exactly these existing tracked files:

```text
scripts/preflight_longterm_o0c_runtime_source_provenance.py
tests/test_preflight_longterm_o0c_runtime_source_provenance.py
```

No other production, test, report, config, workflow, documentation, dataset, artifact, run-registry, or infrastructure file is authorized. No new files are authorized. If a third file is required, stop and obtain separate authority.

## 7. Primary correction semantics

Correct `VALIDATOR_OPTIONAL_BACKEND_REACHABILITY_OVERAPPROXIMATION` by limiting optional-backend analysis to control flow relevant and reachable for the CPU/non-CUDA backend-selection question.

Mere optional-symbol presence in `MambaMixer.__init__`, `MambaMixer.warn_slow_implementation`, `MambaMixer.cuda_kernels_forward`, or availability/feature-check expressions in `MambaMixer.forward` must not alone yield `BACKEND_ASSOCIATIVE_OR_KERNEL_PATH_MAY_INTERVENE` where a statically proven CUDA-only fast branch excludes backend calls from the CPU path. Do not globally suppress optional calls or conclude their absence. A genuinely CPU-reachable optional invocation, or one whose CPU reachability cannot be excluded, remains fail-closed and not statically sequential.

## 8. Secondary correction semantics

Correct `VALIDATOR_BACKEND_PATH_PATTERN_FALSE_NEGATIVE` additively. In addition to the legacy direct CPU-positive `If` plus `slow_forward` branch plus raising alternate branch, recognize this AST/control-flow form:

```python
if (
    <fast-path availability>
    and "cuda" in <...>.device.type
    and <other conjuncts>
):
    return cuda_kernels_forward(...)
return slow_forward(...)
```

For the CPU/non-CUDA question, proof requires that the fast branch be statically CUDA-only and that the direct fallthrough statically selects `slow_forward`. Recognition must be semantic AST/control-flow analysis, never name/string matching. The qualifying direct fast-path `If` is an acceptable backend-selection node.

## 9. Legacy-proof preservation

The existing positive `MAMBA_PASS` fixture must continue to classify as `BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN`. The correction is additive; it must not remove the existing direct CPU-positive `If` proof.

## 10. Fail-closed requirements

Do not prove `BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN` for any of the following: no unique `MambaMixer.forward`; no `slow_forward`; ambiguous qualifying dispatch structures; a CPU-reachable optional invocation; optional invocation reachability not excludable; a fast branch not statically CUDA-only; a CUDA condition under OR, negation, or other shape that does not prove CPU exclusion; no statically established slow fallthrough; multiple incompatible dispatch candidates; or a source merely textually similar but not AST/control-flow equivalent.

When proof is insufficient, retain the existing result-neutral fail-closed classifications. Do not force a proven classification.

## 11. Schema/classification preservation

Do not add, remove, rename, or reinterpret `STATUSES`, `BACKEND_CLASSIFICATIONS`, `SCHEMA_VERSION`, artifact schema keys, or `PASS_STATUS`. The default and intended implementation is no schema/classification change: only selection among existing classifications changes for the supported structures.

Prefer a narrow change around `_backend_proof_if`, `_has_optional_backend_path`, and `classify_backend`. Small private helpers are permitted only to reduce ambiguity within the same production file. Do not introduce a broad AST-framework refactor. Preserve `_backend_proof_if` compatibility with `bind_symbol_locations` and its `backend_kernel_selection` location.

## 12. Exact-source-shape regression fixture

Add a deterministic synthetic `MambaMixer` source fixture semantically equivalent to the validated Transformers 5.0.0 selection structure. Its `forward` defines an availability expression containing `selective_state_update`, `selective_scan_fn`, `causal_conv1d_fn`, `causal_conv1d_update`, and `mamba_inner_fn`; then uses an AND condition containing availability, `"cuda" in self.x_proj.weight.device.type`, and `not is_torchdynamo_compiling()` to return `self.cuda_kernels_forward(...)`; and directly falls through to `return self.slow_forward(...)`. Its CUDA helper may contain optional backend calls, while `slow_forward` exists.

The fixture need not reproduce unrelated Transformers code, but must preserve the stated backend-selection AST semantics. Its corrected classification must be `BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN`.

## 13. Primary regression tests

Add deterministic tests establishing all of the following:

1. Optional backend symbols confined to a CUDA-only fast implementation do not block CPU proof.
2. Optional symbols in `__init__` or warning/helper methods do not block proof merely by existing in the module.
3. Optional names appearing only in a fast-path availability expression do not count as CPU-reachable execution.
4. An unguarded optional invocation reachable in `MambaMixer.forward` yields a non-proven result, normally `BACKEND_ASSOCIATIVE_OR_KERNEL_PATH_MAY_INTERVENE` where the exact decision logic supports it.
5. An optional invocation on the CPU-selected path never yields `BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN`.

## 14. Secondary regression tests

Add deterministic tests establishing all of the following:

1. CUDA-gated fast return plus direct `slow_forward` fallthrough is recognized.
2. Removing the CUDA-only gating prevents proof.
3. Replacing the CUDA conjunction with a shape that does not establish non-CUDA exclusion prevents proof.
4. Removing the `slow_forward` fallthrough prevents proof.
5. A fast backend call before or after the guard on a CPU-reachable path prevents proof.
6. Legacy exact CPU-positive `MAMBA_PASS` remains supported.

## 15. Counterfactual regression

Preserve the frozen causal distinction with coverage equivalent to:

```text
backend proof false
optional path false
slow_forward present
=> BACKEND_SEQUENTIAL_PRESENT_SELECTION_UNPROVEN
```

This counterfactual must remain separate from the optional-reachability fix so the independent secondary proof-shape defect is not erased from decision semantics.

## 16. `bind_symbol_locations` regression

Add or update a test showing that the corrected CUDA-fast-return plus slow-fallthrough fixture, with every other required synthetic symbol valid, successfully binds `symbol_locations["backend_kernel_selection"]` deterministically to the qualifying `MambaMixer.forward` backend-selection AST node. Do not hardcode real Transformers line numbers.

## 17. Existing behavior preservation

Do not alter established recurrent-state semantics or source-provenance behavior. Every existing test in `tests/test_preflight_longterm_o0c_runtime_source_provenance.py` must remain passing unless its changed expectation directly encodes root cause `65ebe123832a4a15c2f5e11f81daff878c13469f`; each such change requires explicit justification. No unrelated expectation churn is authorized.

## 18. Future implementation validation

The future implementation must run, at minimum:

```text
python -m pytest -q tests/test_preflight_longterm_o0c_runtime_source_provenance.py
git diff --check
git diff --name-status
git status --short
git rev-parse HEAD
```

If needed, use the existing Windows pytest temp-root workaround with `TEMP`, `TMP`, and `PYTEST_DEBUG_TEMPROOT` under `C:\Users\Home1\.contramamba\pytest-temp`. No training, evaluation, Kaggle, or network-dependent validation is authorized.

## 19. Future expected final state

The future implementation must leave exactly:

```text
M scripts/preflight_longterm_o0c_runtime_source_provenance.py
M tests/test_preflight_longterm_o0c_runtime_source_provenance.py
```

Nothing may be staged; no untracked files or other tracked modifications may exist. Commit and push are `NO`.

## 20. Independent implementation verification

After implementation, an independent read-only verifier is mandatory. It must inspect this exact authority, the exact baseline and two-file scope, both causal defects, scoped optional reachability, the added proof shape, legacy preservation, fail-closed adversarial coverage, counterfactual behavior, `bind_symbol_locations`, full targeted-test output, production/test raw identities, and final Git state. The verifier must not modify files.

## 21. Post-implementation execution boundary

Even a verified implementation PASS does not authorize corrected preflight execution, Kaggle, model execution, training/evaluation, or scientific interpretation. A separate corrected-preflight execution authority must be authored and frozen first. `SCIENTIFIC_CONCLUSION=NONE` remains in force.

## 22. Current authoring validation

This authoring task runs read-only validation only: `git diff --check`, `git diff --name-status`, `git diff --cached --name-status`, `git status --short`, and `git rev-parse HEAD`. The final required authoring state is the frozen HEAD above, no tracked or staged modifications, exactly this untracked candidate, and no temporary files.

## 23. Candidate raw identity

The independent verifier must compute the final raw-byte identity of this unstaged candidate and confirm SHA256, byte count, LF count, CR count, final-LF status, and blank-line-at-EOF status. These values are reported by the authoring result rather than embedded here, because embedding a final-file SHA256 in the file would make the raw identity self-referential.

## 24. Discrepancies/blockers

None identified. The exact starting HEAD, clean starting state, absent candidate path, and both baseline blobs matched. This candidate itself authorizes no implementation until its mandatory independent authority verification succeeds.

## 25. Exact next authorized action

Perform independent read-only verification of `reports/longterm_o0c_backend_path_validator_correction_implementation_authority_spec_candidate.md`: confirm the frozen chain and root cause, both blobs, two-file scope, causally distinct primary/secondary corrections, no overbroad optional-backend suppression, fail-closed and legacy requirements, no schema widening, regression sufficiency, no execution authority, `SCIENTIFIC_CONCLUSION=NONE`, raw identity, and final Git state. No implementation, Kaggle, run registration/execution, training/evaluation, staging, commit, or push is authorized.
