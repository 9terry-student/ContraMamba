# Longterm O0c Recurrent-State Initialization Validator Correction Implementation Authority Spec Candidate

## 1. Overall verdict

Verdict:

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

Phase:

`REPORT_ONLY_RECURRENT_STATE_INITIALIZATION_VALIDATOR_CORRECTION_IMPLEMENTATION_AUTHORITY`

This is a report-only implementation authority/spec candidate for a future narrow correction of the confirmed recurrent-state-initialization validator false negative.

It does not implement the correction. It authorizes no execution, no training, no evaluation, no Kaggle work, no staging, no commit, and no push.

## 2. Starting repo/HEAD/state

Starting-state validation passed before authoring:

- repository root: `C:\o0c-preflight-auth-c551747`
- expected HEAD: `bfc998f9206ff77fcc2b2e80bb94293dc737b13f`
- observed HEAD: `bfc998f9206ff77fcc2b2e80bb94293dc737b13f`
- tracked modifications: none
- staged changes: none
- pre-existing candidate collision at `reports/longterm_o0c_recurrent_state_initialization_validator_correction_implementation_authority_spec_candidate.md`: none
- task-attributable temp files: none observed

Because no mismatch was observed, authoring this single report candidate was permitted.

## 3. Authority chain

Authority order used:

1. Current controller instruction for this report-only implementation authority/spec candidate.
2. Frozen recurrent-state root-cause interpretation authority: commit `bfc998f9206ff77fcc2b2e80bb94293dc737b13f`, `reports/longterm_o0c_recurrent_state_initialization_diagnostic_interpretation_candidate.md`.
3. Frozen recurrent-state-initialization diagnostic execution authority: commit `ca112032841fe316e5e2e7335dc95d89aeedb450`.
4. Frozen corrected-preflight execution authority: commit `52bae21fc5f5ee4e3ba3b5b15a0d5682f86daf0f`.
5. Frozen corrected implementation / diagnostic execution commit: `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`, especially `scripts/preflight_longterm_o0c_runtime_source_provenance.py`.
6. Earlier applicable implementation authority: commit `811ae9c843564e8cddb5fc373761afb618cb7cfd`.
7. Frozen O0c runtime-source preflight authority: commit `8c6a0ccf2a8583b9b7accbdb5ab757d722b6e328`.
8. Frozen O0c native-state authority: commit `242ad9ed70fc995ebda560911a7d0dfd2f18f9b3`.
9. Repository `AGENTS.md`.

No higher-priority conflict was found.

## 4. Frozen interpretation and root cause

The frozen interpretation report at `bfc998f9206ff77fcc2b2e80bb94293dc737b13f:reports/longterm_o0c_recurrent_state_initialization_diagnostic_interpretation_candidate.md` assigns exactly one root-cause classification:

`VALIDATOR_RECURRENT_STATE_INITIALIZATION_PATTERN_FALSE_NEGATIVE`

The frozen narrow root cause remains:

The validator's direct-body-only AST scan fails to descend into the cache/non-cache `If`, so it misses the nested fresh-state initialization:

`ssm_state = torch.zeros(...)`

The nested assignment satisfies the existing statement-level recurrent-initialization predicate, but it is not a direct statement in `MambaMixer.slow_forward.body`.

This report does not reinterpret or broaden that root cause.

Scientific conclusion remains:

`NONE`

## 5. Frozen implementation defect

At frozen corrected implementation / diagnostic execution commit `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`, `_recurrent_proof_nodes` in `scripts/preflight_longterm_o0c_runtime_source_provenance.py` implements recurrent initialization discovery as:

```python
init_nodes = [
    stmt
    for stmt in slow.body
    if isinstance(stmt, (ast.Assign, ast.AnnAssign))
    and _assigns_name(stmt, "ssm_state")
    and _calls_attr(stmt, {"new_zeros", "zeros", "empty"})
]
```

The current logic inspects only direct statements in `slow.body`.

The exact existing cardinality contract is:

- `len(init_nodes) == 0` raises `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED` with blocker `recurrent_state_initialization`.
- `len(init_nodes) > 1` raises `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` with blocker `recurrent_state_initialization`.
- exactly one match is required.

A future correction must preserve these fail-closed cardinality semantics exactly.

## 6. Explicit adjacent nested-loop/update boundary

The remainder of `_recurrent_proof_nodes` separately discovers recurrence-loop candidates with direct-body iteration:

```python
for stmt in slow.body:
    if isinstance(stmt, ast.For):
        ...
```

The exact installed Transformers `5.0.0` source observed by diagnostic v1 places the sequential `for i in range(seq_len)` under the line-400 conditional's `else`, not as a direct `slow.body` statement.

That nested-loop/update surface is adjacent, but it is not authorized for correction by this implementation authority.

The future implementation authorized by this spec must not:

- change recurrence-loop discovery;
- change `_loop_body_update_and_readout`;
- change `recurrent_state_update` binding semantics;
- change post-update readout semantics;
- make the exact Transformers `5.0.0` source pass later checks by proactively descending to its nested loop;
- claim that the whole preflight will pass after this initialization-only fix.

If a corrected preflight later reaches a new `recurrent_state_update` blocker, that result must be treated as a separate observed blocker requiring its own authority, diagnostic, and interpretation sequence as appropriate.

This boundary is mandatory.

## 7. Exact future writable files

Future implementation may modify only:

- `scripts/preflight_longterm_o0c_runtime_source_provenance.py`
- `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

The test file above is the existing dedicated preflight test module discovered from the repository. No new test module should be invented unless a later formal repository structure authority requires it.

No other production files are authorized.

## 8. Exact authorized implementation behavior

The future correction may make only the smallest bounded change necessary for `recurrent_state_initialization` discovery.

The intended behavior is equivalent to:

- search executable statements belonging to the unique `MambaMixer.slow_forward`;
- allow the recurrent-state initialization matcher to descend through statement container blocks such as the cache/non-cache `If`;
- identify the nested `Assign` at installed-source lines `375-378`;
- retain the existing initialization predicate unchanged:
  - `Assign` or `AnnAssign`;
  - assigns bare name `ssm_state`;
  - contains a call whose final attribute/name is one of `new_zeros`, `zeros`, or `empty`;
- retain exactly-one cardinality semantics;
- return and bind the actual matching assignment node, so `symbol_locations.recurrent_state_initialization` points to the nested assignment itself, not merely the enclosing `If`.

The correction must be structural and deterministic. It must not be a semantic rewrite of the preflight.

The author of the future implementation may choose the exact helper shape after inspecting the existing code, but this authority prefers a small deterministic statement traversal helper over broad `ast.walk` usage on the whole function.

## 9. Lexical-scope traversal constraints

The future traversal must not accidentally cross Python lexical scope boundaries.

At minimum, assignments inside nested nodes of these types must not be treated as initialization statements belonging to `MambaMixer.slow_forward`:

- `FunctionDef`
- `AsyncFunctionDef`
- `ClassDef`
- `Lambda`

Permitted descent is limited to executable statement containers within the same lexical function body, such as `If` bodies and `else` bodies, as needed to find a qualifying initialization assignment.

The traversal must bind the matching `Assign` or `AnnAssign` node itself.

## 10. Preserved cardinality/fail-closed semantics

The future implementation must preserve the existing fail-closed recurrent-initialization contract:

- zero qualifying matches must still raise `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED` with blocker `recurrent_state_initialization`;
- more than one qualifying match must still raise `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` with blocker `recurrent_state_initialization`;
- exactly one qualifying match is required before later recurrent-state proof steps are considered.

The future implementation must not change blocker names or suppress subsequent blockers.

## 11. Required regression tests

Future implementation must add focused coverage in `tests/test_preflight_longterm_o0c_runtime_source_provenance.py` proving at least:

A. Exact false-negative regression

A `MambaMixer.slow_forward` fixture with:

```python
if cache_params is not None:
    ssm_state = cache_params.ssm_states[self.layer_idx].clone()
else:
    ssm_state = torch.zeros(...)
```

must produce exactly one recurrent-state-initialization match corresponding to the nested `torch.zeros` assignment.

B. Symbol-location correctness

The bound `recurrent_state_initialization` location must correspond to the nested `Assign` node, not the enclosing `If`.

C. Existing direct-body compatibility

A previously supported direct-body `ssm_state = torch.zeros(...)` must remain recognized.

D. Zero-match fail closed

No qualifying `ssm_state` initialization must still produce `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED` with blocker `recurrent_state_initialization`.

E. Multiple-match fail closed

Two qualifying executable `ssm_state` zero-like initializations must still produce `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` with blocker `recurrent_state_initialization`.

F. Irrelevant zeros ignored

Assignments such as `conv_state = torch.zeros(...)` and `temp = torch.zeros(...)` must not match.

G. Non-zero cache load ignored as fresh initializer

`ssm_state = cache_params.ssm_states[...].clone()` must not satisfy the zero-like initialization predicate.

H. Nested lexical scopes excluded

A `torch.zeros` assignment to `ssm_state` inside a nested function or class defined inside `slow_forward` must not count.

I. Scope preservation

Tests must establish that the initialization correction does not silently alter recurrent-state-update or loop-discovery behavior. If an exact-source-style fixture contains the nested line-408 recurrence loop under the line-400 conditional's `else`, the test must not assert whole-preflight `PASS` merely because initialization is now found.

## 12. Exact forbidden widening

The future implementation must not modify:

- unrelated scripts;
- O0c scientific implementation;
- model code;
- dataset code;
- evaluation code;
- `AGENTS.md`;
- `README.md`;
- `cm.ps1`;
- prior reports;
- run registry;
- imported audit files;
- package environment.

The future implementation must not change:

- runtime version checks;
- source path resolution;
- distribution-root derivation;
- shadowing checks;
- raw source identity;
- schema version;
- status vocabulary;
- output serialization;
- backend selection proof;
- convolution cache proof;
- hidden/output path proof;
- recurrent update dependency semantics;
- scientific O0c semantics.

The future implementation must not special-case:

- Transformers version string;
- exact source SHA;
- exact source line number;
- one Kaggle path.

Explicitly forbidden logic includes:

- `if transformers == "5.0.0"`;
- matching line `375`;
- matching SHA256 `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83`;
- matching literal source text;
- treating `else` position by line number.

The fix must recognize the structural AST shape that the frozen semantic predicate already intends to accept.

## 13. Exact local validation commands

After future implementation, local validation authority is limited to implementation validation. It is not scientific evidence.

Required commands:

```text
python -m pytest -q tests/test_preflight_longterm_o0c_runtime_source_provenance.py
git diff --check
git diff --name-status
git diff --cached --name-status
git status --short
git rev-parse HEAD
```

The future author must also perform exact diff review showing only the authorized files changed:

- `scripts/preflight_longterm_o0c_runtime_source_provenance.py`
- `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

No Kaggle execution is authorized by this implementation authority itself.

No corrected preflight runtime execution is authorized by this implementation authority itself.

## 14. Independent-verification requirement

Because this correction changes provenance/authority validation logic, any future implementation requires independent verification before freeze.

The independent verifier must specifically check:

- traversal does not cross lexical scopes;
- cardinality remains fail-closed;
- only initialization discovery changed;
- no recurrent-update or loop fix was smuggled in;
- no exact-version, hash, line, or source-text hardcoding was introduced;
- regression tests cover nested `If`, ambiguity, and no-match cases.

## 15. Execution/Kaggle/training/evaluation prohibition

This spec does not authorize:

- Kaggle;
- `cm run save`;
- `cm run`;
- `cm collect`;
- `cm import`;
- diagnostic v1 rerun;
- preflight v3 rerun;
- a new preflight run;
- model loading;
- tokenizer loading;
- dataset loading;
- forward pass;
- generation;
- training;
- evaluation;
- staging;
- commit;
- push.

## 16. Evidence-layer separation

Corrected-preflight code correctness:

Existing local suite `PASS` from prior evidence only.

Prior corrected preflight v3 execution:

Validated fail-closed blocker:

`BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED`

with blocker:

`recurrent_state_initialization`

Prior v3 provenance:

`VALID`

Diagnostic v1 execution:

`PASS`, exit `0`

Diagnostic v1 provenance:

`VALID` after `COLLECT PASS` and `IMPORT PASS`

Frozen recurrent-state root-cause interpretation:

`VALIDATOR_RECURRENT_STATE_INITIALIZATION_PATTERN_FALSE_NEGATIVE`

Scientific O0c conclusion:

`NONE`

These layers must not be conflated.

## 17. Scientific conclusion

Scientific conclusion:

`NONE`

This report makes no model result claim and no O0c scientific correctness claim.

## 18. Post-implementation next-step boundary

The future implementation may establish only:

- code-level correction of `recurrent_state_initialization` discovery;
- regression coverage for the confirmed false-negative shape.

It must not claim:

- corrected preflight `PASS` on Kaggle;
- runtime-source identity `PASS`;
- recurrent-state-update correctness for exact Transformers `5.0.0`;
- full O0c source compatibility;
- O0c scientific correctness;
- any model result.

After implementation and local verification, a separate freeze/verification step is required before any corrected preflight execution authority can be considered.

## 19. Git validation results

Validation to run after authoring this candidate:

```text
git diff --check
git diff --name-status
git diff --cached --name-status
git status --short
git rev-parse HEAD
```

The final observed results are reported outside this file in the task response to avoid recursive report mutation.

## 20. Candidate path and raw identity

Candidate path:

`reports/longterm_o0c_recurrent_state_initialization_validator_correction_implementation_authority_spec_candidate.md`

Final raw identity is reported outside this file after final validation:

- SHA256: recompute after final edits
- bytes: recompute after final edits
- LF: recompute after final edits
- CR: recompute after final edits
- final LF: recompute after final edits

## 21. Discrepancies/blockers

No blocking discrepancy was found.

No implementation was performed.

No training, evaluation, Kaggle, diagnostic rerun, corrected preflight runtime execution, staging, commit, or push was performed.
