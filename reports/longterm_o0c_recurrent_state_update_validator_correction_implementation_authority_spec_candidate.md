# ContraMamba O0c Recurrent-State Update Validator Correction Implementation Authority Spec Candidate

## 1. Verdict

`CANDIDATE_AUTHORED_PENDING_INDEPENDENT_VERIFICATION_AND_FREEZE`

Phase: `REPORT_ONLY_RECURRENT_STATE_UPDATE_VALIDATOR_CORRECTION_IMPLEMENTATION_AUTHORITY_AUTHORING`.

This report-only candidate defines a narrow future local/static correction to the diagnosed recurrent-state-update validator false negative. It authorizes no change, test modification, Kaggle action, preflight, diagnostic, model/tokenizer/dataset use, forward, training, evaluation, staging, commit, or push in this authoring task.

## 2. Starting state

- Repository root: `C:\o0c-preflight-auth-c551747`.
- Required and observed HEAD: `3fcd581eb71add577815daf14e827992a3dc3c83`.
- Tracked modifications: none.
- Staged paths: none.
- Untracked paths: none.
- Candidate path before authoring: absent.
- Task temporary files: none identified.

`git status --short` had no entries. Git emitted permission warnings only for pre-existing inaccessible pytest-cache directories; they were neither task artifacts nor modified.

## 3. Authority chain

Authority is resolved in this order:

1. Current controller instruction.
2. Frozen recurrent-state-update diagnostic interpretation: `3fcd581eb71add577815daf14e827992a3dc3c83`.
3. Frozen recurrent-state-update diagnostic execution authority: `85a78574fd9bded7f761d8db314370e940627320`.
4. Last preflight implementation carrying the diagnosed defect: `426ecd2038a1118d3afb9bd7bdae63f340e3c70b`.
5. Frozen recurrent-state initialization validator correction authority: `989708d011efe10fd72e32a6d91dd9d225f00673`.
6. Repository `AGENTS.md`.

The current controller instruction is the implementation-authority source. The initialization authority is supporting historical authority only and does not broaden this task.

## 4. Frozen diagnosis and provenance

The immutable classification is `VALIDATOR_RECURRENT_STATE_UPDATE_PATTERN_FALSE_NEGATIVE`.

At implementation commit `426ecd2038a1118d3afb9bd7bdae63f340e3c70b`, the validated Transformers 5.0.0 runtime source has `MambaMixer.slow_forward` at lines 342-421 with 18 direct-body statements, including `If@400-417`. Frozen recurrence-loop discovery iterates only direct `slow.body` statements and accepts `ast.For`; therefore `FROZEN_DIRECT_FOR_COUNT=0`, `FROZEN_LOOP_CANDIDATE_COUNT=0`, and the frozen blocker is `recurrent_state_update`.

The exact source has one nested sequential `For` at lines 408-411:

`slow.body/If[15]@400-417.orelse/For[1]@408-411`

Its recurrent update is line 409, its post-update readout is line 410, and the diagnostic complete nested loop count is one. This is established diagnostic evidence, not an instruction to reopen or re-diagnose the question.

The permanently consumed validated diagnostic is:

- Run: `longterm-o0c-recurrent-state-update-diagnostic-426ecd2-v1`.
- Execution commit: `426ecd2038a1118d3afb9bd7bdae63f340e3c70b`.
- Command SHA256: `c8c5f9e6b2cd1a1d25d7c5841624f7b2ca68c901704a79115c9fcc9ecaeb9399`.
- Run-log SHA256: `6d6f5dc8cefbe3823677d7038609696e5fb21924c63c33b240ff88f5f10c9c62`.
- Run-meta SHA256: `64760ed45a6ecb0292f3ba130a0c596b6a39dd5c9a985c48d1c991437c250766`.
- Handoff-ZIP SHA256: `9d544d5c2698453b6d925ec6ff70cef6f98668ceec3ee4db350787911d89b930`.
- Import audit: `C:\Users\Home1\.contramamba\imports\longterm-o0c-recurrent-state-update-diagnostic-426ecd2-v1_426ecd2038a1_20260907_161054`.
- Import: `PASS`; artifact/provenance: `VALID`; scientific conclusion: `NONE`.

The diagnostic must not be rerun, reused, overwritten, or aliased.

## 5. Exact future writable files

After independent verification and formal freeze only, a future implementation may modify exactly:

- `scripts/preflight_longterm_o0c_runtime_source_provenance.py`
- `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

No other production, test, report, configuration, artifact, or provenance file is writable under this authority.

## 6. Correction objective

Authorize only the smallest correction necessary for recurrence-loop candidate discovery to inspect executable statements in the same lexical scope of `MambaMixer.slow_forward`, rather than only direct `slow.body` statements. The correction must make the validated nested sequential loop discoverable without crossing into nested lexical/code scopes.

The future implementation may reuse `_same_lexical_scope_statements(...)` if that is the smallest correct solution. An equivalent narrow traversal is permissible only after independent verification that it enforces the same lexical-scope boundary. No source-line, version, source-hash, AST-path, or environment special case is authorized.

## 7. Same-lexical-scope traversal contract

Traversal may descend through executable statement/control-flow containers in the current `MambaMixer.slow_forward` lexical scope, including nested `If`/`else`, `For`, `While`, `Try`/handlers/`else`/`finally`, `With`, and applicable `Match`/case statements.

Traversal must not descend into a body that establishes a new lexical/code scope: `FunctionDef`, `AsyncFunctionDef`, `ClassDef`, or `Lambda`. If comprehensions or generator expressions are encountered, do not expand this correction into arbitrary expression-scope analysis unless the existing frozen predicate requires it; statement traversal is preferred.

## 8. Preserved loop semantic predicate

The correction is loop discovery only. Every discovered `For` candidate must still be evaluated with the existing frozen `_loop_body_update_and_readout(loop)` logic.

Do not change `_loop_body_update_and_readout` semantics, `_expr_dependencies`, `_bind_assignment_dependencies`, `_assignment_value`, `_loop_target_names`, `_is_range_iteration`, the recurrent-update dependency requirement, or the post-update readout requirement.

## 9. Candidate cardinality and fail-closed preservation

Exactly one qualifying loop candidate remains required. Zero qualifying candidates must remain `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED` with blocker `recurrent_state_update`; more than one must remain `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` with blocker `recurrent_state_update`; and an update whose post-update readout is unresolved must retain its existing fail-closed blocker behavior. The implementation must not choose a source-order candidate merely to avoid ambiguity.

## 10. Initialization preservation

Do not alter the frozen recurrent-state initialization correction. Preserve same-lexical-scope initialization discovery, the `Assign`/`AnnAssign` restriction, the bare `ssm_state` assignment predicate, the `new_zeros`/`zeros`/`empty` call predicate, and zero/multiple initialization-candidate fail-closed behavior. The future implementation must not regress the already-corrected nested initialization case.

## 11. Adjacent non-authorization

This authority does not authorize changes to recurrent-state initialization, readout recognition, dependency analysis, backend proof, CPU/kernel dispatch proof, convolution cache binding, cache recurrent-state storage, source resolution, package/distribution-root derivation, output schema, provenance schema, runtime version checks, artifact serialization, or CLI behavior. If broadened discovery exposes another blocker, stop; a new authority is required.

## 12. No source special casing

The future implementation must not branch on Transformers 5.0.0, an exact `modeling_mamba.py` SHA, source lines 400/408/409/410, AST indices `If[15]`/`For[1]`, or Kaggle-specific paths. It must express a generic structural validator rule.

## 13. Required future tests

Use repository-native tests and fixtures where available. Focused regression coverage in the dedicated test file must establish at least:

- A. A valid update and post-update readout in a `For` nested below an `If`/`else` in `slow_forward` bind loop/update/readout.
- B. An existing direct-body loop still succeeds.
- C. Zero qualifying loops remains `recurrent_state_update` unresolved.
- D. Multiple qualifying loops remains `recurrent_state_update` ambiguous.
- E. A loop in a nested `FunctionDef` is ignored.
- F. A loop in a nested `AsyncFunctionDef` is ignored where applicable.
- G. A loop in a nested `ClassDef` is ignored.
- H. Nested loops lacking valid update/readout do not become candidates.
- I. Unresolved post-update readout preserves its existing blocker semantics.
- J. Previously fixed nested recurrent-state-initialization discovery remains passing.
- K. Adjacent backend, readout, and dependency semantics are not broadened.

## 14. Future local validation

Future implementation permits local/static validation only: the targeted dedicated preflight test file, the relevant bounded existing preflight/O0c validator suite, `git diff --check`, and exact diff review. It authorizes no Kaggle action, actual preflight execution, model/tokenizer/dataset use, forward, training, or evaluation.

## 15. Independent implementation verification

Before implementation freeze, an independent verifier must confirm exact writable scope; lexical-scope traversal boundaries; candidate cardinality; unchanged update/readout semantics; unchanged initialization logic; absence of source/version/hash/line special casing; direct/nested/zero/multiple/new-scope-exclusion coverage; complete local test results; exact production/test raw identities; and clean final Git state.

## 16. Expected implementation outcome

Only after successful tests and independent verification may the implementation establish `PASS_SAFE_TO_FREEZE_RECURRENT_STATE_UPDATE_VALIDATOR_CORRECTION_IMPLEMENTATION`. It must not claim whole-preflight PASS, source-provenance PASS, O0c execution readiness, scientific correctness, or a scientific conclusion. A separately authorized corrected-preflight execution remains required.

## 17. Scientific and execution boundary

Current authority authoring and future correction implementation prohibit Kaggle, preflight, diagnostic execution, model, tokenizer, dataset, forward, training, and evaluation. Scientific conclusion remains `NONE`.

## 18. Failure recovery

Stop rather than widen scope if the existing helper cannot safely express same-lexical-scope traversal; discovery repair requires update/readout semantic modification; multiple new defects appear; a test requires source/version-specific behavior; unrelated repository changes exist; or provenance/authority identities mismatch. Any adjacent defect requires a new authority.

## 19. Git validation

After authoring, only read-only/static Git validation is permitted:

```text
git diff --check
git diff --name-status
git diff --cached --name-status
git status --short
git rev-parse HEAD
```

Required final state: HEAD `3fcd581eb71add577815daf14e827992a3dc3c83`; no tracked modifications; nothing staged; exactly one untracked file, `reports/longterm_o0c_recurrent_state_update_validator_correction_implementation_authority_spec_candidate.md`; and no task temporary files.

## 20. Candidate raw identity

The candidate SHA256, byte count, LF count, CR count, and final-LF flag must be computed after final validation and recorded in the task result, outside this self-referential candidate body. The independent authority verifier must independently reproduce those values against this exact path.

## 21. Discrepancies and blockers

No blocker was identified during authority authoring. The only Git inspection noise was the pre-existing inaccessible pytest-cache warnings described in Section 2; no action was taken on those directories.

## 22. Exact next authorized action

An independent verifier may perform the required read-only verification of this candidate, its raw identity, frozen authority/provenance, and clean Git state. No implementation, test modification, execution, staging, commit, or push is authorized until this candidate is independently verified and formally frozen.
