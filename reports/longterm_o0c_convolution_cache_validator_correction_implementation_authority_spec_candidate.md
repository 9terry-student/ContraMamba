# ContraMamba O0c convolution-cache validator correction implementation authority candidate

## 1. Verdict

`PASS_READY_FOR_INDEPENDENT_CONVOLUTION_CACHE_VALIDATOR_CORRECTION_IMPLEMENTATION_AUTHORITY_VERIFICATION`

This is a report-only implementation-authority candidate. It does not modify production code or tests and does not authorize execution, Kaggle, training, evaluation, commit, or push.

## 2. Active authority and phase

Authority order:

1. Current controller instruction.
2. Formally frozen root-cause interpretation commit `c4ec40fc8e4df82243c2facb810146513ec97b55`.
3. Frozen diagnostic execution authority commit `e7f1d8a0c38c13c8a10bbad8b489cd3012ff66e8`.
4. Frozen implementation under correction `eebf4da0207f00d993d7a6ae213e0cc17b4b1dfc`.

Phase:

`CONVOLUTION_CACHE_VALIDATOR_CORRECTION_IMPLEMENTATION_AUTHORITY`

Scientific conclusion remains `NONE`.

## 3. Frozen root cause

The root cause is not reopened.

Primary:

`VALIDATOR_CONVOLUTION_CACHE_SYMBOL_FAMILY_PATTERN_FALSE_NEGATIVE`

Secondary:

`VALIDATOR_CONVOLUTION_CACHE_DIRECT_BODY_SCOPE_FALSE_NEGATIVE`

Frozen evidence requires all of the following to remain true:

- frozen direct-body candidate count was `0`;
- same-lexical-scope recursive assignment count was `3`;
- the recursive assignment set was not unique;
- two `cache_params.update_conv_state` calls were statically proven persistent cache updates;
- `calls_required_determination=PROVEN_REQUIRED`;
- therefore replacing direct scanning with recursive assignment scanning alone is not an adequate correction.

## 4. Authorized future implementation scope

A later bounded implementation task may modify exactly:

- `scripts/preflight_longterm_o0c_runtime_source_provenance.py`
- `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

No other repository file is authorized for modification by the implementation task.

The correction must remain static-source / AST-only.

It must not alter:

- runtime-version checking;
- source-resolution or shadowing logic;
- raw source identity;
- recurrent-state semantic classification;
- backend-path semantic classification;
- `SYMBOL_KEYS`;
- artifact schema version;
- artifact top-level key set;
- location-object key set;
- deterministic serialization;
- output publication/collision behavior;
- CLI arguments;
- execution/provenance machinery.

No model, tokenizer, dataset, tensor forward, generation, training, evaluation, optional-kernel execution, package install, or package mutation is authorized.

## 5. Schema-preserving location rule

The existing artifact schema records one location object per symbol family.

The correction must not broaden `symbol_locations` into a multi-location schema.

For the branch-aware semantic-family path, proof may use multiple assignments, calls, branches, and method definitions. After the semantic proof succeeds, the canonical location for:

`convolution_cache_initialization_update`

must be:

- `qualname=MambaMixer.slow_forward`
- `source_file_key=mamba`
- `source_sha256=<current measured Mamba source SHA256>`
- `start_line/end_line=<actual parsed MambaMixer.slow_forward span>`

The implementation must not arbitrarily select one conv-state assignment or one `update_conv_state` call as the family location.

## 6. Legacy direct-single-assignment compatibility

Existing valid synthetic/simple fixtures must remain supported.

A legacy direct path may succeed when all are true:

- exactly one relevant direct-body `Assign` or `AnnAssign` exists;
- no competing relevant nested convolution-state assignments exist;
- no relevant `cache_params.update_conv_state` call requires broader semantic modeling.

For that narrow legacy case, preserving the existing assignment-node location is permitted.

If broader branch/call evidence exists, the implementation must use the semantic-family proof rather than silently falling back to the direct legacy rule.

## 7. Same-lexical-scope collection rule

Within `MambaMixer.slow_forward`, the implementation may recursively inspect control-flow statements while excluding nested lexical definitions:

- `FunctionDef`
- `AsyncFunctionDef`
- `ClassDef`

It may collect relevant:

- `Assign`
- `AnnAssign`
- `AugAssign`

when assigned paths structurally identify convolution state/cache state, including `conv_state`, `conv_states`, and structurally relevant conv-prefixed state names.

Comments, strings, or token-name presence alone are not proof.

Nodes inside nested function/class scopes must not participate.

## 8. Cache-present branch proof

For the semantic-family path, the implementation must structurally prove a cache-present branch equivalent to:

`cache_params is not None`

A valid reversed identity-comparison representation may be accepted if structurally equivalent.

Cache presence must not be inferred from names alone.

If no qualifying cache-present structure supports the family:

- status `BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`
- note `convolution_cache_initialization_update`

If multiple incompatible qualifying cache-present structures prevent a unique semantic proof:

- status `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`
- note `convolution_cache_initialization_update`

## 9. Prefill/decode branch proof

Within the relevant cache-present path, the implementation must structurally identify a prefill/decode split equivalent to comparison of:

`cache_position.shape[0]`

with:

`self.conv_kernel_size`

For the frozen Transformers 5.0.0 source:

- `BODY` is prefill;
- `ORELSE` is decode.

The proof must establish opposite arms of the same `ast.If`.

Source order, comments, or naming alone are insufficient.

Zero usable split is unresolved. Multiple incompatible splits are ambiguous.

## 10. update_conv_state call collection

Collect same-lexical-scope calls whose call path ends with:

`update_conv_state`

A call is semantically relevant only when its receiver is structurally `cache_params`.

Calls through `self`, `other_cache`, unrelated receivers, or nested function/class scopes must not prove the family.

## 11. Method discovery and linkage

Search `update_conv_state` definitions in both parsed source trees already available to `bind_symbol_locations`:

- Mamba source;
- cache source.

Do not assume the method is defined in `cache_utils.py`.

The frozen current source legitimately defines `MambaCache.update_conv_state` in `modeling_mamba.py`.

When the `cache_params` annotation is available, use it to constrain method linkage. For example, `MambaCache | None` may constrain the relevant class to `MambaCache`.

Zero plausible linked methods is unresolved.

Multiple plausible linked methods is ambiguous.

## 12. Persistent convolution-cache mutation proof

A linked `update_conv_state` method is acceptable only when persistent convolution-cache mutation is statically proven.

Persistent targets or receivers include structural descendants of:

- `self.conv_states`
- `self.conv_state`

Permitted mutation evidence may include:

- `Assign`
- `AnnAssign`
- `AugAssign`
- subscript assignment
- `zero_()`
- `copy_()`
- `index_copy_()`
- `scatter_()`
- another explicit in-place mutating call ending in `_`

A local assignment such as `conv_state = ...` does not prove persistent cache mutation.

A linked method without persistent mutation proof is unresolved.

## 13. Required semantic-family success condition

For the branch-aware semantic path, success requires all of:

1. unique `MambaMixer.slow_forward`;
2. proven cache-present branch;
3. proven prefill/decode opposite-arm structure;
4. relevant prefill local convolution-state construction;
5. prefill `cache_params.update_conv_state` call with `PROVEN` persistent-cache semantics;
6. decode `cache_params.update_conv_state` call with `PROVEN` persistent-cache semantics;
7. exactly one statically linked mutation-proven relevant `update_conv_state` method;
8. no material unresolved or ambiguous call/method linkage.

A decode call-result binding such as:

`conv_state = cache_params.update_conv_state(...)`

is valid decode evidence.

Additional local transforms of `conv_state` are permitted but must not be treated as persistent mutation proof.

On success, bind the family to canonical anchor `MambaMixer.slow_forward`.

The family must not pass merely because some conv assignment or some update call exists.

## 14. Fail-closed adversaries

The implementation must fail closed for at least:

- nested convolution assignments but no persistent update call;
- prefill update present but decode update absent;
- decode update present but prefill update absent;
- `update_conv_state` receiver is not `cache_params`;
- unrelated `update_conv_state` definition;
- zero linked methods;
- multiple plausible linked methods;
- linked method with only local `conv_state` assignment;
- string/comment mentions only;
- relevant call or mutation only inside nested function/class;
- multiple incompatible cache/prefill-decode structures;
- recursive assignments alone producing multiple candidates.

Use existing public statuses only:

- `BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`
- `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`

with note:

`convolution_cache_initialization_update`

No new public blocker status is authorized.

## 15. Required focused test delta

The implementation must add focused tests covering at least:

A. Transformers-5-like branch-aware structure passes and anchors the family at `MambaMixer.slow_forward`.

B. Prefill standalone persistent update plus decode call-result update passes.

C. Missing prefill update fails unresolved.

D. Missing decode update fails unresolved.

E. Local conv-state construction without persistent mutation fails unresolved.

F. Unique linked method that does not mutate `self.conv_state(s)` fails unresolved.

G. Multiple plausible linked `update_conv_state` methods fail ambiguous.

H. `update_conv_state` on a non-`cache_params` receiver does not prove the family.

I. Relevant call/mutation only inside nested function/class does not prove the family.

J. Missing or ambiguous opposite-arm branch structure fails closed.

K. Existing simple direct-single-assignment `MAMBA_PASS` remains valid.

L. Multiple direct conv assignments without proven semantic structure never cause arbitrary selection.

M. Artifact schema and location-object key set remain unchanged.

N. Existing recurrent/backend tests remain unchanged and passing.

Tests must assert exact blocker status and note where a blocker is expected.

## 16. Preserved unrelated behavior

The implementation must preserve:

- `classify_recurrent_semantics`;
- `classify_backend`;
- source-resolution behavior;
- runtime/version behavior;
- deterministic artifact serialization;
- output collision behavior;
- `SYMBOL_KEYS`;
- existing artifact schema;
- existing location-object key set.

Unrelated tests must not be weakened or rewritten merely to accommodate the correction.

## 17. Authorized future validation

The later implementation task may run CPU-only local validation:

`python -m pytest -q tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

and:

`git diff --check`

The known Windows pytest temp-directory ACL workaround may be used if required, without modifying repository files.

A passing test suite establishes code correctness only.

It does not authorize any corrected O0c preflight execution.

## 18. Implementation workflow

Because this correction changes provenance/validator semantics:

1. use one bounded Codex implementer;
2. implementation may modify only the two authorized files;
3. no commit or push during implementation;
4. after implementation, use one independent read-only verifier;
5. verifier must inspect semantic proof, fail-closed behavior, and focused adversarial tests;
6. after verifier PASS, use `cm ship`;
7. stage only the two exact authorized files;
8. user performs manual commit/push;
9. remote commit/file identities must be verified.

No additional report-authoring Codex pass is required.

## 19. Execution boundary after implementation

Even after implementation, tests, verification, commit, push, and remote verification:

- corrected preflight rerun is `NOT_AUTHORIZED`;
- Kaggle execution is `NOT_AUTHORIZED`;
- diagnostic rerun is `NOT_AUTHORIZED`;
- training/evaluation is `NOT_AUTHORIZED`.

A later corrected-preflight execution requires a separate execution authority that freezes the corrected implementation commit, exact file identities, runtime/source requirements, run name, command identity, and collect/import requirements.

## 20. Evidence-layer separation

| Layer | Status |
|---|---|
| Historical corrected implementation at `eebf4da0207f00d993d7a6ae213e0cc17b4b1dfc` | frozen implementation under correction |
| Convolution-cache diagnostic | executed |
| Diagnostic provenance | valid / collect PASS / import PASS |
| Root cause | established |
| Primary root cause | `VALIDATOR_CONVOLUTION_CACHE_SYMBOL_FAMILY_PATTERN_FALSE_NEGATIVE` |
| Secondary root cause | `VALIDATOR_CONVOLUTION_CACHE_DIRECT_BODY_SCOPE_FALSE_NEGATIVE` |
| New validator correction | `NOT_YET_IMPLEMENTED` |
| Corrected-preflight rerun | `NOT_AUTHORIZED` |
| Scientific conclusion | `NONE` |

## 21. Scope boundary

This authority is limited to a bounded validator/test correction.

It does not authorize:

- model execution;
- tokenizer or dataset loading;
- tensor forward;
- generation;
- training;
- evaluation;
- Kaggle;
- preflight execution;
- diagnostic execution;
- package mutation;
- commit;
- push.

## 22. Exact next authorized action after freeze

After independent verification and formal freeze of this authority, the next authorized phase is:

`CONVOLUTION_CACHE_VALIDATOR_CORRECTION_IMPLEMENTATION`

Use one bounded Codex implementer with the two-file scope and local CPU-only test validation defined above.
