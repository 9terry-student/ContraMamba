# ContraMamba O0c cache recurrent-state storage semantic binding implementation authority candidate

## 1. Status

`PASS_READY_FOR_FORMAL_FREEZE_CACHE_RECURRENT_STATE_STORAGE_SEMANTIC_BINDING_IMPLEMENTATION_AUTHORITY`

This is a bounded implementation-authority candidate.

It authorizes a correction to the O0c runtime-source provenance preflight for the formally frozen infrastructure root cause:

`VALIDATOR_CACHE_RECURRENT_STATE_STORAGE_SOURCE_AND_MUTATION_FORM_FALSE_NEGATIVE`

It does not authorize Kaggle execution, corrected-preflight execution, model execution, tokenizer/dataset loading, training, evaluation, package mutation, scientific interpretation, or scientific claims.

`SCIENTIFIC_CONCLUSION: NONE`.

## 2. Authoritative lineage

| Authority / evidence | Frozen identity |
| --- | --- |
| Cache recurrent-state storage root-cause interpretation | `8e42d0b039fe64becac3caf64293c69e810f2b07` |
| Cache recurrent-state storage diagnostic execution authority | `c54bd26ee214a2e75424b40df8059ea5f562a4f5` |
| Cache-guard corrected-preflight execution authority | `59338ca88796cf39dd31fd60a9c6a46e47570761` |
| Cache-guard role-selection corrected implementation | `0063254795aa21011364833c95d25cbce262c0bf` |
| Cache-guard role-selection implementation authority | `1e7630c70a3a7cd85caa64128d89c241ed8b0960` |
| Earlier corrected convolution-cache validator implementation | `6f394792763abb168f49c1cb1957a326d16eed2b` |

The implementation must preserve all unrelated frozen behavior inherited from this lineage.

## 3. Frozen defect

At implementation commit `0063254795aa21011364833c95d25cbce262c0bf`, the required symbol family:

`cache_recurrent_state_storage`

is incorrectly bound by scanning only `transformers.cache_utils` function direct bodies for assignment forms ending in `ssm_state` / `self.ssm_state` that load local `ssm_state`.

Validated diagnostic replay:

- source key: `cache`;
- scanned cache functions: `87`;
- frozen direct assignment candidates: `0`;
- replay classification: `ZERO`.

Validated actual Transformers `5.0.0` runtime semantics:

- source drift: none;
- `MambaMixer.slow_forward` uniquely resolved;
- cache-present recurrent-state read count: `1`;
- sequential recurrent update count: `1`;
- persistent recurrent-state mutation count: `1`;
- persistent mutation:
  `cache_params.ssm_states[self.layer_idx].copy_(ssm_state)`;
- persistent mutation source: Mamba source;
- cache-utils relevant recurrent-storage behavior count: `0`;
- Mamba role complete: `true`.

Therefore both source-role selection and mutation-form recognition must change together.

## 4. Exact implementation scope

The implementation may modify exactly these two existing files:

1. `scripts/preflight_longterm_o0c_runtime_source_provenance.py`
2. `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

No other repository file may be modified, created, deleted, renamed, formatted, or regenerated during implementation.

No report file is part of the implementation delta.

## 5. Required production-code correction

### 5.1 Replace the frozen cache-only assignment family

The existing `cache_recurrent_state_storage` family construction must no longer use the cache-utils-only direct-assignment candidate list.

The corrected family must bind to:

- module: the Mamba source module;
- source file key: `mamba`;
- source SHA256: the resolved Mamba source identity;
- qualname: `MambaMixer.slow_forward`;
- canonical location node: the uniquely proven persistent recurrent-state cache mutation.

For the validated Transformers `5.0.0` source this canonical location is the call structurally equivalent to:

`cache_params.ssm_states[self.layer_idx].copy_(ssm_state)`.

The implementation must not hard-code source line numbers.

### 5.2 Semantic proof requirements

A qualifying `cache_recurrent_state_storage` proof must establish all of the following inside the unique `MambaMixer.slow_forward` lexical scope.

#### A. Unique slow-forward function

`MambaMixer.slow_forward` must resolve uniquely using the existing fail-closed function-resolution behavior.

Missing resolution must remain blocked.

Ambiguous resolution must remain blocked.

#### B. Cache-present recurrent-state read

There must be a cache-present guarded read that initializes or assigns local:

`ssm_state`

from the persistent cache path structurally rooted at:

`cache_params.ssm_states[...]`.

The proof must recognize the AST structure, not source text or line numbers.

The read must occur in the same `MambaMixer.slow_forward` lexical scope and must not be satisfied by nested function/class definitions.

#### C. Sequential recurrent update

The existing recurrent proof must continue to establish a unique sequential recurrent update of local `ssm_state`.

The implementation should reuse existing proven recurrent-state semantics where possible rather than duplicate a weaker independent loop heuristic.

The storage family must not pass when the required recurrent update proof is unresolved or ambiguous.

#### D. Cache-present persistent write

There must be a cache-present guarded persistent mutation back to a target structurally rooted at:

`cache_params.ssm_states[...]`.

For the frozen runtime, the required recognized mutation form is a mutating method call:

`...copy_(ssm_state)`.

The implementation may recognize a narrowly equivalent AST variation only if it preserves the same semantic guarantees.

It must not accept a generic method call solely because the method name contains `copy`, `update`, `set`, `state`, or another token.

#### E. Final local state consumption

The persistent write must consume local `ssm_state`.

For the canonical `copy_` form, the argument supplying the written value must be the local recurrent state, not an unrelated value.

Loose presence of an `ssm_state` token elsewhere in the statement is insufficient.

#### F. Ordering

The persistent write must occur after the unique sequential recurrent update/loop in source order.

A write before the recurrent update must not satisfy the storage proof.

#### G. Lexical-scope exclusion

Nested functions, async functions, classes, and other nested lexical definitions must not provide qualifying read or persistent-write evidence for the enclosing `MambaMixer.slow_forward`.

Use or extend the existing same-lexical-scope traversal semantics rather than unrestricted `ast.walk` where unrestricted traversal could admit nested-definition false positives.

## 6. Fail-closed cardinality

The corrected semantic binder must preserve strict cardinality.

For qualifying complete recurrent-storage proofs:

- zero => `BLOCKED_REQUIRED_SYMBOL_UNRESOLVED` with note/family `cache_recurrent_state_storage`;
- more than one => `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` with note/family `cache_recurrent_state_storage`;
- exactly one => bind the canonical persistent-mutation node.

Do not choose the first candidate.

Do not choose the earliest or latest line as a disambiguation heuristic.

Do not deduplicate semantically distinct complete candidates merely because they share text, call names, or target roots.

## 7. Canonical location representation

On the validated Transformers `5.0.0` source, the corrected symbol-location record must identify:

- module: `transformers.models.mamba.modeling_mamba`;
- qualname: `MambaMixer.slow_forward`;
- source file key: `mamba`;
- source SHA256: the resolved Mamba source SHA256;
- start/end span: the persistent `copy_` mutation node itself.

The production code must derive this from AST/source identity, not hard-coded values.

The existing symbol-location JSON schema must remain unchanged.

## 8. Required tests

The targeted test file must add or revise tests that prove at least the following.

### 8.1 Valid Mamba persistence is accepted

A source fixture structurally matching:

- cache-present read from `cache_params.ssm_states[...]`;
- valid sequential recurrent update;
- guarded `cache_params.ssm_states[...].copy_(ssm_state)` after recurrence;

must bind `cache_recurrent_state_storage` uniquely.

Assert that its location uses:

- Mamba module;
- `source_file_key == "mamba"`;
- Mamba source SHA;
- `qualname == "MambaMixer.slow_forward"`;
- span of the persistent mutation.

### 8.2 Cache-utils assignment is no longer the required role

A cache-utils direct-assignment-shaped decoy must not by itself satisfy the family when the Mamba persistence proof is missing.

The old source-role assumption must not remain as a fallback.

### 8.3 Missing persistent write fails closed

Remove the persistent cache write while leaving other recurrent semantics intact.

Expected blocker:

`BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`

for:

`cache_recurrent_state_storage`.

### 8.4 Wrong mutation receiver fails closed

A `.copy_(ssm_state)` call on an unrelated object/path must not qualify.

### 8.5 Wrong mutation value fails closed

A cache-target `.copy_(other_state)` or equivalent unrelated value must not qualify merely because `ssm_state` appears elsewhere.

### 8.6 Unguarded persistent write fails closed

A write to `cache_params.ssm_states[...]` outside a proven cache-present guard must not qualify.

### 8.7 Pre-recurrence write fails closed

A cache write that occurs before the sequential recurrent update must not qualify as final persistent storage.

### 8.8 Nested lexical decoy fails closed

A qualifying-looking `copy_(ssm_state)` inside a nested function/class must not satisfy the enclosing slow-forward proof.

### 8.9 Multiple complete writes are ambiguous

Two semantically complete qualifying persistent writes must produce:

`BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`

for:

`cache_recurrent_state_storage`.

### 8.10 Existing convolution-cache correction remains intact

Existing tests for:

`convolution_cache_initialization_update`

including cache-guard role selection, prefill/decode semantic proof, linked update method, persistent convolution mutation, unresolved, and ambiguous cases must remain passing without semantic weakening.

### 8.11 Existing recurrent/backend/source-resolution behavior remains intact

Existing tests for:

- recurrent initialization;
- recurrent update/readout;
- backend selection;
- source root/shadowing;
- runtime version checks;
- deterministic artifact serialization;
- output collision;
- forbidden execution/package mutation boundaries;

must remain passing.

## 9. Forbidden implementation shortcuts

The implementation must not use:

- hard-coded line numbers;
- raw source substring matching as the proof mechanism;
- `first` / `last` / earliest-line selection;
- unrestricted generic `ast.walk` across nested lexical definitions for decisive storage proof;
- cache-utils fallback merely to preserve old behavior;
- acceptance of arbitrary methods named `copy`, `update`, or `set`;
- token-only matching on `ssm_state`;
- broad schema changes;
- runtime execution of model code;
- dynamic monkeypatching of Transformers;
- package modification.

## 10. No unrelated semantic change

Do not change the behavior of:

- runtime version resolution;
- distribution/import-root derivation;
- source shadowing checks;
- raw-byte source identity;
- recurrent-state initialization classification;
- recurrent-state update/readout classification;
- backend selection/classification;
- convolution-cache semantic binding;
- O0c indexing classification;
- schema version;
- symbol-key ordering;
- output publication/collision semantics;
- forbidden-operation boundaries.

If implementation appears to require any such change, STOP and report the blocker rather than widening scope.

## 11. Validation

Implementation validation must include all of the following.

### 11.1 Exact targeted suite

Run:

`python -m pytest -q tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

If the known Windows pytest temporary-directory ACL issue recurs, use a fresh explicitly selected external `--basetemp` path. Do not modify production/test semantics to work around filesystem ACL behavior.

### 11.2 Diff check

Run:

`git diff --check`

If PowerShell/core.autocrlf warning behavior interferes, use only process-local Git config overrides for `core.autocrlf=false` and `core.safecrlf=false`.

Do not change global Git configuration.

### 11.3 Scope verification

Verify exactly the two authorized files are modified.

### 11.4 Static semantic review

Explicitly report:

- how cache-present read is proven;
- how recurrent update proof is reused/linked;
- how persistent cache target is recognized;
- how `copy_(ssm_state)` value flow is proven;
- how post-recurrence ordering is proven;
- how nested lexical scopes are excluded;
- how zero/multiple proofs fail closed;
- resulting canonical symbol-location source key/module/qualname/node.

## 12. Independent verification requirement

Because this change alters fail-closed provenance-validator semantics, one independent verifier is required after the bounded implementer finishes.

The verifier must inspect the actual diff and independently verify:

1. source-role correction from cache-utils to Mamba is exact;
2. mutation-form recognition is semantic and narrow;
3. persistent-write ordering is enforced;
4. nested-scope decoys cannot pass;
5. ambiguity remains fail-closed;
6. the canonical location points to the persistent mutation;
7. prior convolution-cache semantics are unchanged;
8. exact targeted tests and diff check pass;
9. no training/evaluation/model execution occurred;
10. only the two authorized files changed.

The verifier must not modify files.

## 13. Codex execution boundary

After this authority is formally frozen and remotely verified, use:

- one bounded implementer;
- one independent read-only verifier.

The implementer task must state:

`Training/Evaluation allowed: NO`

and:

`Commit/Push: NO`.

The verifier task must also state:

`Training/Evaluation allowed: NO`

and:

`Commit/Push: NO`.

Neither agent may run Kaggle.

## 14. Commit and push boundary

This authority does not authorize automatic commit or push.

After implementation and independent verification PASS:

1. controller reviews exact repo state;
2. run `cm ship`;
3. user explicitly stages only the two authorized files;
4. user explicitly commits;
5. user explicitly pushes.

Never use `git add .`.

## 15. Execution boundary after implementation

A successful implementation/test freeze does not authorize a corrected preflight run.

Any future corrected-preflight execution requires a separate execution-authority artifact frozen after implementation.

No prior consumed run name may be reused.

## 16. Stop conditions

STOP without widening scope if any of the following occurs:

- unique `MambaMixer.slow_forward` cannot be preserved;
- validated runtime shape cannot be represented without unrelated semantic changes;
- persistent mutation cannot be tied to local final `ssm_state`;
- multiple complete storage proofs cannot be distinguished fail-closed;
- targeted tests expose a conflict with frozen recurrent/convolution/backend semantics;
- exact two-file scope cannot be maintained;
- runtime/model execution would be required to establish correctness;
- any authority conflict appears.

## 17. Required implementation report

The bounded implementer must report:

- overall verdict;
- exact files changed;
- concise production semantic delta;
- exact tests added/changed;
- targeted pytest command and result;
- `git diff --check` result;
- exact file scope/status;
- confirmation no training/evaluation/model execution;
- confirmation no commit/push;
- any residual risk.

The independent verifier must report:

- `PASS_SAFE_TO_FREEZE_CACHE_RECURRENT_STATE_STORAGE_SEMANTIC_BINDING_IMPLEMENTATION`
  or a precise blocking verdict;
- independent semantic findings;
- validation results;
- exact changed-file scope;
- confirmation no files were modified by verification.

## 18. Evidence-layer separation

| Layer | State after freezing this authority |
| --- | --- |
| Root-cause interpretation | FROZEN |
| Implementation authority | FROZEN |
| Implementation code | NOT_YET_MODIFIED |
| Implementation correctness | NOT_YET_ESTABLISHED |
| Independent verification | NOT_YET_RUN |
| Corrected preflight execution authority | NOT_YET_AUTHORIZED |
| Corrected preflight execution | NOT_YET_AUTHORIZED |
| Scientific conclusion | `NONE` |

## 19. Freeze boundary

Freezing this candidate authorizes only the bounded two-file implementation and static/test verification described above.

It does not authorize:

- Kaggle;
- preflight execution against Kaggle runtime;
- collection/import;
- model/tokenizer/dataset execution;
- training/evaluation;
- scientific interpretation.

## 20. Exact next action after freeze

After formal freeze and remote verification, the controller should issue the bounded Codex implementer task against the frozen authority commit.

Do not begin implementation before that freeze.
