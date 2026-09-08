# ContraMamba O0c convolution-cache cache-guard role-selection correction implementation authority candidate

## 1. Status and purpose

`PASS_READY_FOR_INDEPENDENT_CACHE_GUARD_ROLE_SELECTION_IMPLEMENTATION_AUTHORITY_VERIFICATION`

This is a report-only implementation-authority candidate under the current controller instruction.

It does not itself modify code or tests. It authorizes a future bounded implementation only after this report is independently verified, formally frozen, committed/pushed, and remote-verified.

The sole implementation purpose is to correct the false `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` caused by over-broad cache-present branch candidate formation in the convolution-cache semantic-family validator.

No Kaggle execution, corrected preflight execution, diagnostic execution, model execution, training, evaluation, package mutation, commit, or push is authorized by this authoring step.

`SCIENTIFIC_CONCLUSION: NONE`.

## 2. Frozen authority and evidence chain

| Authority / evidence | Frozen identity |
| --- | --- |
| Cache-guard ambiguity root-cause interpretation | `6fe942ae7d872314d4fd4da2c68ca221e7f45b0e` |
| Linkage-ambiguity diagnostic execution authority | `b20857b78af86d814b48db9c9846a3c8bb049d61` |
| Corrected-preflight execution authority | `755987eea6230bb0ad6f73400e46ae680434e6f6` |
| Current corrected validator implementation | `6f394792763abb168f49c1cb1957a326d16eed2b` |
| Prior validator correction implementation authority | `3f03e5dec1faf2edb443dff10387f02350b02b6f` |
| Prior convolution-cache root-cause interpretation | `c4ec40fc8e4df82243c2facb810146513ec97b55` |

The latest frozen authority establishes:

Primary root cause:

`VALIDATOR_CONVOLUTION_CACHE_CACHE_PRESENT_BRANCH_PREDICATE_OVERAPPROXIMATION`

Secondary:

`VALIDATOR_CONVOLUTION_CACHE_BRANCH_ROLE_DISAMBIGUATION_MISSING`

The exact runtime source is unchanged from the historically validated Transformers-5.0.0 source.

## 3. Established defect

At current implementation commit `6f394792763abb168f49c1cb1957a326d16eed2b`, `_convolution_cache_location`:

1. scans the same lexical scope of `MambaMixer.slow_forward`;
2. selects every `ast.If` whose test satisfies `_is_cache_present_test`;
3. requires the resulting cache-present branch count to be exactly one before inspecting convolution-cache role evidence.

On the exact validated source there are two such predicates:

- the actual convolution-cache control branch containing the prefill/decode split and `update_conv_state` behavior;
- an unrelated later recurrent-state persistence guard that copies `ssm_state` into `cache_params.ssm_states`.

The frozen helper therefore sees two broad cache-present branches and raises `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` before role evidence can disambiguate them.

This is a static validator semantic-selection defect, not source ambiguity and not source drift.

## 4. Frozen imported evidence

The consumed corrected-preflight run `longterm-o0c-runtime-source-provenance-preflight-6f39479-v1` is provenance-valid and imported. It observed execution commit `6f394792763abb168f49c1cb1957a326d16eed2b`, exit code `2`, `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`, blocker `convolution_cache_initialization_update`.

The consumed linkage diagnostic `longterm-o0c-convolution-cache-linkage-ambiguity-diagnostic-6f39479-v1` is provenance-valid and imported. Its artifact SHA256 is `5304643c695ce35251db62964b6856691686af232a7ebed7c9762eedecae9ea4`.

Imported structural facts include:

- `MambaMixer.slow_forward`: `1`;
- broad cache-present branches: `2`;
- prefill/decode splits: `1`;
- prefill conv-state assignments: `1`;
- prefill update calls: `1`;
- decode update calls: `1`;
- raw linked update-method candidates: `1`;
- semantic-deduplicated linked candidates: `1`;
- nested lexical update definitions: `0`;
- frozen replay: `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`.

These two run identities are permanently consumed. No rerun or same-name reuse is authorized.

## 5. Exact implementation scope

A future implementation under this authority may modify exactly these two files:

- `scripts/preflight_longterm_o0c_runtime_source_provenance.py`
- `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

No other repository file may be modified.

New helper functions may be added only inside the production file when directly necessary to express the bounded static branch-role proof.

No generated artifact, report, fixture file, package file, workflow file, `cm` tooling file, or configuration file may be modified.

## 6. Implementation base and Git boundary

Implementation may begin only after this authority candidate is independently verified, formally frozen, committed/pushed, and remote-verified.

The future implementation worktree must start from that exact formal-freeze commit and be clean.

No staging, commit, push, reset, clean, rebase, merge, or destructive Git action is authorized for the implementer.

`Commit/Push: NO`.

The user retains explicit staging/commit/push control after implementation verification.

## 7. Required correction semantics

The broad structural predicate `_is_cache_present_test` is not, by itself, sufficient to identify the convolution-cache control branch.

The correction must move uniqueness from the broad cache-presence predicate level to the complete convolution-cache role-proof level.

For the semantic-family path, the implementation must:

1. enumerate same-lexical-scope cache-present branches using the existing structural cache-presence concept;
2. inspect each such branch for prefill/decode structures within that branch;
3. identify a complete convolution-cache role proof only when one prefill/decode split supplies all currently required semantic evidence:
   - prefill conv-state construction;
   - at least one prefill `cache_params.update_conv_state` call;
   - at least one decode `cache_params.update_conv_state` call;
4. treat the pair of cache-present branch and its complete split proof as a semantic proof candidate;
5. require exactly one complete semantic proof candidate across all cache-present branches;
6. return unresolved when there are zero complete proof candidates;
7. return ambiguous when there are more than one complete proof candidates;
8. after selecting exactly one complete proof, preserve the existing unique linked `update_conv_state` method requirement and persistent convolution-cache mutation proof;
9. preserve the canonical `MambaMixer.slow_forward` location anchor on semantic-family success.

Equivalent code structure is allowed, but these semantics are mandatory.

## 8. Critical non-solutions

The implementation must not choose the first/earliest/longest cache-present branch, hard-code source line numbers, special-case the known later recurrent guard, loosen fail-closed ambiguity behavior, globally ignore later cache-present guards, replace static proof with runtime/model execution, or silently pick among multiple complete convolution-cache proofs.

The defect must be fixed through branch-role evidence, not source-position heuristics.

## 9. Preserve current semantic-family requirements

Unless directly required by the branch-role correction, preserve current behavior for:

- `_is_cache_present_test`;
- `_is_prefill_decode_test`;
- same-lexical-scope traversal;
- prefill conv-state assignment detection;
- `cache_params.update_conv_state` receiver matching;
- annotation-owner extraction;
- module-scope method discovery;
- nested lexical exclusion;
- linked update-method uniqueness;
- persistent `self.conv_state` / `self.conv_states` mutation proof;
- legacy direct-single-assignment compatibility path;
- canonical location output.

This authority does not permit a redesign of unrelated convolution-cache semantics.

## 10. Legacy direct-assignment path

The existing legacy narrow path must remain behaviorally unchanged.

When direct and recursive relevant convolution-state assignments are both exactly one and there are no cache update calls, the legacy direct assignment remains a valid location anchor.

The new branch-role logic applies only when the semantic-family path is required.

## 11. Fail-closed requirements

After correction:

- zero complete role proofs -> `BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`;
- more than one complete role proof -> `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`;
- missing prefill conv-state construction -> unresolved unless another complete proof exists;
- missing prefill update call -> unresolved unless another complete proof exists;
- missing decode update call -> unresolved unless another complete proof exists;
- linked update method count zero -> unresolved;
- linked update method count greater than one -> ambiguous;
- persistent convolution-cache mutation unproven -> unresolved;
- source parsing/resolution failures retain existing blockers.

Do not downgrade genuine ambiguity into PASS.

## 12. Exact regression obligations

At minimum add or adapt tests for:

A. One complete convolution-cache branch plus one later unrelated recurrent-state cache-present guard -> PASS with canonical `MambaMixer.slow_forward`.

B. Unrelated cache-present guard before the relevant convolution-cache branch -> PASS.

C. Two cache-present branches each containing a complete convolution-cache proof -> `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`.

D. One or more unrelated cache-present guards with no complete convolution-cache proof -> unresolved.

E. Valid-looking prefill/decode split outside every cache-present branch -> unresolved.

F. Cache-present branch missing prefill conv-state construction -> incomplete/unresolved.

G. Cache-present branch missing prefill update -> incomplete/unresolved.

H. Cache-present branch missing decode update -> incomplete/unresolved.

I. One cache-present branch containing more than one independently complete convolution-cache split proof -> ambiguous.

J. Recurrent `ssm_states` persistence-only guard -> ignored for convolution-cache role selection.

K. Existing zero/multiple linked-method and persistent-mutation tests remain passing.

L. Existing nested lexical adversaries remain passing.

M. Existing legacy direct-single-assignment fixtures remain passing.

N. Artifact/location schema, keys, status taxonomy, and canonical anchor remain unchanged.

O. Existing recurrent-state and backend classification tests remain passing.

P. No model/tokenizer/dataset load, forward execution, package installation, or package mutation.

## 13. Public interface and schema freeze

Do not change:

- CLI argument names or requirements;
- `SCHEMA_VERSION`;
- `SYMBOL_KEYS`;
- source key names;
- artifact top-level keys;
- symbol-location key schema;
- existing status strings;
- serialization order/determinism behavior;
- output collision behavior;
- public process exit semantics;
- runtime version checking semantics;
- source resolution/provenance semantics;
- recurrent-state classification semantics;
- backend classification semantics.

No new public blocker string is authorized.

Use existing unresolved/ambiguous statuses with the existing family note.

## 14. No execution boundary

Implementation validation is local/CPU/static only.

No Kaggle, corrected preflight, diagnostic execution, `cm run save`, `cm run`, `cm collect`, `cm import`, training, evaluation, or model execution is authorized.

Passing tests establish code correctness only.

A separate corrected-preflight execution authority is mandatory after a later implementation freeze.

## 15. Required local validation

After implementation, run at minimum:

`python -m pytest tests/test_preflight_longterm_o0c_runtime_source_provenance.py -q`

and:

`git diff --check`

The targeted suite must PASS with all newly added branch-role regressions.

The established local pytest temp-root workaround may be used if needed without modifying repository files.

## 16. Independent verification requirement

Because this correction changes provenance-validator semantic selection and fail-closed ambiguity behavior, one independent verifier is mandatory after the bounded implementer finishes.

The verifier must independently check exact two-file scope, branch-role proof rather than first/position heuristics, zero/one/multiple complete-proof semantics, current exact-shape two-cache-guard coverage, reversed irrelevant-guard ordering, genuine multiple-role ambiguity, linked-method semantics, nested lexical boundary, legacy path, schema/public interface, recurrent/backend behavior, targeted pytest, `git diff --check`, and no execution/training/evaluation.

Any material defect requires bounded repair and re-verification before freeze.

## 17. Implementer stop conditions

STOP and report instead of widening scope if:

- a required fix appears to need a third repository file;
- a schema/public status/CLI change appears necessary;
- runtime/model execution appears necessary to choose the branch;
- frozen source/evidence contradicts this authority;
- tests reveal an unrelated defect requiring broader redesign;
- exact authority/base state is mismatched;
- the worktree is dirty before implementation for an unexplained reason.

No opportunistic cleanup or refactor is authorized.

## 18. Codex implementer task boundary

Role:
Implementation engineer.

Authority:
This formally frozen implementation-authority report, after its freeze commit is known and remote-verified.

Phase:
Bounded implementation only.

Goal:
Correct convolution-cache cache-present branch-role selection so unrelated same-predicate recurrent-state guards do not cause false ambiguity, while preserving fail-closed behavior.

Scope:
Only `scripts/preflight_longterm_o0c_runtime_source_provenance.py` and `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`.

Expected delta:
Minimal static-AST production correction plus focused adversarial regression tests.

Do not change:
CLI, schema, status taxonomy, source resolution, runtime checks, recurrent/backend semantics, legacy direct path, linked-method semantics, public serialization/provenance behavior.

Validation:
Targeted pytest file and `git diff --check`.

Training/Evaluation allowed:
NO.

Commit/Push:
NO.

Stop conditions:
Those in section 17.

Required report:
Verdict; starting HEAD; exact changed files; concise semantic delta; exact tests added/changed; full targeted pytest summary; `git diff --check`; raw byte identities of both modified files; Git status; explicit confirmation that nothing was staged/committed/pushed and no training/evaluation/Kaggle execution occurred.

## 19. Independent verifier task boundary

Role:
Independent provenance-validator verifier.

Authority:
This formally frozen authority plus the implementer's resulting unstaged two-file delta.

Phase:
Independent read-only verification.

Goal:
Determine whether the implementation exactly fixes cache-guard role selection without weakening ambiguity or widening provenance semantics.

Scope:
Read the two modified files and relevant frozen authority/evidence only. Do not modify files.

Validation:
Re-run targeted pytest and `git diff --check`; inspect adversarial branch-role cases and preserved boundaries.

Training/Evaluation allowed:
NO.

Commit/Push:
NO.

Stop conditions:
Any material semantic weakness, scope widening, unexpected repository state, or unverified authority mismatch.

Required report:
PASS/FAIL verdict; exact HEAD; exact modified-file identities; semantic review; regression review; validation outputs; Git state; scientific conclusion `NONE`.

## 20. Evidence-layer separation

| Layer | State at authority authoring |
| --- | --- |
| Prior corrected implementation correctness | PASS / frozen at `6f394792...` |
| Corrected-preflight execution | completed |
| Corrected-preflight provenance | VALID / IMPORT PASS |
| Corrected-preflight result | false `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` established |
| Cache-guard root cause | frozen at `6fe942ae...` |
| New branch-role correction implementation | `NOT_YET_IMPLEMENTED` |
| New implementation verification | `NOT_YET_PERFORMED` |
| New corrected-preflight execution | `NOT_AUTHORIZED` |
| Scientific conclusion | `NONE` |

## 21. Authoring boundary and next action

During this authority candidate authoring/freeze phase there is no production/test modification, Codex implementation, Kaggle, run registration/execution, training, or evaluation.

After this candidate is materialized, independently verified, formally frozen, committed/pushed, and remote-verified, the next authorized action is exactly one bounded implementer using section 18, followed by exactly one independent verifier using section 19.

No execution follows automatically.
