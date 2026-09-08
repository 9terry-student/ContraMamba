# ContraMamba O0c convolution-cache cache-guard ambiguity root-cause interpretation candidate

## 1. Status

`PASS_READY_FOR_FORMAL_FREEZE_CONVOLUTION_CACHE_CACHE_GUARD_ROOT_CAUSE_INTERPRETATION`

This report-only candidate interprets the fully imported, provenance-valid O0c convolution-cache linkage-ambiguity diagnostic and exact frozen source/validator behavior.

It is not an implementation authority, execution authority, code change, test change, rerun authorization, Kaggle authorization, training/evaluation authorization, or scientific conclusion.

`SCIENTIFIC_CONCLUSION: NONE`.

## 2. Frozen authority and evidence chain

| Authority / evidence | Frozen identity |
| --- | --- |
| Linkage-ambiguity diagnostic execution authority | `b20857b78af86d814b48db9c9846a3c8bb049d61` |
| Corrected-preflight execution authority | `755987eea6230bb0ad6f73400e46ae680434e6f6` |
| Corrected convolution-cache validator implementation | `6f394792763abb168f49c1cb1957a326d16eed2b` |
| Validator correction implementation authority | `3f03e5dec1faf2edb443dff10387f02350b02b6f` |
| Prior root-cause interpretation | `c4ec40fc8e4df82243c2facb810146513ec97b55` |

The prior root-cause interpretation established that the original direct-body / assignment-only representation was too narrow. The corrected implementation at `6f394792...` fixed that earlier defect but exposed a distinct later ambiguity described here.

## 3. Consumed corrected-preflight result

The corrected preflight run is permanently consumed:

| Field | Value |
| --- | --- |
| Run | `longterm-o0c-runtime-source-provenance-preflight-6f39479-v1` |
| Execution commit | `6f394792763abb168f49c1cb1957a326d16eed2b` |
| Command SHA256 | `bd829449bc98054d03b39d695a23c163c1317d21ff3a339de7b54f59c90c3ad1` |
| Exit code | `2` |
| Status | `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` |
| Blocker | `convolution_cache_initialization_update` |
| Run log SHA256 | `0861b3db816f210e75c16d00f0d364ce27c6e2aab2e5d2cf6551c5b3610a065c` |
| Run meta SHA256 | `bac04aef0b931d0dc37cb37fdc5f770f5e1d70f14434976777cbec8cd2cd1c19` |
| Handoff ZIP SHA256 | `913859035a8632fd5f439aef286f51eee828a6dfdaa8d27e7564224502880fc9` |
| Import | `PASS` |

This establishes the observed fail-closed validator result only. It does not itself establish the root cause.

## 4. Consumed linkage-ambiguity diagnostic

The diagnostic run is permanently consumed:

| Field | Value |
| --- | --- |
| Run | `longterm-o0c-convolution-cache-linkage-ambiguity-diagnostic-6f39479-v1` |
| Execution commit | `6f394792763abb168f49c1cb1957a326d16eed2b` |
| Command SHA256 | `78606a8e0fef31fa2632cf4e0a4de2206b23857875b8109bd7e7f5c92b895031` |
| Exit code | `0` |
| Run log SHA256 | `4135c68b7342e74eff1c7a54a1ee386a2ba90dabe3141df7965deb57583018ef` |
| Run meta SHA256 | `de3fea2bccbef8c7f88d99e17a6e1893ddc6dad7350488716428f224bbb256f1` |
| Handoff ZIP SHA256 | `a2676f68fd31f04c7d9f04c5b1d47f0dff8c5947c7a1440af6d7f96344b2b3ec` |
| Imported artifact | `outputs/longterm-o0c-convolution-cache-linkage-ambiguity-diagnostic-6f39479-v1.json` |
| Artifact SHA256 | `5304643c695ce35251db62964b6856691686af232a7ebed7c9762eedecae9ea4` |
| Import | `PASS` |
| Import audit | `C:\Users\Home1\.contramamba\imports\longterm-o0c-convolution-cache-linkage-ambiguity-diagnostic-6f39479-v1_6f394792763a_20260908_145235` |

The diagnostic initially emitted `DIAGNOSTIC_INCONCLUSIVE` because its classification taxonomy focused on linked-method ambiguity. The imported artifact nevertheless contains the structural counts needed to identify the actual earlier ambiguity gate.

## 5. Runtime/source identity

The imported diagnostic establishes the runtime:

- Python `3.12.13`
- NumPy `2.0.2`
- torch `2.10.0+cpu`
- Transformers `5.0.0`
- CUDA available `false`
- CUDA device count `0`

The exact Mamba source identity is unchanged from the historical validated source:

- path `/usr/local/lib/python3.12/dist-packages/transformers/models/mamba/modeling_mamba.py`
- SHA256 `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83`
- bytes `39500`
- LF `860`
- CR `0`
- final LF `true`

`RUNTIME_SOURCE_IDENTITY_DRIFT` is therefore rejected.

## 6. Imported structural evidence

The imported diagnostic artifact establishes:

| Structural fact | Count/result |
| --- | --- |
| `MambaMixer.slow_forward` | `1` |
| cache-present branches recognized by frozen helper | `2` |
| prefill/decode splits | `1` |
| prefill conv-state assignments | `1` |
| prefill `cache_params.update_conv_state` calls | `1` |
| decode `cache_params.update_conv_state` calls | `1` |
| raw linked update-method candidates | `1` |
| semantic-deduplicated linked candidates | `1` |
| nested lexical update definitions | `0` |
| frozen replay | `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` |

Therefore the observed ambiguity is not caused by:

- source drift;
- multiple `MambaMixer.slow_forward` definitions;
- multiple prefill/decode splits;
- duplicate linked-method representation;
- multiple semantically distinct linked update methods;
- nested lexical method evidence leakage.

## 7. Exact two cache-present branches

The exact validated Transformers-5.0.0 Mamba source contains two same-lexical-scope `if cache_params is not None` branches in `MambaMixer.slow_forward`.

### Branch A: convolution-cache control branch

Span:

`353–379`

This branch owns the convolution-cache behavior. It contains:

- cached `ssm_state` retrieval;
- the prefill/decode split at `359–373`;
- prefill `conv_state` construction at `360–363`;
- prefill `cache_params.update_conv_state` at line `365`;
- decode `cache_params.update_conv_state` at line `368`;
- the uncached convolution alternative in the branch `else`.

This is the intended cache-present branch for `convolution_cache_initialization_update`.

### Branch B: recurrent-state persistence guard

Span:

`416–417`

This branch is:

`if cache_params is not None:`

followed by a write of the recurrent SSM state:

`cache_params.ssm_states[self.layer_idx].copy_(ssm_state)`

It contains no convolution-cache prefill/decode split, no convolution-state construction, and no `update_conv_state` call.

Its semantic role is recurrent-state persistence, not convolution-cache initialization/update.

## 8. Frozen validator behavior

At corrected implementation commit `6f394792...`, `_convolution_cache_location` finds cache branches by scanning all same-lexical-scope nodes below `MambaMixer.slow_forward.body` and selecting every `ast.If` whose test satisfies `_is_cache_present_test`.

`_is_cache_present_test` recognizes the structural predicate `cache_params is not None` without constraining the semantic role of the guarded branch.

The helper then applies:

- unresolved if zero cache branches;
- ambiguous if cache-branch count is not exactly one.

Because both Branch A and Branch B use the same structural cache-presence predicate, the candidate count is `2`.

The function therefore raises:

`BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`

with note:

`convolution_cache_initialization_update`

before it can uniquely continue through the valid convolution-cache branch.

## 9. Root cause classification

Primary:

`VALIDATOR_CONVOLUTION_CACHE_CACHE_PRESENT_BRANCH_PREDICATE_OVERAPPROXIMATION`

Definition:

The corrected validator treats every same-lexical-scope `cache_params is not None` branch as a candidate convolution-cache control branch, even when the branch is semantically unrelated recurrent-state persistence. On the exact validated source this produces two cache-present candidates and a false `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`.

Secondary:

`VALIDATOR_CONVOLUTION_CACHE_BRANCH_ROLE_DISAMBIGUATION_MISSING`

Definition:

The validator lacks a semantic role constraint requiring the selected cache-present branch to contain or dominate the already-required convolution-cache evidence: the unique prefill/decode split, prefill conv-state construction, and prefill/decode `cache_params.update_conv_state` calls.

These are infrastructure-validator classifications only.

## 10. Why the ambiguity is false rather than genuine

The ambiguity is not genuine source ambiguity because the two cache-present branches have distinguishable static roles:

- Branch A contains the complete required convolution-cache evidence;
- Branch B contains only recurrent `ssm_states` persistence.

The exact source therefore provides sufficient static evidence to select Branch A without using runtime/model execution.

The failure is in validator candidate formation, not in the source semantics.

## 11. Required correction semantics for a future implementation authority

This interpretation does not authorize implementation. A later implementation authority should preserve fail-closed behavior while requiring branch-role evidence.

At minimum, the future correction should ensure that a candidate cache-present branch for `convolution_cache_initialization_update` is considered relevant only when the branch structurally contains or dominates the required convolution-cache semantic proof, including:

- exactly one valid prefill/decode split;
- prefill conv-state construction;
- at least one prefill `cache_params.update_conv_state` call;
- at least one decode `cache_params.update_conv_state` call;
- the existing uniquely linked persistent-mutation-proven update method.

The irrelevant recurrent-state persistence guard at `416–417` must not become a convolution-cache branch candidate merely because its test is also `cache_params is not None`.

No source line numbers may be hard-coded in production logic.

## 12. Fail-closed requirements

A future correction must still fail closed when:

- zero role-valid cache-present branches exist;
- more than one role-valid cache-present branch exists;
- prefill/decode structure is missing or ambiguous;
- required update calls are missing;
- linked update method is missing or ambiguous;
- persistent convolution-cache mutation is unproven;
- source/runtime identity or parsing fails.

The correction must not simply choose the first cache-present branch.

## 13. Regression obligations

A later implementation authority should require tests covering at least:

1. exact-shape source with one relevant convolution-cache cache-present branch plus one unrelated recurrent-state cache-present guard: PASS;
2. two role-valid convolution-cache branches: `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`;
3. only unrelated cache-present guards: unresolved;
4. valid prefill/decode split outside the selected cache-present branch: unresolved;
5. unrelated branch containing recurrent `ssm_states` mutation only: ignored for convolution-cache branch selection;
6. existing linked-method, nested-lexical, legacy direct-assignment, recurrent, backend, schema, and no-model tests remain unchanged/passing.

## 14. Evidence-layer separation

| Layer | State |
| --- | --- |
| Corrected validator code correctness at `6f394792...` | prior targeted suite PASS |
| Corrected-preflight execution | completed |
| Corrected-preflight provenance | VALID / IMPORT PASS |
| Corrected-preflight observed result | `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS` |
| Linkage diagnostic execution | completed / exit `0` |
| Linkage diagnostic provenance | VALID / IMPORT PASS |
| Root cause | established by imported structural evidence + exact frozen source inspection |
| Scientific conclusion | `NONE` |

The prior code-correctness PASS did not guarantee runtime-source acceptance; this new defect is a static validator semantic-selection defect exposed by the exact runtime source.

## 15. No execution or repair authority

This interpretation authorizes no:

- code modification;
- test modification;
- Kaggle execution;
- preflight rerun;
- diagnostic rerun;
- model execution;
- training;
- evaluation;
- package mutation.

The two consumed run names remain permanently consumed.

## 16. Exact next authorized phase after freeze

After this interpretation is independently verified and formally frozen, the next phase is a bounded implementation-authority specification for the cache-present branch-role selection correction.

That implementation authority should permit modification only to:

- `scripts/preflight_longterm_o0c_runtime_source_provenance.py`
- `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

No corrected preflight execution follows implementation automatically; a separate execution authority remains required.
