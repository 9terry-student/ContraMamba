# ContraMamba O0c cache recurrent-state storage diagnostic execution authority candidate

## 1. Status and purpose

`PASS_READY_FOR_FORMAL_FREEZE_CACHE_RECURRENT_STATE_STORAGE_DIAGNOSTIC_EXECUTION_AUTHORITY`

This is a report-only diagnostic execution-authority candidate.

It does not modify the validator, tests, runtime packages, or scientific code. It authorizes, only after formal freeze and remote verification, exactly one future CPU-only read-only diagnostic against the exact frozen implementation commit identified below.

The sole purpose is to explain the newly exposed corrected-preflight blocker:

`BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`

with note:

`cache_recurrent_state_storage`.

The diagnostic must collect source/provenance and static-AST facts sufficient to determine whether the blocker is caused by validator source-role assumptions, mutation-form assumptions, runtime source drift, or another narrowly identifiable infrastructure condition.

No training, evaluation, model execution, tokenizer/dataset loading, package mutation, validator modification, scientific execution, or scientific interpretation is authorized.

`SCIENTIFIC_CONCLUSION: NONE`.

## 2. Frozen authority and execution lineage

| Authority / evidence | Frozen identity |
| --- | --- |
| Cache-guard corrected-preflight execution authority | `59338ca88796cf39dd31fd60a9c6a46e47570761` |
| Cache-guard role-selection corrected implementation | `0063254795aa21011364833c95d25cbce262c0bf` |
| Cache-guard role-selection implementation authority | `1e7630c70a3a7cd85caa64128d89c241ed8b0960` |
| Cache-guard ambiguity root-cause interpretation | `6fe942ae7d872314d4fd4da2c68ca221e7f45b0e` |
| Linkage-ambiguity diagnostic execution authority | `b20857b78af86d814b48db9c9846a3c8bb049d61` |
| Earlier corrected-preflight authority | `755987eea6230bb0ad6f73400e46ae680434e6f6` |
| Earlier corrected validator implementation | `6f394792763abb168f49c1cb1957a326d16eed2b` |

The sole implementation commit that may be inspected/exercised by the future diagnostic is:

`0063254795aa21011364833c95d25cbce262c0bf`

The diagnostic may run only from a clean checkout of that commit.

## 3. Consumed corrected-preflight execution

The following run is permanently consumed and must not be rerun, reused, overwritten, or aliased:

`longterm-o0c-runtime-source-provenance-preflight-0063254-v1`

Frozen execution facts:

- expected commit: `0063254795aa21011364833c95d25cbce262c0bf`;
- actual commit: `0063254795aa21011364833c95d25cbce262c0bf`;
- command SHA256: `a3e973cafc52b281a8d60d57b953e204a515fc06b8e577611bd5ee734e6594a7`;
- started: `2026-09-08T08:21:29Z`;
- finished: `2026-09-08T08:21:51Z`;
- exit code: `2`;
- observed preflight status: `BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`;
- observed blocker/note: `cache_recurrent_state_storage`;
- run-log SHA256: `af0cfa6ce59d6100b050a4582c8e10aca406928254510687dccf525abdf1d717`;
- run-meta SHA256: `59293d16919c992fcac9876cc2260c74f3f3f2fd8b886046785d908af07032fe`;
- handoff ZIP SHA256: `ed74729687239d8ef83db8dd8a42072bfe4be12beb67a0ba7a0af5a59bd6f20a`;
- collection: PASS, `FILES_COLLECTED=0`;
- import: PASS, `VALIDATED=0`, `COPIED=0`, `IDENTICAL=0`.

The zero collected-file count is consistent with the failing preflight not publishing its result artifact. Therefore this consumed run proves the execution/provenance envelope and blocker observation, but does not itself freeze the runtime source identities needed for the new root-cause decision.

No rerun of this consumed preflight is authorized.

## 4. Frozen validator behavior under diagnosis

At implementation commit `0063254795aa21011364833c95d25cbce262c0bf`, `cache_recurrent_state_storage` is a required symbol family.

The frozen binding logic constructs this family from the `cache` source module only and searches function direct bodies for `ast.Assign` / `ast.AnnAssign` statements that:

- assign to a path ending in `ssm_state` or `self.ssm_state`; and
- load a local name `ssm_state`.

This family selection is separate from the corrected convolution-cache branch-role logic.

The diagnostic must replay this exact frozen candidate formation without changing it and must report:

- cache-source function count;
- frozen direct candidate count;
- exact frozen candidate qualnames/spans, if any;
- frozen family outcome: zero / one / multiple.

The diagnostic must not alter or reinterpret the frozen algorithm while producing the replay facts.

## 5. Static upstream evidence motivating the diagnostic

The public Transformers `v5.0.0` Mamba source currently inspected for diagnostic design places recurrent-state cache persistence inside `MambaMixer.slow_forward`.

The relevant structural behavior is:

- when cache is present, the recurrent state is initialized from `cache_params.ssm_states[self.layer_idx]`;
- after the sequential recurrence, cache persistence is performed through mutation of `cache_params.ssm_states[self.layer_idx]` using a `copy_(ssm_state)` call.

This static upstream observation is diagnostic-design context only. It does not substitute for runtime source identity collected by the authorized diagnostic.

The diagnostic must determine whether the actual runtime source used in the Kaggle environment has this same structure.

## 6. Diagnostic hypotheses to distinguish

The diagnostic is result-neutral and must distinguish at least the following hypotheses.

### H1 — cache-module source-role false negative

The actual recurrent-state persistence is present and statically provable in the Mamba model source, while the frozen family searches only the cache-utils source.

Potential formal interpretation after validated import:

`VALIDATOR_CACHE_RECURRENT_STATE_STORAGE_SOURCE_ROLE_FALSE_NEGATIVE`

### H2 — mutation-form false negative

The actual persistence is expressed as a mutating call such as:

`cache_params.ssm_states[...].copy_(ssm_state)`

rather than an assignment statement matching the frozen assignment-only predicate.

Potential formal interpretation after validated import:

`VALIDATOR_CACHE_RECURRENT_STATE_STORAGE_MUTATION_FORM_FALSE_NEGATIVE`

### H3 — combined source-role and mutation-form defect

Both H1 and H2 are simultaneously required to explain the zero frozen candidates.

Potential formal interpretation after validated import:

`VALIDATOR_CACHE_RECURRENT_STATE_STORAGE_SOURCE_AND_MUTATION_FORM_FALSE_NEGATIVE`

### H4 — runtime source drift / different source shape

The actual runtime source differs materially from the previously diagnosed Transformers-5.0.0 source shape.

Potential controller state:

`RUNTIME_SOURCE_IDENTITY_DRIFT`

### H5 — diagnostic inconclusive

The observed source contains another structurally relevant form not covered above, or proof is ambiguous.

No implementation correction may be authorized directly from an inconclusive diagnostic.

## 7. Exact diagnostic facts required

The future diagnostic must produce one deterministic JSON artifact containing at least the following.

### 7.1 Execution and runtime identity

- schema/version identifier for this diagnostic artifact;
- run name;
- expected commit;
- actual commit;
- Python version;
- NumPy version;
- torch version;
- Transformers distribution version;
- CUDA availability;
- CUDA device count.

### 7.2 Source identities

For both resolved sources:

- module name;
- canonical path;
- SHA256;
- byte count;
- LF count;
- CR count;
- final-LF state.

Required source keys:

- `mamba`;
- `cache`.

The diagnostic must use the same import/distribution-root resolution principles as the frozen preflight or directly reuse its pure resolution helpers without invoking model/runtime execution.

### 7.3 Frozen family replay

For `cache_recurrent_state_storage`:

- frozen source key used by the validator;
- frozen scanned function count;
- frozen direct-assignment candidate count;
- exact candidate qualname/start/end location for every candidate;
- replay classification: `ZERO`, `UNIQUE`, or `MULTIPLE`.

### 7.4 Cross-source recurrent-storage structural scan

Across both `mamba` and `cache` ASTs, record all same-lexical-scope statements/calls relevant to persistent recurrent-state storage, including:

- assignments whose target path contains or ends in `ssm_state`, `ssm_states`, `self.ssm_state`, or `self.ssm_states`;
- method calls whose receiver path contains `ssm_state` or `ssm_states`;
- mutation calls ending in `copy_`, `copy`, `update`, `set`, or equivalent discovered statically;
- whether the statement/call loads the local name `ssm_state`;
- enclosing qualname;
- start/end line;
- source key;
- normalized target/receiver/call path.

Do not treat lexical descendants inside nested functions/classes as proof for the enclosing function.

### 7.5 `MambaMixer.slow_forward` role facts

If uniquely present, record whether `MambaMixer.slow_forward` contains:

- cache-present read of `cache_params.ssm_states[...]` into recurrent local state;
- sequential recurrent update of local `ssm_state`;
- cache-present persistent mutation back into `cache_params.ssm_states[...]`;
- the exact mutation form used;
- whether the mutation loads the final local `ssm_state`;
- enclosing cache-present guard;
- canonical source span.

### 7.6 Cache-utils role facts

Record whether the actual cache source contains:

- a class or function that owns recurrent `ssm_state` / `ssm_states` persistence;
- a direct assignment matching the frozen predicate;
- a mutation method linked from Mamba recurrent-state persistence;
- no relevant storage behavior.

Do not assume the answer in advance.

### 7.7 Historical-source comparison

Compare actual Mamba/cache raw identities against the previously validated identities:

Mamba:

- SHA256 `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83`;
- bytes `39500`;
- LF `860`;
- CR `0`;
- final LF `true`.

Cache-utils:

- SHA256 `6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc`;
- bytes `60432`;
- LF `1295`;
- CR `0`;
- final LF `true`.

Record material drift separately for each source.

## 8. Diagnostic output classifications

The diagnostic artifact may report one of these observational classifications:

- `CACHE_STORAGE_RUNTIME_SHAPE_MATCHES_HISTORICAL_MAMBA_PERSISTENCE`;
- `CACHE_STORAGE_PRESENT_IN_MAMBA_ONLY`;
- `CACHE_STORAGE_MUTATION_FORM_NOT_ASSIGNMENT`;
- `CACHE_STORAGE_PRESENT_IN_CACHE_SOURCE`;
- `CACHE_STORAGE_MULTIPLE_STRUCTURAL_CANDIDATES`;
- `RUNTIME_SOURCE_IDENTITY_DRIFT`;
- `DIAGNOSTIC_INCONCLUSIVE`.

These are diagnostic observations, not formal root-cause decisions.

The later controller interpretation must consume the imported artifact and decide the formal infrastructure root cause separately.

## 9. Sole reserved future run name

This authority reserves exactly one run name:

`longterm-o0c-cache-recurrent-state-storage-diagnostic-0063254-v1`

No fallback or alternate run name is authorized.

Repository search at authoring time found no exact occurrence of this name.

Before registration, collision checks must be repeated against:

- repository history/current tree, excluding the intentional authority reservation;
- local `cm` run registry;
- local imports;
- downloaded handoffs;
- accessible prior run metadata.

Any genuine collision blocks execution.

## 10. Exact future execution guard

Before registration/execution require:

- Kaggle checkout HEAD exactly `0063254795aa21011364833c95d25cbce262c0bf`;
- clean Kaggle repository worktree;
- local command registration bound to the same implementation commit;
- remote authority freeze independently verified;
- frozen production blob:
  `86f2cf3e942f1429c2a60c3c1358e188e09bf82f`;
- frozen production canonical LF SHA256:
  `34ed1663fb7b6081f42e4e4fe2a3c4c42791aaeff8910f133a3218ed14f6d62a`;
- no run-name collision;
- no command/hash/provenance guard failure.

The diagnostic must not run from the authority commit itself. The inspected code commit remains `0063254795aa21011364833c95d25cbce262c0bf`.

## 11. Runtime and execution boundary

Required runtime boundary:

- Python `3.12.13`;
- NumPy `2.0.2`;
- torch `2.10.0+cpu`;
- Transformers `5.0.0`;
- CUDA available `False`;
- CUDA device count `0`;
- Kaggle Accelerator `None`;
- GPU `OFF`.

The diagnostic is CPU/static/provenance only.

It may:

- import standard-library parsing/introspection modules;
- query installed distribution metadata;
- resolve module source paths;
- read source bytes;
- parse source with `ast`;
- import the frozen preflight module only if doing so does not execute model/package mutation behavior and only pure helper functions are used.

It must not:

- instantiate a model;
- load a model checkpoint;
- load a tokenizer;
- load a dataset;
- run tensor/model forward;
- run generation;
- train;
- evaluate;
- install or mutate packages;
- modify repository files;
- modify installed Transformers files.

## 12. Command-transport boundary

The exact shell command and command SHA256 are intentionally not frozen in this candidate.

They may be generated only after:

1. this authority is materialized;
2. statically reviewed;
3. formally frozen;
4. committed/pushed;
5. independently remote-verified;
6. `cm kaggle` confirms the exact implementation checkout.

The command must write its deterministic diagnostic JSON outside or within the repository only at the explicitly authorized output path below:

`reports/longterm_o0c_cache_recurrent_state_storage_diagnostic_0063254_result.json`

This output is an execution artifact, not a source modification.

## 13. Collection and import

After the one authorized diagnostic completes, use:

`cm collect longterm-o0c-cache-recurrent-state-storage-diagnostic-0063254-v1`

Run the generated collector in the same Kaggle session, download the ZIP, and locally run:

`cm import <handoff.zip>`

Interpretation is blocked until import PASS.

Validate at least:

- run name;
- expected/actual commit;
- command SHA;
- start/finish timestamps;
- exit code;
- run-log SHA;
- run-meta SHA;
- handoff ZIP SHA;
- output diagnostic artifact SHA;
- source identities;
- frozen-family replay facts;
- cross-source structural facts;
- historical-source comparison.

## 14. Failure and recovery

Any command transport, collision, execution, collection, handoff, import, provenance, runtime-version, or source-resolution failure requires STOP.

Do not rerun the consumed diagnostic name.

Do not patch the validator or create a new execution without a separate authority decision.

A successful diagnostic does not itself authorize code changes.

## 15. Evidence-layer separation

| Layer | State at authority authoring |
| --- | --- |
| A. Cache-guard role-selection implementation correctness | PASS / frozen at `0063254795aa21011364833c95d25cbce262c0bf` |
| B. Corrected preflight execution | COMPLETED / exit `2` |
| C. Corrected preflight provenance validity | PASS |
| D. Observed corrected-preflight blocker | `cache_recurrent_state_storage` unresolved |
| E. New diagnostic execution | `NOT_YET_EXECUTED` |
| F. New diagnostic artifact/provenance validity | `NOT_YET_ESTABLISHED` |
| G. Formal new infrastructure root cause | `NOT_YET_FROZEN` |
| H. Scientific conclusion | `NONE` |

## 16. Formal-freeze boundary

During this candidate authoring/materialization/freeze:

- no diagnostic execution;
- no `cm kaggle`;
- no run registration;
- no `cm run`;
- no collection/import;
- no code modification;
- no model execution;
- no training/evaluation.

Only after formal freeze and remote verification may the controller construct and register the one read-only diagnostic command.

## 17. Exact next action

Materialize this candidate in the current short authority worktree based on remote `59338ca88796cf39dd31fd60a9c6a46e47570761`.

Then perform exact-byte staging and `cm ship`.

Do not execute the diagnostic before authority freeze.
