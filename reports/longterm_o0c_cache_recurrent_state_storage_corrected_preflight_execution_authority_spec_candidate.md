# ContraMamba O0c cache recurrent-state storage corrected-preflight execution authority candidate

## 1. Formal status

`PASS_READY_FOR_FORMAL_FREEZE_CACHE_RECURRENT_STATE_STORAGE_CORRECTED_PREFLIGHT_EXECUTION_AUTHORITY`

This report is an execution-authority candidate for one CPU-only deterministic O0c runtime-source provenance preflight after the frozen `cache_recurrent_state_storage` semantic-binding correction.

It authorizes only the exact preflight execution described here.

It does not authorize training, evaluation, model execution, tokenizer/dataset loading, generation, scientific interpretation, package mutation, source modification, or any broader execution.

`SCIENTIFIC_CONCLUSION: NONE`.

## 2. Frozen authority lineage

| Authority / implementation | Frozen identity |
| --- | --- |
| Cache recurrent-state storage semantic-binding implementation | `84fba077d95c7691fd76038c8cb817ab1fde3b7a` |
| Semantic-binding implementation authority | `a8500976faec4ee80421827e5f13d00db2be5591` |
| Cache recurrent-state storage root-cause interpretation | `8e42d0b039fe64becac3caf64293c69e810f2b07` |
| Cache recurrent-state storage diagnostic execution authority | `c54bd26ee214a2e75424b40df8059ea5f562a4f5` |
| Prior cache-guard corrected-preflight execution authority | `59338ca88796cf39dd31fd60a9c6a46e47570761` |
| Prior cache-guard role-selection corrected implementation | `0063254795aa21011364833c95d25cbce262c0bf` |

The sole implementation commit authorized for execution by this report is:

`84fba077d95c7691fd76038c8cb817ab1fde3b7a`.

No descendant, ancestor, dirty worktree, reconstructed patch, or alternate commit is authorized.

## 3. Frozen implementation identity

Production file:

`scripts/preflight_longterm_o0c_runtime_source_provenance.py`

Frozen identity:

- SHA256: `b75fe23c22fe2570f32261fc0a9498a18bc9d94cb5bef3d28a88e0594e9c4497`;
- bytes: `44656`;
- LF: `1152`;
- CR: `0`;
- final LF: `true`;
- Git blob: `5c177003b21c26e391bc22c41f3e8f0f1f1b6813`.

Targeted test file:

`tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

Frozen identity:

- SHA256: `76b555a87d67038663923d85e1b0cff779c5f827e1130c539e720a74fce06deb`;
- bytes: `60467`;
- LF: `1418`;
- CR: `0`;
- final LF: `true`;
- Git blob: `af4d1d94ca19efd2229e79c0c16c8478316e50e3`.

Implementation validation:

- independent verifier verdict:
  `PASS_SAFE_TO_FREEZE_CACHE_RECURRENT_STATE_STORAGE_SEMANTIC_BINDING_IMPLEMENTATION`;
- targeted pytest:
  `96 passed, 3 skipped`;
- staged/commit scope: exactly the production and targeted test files above;
- no training/evaluation/model execution/Kaggle activity occurred during implementation or verification.

## 4. Frozen correction being exercised

The prior implementation at `0063254795aa21011364833c95d25cbce262c0bf` blocked on:

`BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`

with note:

`cache_recurrent_state_storage`.

Validated diagnostic evidence established the formal root cause:

`VALIDATOR_CACHE_RECURRENT_STATE_STORAGE_SOURCE_AND_MUTATION_FORM_FALSE_NEGATIVE`.

The corrected implementation at `84fba077d95c7691fd76038c8cb817ab1fde3b7a` now requires a fail-closed semantic proof in `MambaMixer.slow_forward` and binds the canonical recurrent-state storage location to the unique guarded Mamba persistent mutation structurally equivalent to:

`cache_params.ssm_states[self.layer_idx].copy_(ssm_state)`.

The corrected binder:

- uses Mamba source/module/source SHA;
- requires a guarded recurrent-state cache read;
- reuses the existing unique sequential recurrent proof;
- requires a guarded `copy_(ssm_state)` persistent write after the recurrent loop;
- excludes nested lexical definitions;
- resolves zero candidates as unresolved;
- resolves multiple qualifying writes as ambiguous;
- preserves the existing symbol-location schema.

This execution authority does not reinterpret or widen that correction.

## 5. Runtime boundary

The execution is authorized only under this exact expected runtime boundary:

- Python: `3.12.13`;
- NumPy: `2.0.2`;
- torch: `2.10.0+cpu`;
- Transformers: `5.0.0`;
- CUDA available: `false`;
- CUDA device count: `0`.

Kaggle Accelerator must be:

`None`.

GPU must remain OFF.

Any runtime-version mismatch or CUDA availability is a blocker and must not be bypassed.

## 6. Historical source identities expected by the validator

Historical Mamba source identity:

- module: `transformers.models.mamba.modeling_mamba`;
- path: `/usr/local/lib/python3.12/dist-packages/transformers/models/mamba/modeling_mamba.py`;
- SHA256: `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83`;
- bytes: `39500`;
- LF: `860`;
- CR: `0`;
- final LF: `true`.

Historical cache source identity:

- module: `transformers.cache_utils`;
- path: `/usr/local/lib/python3.12/dist-packages/transformers/cache_utils.py`;
- SHA256: `6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc`;
- bytes: `60432`;
- LF: `1295`;
- CR: `0`;
- final LF: `true`.

The preflight remains responsible for resolving and recording actual runtime source identities. This authority does not permit hard-coded substitution for observed runtime facts.

## 7. Sole reserved run

The sole reserved run name is:

`longterm-o0c-runtime-source-provenance-preflight-84fba07-v1`

This run name must not be changed or reused after execution.

If execution starts, the run name is consumed regardless of exit code.

A failed or blocked execution must be collected/imported when appropriate; it must not be rerun under the same name.

## 8. Authorized output

The sole intended validator output is:

`reports/longterm_o0c_runtime_source_provenance_preflight_84fba07_result.json`

The output must not pre-exist in the Kaggle checkout before execution.

Collision is a blocker.

No alternative output filename is authorized under this report.

## 9. Authorized command semantics

After this authority is formally frozen and remotely verified, and only after `cm kaggle` verifies the exact implementation commit, the controller may generate/register one pinned command equivalent to:

`python scripts/preflight_longterm_o0c_runtime_source_provenance.py --output reports/longterm_o0c_runtime_source_provenance_preflight_84fba07_result.json --expected-python 3.12.13 --expected-numpy 2.0.2 --expected-torch 2.10.0+cpu --expected-transformers 5.0.0`

The exact command bytes and SHA256 must be frozen through the run registry before execution.

The run registry must record:

- run name exactly `longterm-o0c-runtime-source-provenance-preflight-84fba07-v1`;
- HEAD exactly `84fba077d95c7691fd76038c8cb817ab1fde3b7a`;
- the exact registered command SHA256.

Do not manually reconstruct a Kaggle cell after registration. Use the generated pinned run cell.

## 10. Pre-execution guards

Before command registration/execution, all of the following must pass.

1. Local implementation worktree HEAD equals:
   `84fba077d95c7691fd76038c8cb817ab1fde3b7a`.
2. Remote `main` contains the formally frozen execution authority as the latest applicable authority.
3. The implementation worktree is clean.
4. `cm kaggle` bootstraps/checks out exactly:
   `84fba077d95c7691fd76038c8cb817ab1fde3b7a`.
5. Accelerator is `None`.
6. GPU is OFF.
7. The reserved run name is absent from the run registry before save.
8. No handoff/import/download collision exists for this reserved run.
9. The authorized output path does not pre-exist in the Kaggle checkout.
10. The generated pinned run cell reports the exact expected commit and registered command SHA256.

Any failed guard blocks execution.

## 11. Authorized execution class

This is infrastructure-only, CPU/static/provenance validation.

Allowed:

- importing the frozen preflight script;
- Python/NumPy/torch/Transformers version inspection;
- CUDA availability/device-count inspection;
- filesystem/source-root resolution;
- reading package source files as raw bytes/text;
- AST parsing and static classification;
- deterministic JSON serialization/publication;
- Git/run provenance capture;
- collection and import of produced artifacts/log/meta.

Not allowed:

- model instantiation;
- tokenizer loading;
- dataset loading;
- tensor/model forward passes;
- generation;
- training;
- evaluation;
- optimizer/scheduler creation;
- package installation/upgrading/downgrading;
- source modification;
- monkeypatching Transformers;
- GPU use;
- scientific result interpretation.

## 12. Expected execution interpretation boundary

This authority does not predetermine the validator outcome.

Possible validator outcomes include PASS or a fail-closed blocker.

A PASS means only that the frozen preflight accepted the inspected runtime/source/provenance conditions under its rules.

A blocker means the observed condition must be collected/imported and interpreted under a separate report/authority as needed.

Neither outcome establishes a scientific claim.

## 13. Prior consumed runs

The following run names are consumed and must not be reused:

- `longterm-o0c-runtime-source-provenance-preflight-0063254-v1`;
- `longterm-o0c-cache-recurrent-state-storage-diagnostic-0063254-v1`.

Their outputs/provenance remain historical evidence only.

No command, artifact, or handoff from either run may be substituted for the new authorized run.

## 14. Collection and import

After the run finishes, regardless of PASS or fail-closed validator exit when collection is appropriate:

1. run:
   `cm collect longterm-o0c-runtime-source-provenance-preflight-84fba07-v1`;
2. execute the generated collector in the same relevant Kaggle session;
3. download the produced handoff ZIP;
4. locally run:
   `cm import <handoff.zip>`.

Interpretation of validator evidence is blocked until import provenance validation passes.

A run success alone is not sufficient.

## 15. Provenance blockers

Treat any of the following as a hard blocker:

- expected/actual commit mismatch;
- run registry commit mismatch;
- command SHA mismatch;
- dirty/reconstructed implementation;
- runtime boundary mismatch;
- CUDA available or device count nonzero;
- output collision;
- run-name collision/reuse;
- source/provenance hash mismatch;
- collection manifest mismatch;
- handoff ZIP provenance mismatch;
- import validation failure.

Do not bypass, rename around, or manually repair such a blocker under this authority.

## 16. Evidence-layer separation

| Layer | State at authority freeze |
| --- | --- |
| Root-cause interpretation | FROZEN |
| Semantic-binding implementation authority | FROZEN |
| Semantic-binding implementation | FROZEN |
| Independent implementation verification | PASS |
| Corrected-preflight execution authority | THIS REPORT |
| Corrected-preflight execution | NOT_YET_RUN |
| Execution provenance | NOT_YET_ESTABLISHED |
| Validator result | NONE |
| Scientific conclusion | `NONE` |

## 17. Stop conditions

STOP without execution or rerun if:

- exact implementation commit cannot be established;
- remote authority state conflicts;
- worktree is dirty;
- `cm kaggle` does not land on the exact commit;
- Accelerator/GPU boundary is violated;
- reserved run already exists unexpectedly;
- output/handoff/import collision exists;
- command identity changes after registration;
- execution has already started under the reserved run name;
- any broader code/package change appears necessary.

## 18. Freeze boundary

Formal freeze of this report authorizes exactly one pinned CPU-only corrected-preflight run at implementation commit:

`84fba077d95c7691fd76038c8cb817ab1fde3b7a`

under run name:

`longterm-o0c-runtime-source-provenance-preflight-84fba07-v1`.

It does not authorize execution before formal freeze/remote verification and successful `cm kaggle`.

It does not authorize any subsequent run, correction, training/evaluation, or scientific claim.

## 19. Exact next action after freeze

After formal freeze and remote verification:

1. verify the clean implementation worktree at `84fba077d95c7691fd76038c8cb817ab1fde3b7a`;
2. run `cm kaggle`;
3. perform collision checks;
4. register the exact command with `cm run save`;
5. generate the pinned cell with `cm run longterm-o0c-runtime-source-provenance-preflight-84fba07-v1`;
6. execute only that generated cell.

No Kaggle execution is authorized before those guards pass.
