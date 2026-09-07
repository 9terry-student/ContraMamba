# ContraMamba O0c Corrected Preflight Execution Authority Spec Candidate

## 1. Overall verdict

CANDIDATE_AUTHORED_PENDING_INDEPENDENT_VERIFICATION_AND_FREEZE

This report-only candidate may, after independent verification and formal freeze, authorize exactly one new CPU-only O0c runtime-source provenance preflight run. It does not execute that run and does not authorize scientific execution, implementation repair, or environment mutation.

The candidate is tied to implementation commit 426ecd2038a1118d3afb9bd7bdae63f340e3c70b and reserves the run name longterm-o0c-runtime-source-provenance-preflight-426ecd2-v4.

## 2. Starting repository, HEAD, and state

- Repository root: C:\o0c-preflight-auth-c551747
- Starting HEAD: 426ecd2038a1118d3afb9bd7bdae63f340e3c70b
- Tracked modifications before authoring: none
- Staged paths before authoring: none
- Untracked paths before authoring: none reported by Git
- Candidate path before authoring: absent
- Task temp files: none identified

Git status inspection emitted permission warnings for pre-existing .pytest_cache / pytest-cache-files-* directories; no status entries were reported. Those directories were not modified or used as task artifacts.

## 3. Authority chain

The current controller instruction is the active authority for this report-only authoring phase. The frozen authority chain supplied by that controller is:

1. Primary frozen corrected implementation commit: 426ecd2038a1118d3afb9bd7bdae63f340e3c70b
2. Frozen recurrent-state initialization correction implementation authority: 989708d011efe10fd72e32a6d91dd9d225f00673
3. Frozen recurrent-state root-cause interpretation: bfc998f9206ff77fcc2b2e80bb94293dc737b13f
4. Prior corrected-preflight execution authority: 52bae21fc5f5ee4e3ba3b5b15a0d5682f86daf0f
5. Prior corrected implementation / consumed v3 execution commit: c551747180ce1e8fe4eed5f7aa5ab6294cd89948
6. Frozen diagnostic execution authority: ca112032841fe316e5e2e7335dc95d89aeedb450
7. Frozen O0c runtime-source preflight implementation authority: 811ae9c843564e8cddb5fc373761afb618cb7cfd
8. Frozen O0c runtime-source preflight authority: 8c6a0ccf2a8583b9b7accbdb5ab757d722b6e328
9. Frozen O0c native-state authority: 242ad9ed70fc995ebda560911a7d0dfd2f18f9b3

No authority conflict was identified during this authoring check.

## 4. Frozen implementation commit and scope

Commit 426ecd2038a1118d3afb9bd7bdae63f340e3c70b has parent 989708d011efe10fd72e32a6d91dd9d225f00673. Its exact commit scope is only:

- scripts/preflight_longterm_o0c_runtime_source_provenance.py
- tests/test_preflight_longterm_o0c_runtime_source_provenance.py

The production change remains narrow and is recorded here without extending its interpretation:

- recurrent initialization discovery uses same-lexical-scope traversal;
- the initialization predicate is unchanged;
- zero/multiple initialization cardinality behavior is unchanged;
- recurrent loop/update discovery remains direct-body-only;
- no runtime, version, hash, or line-number special case exists.

The direct commit diff is 13 insertions and 1 deletion in the script, and 171 insertions and 1 deletion in its dedicated test file.

## 5. Raw Git-object identities

Identities below were independently recomputed from git show bytes using a binary-safe Python subprocess read. They are Git-object identities, not Windows checkout identities.

### Production script

Path: scripts/preflight_longterm_o0c_runtime_source_provenance.py

- SHA256: a55a69cd0d6bd6f868e18e2d8c48cfb40fc85b0b52fb642c1a3964645c26a5ba
- Bytes: 32448
- LF: 842
- CR: 0
- Final LF: true
- Git blob SHA: 06a37080ff5971bad4472a7ed4004b6deeceeaf8

### Dedicated test file

Path: tests/test_preflight_longterm_o0c_runtime_source_provenance.py

- SHA256: ea3079ef326681cb549fd2de4d58e783a003422360bf3d245cc3afa39d7c59f6
- Bytes: 35346
- LF: 932
- CR: 0
- Final LF: true
- Git blob SHA: aad3580164ec110086815cf948d9c982e3b1c901

If an independent recomputation differs, this candidate is blocked.

## 6. Evidence-layer separation

### A. Code correctness

- Local dedicated suite: 50 passed, 3 skipped
- Independent implementation verifier: PASS_SAFE_TO_FREEZE_RECURRENT_STATE_INITIALIZATION_VALIDATOR_CORRECTION_IMPLEMENTATION

This is local code-correctness evidence only.

### B. Implementation freeze

- Frozen implementation commit: 426ecd2038a1118d3afb9bd7bdae63f340e3c70b

### C. Runtime execution on 426ecd

- NOT_YET_ESTABLISHED

### D. Artifact/provenance validity for a future 426ecd run

- NOT_YET_ESTABLISHED

### E. Scientific O0c conclusion

- NONE

These layers must remain separate. The local suite and verifier do not constitute runtime evidence for the exact Kaggle Transformers 5.0.0 source.

## 7. Consumed prior runs

The following runs are consumed and are never reusable, rerunnable, or overwritable:

- longterm-o0c-runtime-source-provenance-preflight-de874a2-v2
- longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1
- longterm-o0c-runtime-source-provenance-preflight-c551747-v3
- longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1

### Consumed v3 preflight

- Run: longterm-o0c-runtime-source-provenance-preflight-c551747-v3
- Execution commit: c551747180ce1e8fe4eed5f7aa5ab6294cd89948
- Command SHA256: 369bd49fda005c35d22894e0dca0a105472f72daa6eab55a19821c7b697c9db0
- Validated runtime result: BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED
- Blocker: recurrent_state_initialization
- Provenance: VALID after collect/import

### Consumed diagnostic v1

- Run: longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1
- Execution commit: c551747180ce1e8fe4eed5f7aa5ab6294cd89948
- Command SHA256: 0795668fa8e7770bed5dabde1991d0447f5c42d30d6062f499ff0033ca016247
- Exit: 0
- Later-frozen classification: VALIDATOR_RECURRENT_STATE_INITIALIZATION_PATTERN_FALSE_NEGATIVE
- Provenance: VALID after collect/import

These runs are historical evidence only and are not substituted for the future v4 execution.

## 8. Reserved run name and collision check

Reserved run name:

longterm-o0c-runtime-source-provenance-preflight-426ecd2-v4

Collision checks performed:

- Repository text: no exact v4 run-name hit. Historical v3 references exist and are preserved; they are not v4 collisions.
- Local .contramamba run registry: .contramamba/runs was absent or inaccessible; no v4 registration was found.
- Local import audits: C:\Users\Home1\.contramamba\imports was accessible; no exact v4 import directory was found.

No exact run-name collision was identified. If an exact v4 name is later found to be registered, executed, imported, or consumed in an incompatible context, stop and block rather than selecting another name.

## 9. Exact future execution commit

The future execution commit must be exactly 426ecd2038a1118d3afb9bd7bdae63f340e3c70b.

Before future registration and execution, require exact HEAD match, a clean tracked tree, no staged paths, no untracked execution-affecting files, and no execution from any earlier or later commit. No dirty-tree or hash blocker may be bypassed.

## 10. Runtime and CPU/GPU boundary

Freeze the intended Kaggle runtime:

- Python 3.12.13
- NumPy 2.0.2
- torch 2.10.0+cpu
- Transformers 5.0.0
- CPU only
- Kaggle accelerator: none / GPU OFF

No package install, uninstall, upgrade, downgrade, CUDA enablement, optional-kernel enablement, or environment repair is authorized. Runtime mismatch, GPU exposure, or package/environment mismatch must fail closed.

## 11. Future command construction and identity contract

No command.sh is materialized by this authoring task. The later exact shell command must be newly generated for this authority and must not reuse the consumed c551747-v3 command identity.

The future command must pin:

- Run name: longterm-o0c-runtime-source-provenance-preflight-426ecd2-v4
- Commit: 426ecd2038a1118d3afb9bd7bdae63f340e3c70b
- Script: scripts/preflight_longterm_o0c_runtime_source_provenance.py
- Script SHA256: a55a69cd0d6bd6f868e18e2d8c48cfb40fc85b0b52fb642c1a3964645c26a5ba
- Script bytes: 32448
- Script LF: 842
- Script CR: 0
- Script final LF: true

It must verify exact commit, clean tree, exact script raw identity, runtime versions, GPU OFF / no NVIDIA exposure, and invoke only the preflight script. It must preserve the existing output schema/path and fail-closed semantics, must not load a model, tokenizer, or dataset, and must not train or evaluate. It must emit deterministic wrapper/run evidence. Before cm run save, compute and freeze the newly generated command's exact raw SHA256, byte count, LF/CR counts, and final-LF identity. Command identity is separate from Git commit and script identity; no CRLF-normalized alternate may be accepted silently.

## 12. Allowed future outcomes

The future runtime result is deliberately not predeclared. The local regression suite suggests initialization discovery is corrected and a nested recurrence-loop shape may subsequently reach a recurrent_state_update blocker, but that is not runtime evidence for the exact Kaggle Transformers 5.0.0 source.

Allowed future outcomes are only:

- PASS_SOURCE_IDENTITY_FROZEN
- any existing frozen fail-closed preflight blocker
- wrapper/provenance fail-closed blocker

If a new semantic blocker occurs, stop. Do not patch or reinterpret it under this authority.

## 13. Schema and output contract

The existing schema remains o0c_runtime_source_provenance_preflight_v1.

The existing canonical JSON output path remains:

reports/longterm_o0c_runtime_source_provenance_preflight.json

No schema expansion, manual JSON synthesis, append, overwrite of unrelated artifacts, timestamp fallback, random fallback, or alternate output path is authorized. If the preflight blocks before JSON publication, missing JSON is interpreted according to frozen script semantics and is not automatically collector failure.

## 14. Artifact and provenance contract

Future execution provenance must connect, without a broken link:

run name -> full 40-character commit SHA -> exact command SHA256/raw identity -> exact script raw identity -> runtime versions -> start/finish UTC -> exit code -> run.log -> run metadata -> canonical JSON only if published -> collector result -> handoff ZIP SHA256 -> local import audit

Any broken link fails closed.

## 15. Future Kaggle/run/collect/import sequence

Only after authority freeze and all pre-execution gates pass, the authorized sequence may specify:

1. cm kaggle
2. The future controller-provided exact shell command.
3. cm run save longterm-o0c-runtime-source-provenance-preflight-426ecd2-v4
4. cm run longterm-o0c-runtime-source-provenance-preflight-426ecd2-v4
5. When appropriate, cm collect longterm-o0c-runtime-source-provenance-preflight-426ecd2-v4.
6. Kaggle collector, download of the handoff ZIP, then cm import <handoff.zip>.

This candidate performs none of those actions and authorizes no bypass of HEAD, hash, dirty-tree, runtime, GPU, collector, import, or provenance blockers.

## 16. Failure-recovery boundary

The future authority must stop rather than patch, rerun, or improvise on:

- HEAD mismatch
- dirty tree
- run-name collision
- command identity mismatch
- script identity mismatch
- runtime mismatch
- GPU exposure
- package/environment mismatch
- unexpected schema
- collector/import mismatch
- provenance mismatch
- any newly reached fail-closed semantic blocker

This authority does not authorize repair.

## 17. Implementation and scientific non-authorization

The future run must execute the frozen implementation exactly. This execution authority does not authorize modifying the preflight script, its dedicated tests, any report, the run registry except for future explicit registration, imported audits, or the package environment.

It does not authorize correcting the adjacent recurrent_state_update / nested-loop surface. If execution exposes that blocker, implementation work requires a new formal authority sequence.

The future run is infrastructure/provenance validation only. It does not authorize O0c scientific execution, model/tokenizer/dataset loading, forward passes, generation, training, evaluation, model selection, threshold tuning, promotion, or scientific interpretation. Scientific conclusion remains NONE.

## 18. Independent verification requirement

Before freeze, an independent verifier must check:

- exact parent and current authority chain;
- exact implementation commit and two-file scope;
- raw Git-object script and test identities;
- run-name collision status;
- consumed-run preservation;
- runtime and CPU/GPU boundary;
- command-construction and separate command-identity contract;
- absence of a predeclared runtime result;
- absence of implementation and scientific authorization;
- candidate raw identity;
- clean Git state.

## 19. Candidate authoring validation

The following report/static validation is required after writing this candidate:

    git diff --check
    git diff --name-status
    git diff --cached --name-status
    git status --short
    git rev-parse HEAD

Required final state:

- HEAD exactly 426ecd2038a1118d3afb9bd7bdae63f340e3c70b;
- tracked modifications: none;
- staged paths: none;
- exactly one untracked path: reports/longterm_o0c_recurrent_state_initialization_corrected_preflight_execution_authority_spec_candidate.md;
- no task temp files;
- no commit and no push.

## 20. Candidate path and raw identity

Candidate path:

reports/longterm_o0c_recurrent_state_initialization_corrected_preflight_execution_authority_spec_candidate.md

Canonical candidate payload raw identity, computed after writing with this self-referential Section 20 identity block excluded:

- SHA256: ae5b5ce1a30752ab11b52b8faba2e733e8385417946d249f3eb587ef9d158380
- Bytes: 14725
- LF: 311
- CR: 0
- Final LF: true

## 21. Discrepancies and blockers

No authority, commit, implementation-scope, raw Git-object, or exact run-name collision discrepancy was identified during authoring checks.

The pre-existing cache-directory permission warnings are environmental inspection warnings only; they did not produce Git status entries and were not modified. The external run-registry path was absent or inaccessible, so that registry could not provide positive evidence beyond the absence of a discoverable v4 registration. Any independent verifier that cannot resolve that registry consistently must fail closed before freeze.

Runtime execution, artifact/provenance validity, and scientific O0c results remain unestablished by design.

## 22. Exact next authorized action

An independent verifier may now perform the checks in Section 18 and the post-authoring validation in Section 19. Only after independent verification passes and this candidate is formally frozen may a future controller generate the exact command, compute and freeze its raw identity, and consider the single v4 CPU-only preflight sequence in Section 15. This task itself ends before all execution, registration, collection, import, staging, commit, and push actions.
