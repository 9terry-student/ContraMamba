# ContraMamba O0c Recurrent-State Update Blocker Diagnostic Execution Authority Spec Candidate

## 1. Verdict

`CANDIDATE_AUTHORED_PENDING_INDEPENDENT_VERIFICATION_AND_FREEZE`

Phase: `REPORT_ONLY_RECURRENT_STATE_UPDATE_DIAGNOSTIC_EXECUTION_AUTHORITY_AUTHORING`.

After independent verification and formal freeze only, this candidate may authorize exactly one CPU-only, read-only diagnostic run. Its sole question is why the frozen corrected preflight at commit `426ecd2038a1118d3afb9bd7bdae63f340e3c70b` emitted `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED` with `blocker=recurrent_state_update`. This candidate does not execute that diagnostic, predeclare its result, authorize a repair, or authorize scientific execution.

## 2. Starting state

- Repository root: `C:\o0c-preflight-auth-c551747`
- Expected and observed HEAD: `51a4c450a9c2d7354df48d2e55fb6327899fdaeb`
- Tracked modifications before authoring: none
- Staged paths before authoring: none
- Untracked paths before authoring: none
- Candidate path before authoring: absent
- Task temporary files before authoring: none identified

`git status --short` emitted no entries. Git inspection emitted only permission warnings for pre-existing `.pytest_cache` and `pytest-cache-files-*` directories; those directories were not read as task artifacts or modified.

## 3. Authority chain

The current controller instruction is the active authority. The frozen supporting chain is:

1. Corrected-preflight execution authority: `51a4c450a9c2d7354df48d2e55fb6327899fdaeb`.
2. Corrected implementation: `426ecd2038a1118d3afb9bd7bdae63f340e3c70b`.
3. Recurrent-state initialization correction authority: `989708d011efe10fd72e32a6d91dd9d225f00673`.
4. Recurrent-state initialization interpretation: `bfc998f9206ff77fcc2b2e80bb94293dc737b13f` (`Freeze O0c recurrent-state initialization diagnostic interpretation`).
5. Prior recurrent-state initialization diagnostic authority: `ca112032841fe316e5e2e7335dc95d89aeedb450`.
6. Repository `AGENTS.md`.

The initialization interpretation is historical evidence for its own false-negative question only. It is not an authority to classify the separate recurrent-state-update blocker.

## 4. Validated v4 evidence

Consumed run: `longterm-o0c-runtime-source-provenance-preflight-426ecd2-v4`.

- Execution commit: `426ecd2038a1118d3afb9bd7bdae63f340e3c70b`
- Command SHA256: `0721e3161a17dea58a6be6eccfdaf9ff5db0388370c1004fa7f378d7b188d029`
- Runtime: Python `3.12.13`; NumPy `2.0.2`; torch `2.10.0+cpu`; Transformers `5.0.0`
- GPU/CUDA: `TORCH_CUDA_AVAILABLE=False`; `TORCH_CUDA_DEVICE_COUNT=0`
- Started UTC: `2026-09-07T06:12:40Z`
- Finished UTC: `2026-09-07T06:13:05Z`
- Exit code: `2`
- Validated runtime result: `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED`
- Blocker: `recurrent_state_update`
- Collector: `PASS`; `FILES_COLLECTED=0`
- Run-log SHA256: `e1eaf8288e5c9cc64b3289185c1855c404fa650a204f2690ea71646ce82d7353`
- Run-meta SHA256: `73bd59e80b84b67a92eebb3c50996e6809a3889cc46c1bebd448b7227f3cb671`
- Handoff ZIP SHA256: `aebc2618082211e1c3adbcbe1e6a4e6404af96c2d3013271c00992d85ceeea94`
- Import audit: `C:\Users\Home1\.contramamba\imports\longterm-o0c-runtime-source-provenance-preflight-426ecd2-v4_426ecd2038a1_20260907_151504`
- Import: `PASS`
- Artifact/provenance validity: `VALID`
- Scientific conclusion: `NONE`

The imported `import.json`, `manifest.json`, `run.meta`, and `run.log` reconcile these facts, including the exact commit, command hash, timestamps, exit code, zero collected files, and blocker.

## 5. Evidence-layer separation

The following are deliberately separate:

- Implementation correctness: `PASS` from the prior local suite and verifier.
- V4 execution: completed and reached `recurrent_state_update`.
- V4 provenance: `VALID` after collection and import.
- Root cause of the `recurrent_state_update` blocker: `NOT_YET_ESTABLISHED`.
- Implementation defect: `NOT_YET_AUTHORIZED_TO_DECLARE`.
- Scientific conclusion: `NONE`.

Prior tests do not establish this diagnostic conclusion.

## 6. Consumed-run preservation

V4 is permanently consumed. It must not be rerun, reused, overwritten, aliased, or reinterpreted as a diagnostic result beyond its actual blocker evidence. This candidate reserves a distinct future diagnostic name and does not convert the v4 blocker into a source or validator conclusion.

## 7. Diagnostic question

The one future diagnostic must answer:

> Why does the frozen corrected preflight at commit `426ecd2038a1118d3afb9bd7bdae63f340e3c70b` emit `blocker=recurrent_state_update` against the exact Kaggle Transformers `5.0.0` runtime source?

It must determine, without mutation:

- exact runtime source-file identity and exact `MambaMixer.slow_forward` span;
- direct `slow.body` statement structure;
- whether the relevant recurrence loop is direct-body or nested, its exact nesting path, and its sequential-loop location;
- recurrent-update assignment location and post-update readout location;
- replay result of frozen recurrence-loop/update discovery;
- whether direct-body candidate loops number zero, one, or multiple;
- whether an update exists but is unreachable to frozen direct-body discovery; and
- whether another frozen predicate or cardinality condition causes the blocker.

## 8. Diagnostic scope and non-goals

The future diagnostic is read-only source/AST inspection only. It must not modify the preflight implementation or tests; patch AST traversal; rewrite source; load a model, tokenizer, or dataset; invoke a model forward; train/evaluate; install, uninstall, or upgrade packages; enable GPU/CUDA or optional kernels; synthesize scientific evidence; declare a correction authority; or claim an exact repair.

No Kaggle action, run registration, preflight rerun, diagnostic execution, collection, import, implementation work, test change, package mutation, staging, commit, or push occurs in this authoring task.

## 9. Reserved run name and collision check

The sole reserved run name is:

`longterm-o0c-recurrent-state-update-diagnostic-426ecd2-v1`

Collision checks before authoring found no exact occurrence in repository text (excluding this not-yet-created candidate), no local `.contramamba` registry occurrence because no local registry exists, and no local import-audit occurrence under `C:\Users\Home1\.contramamba\imports`. A future collision in any registry, execution record, or import audit blocks execution; no substitute name may be chosen.

## 10. Exact execution commit

The future diagnostic execution repository commit must be exactly `426ecd2038a1118d3afb9bd7bdae63f340e3c70b`, the same frozen implementation commit that produced v4. It must not execute at authority commit `51a4c450a9c2d7354df48d2e55fb6327899fdaeb`.

Before future registration/execution, require exact HEAD, clean tracked tree, nothing staged, no execution-affecting untracked files, exact runtime version match, and GPU OFF.

## 11. Runtime, CPU, and GPU boundary

Freeze Python `3.12.13`, NumPy `2.0.2`, torch `2.10.0+cpu`, and Transformers `5.0.0`. Execution is CPU-only with Kaggle accelerator `None` / GPU OFF. No environment mutation is authorized. Any runtime mismatch or GPU exposure fails closed.

## 12. Runtime-source identity contract

The future diagnostic must compute and report raw-byte identity for the actual installed source it inspects:

- canonical path;
- SHA256;
- byte count;
- LF count;
- CR count;
- final-LF flag; and
- Transformers distribution version.

It must bind the inspected `MambaMixer.slow_forward` to that exact source identity. The historical Transformers `5.0.0` `modeling_mamba.py` SHA256 `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83` may be reported only as a comparison, not as a required outcome. If the current source differs, record the difference and stop interpretation until provenance is clear.

## 13. Frozen-validator replay contract

The diagnostic must independently reproduce enough of frozen `426ecd` logic to explain the blocker without changing source. At minimum it must inspect/replay `_find_unique_function(..., "MambaMixer.slow_forward")`, the direct-body recurrence-loop candidate search in `_recurrent_proof_nodes`, relevant update-binding logic, and `_loop_body_update_and_readout` only when a loop candidate is obtained.

It must label source structure, frozen-validator observation, and diagnostic interpretation separately. It must not broaden traversal and present that result as frozen-validator behavior. A diagnostic-only nested-scope enumeration is allowed solely to show extant source nodes and must be labeled diagnostic observation.

The prior initialization diagnostic observed the Transformers `5.0.0` cache branch, zero-state else initialization, training/pscan conditional, sequential recurrence under a nested `else`, recurrent update, post-update readout, and final cache-state copy. That evidence was frozen only for initialization and must not be used to simply declare `VALIDATOR_RECURRENT_STATE_UPDATE_PATTERN_FALSE_NEGATIVE`.

## 14. Allowed non-predeclared classifications

The future diagnostic may conclude only one evidence-supported classification from:

- `VALIDATOR_RECURRENT_STATE_UPDATE_PATTERN_FALSE_NEGATIVE`
- `SOURCE_RECURRENT_STATE_UPDATE_TRULY_UNRESOLVED`
- `VALIDATOR_RECURRENT_STATE_UPDATE_AMBIGUITY_CONFIRMED`
- `RUNTIME_SOURCE_IDENTITY_CHANGED_REQUIRES_REVIEW`
- `DIAGNOSTIC_INCONCLUSIVE`

It may propose a more precise non-scientific validator classification when evidence supports it, but may not convert that proposal into implementation authority.

## 15. Provenance contract

Future provenance must connect, without a broken link:

`run name -> exact 426ecd commit -> exact command identity -> exact runtime source identity -> runtime versions -> timestamps -> exit code -> diagnostic stdout/log -> metadata -> collector -> handoff ZIP SHA256 -> local import audit`.

Any broken link fails closed.

## 16. Future run, collect, and import sequence

Only after freeze and all future pre-execution guards pass, the authorized sequence is:

1. `cm kaggle`
2. Controller-provided exact diagnostic shell command.
3. `cm run save longterm-o0c-recurrent-state-update-diagnostic-426ecd2-v1`
4. `cm run longterm-o0c-recurrent-state-update-diagnostic-426ecd2-v1`
5. `cm collect longterm-o0c-recurrent-state-update-diagnostic-426ecd2-v1`
6. Kaggle collector to ZIP, then `cm import <handoff.zip>`.

This authoring task performs none of these steps and authorizes no bypass.

## 17. Failure recovery

Stop rather than repair, retry, or improvise on HEAD mismatch, dirty tree, run-name collision, command identity mismatch, runtime mismatch, GPU exposure, source-identity ambiguity, inability to bind `MambaMixer.slow_forward` uniquely, collector/import mismatch, provenance mismatch, or any unexpected diagnostic condition. No implementation work is authorized.

## 18. Scientific and implementation non-authorization

This is infrastructure/validator analysis only. It authorizes no O0c scientific execution, model inference, model selection, training, evaluation, scientific claim, implementation change, test change, source rewrite, or correction authority. Scientific conclusion remains `NONE`.

## 19. Independent verification requirement

Before freeze, an independent verifier must check v4 imported provenance facts; v4 consumed status; the exact diagnostic question; execution commit `426ecd`; runtime/CPU/GPU boundary; source-identity contract; frozen-validator replay separation; absence of a predeclared result; absence of implementation authorization; run-name collision; candidate bytes; and clean Git state.

## 20. Git validation

After writing, this authoring task must run only:

```text
git diff --check
git diff --name-status
git diff --cached --name-status
git status --short
git rev-parse HEAD
```

Required final state: HEAD `51a4c450a9c2d7354df48d2e55fb6327899fdaeb`; no tracked modifications; nothing staged; exactly one untracked file at this candidate path; and no task temp files. The candidate remains unstaged.

## 21. Candidate raw identity

Candidate path:

`reports/longterm_o0c_recurrent_state_update_diagnostic_execution_authority_spec_candidate.md`

Raw identity is computed after the final write and validation, separately from this non-self-referential candidate body: SHA256, bytes, LF, CR, and final-LF flag.

## 22. Discrepancies and blockers

No blocking discrepancy was found during authoring guards. The corrected immutable interpretation SHA resolved exactly. The only inspection warnings were the pre-existing inaccessible pytest-cache directories noted in Section 2; they generated no Git status entry and were not modified.

## 23. Exact next authorized action

An independent verifier may inspect this candidate, its raw identity, the v4 import audit, collision checks, and clean Git state. Only after independent verification passes and this candidate is formally frozen may a future controller provide and freeze the exact one-run diagnostic command for commit `426ecd2038a1118d3afb9bd7bdae63f340e3c70b`.
