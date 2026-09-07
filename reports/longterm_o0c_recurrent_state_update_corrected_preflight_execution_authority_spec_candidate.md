# ContraMamba O0c Recurrent-State Update Corrected Preflight Execution Authority Spec Candidate

## 1. Verdict

`CANDIDATE_AUTHORED_PENDING_INDEPENDENT_VERIFICATION_AND_FREEZE`

Phase: `REPORT_ONLY_CORRECTED_PREFLIGHT_EXECUTION_AUTHORITY_AUTHORING`.

This candidate authorizes nothing during authoring. Only after independent verification and formal freeze, it may authorize exactly one future CPU-only, read-only corrected O0c runtime-source-provenance preflight at the exact frozen implementation commit specified below. It does not presuppose `PASS`.

## 2. Starting state

- Repository root: `C:\\o0c-preflight-exec-auth-686b745`.
- Required and observed HEAD: `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`.
- Tracked modifications: none.
- Staged paths: none.
- Untracked paths before authoring: none.
- Candidate path before authoring: absent.
- Task temporary files: none identified.

Any mismatch in root, HEAD, tracked state, staging state, untracked state, candidate absence, or task-temporary-file state blocks authoring without changes.

## 3. Authority chain

Authority is resolved in this order:

1. Current controller instruction.
2. Frozen corrected implementation authority: `944f8388a251a2827e2a17409a43678ac434772b`.
3. Frozen diagnostic interpretation: `3fcd581eb71add577815daf14e827992a3dc3c83`.
4. Frozen corrected implementation: `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`.
5. Repository `AGENTS.md`.

The controller instruction is the active authority. The diagnostic interpretation and parent implementation authority are frozen supporting authority only and do not broaden this execution candidate.

## 4. Frozen implementation identity

`IMPLEMENTATION_COMMIT=686b7457ed220e7d74ecdf41eb7ec500d24bb23f`

Parent implementation authority: `944f8388a251a2827e2a17409a43678ac434772b`.

Frozen Git-object identities, measured from the exact commit rather than CRLF-normalized checkout bytes, are:

| File | SHA256 | Bytes | LF | CR | Final LF | Git blob |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `scripts/preflight_longterm_o0c_runtime_source_provenance.py` | `5a7e9a8ddc4a0ae25c9f3c7304623100b082a331abf247f77d9b6610d155858d` | 32480 | 842 | 0 | true | `2dc72aaed31a1932e77ec93273918b83835a12e1` |
| `tests/test_preflight_longterm_o0c_runtime_source_provenance.py` | `a07c9db7057cac1770cba571187f7698331f3502b10cb97a93596920d7b6178e` | 38679 | 1003 | 0 | true | `a009e375cef8de0a3f3d93c400c09523fd24a513` |

Independent implementation verdict: `PASS_SAFE_TO_FREEZE_RECURRENT_STATE_UPDATE_VALIDATOR_CORRECTION_IMPLEMENTATION`.

Frozen targeted local suite result: `59 passed, 3 skipped`. This is already-established implementation evidence only; no suite is run by this report-authoring task.

## 5. Frozen correction summary

The correction is recorded narrowly:

- recurrence-loop discovery now uses `_same_lexical_scope_statements(slow.body)` instead of direct `slow.body` iteration;
- `_loop_body_update_and_readout` and dependency/readout semantics are unchanged;
- recurrent-state initialization semantics are unchanged;
- no source/version/hash/line/AST-path/Kaggle-path special casing was introduced.

This candidate does not reinterpret or expand that implementation claim.

## 6. Prior diagnostic provenance

Frozen diagnosis: `VALIDATOR_RECURRENT_STATE_UPDATE_PATTERN_FALSE_NEGATIVE`.

Validated consumed diagnostic:

- Run: `longterm-o0c-recurrent-state-update-diagnostic-426ecd2-v1`.
- Execution commit: `426ecd2038a1118d3afb9bd7bdae63f340e3c70b`.
- Command SHA256: `c8c5f9e6b2cd1a1d25d7c5841624f7b2ca68c901704a79115c9fcc9ecaeb9399`.
- Run-log SHA256: `6d6f5dc8cefbe3823677d7038609696e5fb21924c63c33b240ff88f5f10c9c62`.
- Run-meta SHA256: `64760ed45a6ecb0292f3ba130a0c596b6a39dd5c9a985c48d1c991437c250766`.
- Handoff-ZIP SHA256: `9d544d5c2698453b6d925ec6ff70cef6f98668ceec3ee4db350787911d89b930`.
- Scientific conclusion: `NONE`.

The diagnostic is permanently consumed and must not be rerun, reused, overwritten, or aliased.

## 7. Prior consumed preflight preservation

`longterm-o0c-runtime-source-provenance-preflight-426ecd2-v4` is permanently consumed. It ended `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED` with blocker `recurrent_state_update`.

It must not be rerun, reused, overwritten, aliased, or treated as the corrected run.

## 8. Reserved corrected run

The sole reserved corrected run name is:

`longterm-o0c-runtime-source-provenance-preflight-686b745-v5`

Before registration and before execution, collision checks must cover the run registry, imports, handoffs, and relevant repository records. If this exact name already exists in any of those locations, block and require a separate authority with a new run name; do not reuse it.

Authoring inspection found no exact v5 reference in repository records and no exact v5 entry under accessible local imports. That is not a substitute for the mandatory pre-registration and pre-execution collision checks.

## 9. Exact future execution commit

The future execution commit must be exactly `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`. No earlier, later, or alternate commit is authorized.

Future execution must fail closed on a HEAD mismatch, tracked modification, staged path, untracked execution-affecting file, or otherwise dirty worktree. No HEAD, hash, or dirty-worktree guard may be bypassed.

## 10. Result-neutral execution purpose

Authorize exactly one corrected runtime-source-provenance preflight to determine what the frozen validator reports after the recurrent-state-update discovery correction.

The execution may establish only the next preflight status and blocker. It establishes neither implementation correctness beyond the already frozen evidence nor any scientific result. Result-neutrality is mandatory: this authority does not expect or declare `PASS`.

## 11. Allowed outcomes

Without predeclaring which will occur, the future run may validly produce:

- A. preflight `PASS` according to the frozen validator;
- B. a different fail-closed blocker exposed after `recurrent_state_update` is successfully resolved;
- C. `recurrent_state_update` remains unresolved;
- D. required-symbol ambiguity;
- E. source/runtime/provenance identity mismatch;
- F. another existing frozen preflight blocker; or
- G. infrastructure/transport failure requiring separate recovery authority.

No outcome authorizes implementation repair, rerun, scientific interpretation, or authority expansion.

## 12. Runtime/source guards

Future execution must be CPU-only and independently verify the frozen expected environment:

| Component | Required value |
| --- | --- |
| Python | `3.12.13` |
| NumPy | `2.0.2` |
| torch | `2.10.0+cpu` |
| Transformers | `5.0.0` |
| CUDA available | `False` |
| CUDA device count | `0` |

Any environment mismatch fails closed. GPU use is forbidden.

The preflight must independently measure, rather than hard-code as an execution result, the current Transformers source identity. Historical comparison identity:

- Source: `/usr/local/lib/python3.12/dist-packages/transformers/models/mamba/modeling_mamba.py`.
- SHA256: `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83`.
- Bytes: `39500`; LF: `860`; CR: `0`; final LF: `true`.
- Distribution/import root: `/usr/local/lib/python3.12/dist-packages/transformers`.

If independently measured source/runtime identity differs, the frozen preflight logic must fail closed.

## 13. Execution boundary

The future run may execute only the frozen preflight CLI/script behavior at commit `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`.

It authorizes no code or test modification, package installation or mutation, dataset use, tokenizer use, scientific forward, training, evaluation, or diagnostic instrumentation. It may not load a model beyond whatever the frozen preflight itself performs; if the frozen preflight is designed not to load a model, that boundary is preserved.

## 14. Kaggle boundary

Kaggle is not authorized by this authoring task. Future Kaggle execution is permitted only after this candidate is independently verified, formally frozen, committed and pushed, and the exact remote commit is verified.

Then the future workflow requires `cm kaggle` and a fresh/appropriate Kaggle session at the exact frozen execution commit. Accelerator must be None and GPU must be OFF.

## 15. Command transport/provenance

Use the existing `cm` run workflow only:

- transport one exact command line;
- use `utf8-final-lf-v1` where applicable;
- freeze the command SHA256 before execution;
- use `cm run save <run-name>` followed by `cm run <run-name>`;
- require the runner to recompute the command SHA and refuse a mismatch; and
- require run metadata to bind the exact run name, commit, command identity, timestamps, exit code, and paths.

No HEAD, hash, or dirty-worktree blocker may be bypassed.

## 16. Collection/import

After execution, regardless of `PASS` or a fail-closed scientific-preflight status where the collector is valid, run:

`cm collect longterm-o0c-runtime-source-provenance-preflight-686b745-v5`

Run the collector in the same Kaggle session, download the handoff ZIP, then locally run `cm import <handoff.zip>`. Interpret the result only after `IMPORT PASS`.

## 17. Evidence-layer separation

| Layer | Status |
| --- | --- |
| A. Corrected implementation correctness | already `PASS` and frozen at `686b745...` |
| B. Corrected-preflight execution | `NOT_YET_EXECUTED` |
| C. Artifact/provenance validity | `NOT_YET_ESTABLISHED` |
| D. Resulting preflight status | `UNKNOWN` |
| E. Scientific conclusion | `NONE` |

These layers must remain distinct.

## 18. Scientific boundary

Even if the corrected preflight returns `PASS`, it alone establishes no O0c scientific claim. Scientific conclusion remains `NONE`. Any scientific execution or conclusion requires separate authority.

## 19. Failure recovery

If command transport, environment, package availability, source identity, provenance, Kaggle infrastructure, metadata, collector, or import fails, stop.

Do not rerun under the same consumed name unless frozen failure-recovery rules explicitly authorize that exact action. Prefer a separate recovery authority and run identity when required.

## 20. Git validation

After authoring, run only:

```text
git diff --check
git diff --name-status
git diff --cached --name-status
git status --short
git rev-parse HEAD
```

Required final state: HEAD `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`; no tracked modifications; nothing staged; exactly one untracked path, `reports/longterm_o0c_recurrent_state_update_corrected_preflight_execution_authority_spec_candidate.md`; and no task temporary files. Do not stage, commit, or push.

## 21. Candidate raw identity

After final Git validation, compute this exact candidate's raw SHA256, byte count, LF count, CR count, and final-LF flag. Record those values in the task result outside this self-referential report body. An independent verifier must reproduce them against this exact path.

## 22. Discrepancies/blockers

No authoring blocker was identified. Frozen Git-object identities match the required values. The Windows checkout's CRLF working-tree representation is not a discrepancy because the frozen raw identities are explicitly Git-object bytes and match exactly.

Corrected-preflight execution, artifact/provenance validity, resulting status, and scientific conclusion remain unestablished by design.

## 23. Exact next authorized action

An independent verifier may conduct read-only verification of the authority chain, frozen implementation and raw identities, frozen test result, consumed diagnostic and v4 preservation, v5 reservation/collision state, future execution guardrails, candidate raw identity, and clean Git state.

No preflight execution, Kaggle action, `cm run save`, `cm run`, `cm collect`, `cm import`, implementation, test modification, package mutation, model/tokenizer/dataset activity, forward, training, evaluation, staging, commit, or push is authorized until independent verification and formal freeze.
