# Longterm O0c Corrected Runtime-Source Provenance Preflight Execution Authority Spec Candidate

## 1. Verdict

`PASS_READY_FOR_REVERIFICATION`

Phase:

`REPORT_ONLY_CORRECTED_PREFLIGHT_EXECUTION_AUTHORITY_AUTHORING`

This candidate authorizes exactly one future CPU-only corrected O0c runtime-source provenance preflight run, after this authority is independently verified and frozen.

This authoring task does not execute the preflight, does not invoke Kaggle, does not register a run, does not collect/import a run, does not load a model/tokenizer/dataset, does not train/evaluate, does not modify implementation/tests, does not stage, does not commit, and does not push.

## 2. Starting And Final HEAD

Expected starting HEAD:

`c551747180ce1e8fe4eed5f7aa5ab6294cd89948`

Starting HEAD observed before authoring:

`c551747180ce1e8fe4eed5f7aa5ab6294cd89948`

Final HEAD observed after authoring validation:

`c551747180ce1e8fe4eed5f7aa5ab6294cd89948`

If final HEAD differs, this candidate is `BLOCKED`.

## 3. Authority Chain Inspected

Authority order used:

1. Current controller instruction for this task.
2. Primary frozen corrected implementation: `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`.
3. Supporting implementation authority: `387f0d2dda27ae0448fc7c0e66533306d9743e21`.
4. Supporting diagnostic interpretation: `3fc6fbbe9aed00594677c6f35029200ab055e883`.
5. Frozen original preflight implementation lineage: `de874a22df4f60adbdc5efbcf294961c7b3a48a5`.
6. Prior O0c preflight execution authority: `reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md`.
7. Command-transport recovery authority: `reports/longterm_o0c_runtime_source_provenance_preflight_command_transport_recovery_execution_authority_spec_candidate.md`.
8. PowerShell hash compatibility correction authority: `reports/longterm_o0c_runtime_source_provenance_preflight_command_transport_recovery_powershell_hash_compatibility_correction_authority_spec_candidate.md`.
9. Diagnostic execution authority: `reports/longterm_o0c_transformers_distribution_root_ambiguity_diagnostic_execution_authority_spec_candidate.md`.
10. `scripts/preflight_longterm_o0c_runtime_source_provenance.py` at `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`.
11. `AGENTS.md` and applicable workflow/Kaggle runbook constraints.

No contradictory higher-priority authority was found during authoring.

## 4. Corrected Implementation Identity

Corrected execution commit:

`c551747180ce1e8fe4eed5f7aa5ab6294cd89948`

Corrected script:

`scripts/preflight_longterm_o0c_runtime_source_provenance.py`

Authoritative corrected-preflight script identity:

- SHA256: `0930a55ad118588fe2928655683ef79ef5729bda3cb54f01025685f54daba578`
- bytes: `31981`
- LF count: `830`
- CR count: `0`
- final LF: `true`

This identity was computed from raw Git-object bytes for:

`c551747180ce1e8fe4eed5f7aa5ab6294cd89948:scripts/preflight_longterm_o0c_runtime_source_provenance.py`

The authoritative corrected-preflight script identity is the raw blob content committed at `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`, not whichever line-ending representation a local checkout happens to materialize.

Windows CRLF checkout facts are non-authoritative authoring-environment facts. They must not be used as the expected Kaggle script identity. A platform checkout conversion must never silently redefine the frozen source identity.

The inspected corrected source uses uniform distribution path-depth root derivation:

```python
root = located.parents[len(parts) - 2]
```

This replaces the frozen original nested `__init__.py` special case. Local Python 3.13.x unit-test/code-correctness evidence must not be substituted for the required later Kaggle runtime evidence.

## 5. Separation Of Evidence

Preserved separation:

- code correctness: already `PASS` locally under the corrected implementation lineage;
- corrected runtime execution: `NOT_YET_ESTABLISHED`;
- corrected runtime artifact/provenance validity: `NOT_YET_ESTABLISHED`;
- scientific conclusion: `NONE`.

The future run is infrastructure/provenance validation only.

It does not authorize:

- scientific O0c execution;
- training/evaluation;
- model/tokenizer/dataset loading;
- implementation/test changes;
- scientific interpretation.

## 6. Reserved Run Name And Collision Check

Exact reserved run name:

`longterm-o0c-runtime-source-provenance-preflight-c551747-v3`

Repository text search and local `.contramamba` run-registration/import inspection found no consumed or registered occurrence of this exact run name at authoring time.

If independent verification or later registration finds this exact name already consumed or registered incompatibly:

`BLOCK`

Do not silently choose another name.

## 7. Consumed-Run Preservation

The following consumed runs must not be rerun, reused, overwritten, deleted, aliased, or reinterpreted:

- `longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`
- `longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1`

Observed consumed-run evidence includes:

- `longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`: execution commit `de874a22df4f60adbdc5efbcf294961c7b3a48a5`, command SHA256 `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`, exit code `2`, status `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS`, blocker `distribution roots`.
- `longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1`: execution commit `de874a22df4f60adbdc5efbcf294961c7b3a48a5`, command SHA256 `5d4fd534e1d0dd96ea02a1393a43aa4f0379219301e550632d549040b39e2fc1`, exit code `0`, diagnostic classification later interpreted as `VALIDATOR_DISTRIBUTION_ROOT_DERIVATION_FALSE_AMBIGUITY`.

The prior `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS` result must not be declared resolved at runtime from local tests alone.

## 8. Execution Commit And Tree State

The later execution commit must be exactly:

`c551747180ce1e8fe4eed5f7aa5ab6294cd89948`

The later execution tree must be clean before command registration and before preflight execution:

- exact HEAD match required;
- no dirty tracked files;
- nothing staged;
- no execution from a different commit;
- no reuse of any `de874a22`-pinned command.

Any mismatch is a fail-closed blocker and does not authorize repair.

## 9. Runtime / CPU / GPU Boundary

Intended runtime:

- Python `3.12.13`
- NumPy `2.0.2`
- torch `2.10.0+cpu`
- Transformers `5.0.0`
- CPU only
- GPU OFF

Kaggle accelerator must be `None` before and during this CPU-only preflight. The command and wrapper evidence must fail closed on runtime mismatch or GPU exposure. No package install, uninstall, upgrade, downgrade, optional-kernel enablement, CUDA enablement, or environment repair is authorized.

## 10. Command Construction And Byte-Identity Contract

The later command must be a new `c551747` command identity, not the old `de874a22` command identity.

Preserved identity separation:

- Git commit/tree/blob identity;
- raw script-byte identity;
- command raw-byte identity.

These are separate provenance facts. The Git blob SHA must not be conflated with the SHA256 of the raw script content.

It must:

- pin `c551747180ce1e8fe4eed5f7aa5ab6294cd89948` exactly;
- execute `scripts/preflight_longterm_o0c_runtime_source_provenance.py` from that exact commit;
- verify exact HEAD and a clean tree;
- verify deterministic command raw-byte identity before `cm run save`;
- verify corrected script raw-byte SHA256, byte count, LF count, CR count, and final-LF fact against the frozen Git-object identity;
- preserve CPU-only behavior and GPU OFF checks;
- preserve runtime-version checks;
- preserve all frozen preflight fail-closed semantics;
- never load model/tokenizer/dataset;
- never train/evaluate;
- use the existing output schema only;
- avoid static `[SHA256]::HashData` dependency in any PowerShell registration helper;
- use `SHA256.Create().ComputeHash(...)` if PowerShell hashing is needed locally;
- preserve LF/no-CR command-byte identity unless a directly inspected frozen authority supersedes it.

Do not materialize `command.sh` during authority authoring. This candidate did not materialize `command.sh`.

The later exact generated shell command must include the same semantic guards as the prior frozen command, updated only for:

- run name `longterm-o0c-runtime-source-provenance-preflight-c551747-v3`;
- execution commit `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`;
- corrected script SHA256 `0930a55ad118588fe2928655683ef79ef5729bda3cb54f01025685f54daba578`;
- corrected script byte count `31981`;
- corrected script LF count `830`;
- corrected script CR count `0`;
- corrected script final-LF fact `true`;
- a new command SHA256 computed from the exact generated bytes.

If execution checkout bytes differ from the frozen Git-object identity above, execution must fail closed with a script identity mismatch rather than normalize, rewrite, or accept alternate bytes.

Any command hash/byte mismatch blocks.

## 11. Allowed Execution Outcomes

Do not predeclare the corrected runtime result.

Allowed later outcomes include:

- `PASS_SOURCE_IDENTITY_FROZEN`;
- any existing frozen fail-closed blocker emitted by the corrected preflight;
- command/wrapper fail-closed blockers for HEAD, dirty tree, script hash, command identity, runtime, GPU, collector/import, or provenance mismatch.

The later run must not claim `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS` is resolved at runtime solely from local tests.

## 12. Artifact And Provenance Contract

Later provenance capture must include:

- run name;
- execution commit;
- command SHA256;
- command bytes;
- LF count;
- CR count;
- final-LF fact;
- corrected script SHA256;
- runtime versions;
- start UTC;
- finish UTC;
- exit code;
- `run.log`;
- run metadata;
- preflight JSON if and only if the frozen corrected script publishes it;
- collector result;
- handoff ZIP SHA256;
- import audit/provenance identity.

If the corrected script blocks before JSON publication, absence of JSON must be interpreted according to frozen script semantics rather than automatically as collector/import corruption.

The existing output schema remains:

`o0c_runtime_source_provenance_preflight_v1`

The existing canonical JSON output path remains:

`reports/longterm_o0c_runtime_source_provenance_preflight.json`

No schema expansion, manual JSON synthesis, overwrite, append, timestamped fallback, or random fallback is authorized.

## 13. Later Kaggle Run / Collect / Import Sequence

Only after this authority itself is independently verified and frozen, the later sequence is:

```text
cm kaggle
```

Then paste/run the exact generated shell command only.

Then:

```text
cm run save longterm-o0c-runtime-source-provenance-preflight-c551747-v3
cm run longterm-o0c-runtime-source-provenance-preflight-c551747-v3
```

After execution, when appropriate:

```text
cm collect longterm-o0c-runtime-source-provenance-preflight-c551747-v3
```

Then run the Kaggle collector, download the handoff ZIP, and import:

```text
cm import <handoff.zip>
```

No hash, HEAD, dirty-tree, runtime, GPU, collector, import, or provenance blocker may be bypassed.

## 14. Failure Recovery

Stop rather than patch, improvise, rerun, or reinterpret on:

- HEAD mismatch;
- dirty tree;
- run-name collision;
- command hash/byte mismatch;
- script hash mismatch;
- runtime mismatch;
- GPU exposure;
- provenance mismatch;
- unexpected artifact schema;
- collector/import commit mismatch;
- any new fail-closed preflight blocker.

This execution authority does not authorize implementation repair.

## 15. Scientific Non-Authorization

Training/evaluation allowed:

`NO`

Scientific execution allowed:

`NO`

Corrected preflight execution during this task:

`NO`

Kaggle during this task:

`NO`

Commit/push:

`NO`

This candidate is not evidence for an O0c scientific conclusion and must not be used for model selection, threshold tuning, candidate selection, promotion, or scientific interpretation.

## 16. Candidate Path And Raw Identity

Candidate path:

`reports/longterm_o0c_runtime_source_provenance_corrected_preflight_execution_authority_spec_candidate.md`

Raw identity must be computed after final authoring edits:

- SHA256;
- bytes;
- LF count;
- CR count;
- final-LF fact.

The final raw identity is reported by the authoring agent after validation, rather than embedded here, to avoid self-referential hash churn.

## 17. Validation For This Authoring Task

Required validation commands for this authoring task:

```powershell
git diff --check
git diff --name-status
git diff --cached --name-status
git status --short
```

Required state:

- HEAD unchanged at `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`;
- no tracked modifications;
- nothing staged;
- exactly one task-attributable new untracked report:
  `reports/longterm_o0c_runtime_source_provenance_corrected_preflight_execution_authority_spec_candidate.md`.

Observed validation after authoring:

- `git diff --check`: passed with no output;
- `git diff --name-status`: no tracked modifications;
- `git diff --cached --name-status`: no staged changes;
- `git status --short`: exactly `?? reports/longterm_o0c_runtime_source_provenance_corrected_preflight_execution_authority_spec_candidate.md`.

## 18. Explicit Non-Execution Attestation

NO PREFLIGHT EXECUTION.

NO KAGGLE.

NO `cm run save`.

NO `cm run`.

NO `cm collect`.

NO `cm import`.

NO MODEL TOKENIZER LOADING.

NO TOKENIZER INVOCATION.

NO PRETRAINED MODEL LOADING.

NO MODEL FORWARD.

NO DATASET LOADING.

NO GENERATION.

NO TRAINING.

NO EVALUATION.

NO PACKAGE INSTALL.

NO PACKAGE UNINSTALL.

NO PACKAGE UPGRADE OR DOWNGRADE.

NO OPTIONAL KERNEL ENABLEMENT.

NO ENVIRONMENT MUTATION.

NO IMPLEMENTATION.

NO TEST MODIFICATION.

NO STAGING.

NO COMMIT.

NO PUSH.

## 19. Discrepancies

No blocking discrepancy was found during correction authoring.

The prior candidate incorrectly recorded the Windows checked-out CRLF script identity as normative:

- SHA256: `e921000647add11c493167aab86f29854f00994a6dbdab7d40f283a29646636b`
- bytes: `32811`
- LF count: `830`
- CR count: `830`
- final LF: `true`

That CRLF checkout identity is a non-authoritative authoring-environment fact only. It is not the frozen execution-source identity and must not be used as the expected Kaggle script identity.

## 20. Next Authorized Action

Independent verification of this candidate's exact bytes, authority sufficiency, collision check, and git state.

Only after independent verification and controller activation may the corrected preflight run be registered/executed under:

`longterm-o0c-runtime-source-provenance-preflight-c551747-v3`
