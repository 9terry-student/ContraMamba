# ContraMamba O0b Scientific Runtime-Version Recovery Execution Authority Candidate

Status: CANDIDATE ONLY.

This document authorizes no execution by itself. It is a candidate authority for exactly one fresh O0b scientific recovery attempt using the repaired observer implementation at `44c5ba4f2204167f91c7f5564c6dbfcd82304035`, subject to the activation rule below.

## 1. Authority And Phase

Authority order:

1. Current controller instruction.
2. Frozen runtime-version recovery authority: `845a259f62d52e86b63a52b922706e0da06e0e3d`.
3. Frozen repaired observer implementation: `44c5ba4f2204167f91c7f5564c6dbfcd82304035`.
4. Frozen original O0b scientific execution authority: `49d3361aa96cd1aea958bd0e85f462811b92540c`.
5. Consumed failed scientific attempt: `longterm-o0b-token-aligned-native-mamba-state-dynamics-9a249c0-v1`.
6. Upstream O0b scientific-design / boundary / input / observer authority chain.
7. Repository `AGENTS.md`.

Phase: STATIC SCIENTIFIC RECOVERY EXECUTION-AUTHORITY AUTHORING ONLY.

Prohibitions for this authoring phase:

- No code edit.
- No test edit.
- No data edit.
- No tokenizer invocation.
- No model invocation.
- No forward pass.
- No training.
- No evaluation.
- No Kaggle invocation.
- No `cm kaggle`.
- No `cm run save`.
- No `cm run`.
- No `cm collect`.
- No `cm import`.
- No stage.
- No commit.
- No push.

## 2. Repository State Preconditions

Canonical repository: `C:\Users\Home1\Desktop\ContraMamba`

Expected HEAD:

`44c5ba4f2204167f91c7f5564c6dbfcd82304035`

Required validation commands before accepting this candidate for independent verification:

- `git rev-parse HEAD`
- `git diff --check`
- `git diff --name-status`
- `git diff --cached --name-status`
- `git status --short`

Expected task-attributable delta after authoring: exactly one untracked file:

`reports/longterm_o0b_scientific_runtime_version_recovery_execution_authority_spec_candidate.md`

Protected state:

- Do not touch `C:\o0b-scientific-v1`.
- Do not clean, reset, stash, delete, or otherwise alter protected unrelated state.

## 3. Corrected Repaired Implementation Identity

Implementation commit:

`44c5ba4f2204167f91c7f5564c6dbfcd82304035`

Observer path:

`scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`

Committed observer Git blob SHA1:

`e2a49bdcf59d8bcc148174167ac9f0fcb97d0dec`

Committed observer SHA256:

`fa4935a57baebf6b726b8e94c682afa7bcffc3d22b9cf6552a17eeb26ba3a63a`

Committed observer bytes: `45259`

Committed observer line-ending facts:

- CR: `0`
- LF: `442`
- FINAL_LF: `True`

Test path:

`tests/test_observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`

Committed test Git blob SHA1:

`b17d23187da49d066b9b1f128b94939dac0f82b5`

Committed test SHA256:

`e6cef994a9bae52c3e6d055f9fe1f8800369c7996c6b9fd24c86e4924e500049`

Committed test bytes: `40294`

Committed test line-ending facts:

- CR: `0`
- LF: `491`
- FINAL_LF: `True`

The earlier working-tree observer identity `1b001a8d51d69674932813a298af706470cd1ec0d2b3df61480a47a37cd1e90f` with `45674` bytes is not the committed execution identity and must not appear as the frozen `observer_script_sha256` for this execution authority.

Identity derivation requirement:

- Derive identities independently from `git cat-file blob <commit>:<path>`.
- Require exact match to the committed identities above.

Parent-to-implementation delta:

- In `validate_runtime_versions()`, exactly one semantic implementation change is frozen:
  - from `type(value) is str`
  - to `isinstance(value, str)`
- Exactly one synthetic str-subclass non-coercion regression test is frozen.
- No other semantic delta is permitted.

## 4. Consumed V1 Failure Boundary

Consumed run:

`longterm-o0b-token-aligned-native-mamba-state-dynamics-9a249c0-v1`

Execution HEAD:

`9a249c071b76fbf693f63b36ba8ec1036c69b2ba`

Layer-1 SHA256:

`49a4e753d074bf353ffbff1677605d0078bd9a802182342e72cd512ae66c026b`

Exit code: `1`

Runtime precheck: `PASS`

Actual versions:

- python: `3.12.13`
- numpy: `2.0.2`
- torch: `2.10.0+cpu`
- transformers: `5.0.0`

Failure:

`ContractError: runtime version torch_version`

FILES_COLLECTED: `0`

Failure handoff import: `PASS`

Boundary:

- v1 cannot be rerun, resumed, or reused.
- The old v1 run name cannot be reused.
- The old v1 output directory cannot be reused.
- The old Layer-1 command cannot be reused.
- No v1 in-memory numerical output is reusable.
- v1 establishes no scientific conclusion.

## 5. Fresh V2 Run Identity

RUN_NAME:

`longterm-o0b-token-aligned-native-mamba-state-dynamics-44c5ba4-v2`

OUTPUT_DIR:

`reports/longterm_o0b_token_aligned_native_mamba_state_dynamics_44c5ba4_v2`

This must be a complete fresh execution from scratch.

No state or artifact from v1 may be copied, resumed, seeded, or consumed.

## 6. Frozen Scientific Design Invariants

Model/tokenizer:

`state-spaces/mamba-130m-hf`

Revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

Execution/runtime invariants:

- CPU.
- `float32`.
- `model.eval()`.
- Frozen parameters.
- `torch.inference_mode()`.
- `add_special_tokens=False`.
- `use_fast=True`.
- `trust_remote_code=False`.
- `use_cache=False`.
- `output_hidden_states=True`.
- `return_dict=True`.
- Exactly 3 pair IDs.
- Exactly 4 conditions per pair.
- Exactly 12 full-sequence forwards.
- One sequence per forward.
- No padding.
- No generation.
- No optimizer.
- No backward.
- No `cache_params` instrumentation.
- No A/B/C/Delta instrumentation.

Hidden-state definition:

- Native pretrained Mamba hidden-state proxies only.
- Not selective-SSM recurrent state.

Comparisons:

- A = `insufficient_matched` vs `reference_sufficient`.
- B = `paraphrase_sufficient` vs `reference_sufficient`.
- C = `surface_null_matched` vs `reference_sufficient`.

Preserved analysis constraints:

- Pair-relative divergence coordinates.
- Frozen anchor semantics.
- Pre-divergence `rtol=0`.
- Pre-divergence `atol=1e-6`.
- Cosine redundancy `atol=1e-12`.
- Normalized L2/cosine definitions.
- Deterministic serialization.
- Exact seven-artifact bundle.
- No learned aggregate.
- No best-layer selection.
- No best-anchor selection.
- No hard scientific PASS threshold.
- No significance claim.
- No generalization claim.

## 7. Frozen Inputs

Dataset:

`data/longterm_o0b_matched_controls_v1.jsonl`

Dataset SHA256:

`75a675bee49cb26eb0935d364f0f5d090922dd01576dfc23294961b28394aec2`

Validation artifact:

`reports/longterm_o0b_matched_controls_v1_validation.json`

Validation artifact SHA256:

`e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff`

No regeneration is authorized.

## 8. Runtime Contract

Expected exact runtime strings:

- python: `3.12.13`
- numpy: `2.0.2`
- torch: `2.10.0+cpu`
- transformers: `5.0.0`

Layer-1 must perform a fail-closed runtime precheck before observer invocation.

Runtime guard requirements:

- No model/tokenizer import in guard.
- No install.
- No environment mutation.
- Exact string comparison.
- Print `RUNTIME_VERSION_PRECHECK=PASS` or `RUNTIME_VERSION_PRECHECK=FAIL`.
- Print actual version mapping.
- Exit `70` on mismatch.
- Invoke observer only after `PASS`.

## 9. Exact Layer-2 Canonical Argv

Freeze exactly:

```json
["scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py","--output-dir","reports/longterm_o0b_token_aligned_native_mamba_state_dynamics_44c5ba4_v2","--run-name","longterm-o0b-token-aligned-native-mamba-state-dynamics-44c5ba4-v2","--observer-implementation-commit","44c5ba4f2204167f91c7f5564c6dbfcd82304035","--observer-script-sha256","fa4935a57baebf6b726b8e94c682afa7bcffc3d22b9cf6552a17eeb26ba3a63a"]
```

Rules:

- Actual argv has exactly one `--exact-command <payload>` pair.
- Payload equals the JSON above exactly.
- Canonicalization removes exactly that pair.
- Preserve actual `sys.argv[0]`.
- Do not prepend `python` or `-u`.
- Preserve argument order.

## 10. Exact Layer-1 Command

The following command is a regenerated v2 command using the v1 exact-command authority only as structural template, with the corrected committed observer identity.

BEGIN_EXACT_COMMAND
python -c "import importlib.metadata as m,platform,subprocess,sys; expected={'python':'3.12.13','numpy':'2.0.2','torch':'2.10.0+cpu','transformers':'5.0.0'}; actual={'python':platform.python_version(),'numpy':m.version('numpy'),'torch':m.version('torch'),'transformers':m.version('transformers')}; print('RUNTIME_VERSION_ACTUAL='+repr(actual)); ok=actual==expected; print('RUNTIME_VERSION_PRECHECK='+('PASS' if ok else 'FAIL')); sys.exit(70 if not ok else subprocess.run([sys.executable,'scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py','--output-dir','reports/longterm_o0b_token_aligned_native_mamba_state_dynamics_44c5ba4_v2','--run-name','longterm-o0b-token-aligned-native-mamba-state-dynamics-44c5ba4-v2','--exact-command','[\"scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py\",\"--output-dir\",\"reports/longterm_o0b_token_aligned_native_mamba_state_dynamics_44c5ba4_v2\",\"--run-name\",\"longterm-o0b-token-aligned-native-mamba-state-dynamics-44c5ba4-v2\",\"--observer-implementation-commit\",\"44c5ba4f2204167f91c7f5564c6dbfcd82304035\",\"--observer-script-sha256\",\"fa4935a57baebf6b726b8e94c682afa7bcffc3d22b9cf6552a17eeb26ba3a63a\"]','--observer-implementation-commit','44c5ba4f2204167f91c7f5564c6dbfcd82304035','--observer-script-sha256','fa4935a57baebf6b726b8e94c682afa7bcffc3d22b9cf6552a17eeb26ba3a63a'],check=False).returncode)"
END_EXACT_COMMAND

Normative extraction:

- Start immediately after `b"BEGIN_EXACT_COMMAND\n"`.
- End immediately before `b"\nEND_EXACT_COMMAND"`.
- Delimiter LF is document structure, not command data.

Exact-command byte contract:

- SHA256: `1a76f1c14798c27db89a55aea1a32682964487bfe35ec74cbb614ef017f455bd`
- Byte count: `1393`
- CR count: `0`
- LF count: `0`
- FINAL_LF: `False`
- First byte: `0x70`
- Last byte: `0x22`

Default `cm run save` `.Trim()` must preserve bytes exactly.

Do not use `CONTRAMAMBA_RUN_COMMAND_BYTE_MODE=utf8-final-lf-v1`.

Layer-1 command requirements:

- It begins exactly with `python -c "`.
- It ends with the final closing double quote.
- Observer subprocess argv contains `sys.executable`.
- Observer subprocess argv contains `scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`.
- Observer subprocess argv contains the frozen v2 output directory.
- Observer subprocess argv contains the frozen v2 run name.
- Observer subprocess argv contains exactly one `--exact-command` payload equal to the frozen Layer-2 JSON.
- Observer subprocess argv contains `--observer-implementation-commit 44c5ba4f2204167f91c7f5564c6dbfcd82304035`.
- Observer subprocess argv contains `--observer-script-sha256 fa4935a57baebf6b726b8e94c682afa7bcffc3d22b9cf6552a17eeb26ba3a63a`.
- No shell redirection.
- No pipe.
- No install.
- No external chaining.
- No multiline command.

## 11. Exact Artifact Contract

Exactly seven artifacts:

1. `manifest.json`
2. `anchor_observations.jsonl`
3. `anchor_hidden_states.npz`
4. `paired_distances.jsonl`
5. `summary.json`
6. `report.md`
7. `SHA256SUMS.txt`

Manifest must bind:

- `observer_implementation_commit = 44c5ba4f2204167f91c7f5564c6dbfcd82304035`
- `observer_script_sha256 = fa4935a57baebf6b726b8e94c682afa7bcffc3d22b9cf6552a17eeb26ba3a63a`
- `run_name = longterm-o0b-token-aligned-native-mamba-state-dynamics-44c5ba4-v2`

The manifest must also bind all frozen fields required by the upstream scientific-design / boundary / input / observer authority chain.

## 12. Attempt Consumption Rule

Before pinned wrapper execution actually begins, v2 is not consumed.

Once the wrapper creates its start marker or begins wrapped execution, v2 is CONSUMED regardless of success or failure.

If started v2 fails:

- Do not rerun the same name.
- Do not overwrite the same output directory.
- Collect/import failure provenance where applicable.
- Any later attempt requires a new authority and fresh identity.

## 13. Interpretation Boundary

`CONTRAMAMBA RUN PASS` means execution success only.

Scientific interpretation requires a later controller transition after all of:

1. Collect PASS.
2. Import PASS.
3. Seven-artifact validation.
4. Hash/provenance validation.
5. Manifest validation.
6. Controller interpretation transition.

Keep separate:

- Code correctness.
- Execution success.
- Artifact-provenance validity.
- Scientific conclusion.

## 14. Activation Rule

This candidate itself authorizes no execution.

Before any Kaggle scientific run, all of the following are required:

1. Exact candidate byte identity.
2. Independent verifier PASS.
3. Commit/push unchanged.
4. Controller records committed authority plus Git-object identity.
5. Controller independently verifies committed exact Layer-1 bytes.
6. Controller explicitly transitions to scientific execution.

Until then: NO KAGGLE SCIENTIFIC RUN.

## 15. Stop Conditions

Block if:

- HEAD mismatches `44c5ba4f2204167f91c7f5564c6dbfcd82304035`.
- Committed identities mismatch the corrected values above.
- Broader implementation semantic delta exists.
- v1 name/output/command would be reused.
- Layer-1 cannot be byte-clean single-line.
- Scientific design broadens.
- More than one candidate file changes as task-attributable delta.
- Protected state needs alteration.

## 16. Non-Execution Attestation

This authoring candidate does not authorize and did not require model/tokenizer import, model execution, forward passes, training, evaluation, Kaggle, `cm run save`, `cm run`, `cm collect`, `cm import`, staging, committing, or pushing.
