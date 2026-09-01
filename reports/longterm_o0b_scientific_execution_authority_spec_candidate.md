# Long-Term O0b Scientific Execution Authority Spec Candidate

## Verdict

PASS_READY_FOR_INDEPENDENT_VERIFICATION

This candidate is an execution-authority candidate only. It does not itself authorize execution. It binds the frozen O0b observer implementation and validated v4 CPU package-preflight evidence to exactly one future provenance-pinned CPU-only O0b scientific observer attempt, subject to the activation rule below.

## Authority Chain

1. Current controller instruction.
2. O0b scientific-design authority: `df461469cb087f7f5db1e41a2b08e65ea517ad8a`.
3. O0b full-sequence offset boundary-recovery authority: `2ed4439e511f7534186cbd5df9110e45fdc1d66c`.
4. Repaired matched-control/input implementation: `7ce4e0cd05d87118c29526a53ab5178dc722db27`.
5. O0b observer implementation authority: `65881cf398d26b136e4984686b14f7d40b939c3e`.
6. Runtime-package provenance recovery authority: `27515b7cde33e02f992b093c70fec08d92e1b721`.
7. Exact-command/execution-provenance recovery authority: `67cc985963aa44df952978fd98b1ed18dfc9e13c`.
8. Final exact-command repaired observer implementation: `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`.
9. Frozen v3 package-preflight authority: `a0ee0a260369b99db160a117bef842ba6c0e945c`.
10. Frozen v4 invalid-torch-version recovery authority: `d479d602f868e65614d67968c429d89d91f4f878`.
11. Validated/imported v4 package-preflight evidence supplied by the controller.
12. Repository `AGENTS.md`.

Authority assessment: no conflict was identified among the controller instruction, upstream O0b authorities, repaired observer/input provenance, v4 package-preflight evidence, and repository-wide research-integrity rules. This file is authored in the final scientific execution-authority authoring phase only.

## Frozen Repository Identities

Expected authoring repository HEAD: `d479d602f868e65614d67968c429d89d91f4f878`.

Future scientific execution repository HEAD: `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`.

Observer script path: `scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`.

Observer script Git-object SHA256: `7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375`.

Observer script committed byte size: `45255`.

Final observer test SHA256: `cb2ad5723ea36ddc118b2cda3f6560057f4040e95100b3cb87c112e90373cea1`.

Dataset path: `data/longterm_o0b_matched_controls_v1.jsonl`.

Dataset SHA256: `75a675bee49cb26eb0935d364f0f5d090922dd01576dfc23294961b28394aec2`.

Validation artifact path: `reports/longterm_o0b_matched_controls_v1_validation.json`.

Validation artifact SHA256: `e8344ea3df54a3393aa8fa82dba19eb2baade9af9366687bb105f4ad348979ff`.

Validator SHA256: `e4b488c8f7a16a7004b27f0bd47e712785b9f9f9fe40def81cd9836e7d25ff67`.

Validation-test SHA256: `558f1f718d9c0024d18b46a5da91cf89a6a98a55b91e840615782243c13205e0`.

Model/tokenizer ID: `state-spaces/mamba-130m-hf`.

Immutable model/tokenizer revision: `5708daa364c50b880e7bd92eab456e0d34492ee9`.

## Frozen Runtime Package Evidence

v4 preflight authority freeze: `d479d602f868e65614d67968c429d89d91f4f878`.

Preflight run: `longterm-o0b-cpu-package-version-preflight-9a249c0-v4`.

Preflight execution HEAD: `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`.

Preflight exact command SHA256: `68d4b5b77b87bb2d531a67dee009b3383cd582c44f3e2e29328823dbac90e08b`.

Preflight exit code: `0`.

Preflight start UTC: `2026-09-01T17:19:44Z`.

Preflight finish UTC: `2026-09-01T17:20:05Z`.

Preflight run log SHA256: `3d47fd703cf68bb357093bd091537f79aeb5f98a05ce95c1d57c8ae5815c85a8`.

Preflight run meta SHA256: `be6af9d8bedb6bc5818a84e32976aaacfe7abe6afdae4764a3ca868d90a451f5`.

Imported handoff ZIP SHA256: `727a34d3c8c9e36b37ee3cf0474fa4a09e8dac43969a80209945abff9729d346`.

Imported artifact path: `reports/longterm_o0b_cpu_package_version_preflight_9a249c0_v4.json`.

Imported artifact SHA256: `89085153bf9e441f9a2538b5e3f4d421298698afe4e73e89a3a0707abceec827`.

Imported artifact bytes: `306`.

The imported preflight artifact is not consumed as a repository modification by this candidate.

The eventual scientific `manifest.json` must match all four runtime strings exactly:

- `python_version=3.12.13`
- `numpy_version=2.0.2`
- `torch_version=2.10.0+cpu`
- `transformers_version=5.0.0`

Any mismatch is provenance invalidity regardless of exit code, artifact completeness, or scientifically interesting-looking metrics.

## Scientific Run Identity

Run name: `longterm-o0b-token-aligned-native-mamba-state-dynamics-9a249c0-v1`.

Output directory: `reports/longterm_o0b_token_aligned_native_mamba_state_dynamics_9a249c0_v1`.

This is the first authorized O0b scientific observer attempt. Once actual wrapped execution begins, this run name is consumed. No rerun, retry, or resume is authorized under the same run name after any started failure or partial attempt. Any later retry requires separate recovery authority and a new run identity.

## Required Execution Semantics

The future run must preserve:

- `device=cpu`
- `dtype=float32`
- `model.eval()`
- frozen parameters
- `torch.inference_mode()`
- `add_special_tokens=False`
- tokenizer `use_fast=True`
- `trust_remote_code=False`
- `use_cache=False`
- exactly 12 full-sequence forwards
- one sequence per forward
- no padding
- `output_hidden_states=True`
- `return_dict=True`
- no generation
- no training
- no optimizer/backward
- no `cache_params` / A/B/C/Delta instrumentation
- hidden states are native pretrained Mamba hidden-state proxies, not selective-SSM recurrent state

The run must preserve all frozen pair IDs, condition semantics, exact divergence coordinates, anchor schedules, pre-divergence tolerance, metric definitions, deterministic artifact rules, and interpretation boundaries from upstream authorities.

No new thresholds, selected best layers, selected best anchors, learned scores, significance tests, or promotion rules are authorized.

## Exact Scientific Command

The following is the one complete future shell command. It is intended for the existing default cm registry workflow. Do not set or require `CONTRAMAMBA_RUN_COMMAND_BYTE_MODE=utf8-final-lf-v1`.

Normative extraction rule: the authority-defined Layer-1 command bytes are the UTF-8 bytes immediately after `b"BEGIN_EXACT_COMMAND\n"` and immediately before `b"\nEND_EXACT_COMMAND"`. The structural delimiter LF immediately before `END_EXACT_COMMAND` is not command data.

BEGIN_EXACT_COMMAND
python -c "import subprocess,sys,numpy,torch,transformers; expected={'python':'3.12.13','numpy':'2.0.2','torch':'2.10.0+cpu','transformers':'5.0.0'}; actual={'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'transformers':transformers.__version__}; bad={k:(expected[k],actual[k]) for k in expected if actual[k]!=expected[k]}; print('RUNTIME_VERSION_PRECHECK='+('PASS' if not bad else 'FAIL')); print('RUNTIME_VERSION_ACTUAL='+repr(actual)); sys.exit(70 if bad else subprocess.call([sys.executable,'scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py','--output-dir','reports/longterm_o0b_token_aligned_native_mamba_state_dynamics_9a249c0_v1','--run-name','longterm-o0b-token-aligned-native-mamba-state-dynamics-9a249c0-v1','--exact-command','[\"scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py\",\"--output-dir\",\"reports/longterm_o0b_token_aligned_native_mamba_state_dynamics_9a249c0_v1\",\"--run-name\",\"longterm-o0b-token-aligned-native-mamba-state-dynamics-9a249c0-v1\",\"--observer-implementation-commit\",\"9a249c071b76fbf693f63b36ba8ec1036c69b2ba\",\"--observer-script-sha256\",\"7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375\"]','--observer-implementation-commit','9a249c071b76fbf693f63b36ba8ec1036c69b2ba','--observer-script-sha256','7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375']))"
END_EXACT_COMMAND

Layer-1 command SHA256 over the normative extracted UTF-8 bytes with no final LF: `49a4e753d074bf353ffbff1677605d0078bd9a802182342e72cd512ae66c026b`.

Layer-1 command byte size: `1415`.

Layer-1 command CR count: `0`.

Layer-1 command LF count: `0`.

Layer-1 first byte: `112`.

Layer-1 final byte: `34`.

Layer-1 final LF present: `false`.

Layer-2 canonical observer argv payload:

```json
["scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py","--output-dir","reports/longterm_o0b_token_aligned_native_mamba_state_dynamics_9a249c0_v1","--run-name","longterm-o0b-token-aligned-native-mamba-state-dynamics-9a249c0-v1","--observer-implementation-commit","9a249c071b76fbf693f63b36ba8ec1036c69b2ba","--observer-script-sha256","7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375"]
```

The observer `--exact-command` payload is the compact canonical JSON array produced from the actual observer `sys.argv` after removing exactly the one `--exact-command <payload>` pair, retaining `sys.argv[0]`, and preserving every remaining argv element exactly and in order. It does not include `python`, `-u`, or the external pre-invocation guard in Layer 2.

## Runtime Version Guard

The external Layer-1 command performs a pure pre-invocation runtime package guard before observer invocation. It imports only `sys`, `numpy`, `torch`, and `transformers` to read version identity, compares exact strings to `3.12.13`, `2.0.2`, `2.10.0+cpu`, and `5.0.0`, prints the actual version dictionary, and exits with code `70` if any string differs.

Only after the equality check passes does the guard invoke the observer through `subprocess.call([sys.executable, ...])` with the exact observer argv listed above. The guard performs no package installation, no fallback inference, no environment mutation, no tokenizer execution, no tokenizer/model loading, no model download, and no scientific input change.

Kaggle runtime for the future scientific run must use CPU only / Accelerator None. Model/tokenizer network retrieval is permitted only as required to fetch the exact immutable `state-spaces/mamba-130m-hf` revision during the later authorized scientific execution. No package install/update is authorized.

## Required Scientific Artifacts

The future scientific run must produce exactly one complete bundle containing:

1. `manifest.json`
2. `anchor_observations.jsonl`
3. `anchor_hidden_states.npz`
4. `paired_distances.jsonl`
5. `summary.json`
6. `report.md`
7. `SHA256SUMS.txt`

`SHA256SUMS.txt` must cover every other required artifact exactly according to the frozen observer contract. No partial artifact set may support interpretation.

## Required Post-Run Gates

Before any scientific interpretation, collection/import evidence must establish at minimum:

- exact registered run name
- expected commit equals actual commit equals `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`
- exact Layer-1 command SHA256 `49a4e753d074bf353ffbff1677605d0078bd9a802182342e72cd512ae66c026b`
- exact observer implementation commit
- exact observer Git-object SHA256
- exact dataset SHA256
- exact validation artifact SHA256
- exact model/tokenizer ID
- exact immutable model/tokenizer revision
- CPU / float32
- exact four runtime versions
- exact canonical Layer-2 observer argv
- exact run name
- `execution_status=COMPLETE`
- exact seven required artifacts
- `SHA256SUMS.txt` completeness
- cm wrapper log/meta/command provenance
- successful cm collect
- successful cm import

The authority distinguishes code correctness, execution success, artifact/provenance validity, and scientific conclusion. `CONTRAMAMBA RUN PASS` alone does not establish the scientific conclusion. No scientific interpretation is authorized until collection/import and all authority-bound provenance/artifact gates pass.

## Scientific Interpretation Boundary

Primary comparisons remain:

- A = `insufficient_matched` vs `reference_sufficient`
- B = `paraphrase_sufficient` vs `reference_sufficient`
- C = `surface_null_matched` vs `reference_sufficient`

No hard scientific PASS threshold is authorized. No favorable layer or anchor selection is authorized post hoc.

Permitted later interpretation is limited to whether A shows a consistently larger aligned-position response than B and C across the already-frozen pairs, layers, and anchors, together with the alternative/falsification interpretations already defined by the scientific-design authority.

No population, generalization, or significance claim is authorized. A pre-divergence invariant violation or provenance mismatch invalidates the run rather than becoming scientific signal.

## CM / External Tool Provenance

Live cm path inspected read-only: `C:\Users\Home1\.contramamba\cm.ps1`.

Current live `cm.ps1` SHA256: `b15d70832e7c76c05fea6a9955bd199edcf9fb633fe0fe34266c44788260f570`.

The current live hash matches the last independently verified identity supplied by the controller.

The inspected default `cm run save` semantics are compatible with this command: cm reads the clipboard with `Get-Clipboard -Raw`, applies `.Trim()`, rejects Markdown code fences, optionally strips and trims a leading `%%bash` cell marker, and hashes the resulting UTF-8 command string without appending a final LF when no command byte mode is set. The normative command bytes extracted by this candidate are byte-for-byte identical to the bytes the current default cm registry path would store. This candidate does not execute `cm run save`, `cm run`, `cm collect`, or `cm import`.

## Activation Rule

This candidate must not authorize execution merely by existing. Activation requires all five:

1. Independent verifier PASS over exact candidate bytes and command.
2. Candidate committed/pushed unchanged.
3. Controller records full authority freeze commit and committed candidate Git-object identity.
4. Controller independently verifies exact scientific command bytes/SHA from committed Git object.
5. Explicit controller transition to scientific execution.

Only after all five may one O0b scientific attempt be registered/run.

## Non-Execution Attestation

For this authoring task:

- NO TOKENIZER EXECUTION
- NO MODEL DOWNLOAD
- NO MODEL LOAD
- NO MODEL WEIGHTS
- NO HIDDEN-STATE FORWARD
- NO DATASET REGENERATION
- NO TRAINING
- NO EVALUATION
- NO KAGGLE
- NO CM RUN SAVE
- NO CM RUN
- NO CM COLLECT
- NO CM IMPORT
- NO COMMIT
- NO PUSH
