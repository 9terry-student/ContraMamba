# Gen5 Native Recurrent Generator-Anchor Pair Localization — Bounded Execution Gate

Status: `EXECUTION_GATE_CANDIDATE_READY_FOR_FREEZE`

This is a **single-stage, recurrence-only scientific execution gate**. It becomes operative only when this exact file is committed on `gen5-causal-role-state-ownership`, the code identities below still match, and the new commit's pinned Kaggle authentication/preflight succeeds. It is not evidence that the scientific run succeeded.

## 1. Authority and fixed implementation

- Scientific design: `reports/native_recurrent_generator_anchor_pair_localization_design_candidate.md`, frozen at `55b8784aa154d766cd189f60b2919451720598de`, Git blob `39690866075d65366db5e7c631e8c0e2415f7e16`.
- Validated serialization-region evidence freeze: `ee61ac1f70816d7150ab9dc5af57a4b9f0714515`.
- Exact corrected implementation base commit: `e5f4c8635e922a88140a03df4bb5a13cc49f2d26`, on branch `gen5-causal-role-state-ownership`.
- Audit script: `scripts/audit_native_recurrent_generator_anchor_pair_localization.py`, Git blob at implementation base `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`, observed working-file SHA256 after correction `cc1d3e1ed6b6288084c16fd578d40c7a8da99d91a2b8ea194aa99d6c08ae29c7`.
- Tests: `tests/test_native_recurrent_generator_anchor_pair_localization.py`, Git blob at implementation base `26c90ada9a74a2ac2716c880e58d15df5bd01704`, observed working-file SHA256 after correction `95310cecc1f2b3402e6d1db1b50e973f915f461ad51f7f2bb9f59e3d3d8b9afe`.
- The only arithmetic correction versus `f17c4a3660d4557cbdbd20a3c71b6e6015a65edf` is promoting complex FFT coefficients to `complex128` **before** the fine class-pair spectral contraction. The recurrence estimand, pair definitions, normalization and frozen tolerance tests remain unchanged.

Do not alter either implementation file, the validated coarse reducer, data, splits, seeds, tokenizer/encoding, model/checkpoint, frozen projector, tolerance criteria, or output schemas under this gate. If any identity drifts, stop and obtain a new review/gate rather than treating this candidate as transferable authority.

## 2. Completed evidence, separated by kind

### Static code tests

User-supplied local outputs at `e5f4c86`'s immediately preceding working tree, then committed without further changes:

- `24 passed in 4.26s` in `tests/test_native_recurrent_generator_anchor_pair_localization.py`;
- `GEN5_NATIVE_RECURRENT_GENERATOR_ANCHOR_STATIC_CONTRACT_PASS`;
- `git diff --check` and `cm ship` PASS; the resulting commit was pushed and clean (`BEHIND=0`, `AHEAD=0`, `OTHER_CHANGES=0`).

### Pinned Kaggle static synchronization

User-supplied fresh static outputs at `e5f4c8635e922a88140a03df4bb5a13cc49f2d26`:

- `GEN5_E5F4C86_KAGGLE_STATIC_SYNC_PASS`;
- active fine classes: `A_TITLE,A_NAME,A_ROLE,A_PREDICATE,RESIDUAL`;
- 3 parent pairs; 75 fine cells; 840 dev rows; 60,094 valid tokens;
- CUDA/scientific training not executed, Kaggle worktree clean.

### Post-correction CUDA numerical and memory regression

User-supplied outputs of `GEN5_E5F4C86_CUDA_NUMERICAL_REGRESSION_PASS` using the same external v2 CUDA gate helper (SHA256 `76075e61a0c90f8a74d4c476d13c9a66f84243e4914f4e53b2706be03a115d20`), pinned to `e5f4c8635e922a88140a03df4bb5a13cc49f2d26`:

| Measurement | Worker 0 / physical T4 0 | Worker 1 / physical T4 1 |
| --- | ---: | ---: |
| Anchor partition rows | 416 | 424 |
| Batch / k-count / state width | 32 / 4 / 24,576 | 32 / 4 / 24,576 |
| Expected source-gradient forwards | 117 | 126 |
| Stress fine-parent maximum absolute reconstruction error | `7.80035244458339e-08` | `7.558307553726088e-08` |
| CUDA peak allocated bytes | `5762983424` | `5762983424` |
| CUDA peak reserved bytes | `6159335424` | `6159335424` |
| Total GPU memory bytes | `15636037632` | `15636037632` |
| Fast path / large-span fallback | PASS / PASS | PASS / PASS |

The worker process reports logical `cuda:0` after `CUDA_VISIBLE_DEVICES` remapping; physical device assignment is established by the parent shell's `GPU=0`/`GPU=1` mapping. No scientific model forward, optimizer or training was executed by this stress. This is evidence for the numerical/memory **gate geometry**, not proof of full native scientific-runtime memory sufficiency or scientific outcome. Very large reported `max_rel` values arise on near-zero comparison entries; the frozen reconstruction contract is a per-entry **absolute OR relative** criterion. Do not convert the reported global maxima into a new tolerance rule.

### Retained limitations and failed synthetic diagnostics

- At `f17c4a3`, uniform-positive synthetic stress failed GPU1 fine-parent replay with absolute error `2.0195381391224787e-05`; the **same v2 helper** subsequently passed on both GPUs after the `e5f4c86` precision correction.
- At `f17c4a3`, a separate signed, random synthetic component (`Normal(0,0.1)`, seed 1903) failed **upstream pair-gap cross-term replay**, with absolute error `1.277834607015393` and relative error `0.00017229821150320295`. That stress has **not** been resolved or requalified and is not evidence that the fine reducer itself failed. Do not conceal the failure or claim arbitrary-input numerical robustness.
- A full native batch with actual projected components and all 243 source-gradient forwards has **not yet** run at this commit. All native per-batch replay guards must remain fail-closed; a failure stops the scientific run and cannot be bypassed by input rescaling, amplitude reduction, increased tolerances, or dropping shards/cells.

## 3. Narrow scientific run permitted after freezing this gate

Only one original scientific question is in scope: exact generator-anchor localization of frozen seed180 Phase3A recurrent interference. This is descriptive structural localization, not token intervention/causal identification.

- Execution implementation: the **unchanged** corrected audit script above.
- Population: the original 840-row, 60,094-token dev population, dev order SHA256 `b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25`, active encoding SHA256 `e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51`.
- Provenance: frozen model/tokenizer `state-spaces/mamba-130m-hf` revision `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`; seed180 selected checkpoint SHA256 `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`. Preserve the previously authenticated runtime binary identities; authenticate them again if session/runtime is reset.
- Fixed topology: **two independent single-GPU Tesla T4 workers**, `CUDA_VISIBLE_DEVICES=0` and `1`, no DDP.
- Fixed shard ranges: worker0 `[0,416)`, worker1 `[416,840)`; no overlap/omission.
- Fixed batch rows: `32`; worker batches `13` and `14`.
- Fixed source-gradient forwards: worker0 `117`, worker1 `126`, total `243`.
- Fixed group decompositions: worker0 `234`, worker1 `252`, total `486`.
- Fixed fine cells: exactly `3 coarse parents x 25 fine anchor pairs = 75`, both visible and complement components. Frozen windows `gap1_8`, `gap9_16`, `gap17_32`, `gap33_64`, `gap65_127`.
- Zero downstream transport stages, zero final-head transports, zero training/optimizer/checkpoint mutation, no confirmatory seeds `9601/9900`, no VitaminC, no text-decoding based label changes.
- Preserve the exact arithmetic/replay guards, all 75 cells, and coarse/fine reconstruction.

## 4. Activation, runtime and artifact gate

1. Commit **only this execution-gate report** to `gen5-causal-role-state-ownership` after clean/staged-scope review; the resulting *new* full commit SHA is the exact execution HEAD `H`. Verify the script/test Git blobs at `H` are identical to the fixed blobs in section 1. Do not guess `H` in advance.
2. Fresh authenticate Kaggle checkout at `H`, clean worktree, provisioned runtime, exact tokenizer/model/checkpoint/kernel binary identity; rerun static contract and CUDA preflight at `H`. A previous static/preflight at `e5f4c86` does not substitute for HEAD authentication at `H`.
3. Record the **same** two-GPU numerical stress contract and memory headroom at the new HEAD. Because the gate commit is report-only and leaves the scientific code blobs unchanged, a read-only reauthentication of the exact helper/code hashes is required; if runtime/binaries changed, redo the CUDA numerical gate rather than importing old measurements.
4. Choose a fresh, previously unused run name and **single repo-relative dedicated artifact root** before execution. Set `scratch_root=<artifact_root>/scratch` and `output_root=<artifact_root>/results` as siblings (neither may preexist). The entire root is the collection boundary. Refuse dirty repository, stale paths, mismatched command SHA, or ambiguous provenance.
5. Generate/run the exact command with `--run --expected-head H --implementation-freeze-commit H --model-snapshot <frozen> --tokenizer-snapshot <frozen> --checkpoint <frozen> --scratch-root <new_root>/scratch --output-root <new_root>/results`. Use `cm run save <new-run-name>` then `cm run <new-run-name>`; do not reuse a prior-commit run.
6. Preserve completed per-worker JSON and `.sha256` sidecars. If worker or merge fails, retain artifacts; do not delete/restart blindly. Recompute only missing/invalid shards under a separately justified recovery action. Check row coverage, hashes, active encoding, coarse validated replay, fine-parent replay, all counts and completeness before promoting evidence.
7. After successful GPU work, GPU OFF, `cm collect <run-name>`, Kaggle collector, download handoff ZIP, `cm import <handoff.zip>`. Require exact commit/command/root/hash/provenance consistency. Scientific interpretation requires imported validated evidence, not merely a PASS status line.

## 5. Stop conditions

Stop on any mismatch of authority/design/frozen artifacts, implementation blobs, runtime/model/tokenizer/checkpoint/kernel hashes, branch/HEAD, clean worktree, 2x T4 mapping, BATCH_ROWS=32, source-gradient forwards/shard coverage, frozen reconstruction guard, GPU memory, artifact root or handoff chain.

This gate authorizes **only** the bounded recurrence-only execution above once committed and authenticated. It does not authorize further implementation changes, training, classifier evaluation, new populations, post-hoc label/anchor changes, hyperparameter sweeps, or a scientific conclusion without validated imported artifacts.
