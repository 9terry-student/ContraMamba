# Gen5 v10 — Stage E0 Source-Bound Worker Adapter Contract

**Status:** `STAGE_E0_STATIC_SOURCE_BINDING_CANDIDATE / CPU_ONLY / REAL_WORKER_NOT_EXECUTABLE / NO_GPU_GATE`

**Frozen predecessor HEAD:** `59bf37f426ccc1620a66be21e131b3a3d6b4fdd8` on branch `gen5-causal-role-state-ownership`, worktree `C:\g5-state`. The reported 3-file D4 freeze/push was independently confirmed from the remote commit. **This scope is not an execution authority.**

## Objective and non-repetition boundary

D4 proved only that 17 injected CPU test callbacks, with synthetic-width tensors, can be called and value-hashed in order. It did not inspect the actual original worker's source-level callsite nesting or establish that a future bounded worker will not accidentally enter the historical 840-row/36-orientation execution loop. Another self-asserted D3 event/metadata contract does not advance scientific E0. Stage E0 therefore creates a **read-only source-bound adapter contract** that checks the original frozen worker from the exact committed Git blob and identifies the future legal cut points; it does not merely accept caller-supplied event labels.

E0 is **not** yet an executable real-worker adapter. Creating an actually executable bounded CUDA path will require an independently reviewed, separately versioned implementation and explicit execution gate after E0's original-blob integration check passes. Do not call the original `run_worker()`, `_spawn_worker()` or `run_full()`: they remain full-population/frozen dispatch paths.

## Exact candidate whitelist: only three new files

1. `reports/native_recurrent_generator_anchor_pair_localization_v10_stage_e0_source_bound_worker_adapter_scope.md` — this contract.
2. `scripts/gen5_v10_stage_e0_frozen_locus_adapter_candidate.py` — source-blob/AST integration validator, read-only `git show`, no import or execute of the original worker.
3. `tests/test_gen5_v10_stage_e0_frozen_locus_adapter_candidate.py` — deterministic CPU source fixtures and negative structural tests, no model/GPU calls.

Do not edit original worker, Stage A/B, C, D1–D4, hybrid/GEMM, checkpoint/model/tokenizer/data, other worktrees or prior reports. No Codex. No training/evaluation/Kaggle/GPU/forward/backward. No commit/push before local integration and independent review.

## Frozen sources and what AST can actually establish

The original source `scripts/audit_native_recurrent_generator_anchor_pair_localization.py` is pinned at Git blob `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`. Using **bytes returned by `git show HEAD:<path>`**, the reader independently checks the Git blob SHA1 and parses Python `run_worker(args)` without Python module import. It checks a single batch loop `enumerate(range(row_start,row_stop,BATCH_ROWS), start=1)`, a separate `FULL_FACTORIAL_CELLS` raw-write loop under that batch, a source gradient/projector loop, and the target-group loop under the source. It requires one each of the expected 13 frozen callsites in strict lexical order and structural nesting. It confirms `torch.cat((visible, complement), dim=0)`, exact seven-argument `_component_fine_pair_stats` invocation, and the original `RAW_RECONSTRUCTION_AUTH`, `VISIBLE_RAW_ENERGY_INTERNAL_REPLAY`, `COMPLEMENT_RAW_ENERGY_INTERNAL_REPLAY` guard literals after stats. The original `_validate_runtime_head()` is required before `_runtime_inputs()`.

AST identity, lexical order and nesting are **static source assertions**; they do not prove those functions ran, the original guard actually passed, or its module/model/checkpoint/tensors match runtime memory. Git blob SHA1 verifies *Git object identity*, not runtime provenance or a cryptographic attestation of loaded data. The local Git read path requires exact current HEAD/branch and no tracked changes. The three candidate files may be untracked for installation testing; that exception does not authorize unrelated changes.

The future bounded locus remains strictly worker 1, batch 13 (one-based), global dev rows `[800,832)`, source `(6201,6201)`, target group `(6201,6202)` then `(6201,6203)` in `CONTROL_R`, with K4/B32/S128/W24576. Stage E0 **must not** run the original batch/source/group loops. An eventual new adapter must bound the row/group **before any model/gradient work**, perform authentic nine raw writes, one selected source gradient/projector, one K4 group, three 17-key input captures, original frozen FFT plus v10 Hybrid, and independently replay all guards without monkeypatching frozen globals. It must bind to actual `FULL_FACTORIAL_CELLS`/`all_orientations()` semantics and independently authenticate code, checkpoint, tokenizer, dev order, active encoding, and physical GPU environment.

## Runtime artifact/memory linkage for later implementation, not E0 output

D2 established exact *logical* B32 17-tensor bytes `2,415,976,448` per capture and `7,247,929,344` across three captures. This does **not** bound peak GPU allocation. Future genuine instrumentation must record actual torch/CUDA allocator and device samples around model/context/raw writes, original FFT, Hybrid GEMM, and each copy/hash, including failure paths. No concurrent full GPU clones and no whole GPU-to-CPU materialization. The D1 immutable artifact writer cannot attest its own later publication inside its pre-published trace; any publication receipt belongs in a separate externally verifiable post-publish record. Do not convert claimed D3 memory samples into GPU readiness.

Preserve Stage C V3's strictly untrusted runtime mode and Stage D1's write-once artifact integrity semantics. E0 does not run them or manufacture V3 bundle records. Even a passing original-source AST check returns `E0=E1=E2=E3=NOT_EVALUATED`, `GPU_EXECUTION=SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`, `PRODUCTION=BLOCKED`. There is no Kaggle execution authority and no valid readiness threshold.

## CPU validation and mandatory real-source integration

The new module exposes **only** `--static-contract-check` and `--inspect-local-frozen-source`. Both require `CUDA_VISIBLE_DEVICES` empty or `-1`. The latter reads the frozen committed blob by Git, performs original-source AST checks, and prints `GEN5_V10_STAGE_E0_FROZEN_SOURCE_AST_PASS` only when all constraints hold. It performs no model import or GPU query.

Required local checks after exact ZIP installation on the frozen predecessor HEAD:

1. Exactly the new `TestE0SourceBoundCPU` cases must pass without skip.
2. `python -B -m scripts.gen5_v10_stage_e0_frozen_locus_adapter_candidate --static-contract-check` must pass.
3. `python -B -m scripts.gen5_v10_stage_e0_frozen_locus_adapter_candidate --inspect-local-frozen-source` must inspect the **actual original Git blob** and pass. This check was not possible in the independent synthetic-only test container and **cannot be claimed as verified before the user runs it**.
4. Compare ZIP source SHA256, whitespace/compile and exact three-file untracked scope. No staging/commit/push until local source inspection and independent review succeed.

Stop on any AST mismatch or missing callsite, unknown real frozen shape/loop, nonpinned commit, potential execution of original worker, required frozen-file edit, unbounded workload, unauthenticated lineage promoted to PASS, or missing evidence. **Do not relax the original AST verifier merely to get a green integration result**; inspect a failing source clause first.

**Disposition:** `D4=FROZEN_AT_59bf37f`; `E0=SOURCE_BOUND_STATIC_CANDIDATE_ONLY`; `CODEX=NO`; `GPU=OFF`; `TRAINING=NO`; `SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`; `COMMIT_PUSH=NO`.
