# Gen5 v10 Stage E1 — One-Group Original-Helper Call Budget (CPU Only)

**Status:** `E1_CPU_INJECTED_HELPER_BUDGET_CANDIDATE / NO_LIVE_MODEL / NO_CUDA_GATE / NO_SCIENTIFIC_EXECUTION`

**Base:** `c3c75af0e232252c6e1c0ec0ff3f0d1544c3a8fd`; branch `gen5-causal-role-state-ownership`; worktree `C:\g5-state`. Prior E0 freeze was reported clean/pushed and independently seen on the remote. This scope is neither a real CUDA run permit nor scientific experiment authorization.

## Objective

E0 verified the frozen original worker's AST and exact Git blob, but only statically. Stage C had already implemented a dependency-injected CPU synthetic-width single-group sequence. E1 **does not reimplement Stage C's numerical paths or claim real model access**. Instead, it inserts a non-mutating, instrumented proxy around the existing Stage C callable and gives each upstream original-helper call a hard count and call-order budget. This is a meaningful early-stop protection against accidentally invoking the frozen 840-row / all-source loops in the **controlled CPU helper path**. It is not a sandbox: an untrusted callback may have side effects inside its body, including importing GPU libraries. Tests use controlled CPU fixtures only.

## Exact new-file scope

1. `reports/native_recurrent_generator_anchor_pair_localization_v10_stage_e1_bounded_cpu_dispatch_scope.md`
2. `scripts/gen5_v10_stage_e1_bounded_cpu_dispatch_candidate.py`
3. `tests/test_gen5_v10_stage_e1_bounded_cpu_dispatch_candidate.py`

**Immutable predecessors:** frozen FFT/original worker, Hybrid/GEMM, Stage A/B/C, D1–D4, E0 and their tests, model/checkpoint/tokenizer/dev row sources. Never edit an earlier stage, run Codex, load a real model, touch Kaggle, call CUDA, or create science/artifact output in this phase. No newly runnable `--run-bounded` or `run_worker` command.

## Fixed work and binding

One selected locus only: worker 1, one-based batch 13, global rows `[800,832)`, source `(6201,6201)`, ordered targets `(6201,6202)` and `(6201,6203)`, K4/B32/S128. The *future* real geometry is W24576, but **E1 explicitly rejects W24576**, allowing CPU synthetic widths only. `row_ids` must be 32 distinct strings; already-injected `active` and `targets` must contain at least 832 elements. `inspect_local_source` separately checks the exact base HEAD, branch, frozen committed Git blob and E0 AST before allowing a static source PASS. These checks do not authenticate loaded runtime model bytes or the user-supplied CPU fixture object.

The proxy wraps (without monkeypatching original objects): exactly one batch-features call, one anchor partition, one fine-mask slice, one context preparation, exactly nine factorial raw-write calls **before** one selected source gradient and projector, exactly two original orientation accumulators, one projected target group, and one frozen full reducer. The required budget contains **19 upstream helper invocations**. Upstream functions that return an exception produce no successful complete trace; any missing, out-of-order or extra intercepted call is blocked. Call budget covers the named helper boundaries, **not arbitrary internal work inside a helper body**. Stage C independently checks original mask/data inputs, input immutability over 3×17 tensor captures, genuine frozen + v10 reducer record structure, and original raw/projector guard CPU replay.

## Verification boundary

Differential CPU mock tests compare E1 and frozen Stage C on the same fixture: selected locus, exact Stage A input manifest, K4 frozen/v10 records and dispatch. Negative cases include wrong source order, bad batch capacity/row IDs, production-width injection, raw-write/gradient/projector/frozen failure, wrong call order/budget, upstream guard failure, Git HEAD/byte mismatch, unknown execution CLI and visible CUDA. The adapter deliberately returns `CPU_MOCK_REPLAY_ONLY_NOT_AUTHENTICATED`, `GPU_EXECUTION=NOT_AUTHORIZED`, `V10_E0_PROVENANCE=V10_E1_FROZEN_GUARDS=V10_E2_PAYLOAD=V10_E3_DIAGNOSTICS=NOT_EVALUATED`, `PRODUCTION=BLOCKED`. It does not produce D1 artifacts or sign any provenance evidence.

**Outstanding real execution blockers:** no authenticated physical loaded checkpoint/dev encoding/row lineage, no pinned actual execution process, no real GPU tensor value capture, no measured frozen FFT/v10 GEMM peak and model concurrency, no guaranteed original upstream guard coverage, and no trusted external artifact receipt. Stage D2's 2,415,976,448 logical bytes per capture are *not* a GPU memory headroom measurement.

## Validation

From `C:\g5-state`, with `CUDA_VISIBLE_DEVICES=''` and `PYTHONDONTWRITEBYTECODE=1`, run new `TestE1BoundedCPU` only (31 test cases, zero skips), `python -B -m scripts.gen5_v10_stage_e1_bounded_cpu_dispatch_candidate --static-contract-check`, and mandatory `--inspect-local-frozen-source` against the frozen committed source. Require exact base `c3c75af0...`, exact new-file SHA256, no tracked changes and the three expected untracked files. **No commit/push pending independent review.** Any local AST, unit, SHA, or scope failure blocks freeze.

**Future boundary:** a new, independently reviewed one-group real execution adapter with loading provenance, actual hardware measurements, independent guard observation and a separately approved explicit GPU execution authority. Do not treat E1 CPU code freeze or callable budget compliance as permission to execute on Kaggle.
