# Gen5 v10 — Stage D3 Untrusted Bounded Runtime Observation Contract

**Status:** `D3_CPU_STATIC_OBSERVATION_CANDIDATE / CUDA_OFF / NO_EXECUTION_GATE`

**Frozen predecessor HEAD:** `8b2a9ad0be96c45e18883cab951a6414bfb179c3`, branch `gen5-causal-role-state-ownership`, worktree `C:\g5-state`. The D2 freeze was user-reported clean and pushed, and its exact three-file commit was verified on GitHub. This document is an implementation-scope specification, **not** CUDA execution authority.

## Scientific boundary

D2 calculated exactly 2,415,976,448 logical input bytes per K4/B32/S128/W24576 capture and 7,247,929,344 bytes for three phases. Neither figure is a peak GPU allocation, free VRAM measurement, or scientific result. D1's immutable writer checks local artifact integrity only. V3 validates self-described runtime evidence but cannot independently authenticate input origin or source-code execution. D3 **does not** lower any of these boundaries.

D3 prepares a narrowly scoped CPU-only contract for the *locations and validation requirements* of eventual runtime instrumentation. It does **not** instrument live CUDA, invoke the frozen worker, run model forward, capture real hardware counters, authorize Kaggle, or construct a runnable `--run-bounded` CLI. A complete D3 observation is still self-asserted, even when it contains matching hashes, self-reported guard values and memory samples.

## Exact three-file whitelist

1. `reports/native_recurrent_generator_anchor_pair_localization_v10_stage_d3_untrusted_observation_scope.md` (this document)
2. `scripts/gen5_v10_stage_d3_untrusted_instrumentation_candidate.py` — CPU synthetic observation constructor, strict validator, static-contract CLI, no torch import or runtime I/O
3. `tests/test_gen5_v10_stage_d3_untrusted_instrumentation_candidate.py` — CPU-only mocked observations and negative/tamper tests, zero GPU or model calls

Do not modify any frozen original FFT/coarse audit, Hybrid/GEMM, Stage A manifest, Stage B worker/V2, Stage C bridge/V3, D1 writer, D2 preflight, checkpoint, split, data or preexisting artifacts. Do not commit/push until candidate tests and local SHA/scope review succeed. No Codex.

## Frozen worker seam and observation sequence

Review basis: `scripts/audit_native_recurrent_generator_anchor_pair_localization.py`, immutable Git blob `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`. The original worker builds batch features/role+fine masks, context/log_a/survival, **nine** factorial raw writes, then first source gradient and projector, then projects one pair of targets; the selected `combined` has K4. The planned subsequent paired seam is frozen full FFT then v10 Hybrid on identical values, with three input captures and original guard checks. The required ordered 17 observation labels in code encode this flow. These labels are a **future instrumentation plan**, not proof that those functions ran.

The sole bounded locus is worker 1, one-based batch 13, global dev `[800,832)`, source `6201_6201`, targets `6201_6202` and `6201_6203` in order, K4/B32/S128/W24576. Only one source gradient/projector and one target group are permitted. B8, other groups, and all 840 rows remain excluded.

## Observation schema, links, and nonpromotion

`GEN5_V10_D3_UNTRUSTED_RUNTIME_OBSERVATION_V1` has an exact field set: frozen branch/base, a **claimed** new execution code head, exact selected locus, explicit source kind, 17 ordered events (strictly increasing caller-reported elapsed nanoseconds), declared call counts, 9 guard labels, source claims, and a domain-separated canonical JSON SHA256. The three named input-capture checkpoints contain a 64-hex **claimed capture-set digest**. Values must match across phases, but D3 **cannot derive these hashes from tensor bytes**. Actual future work must link them to Stage A 17-key canonical input bytes and V3's per-tensor capture entries before any provenance assessment.

CPU fixtures must not assert any GPU memory; `UNVERIFIED_RUNTIME_CLAIM` records can contain caller-provided allocated/reserved/peak/free/total byte counters. D3 validates types, obvious consistency and monotonically nondecreasing claimed peak within a single uninterrupted sampling window. It cannot establish that any claimed sample was actually collected, covered the relevant allocation lifetime, captured all FFT/GEMM scratch allocations, or reflected the physical T4. No GPU readiness decision is derived from counters.

Runtime source claims must pass the unchanged D2 `validate_source_claims()` for frozen model/checkpoint/dev-order/encoding literals, code-head consistency and TF32-off assertion. D2's own return value is **not** source attestation. Guard entries accept only `NOT_AUTHENTICATED` or `OBSERVED_CLAIM_ONLY`, never `PASS`, regardless of their apparent completeness. A self-consistent SHA256 or all nine guard labels cannot promote the original guard coverage E1.

The result must always state `GPU_EXECUTION=NOT_AUTHORIZED`, `E0=E1=E2=E3=NOT_EVALUATED`, and `readiness=BLOCKED_INDEPENDENT_RUNTIME_AND_MEMORY_EVIDENCE`. A forged readiness status blocks even when its hash is recomputed. The synthetic helper exists exclusively for tests/static CLI; never use its counters or hashes as experiment evidence.

## Validation and stop conditions

From `C:\g5-state` with `CUDA_VISIBLE_DEVICES=''` and `PYTHONDONTWRITEBYTECODE=1`, run only the new 44-case CPU unittest module and `python -B -m scripts.gen5_v10_stage_d3_untrusted_instrumentation_candidate --static-contract-check`; execute `git diff --check` after explicit three-file staging, not before. Code/head/source SHA256 and actual worktree status require separate independent checks. No GPU model import, Kaggle run, model inference, numeric science execution or promotion is permitted.

Stop if source-head/blob drifts, a prior file changes, missing/reordered/duplicated events are accepted, missing guard/capture evidence passes, memory figures become a VRAM safety claim, input/loaded checkpoint origin is inferred from self-reported JSON, or an executable CUDA CLI appears.

**Next boundary after an independent D3 code freeze:** implement a separately versioned, fail-closed *in-process real-event wiring and trusted lineage acquisition* mechanism and verify authentic original worker guards; independently review and sign off a bounded GPU memory probe/execution gate only when evidence supports it. No automatic authorization follows from D3.
