# Gen5 v10 — Stage D4 CPU In-Process Wiring Scope

**Status:** `D4_CPU_CALLBACK_WIRING_CANDIDATE / NO_REAL_WORKER / NO_CUDA_EXECUTION_GATE`

**Frozen predecessor:** `da6c5480e4506afa840ccefac72dbd1bd0fbc320`, branch `gen5-causal-role-state-ownership`, worktree `C:\g5-state`; pushed/clean status reported by the operator and the exact commit verified against GitHub. This scope is not authority for model load, Kaggle, CUDA, or scientific execution.

## 1. Purpose and exact scope

Stage D3 supplies 17 ordered *self-asserted* events. Stage A supplies canonical SHA256 of actual **provided CPU tensor values**. Neither the D3 events nor a self-sealed D1 artifact establishes code/model/source provenance. Stage D4 closes a narrow *CPU synthetic* gap: ordered in-process callbacks are invoked exactly once, an event is recorded only after a callback successfully returns, and the PRE_FROZEN / POST_FROZEN / POST_V10 input manifests are freshly computed from the borrowed 17 tensors with the **unchanged Stage A canonicalizer**. An additional final rehash detects mutations after POST_V10 by guard/offline/publication callbacks.

This is **not** a callable real frozen worker, GPU input observation, physical-device verification, model checkpoint attestation, or a scientifically validated run. The 17 callbacks are explicitly **CPU injected test doubles**, never independently authenticated worker events. The code reports the scientific and runtime floor as `NOT_EVALUATED / NOT_AUTHORIZED / BLOCKED` without exception.

**Exactly three new tracked files:**

1. `reports/native_recurrent_generator_anchor_pair_localization_v10_stage_d4_cpu_inprocess_wiring_scope.md` (this report).
2. `scripts/gen5_v10_stage_d4_inprocess_cpu_wiring_candidate.py` (CPU fixture callback orchestrator, direct Stage A manifest recomputation, D3 untrusted validation, static-only CLI).
3. `tests/test_gen5_v10_stage_d4_inprocess_cpu_wiring_candidate.py` (strict positive/negative CPU injected fixture tests, zero skips).

Do not edit any existing FFT, Hybrid/GEMM, Stage A–D3, V1/V2/V3 verifier, model, checkpoint, tokenizer, label, split, data, or prior evidence. No Codex, Kaggle, GPU activation, new execution CLI, git commit or push until separate local verification and operator instruction.

## 2. Frozen seam and provenance constraints

The frozen original `scripts/audit_native_recurrent_generator_anchor_pair_localization.py` (`f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`) constructs batch role/fine masks, context and recurrence, all nine factorial raw writes, a single first-source gradient and projector, two first-group target projections, and a K4 `combined` for two independent reducers. Original CPU proxy tests in Stage C already exercised genuine frozen `_component_fine_pair_stats` and replay guard twice. Stage D4 does **not** transcribe the worker or monkeypatch original module/global symbols; it accepts exactly 17 named injected CPU callbacks in D3 event order, validates nine ordered raw-write cells and K4 concatenation, and separately validates each reducer's K4 record against the frozen V3 **record** contract.

Each capture callback must return the same CPU `component`, `log_a`, `survival` and `attention_mask` objects already supplied at earlier stages, plus complete role/fine masks and shape. Stage A validates real provided CPU float32/bool bytes, geometry, mask partition, finite signs, and domain-separated binding. The observer compares the complete Stage A manifests at all three phases, then rehashes after the final callback. No user-supplied capture digest or presealed D3 observation is accepted as a replacement for these recomputations. The observer compares normalized H counts using the existing V3 helper on synthetic K4 records; no V3/E0 science PASS is reported.

The current Stage D3 schema includes `D1_ARTIFACT_PUBLISHED` as its final event. A self-contained D1 directory cannot contain a trace proving its *own* subsequent publication without a circular dependency. For this CPU fixture only, D4 requires the literal `CPU_SIMULATED_PUBLICATION_NO_FILE_WRITTEN`, sets `artifact_written=False` and never calls the D1 writer. Actual post-publication receipts need a separately versioned, external append-only/immutable evidence mechanism, not edits to an existing D1 artifact.

## 3. Strict negative cases and scientific floor

A missing/additional/duplicate callback, repeated frozen/v10 reducer callback object, wrong source or target order, short/raw-write reorder, wrong batch/mask dtype, non-CPU tensor, failed projector/guard callback, invalid K4 concatenation, incomplete Stage A capture, input tensor mutation before or after either reducer (including late callbacks), forged V3 diagnostic, nonmonotonic clock, or false published-artifact marker stops the fixture without completing an observation.

The observer returns only `source_kind=CPU_SYNTHETIC_FIXTURE`, `attestation=SELF_ASSERTED_UNAUTHENTICATED`, `V10_E0_PROVENANCE=V10_E1_FROZEN_GUARDS=V10_E2_PAYLOAD=V10_E3_DIAGNOSTICS=NOT_EVALUATED`, `GPU_EXECUTION=SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`, `PRODUCTION=BLOCKED`. A forged CPU callback could still have arbitrary side effects outside this module; this is a restricted test-helper API, **not a sandbox or safe wrapper around an untrusted live callable**. Its static CLI imports no actual model and never invokes injected callbacks. D3 guard labels can say at most `OBSERVED_CLAIM_ONLY`, never real guard PASS.

The sole planned production locus remains worker1/batch13 `[800,832)`, source `(6201,6201)`, first two targets `(6201,6202)/(6201,6203)`, K4/B32/S128/W24576. D4 uses explicitly smaller CPU synthetic width only. Logical production input bytes per capture `2,415,976,448`, three-capture traffic `7,247,929,344` are **not** runtime GPU memory measurements. The actual original FFT/GEMM peak, model/activation concurrence, device/loaded-checkpoint lineage and complete authenticated upstream guard coverage are unresolved.

## 4. Validation and next boundary

Validation is only CPU: `python -B -m unittest ... TestD4WiredCPU`, `python -B -m scripts.gen5_v10_stage_d4_inprocess_cpu_wiring_candidate --static-contract-check`, Python syntax/whitespace and the relevant D2/D3 CPU regressions. A clean Git HEAD and strict new-file SHA256 scope must be checked before a later code-only freeze. No GPU or scientific worker execution may be triggered by the tests.

After a D4 CPU-only freeze, stop creating additional self-asserted pass-through metadata as if that advanced E0. The remaining meaningful implementation is a **new separately reviewed authentic bounded-worker adapter** with physically measured CUDA memory, actual source lineage, independent original guard attribution and external immutable artifact linkage. That requires its own authority/gate, pinned new code commit and explicit user authorization before any Kaggle/GPU run. D4 does not grant this authorization.

**Disposition:** `D3_FREEZE=da6c5480e4506afa840ccefac72dbd1bd0fbc320`; `D4=CPU_CALLBACKS_ONLY`; `CODEX=NO`; `GPU=OFF`; `SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`; `COMMIT_PUSH=NO_PENDING_LOCAL_VALIDATION`.
