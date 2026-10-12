# Gen5 v10 — Single-Group Real CUDA Runner Candidate

**Status:** `IMPLEMENTED_CANDIDATE / CPU_TESTED_ONLY / CUDA_EXECUTION_NOT_AUTHORIZED`. Base: `f01dabd0d73014423d5c1a3c5f8d7f8c1187ae28` on `gen5-causal-role-state-ownership`. This is not a new research claim or a Kaggle run permit.

## Execution objective

This is the final implementation bridge toward a **real**, separately authorized, full-width bounded GPU experiment. Unlike D/E CPU mock modules, `scripts/gen5_v10_single_group_cuda_runner_candidate.py` contains an actual `--run-bounded` route, but it is deliberately locked behind a clean pushed commit, a full-SHA execution pin, exact source blobs, one visible GPU, both explicit approval switches, and local input byte validation. **Do not activate GPU yet.**

The only legal locus: worker 1, one-based batch 13, dev rows 800–831 (`[800,832)`), first source `(6201,6201)`, first two targets `(6201,6202)` and `(6201,6203)`, K4/B32/S128/W24576. There is no call to the original `run_worker()`, `run_full()` or full source/batch/orientation loop.

## Real runtime path (only after separate explicit approval)

1. Pin Git HEAD and original runtime helper Git blobs, reject dirty trees and modified runner bytes.
2. CPU-verify actual checkpoint and model snapshot SHA256 via frozen E2; original `transport._runtime_inputs()` validates/dev-encodes and loads the actual checkpoint/model; E2 independently checks generated 840-row order and CPU encoding hash.
3. Load dev features once. Execute **only** the specified batch. Build original role/fine masks and common context, compute `log_a`/survival, all nine frozen raw writes, one source gradient, one projector and the selected two-target projection. Count actual successful upstream helper returns against E1's 19-call order.
4. Form K4, capture the same 17 live CUDA tensor *values* three times using Stage B chunk-bounded copies, execute the frozen full FFT reducer, the v10 Hybrid/GEMM reducer with the original width=256 chunk semantics, and frozen raw/projector guards. Recheck hashes and run V3 individual record validators.
5. Synchronize and sample PyTorch allocated/reserved/current peak and CUDA driver free/total memory at all meaningful phases. Pre-model device headroom must be at least 12 GiB and between phases 1.5 GiB; an OOM or failure produces a separate **failed** receipt. There is no automatic batch fallback.
6. Write exclusively to a fresh run-specific path under `reports/native_recurrent_generator_anchor_pair_localization_v10_bounded_cuda_runs/` with `.sha256` sidecars. A result is a *self-recorded execution candidate*, **not** third-party attestation, E0–E3 PASS or publication proof.

## Correctness/authority boundaries

**22/22 CPU tests** exercise entrypoint approval lock, run never reached without approvals, selected locus, Git/dirty/frozen blob constraints (mocked), output collision, hash, capture mismatch and memory counter behavior (mocked). The static CLI imports no CUDA/model. The real frozen AST integration must be performed on the operator's checkout; CPU tests do **not** prove the real W24576 path executes successfully or fits a T4. They do not verify native CUDA memory, numerical equivalence or actual model loading.

The first genuinely useful next gate after local code review is a **single authorized T4 bounded attempt at a fixed pushed commit**, not another series of documentation stages. A failed run is a diagnostic outcome with measured memory samples, not a reason to assume scientific equivalence. The operator controls GPU ON/OFF and full launch; approve the run separately only after code commit, bootstrap, CPU provisioning and headroom checks. No Codex. No training/evaluation until separately authorized.

## Exact candidate whitelist

- `reports/native_recurrent_generator_anchor_pair_localization_v10_single_group_cuda_runner_scope.md`
- `scripts/gen5_v10_single_group_cuda_runner_candidate.py`
- `tests/test_gen5_v10_single_group_cuda_runner_candidate.py`

No predecessor file, historic science artifact, model data, checkpoint, tokenizer, or split is modified. Any new failure means STOP and bounded correction; neither E0 science nor GPU execution approval follows from code-only PASS.
