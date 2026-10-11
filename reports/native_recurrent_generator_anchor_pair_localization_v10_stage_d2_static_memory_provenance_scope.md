# Gen5 v10 — Stage D2 CPU Static Memory and Runtime Provenance Preflight Scope

**Status:** `D2_CPU_STATIC_PREFLIGHT_CANDIDATE / GPU_OFF / EXECUTION_NOT_AUTHORIZED`

**Frozen predecessor:** `6927a3ae96985a50ed1017edd522d6bebd679b6a` on `gen5-causal-role-state-ownership`, user-reported clean and pushed, remote commit verified 2026-10-11. **Worktree:** `C:\g5-state`. **Phase:** CPU static implementation and tests only, not an execution gate.

## 1. Purpose and limits

Stage D1 froze an immutable, external V3/failure artifact writer. Stage C's runtime bridge remains CPU-mock-only and Stage B's CUDA-capture function has no authorized real workload entrypoint. Static code validity does not establish GPU memory headroom, actual model/tensor lineage, or original guard coverage. Stage D2 adds exact arithmetic of *logical* capture bytes and a separate read-only local Git source identity check, with explicit nonpromotion of E0–E3 and CUDA readiness.

All new code is CPU-only; do not load a model, checkpoint, tokenizer, run the worker, query CUDA, synchronize a GPU, construct synthetic scientific results, access Kaggle, or write experimental outputs. Do not run `cm kaggle` or `cm run`.

## 2. Only authorized additions (three new files)

1. `reports/native_recurrent_generator_anchor_pair_localization_v10_stage_d2_static_memory_provenance_scope.md` (this document)
2. `scripts/gen5_v10_stage_d2_static_runtime_preflight_candidate.py` — CPU integer arithmetic, shape/budget validation, pinned source claims, optionally read-only local Git HEAD/branch/status/blob verification, and `--static-contract-check` only
3. `tests/test_gen5_v10_stage_d2_static_runtime_preflight_candidate.py` — deterministic CPU tests; unit-test injected simulated Git results; fail-closed negative cases; zero skips

**Never modify:** frozen original FFT/coarse audit/tests, v10 hybrid/GEMM, Stage A manifest, Stage B V2/static worker, Stage C bridge/V3, Stage D1 writer, execution guards, checkpoint/model/tokenizer/splits/data, or previous report/artifact. No report claims an E0/E1/E2/E3 pass.

## 3. Fixed target and exact logical byte accounting

Only the intended single group: worker 1; one-based batch 13; rows `[800,832)`; `(6201,6201)` source and `(6201,6202)`, `(6201,6203)` targets in that order; K4, B32, S128, W24576. Require `shape.intermediate_size * shape.state_size == 24576`. Stage A's 17 tensors contain 3 float32 inputs and 14 bool masks. Their row-major *logical* byte sizes are:

| Input | Bytes | Binary GiB |
| --- | ---: | ---: |
| `component` K4×B32×S128×W24576×float32 | 1,610,612,736 | 1.5 |
| `log_a` B32×S128×W24576×float32 | 402,653,184 | 0.375 |
| `survival` B32×S128×W24576×float32 | 402,653,184 | 0.375 |
| 14 bool masks B32×S128×1 byte | 57,344 | approx. 0.0000534 |
| **One complete 17-tensor capture** | **2,415,976,448** | approx. 2.2500534 |
| **Three full capture phases** | **7,247,929,344** | approx. 6.7501602 |

Phases are `PRE_FROZEN`, `POST_FROZEN`, `POST_V10`. These are exact logical bytes to be processed, **not measured bandwidth, wall time, peak GPU resident memory, incremental model memory, transfer allocation, or GPU headroom**. The Stage B capture leaf limit is at most 8 MiB; its leaf cap does not bound the frozen FFT/GEMM working set, coexisting model activations, or allocator fragmentation. Do not create full GPU copies of all 17 tensors. B8 tail, full 840 rows, alternative groups, and any actual GPU run remain out of scope.

## 4. Runtime/source verification contract

Constants cross-checked against frozen source/V3: checkpoint SHA256 `1ff3fcf2ebd754ab6f9483d6a9982b9b04b3357f9f8cdbe2b30399e7d2f`; dev order `b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25`; active encoding `e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51`; model revision `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`. These **pinned string identities are not proof of actual loaded bytes**.

`validate_source_claims()` checks a supplied, explicitly untrusted metadata object against these literals and rejects wrong head/branch/row order/TF32 claims. It **always** reports source not authenticated. `inspect_clean_local_git()` separately checks, at a caller-pinned exact 40-digit commit, the matching branch, clean tracked+untracked state, D1 ancestry, and seven immutable source Git blob hashes. This establishes only local Git identity **at the time of check**, not live GPU process state or model/dataset/token bytes. During candidate installation the new files are untracked and `inspect_clean_local_git()` would correctly reject a dirty worktree; static CLI never invokes it. It is an optional callable for a later independently authorized clean/committed environment, not a GPU-ready CLI mode.

Missing or unverified data that block an eventual execution gate: actual physical CUDA device and mapping; independently captured torch/CUDA/cuBLAS/TF32/determinism state; loaded checkpoint/tokenizer/raw rows and active encoding hash verification; real CUDA `mem_get_info`/peak allocation traces **for the authentic bounded seam**; frozen full-width FFT, v10 Fast/GEMM, capture scratch, concurrent model activation and failure overhead; original source-gradient/projector/raw/coarse/fine guard attribution; true paired source tensor byte lineage, authentic code HEAD and exact row IDs. No GPU memory threshold can be selected from CPU-only code.

## 5. Fail-closed verdict boundary and next gate

The static report schema is `GEN5_V10_D2_CPU_STATIC_RUNTIME_PREFLIGHT_V1`. Always output `readiness=BLOCKED_PENDING_INDEPENDENT_RUNTIME_EVIDENCE`, `GPU_EXECUTION=NOT_AUTHORIZED`, `V10_E0_PROVENANCE=NOT_EVALUATED`, E1/E2/E3=`NOT_EVALUATED`, `PRODUCTION=BLOCKED`. A canonical SHA256 on a self-issued plan is merely a local integrity aid; edited metadata plus recomputed hash never grants a PASS. Reject missing/additional fields or forged pass values by recomputation.

Next, after this CPU-only D2 freeze, independently audit a *new* narrowly bounded runtime preflight instrument and authentication chain before proposing a separately approved GPU execution gate. Do not retrofit `--run-bounded` into any static script or infer permission from a clean Git status.

## 6. Validation and stop conditions

Run from worktree with `CUDA_VISIBLE_DEVICES=''` and `PYTHONDONTWRITEBYTECODE=1`:

- `python -B -m unittest discover -s tests -p test_gen5_v10_stage_d2_static_runtime_preflight_candidate.py -v`
- `python -B -m scripts.gen5_v10_stage_d2_static_runtime_preflight_candidate --static-contract-check`
- `git diff --check` (after exact three-file staging) and SHA256/source scope inspection.

No GPU, live model, runtime probe, Kaggle invocation, artifact export, training/evaluation, commit/push until subsequent manual review. Stop if tests fail/skip, HEAD or frozen blob drifts, any prior file changes, numeric metadata becomes self-authenticated, or static preflight emits a GPU-ready or E0–E3 PASS verdict.

**Disposition:** `D1_FREEZE=6927a3a`; `D2_CPU_STATIC_ONLY`; `CODEX=NO`; `GPU=OFF`; `COMMIT_PUSH=NO_PENDING_LOCAL_VALIDATION`; `SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`.
