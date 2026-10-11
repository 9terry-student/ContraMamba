# Gen5 v10 Stage B — Versioned Bounded Worker/Exporter and V2 Evidence Verifier Scope v1

**Status:** `STAGE_B_STATIC_IMPLEMENTATION_SCOPE_CANDIDATE / NO_GPU / NO_EXECUTION_GATE`
**Date:** 2026-10-11 (Asia/Seoul)
**Branch / local worktree:** `gen5-causal-role-state-ownership` / `C:\g5-state`
**Exact predecessor HEAD:** `f8e60f173d3a72aa17779449ec9b931c7fa4d043` (pushed, clean, user-reported and remote-confirmed).

## 1. Governing identities and non-negotiable boundaries

- Frozen original audit: `scripts/audit_native_recurrent_generator_anchor_pair_localization.py`, Git blob `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`.
- Frozen original tests: `tests/test_native_recurrent_generator_anchor_pair_localization.py`, blob `26c90ada9a74a2ac2716c880e58d15df5bd01704`.
- Frozen v10 Hybrid / shared GEMM: blobs `56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0` / `9003fb9bd23b2a6680ef210a51845058622badd2`.
- Frozen v10 bounded CPU record verifier / CPU paired adapter: blobs `7f547adc42a6e113dd179cb0f4988336ee476b17` / `d2f1535fe19b07483911d3293be5d0be9d400732`.
- Newly frozen Stage A provenance canonicalizer: `scripts/gen5_v10_bounded_tensor_provenance_candidate.py`, blob `703e0f6faf15ee499ec6a97f1aa8edc8d2a93333`, schema `GEN5_V10_BOUNDED_INPUT_MANIFEST_V1` with exactly 17 logical tensor keys; its tests blob `6ddc7df3d87536fbb94383ba4520501b405a2f30`.
- Approved E2/E3 distinction: `reports/native_recurrent_generator_anchor_pair_localization_v10_numeric_contract_and_candidate_freeze_spec.md`, blob `661c627f09034c34f9830b18b7e9938c72c05925`.
- Prior Stage A/B design: `reports/native_recurrent_generator_anchor_pair_localization_v10_bounded_worker_export_static_design_v1.md`, blob `d48f520a21bff6f53612c04ed0edcef01463ec9d`.

All of these source blobs, all original data/model/checkpoint/tokenization/split identities, both pre-existing worker/output schemas, original numerical tolerance (`abs <= 2e-5 OR relative <= 2e-5` with original denominator), and all original upstream and fine-parent guards stay unchanged. The original FFT scientific-execution gate does **not** grant a v10 run.

## 2. One bounded scientific question, not a population run

The eventual first real-input paired check compares **the exact same native projected `combined` input** under frozen FFT and v10 Hybrid, retaining full K4 raw/coarse/fine outputs separately. The selected bounded locus is worker **1**, its **13th one-based batch**, dev row interval **[800,832)** (B32), one source from the frozen `FULL_FACTORIAL_CELLS` ordering and its **first two-target group** as determined by the original `all_orientations()`/`TARGETS_PER_TRANSPORT` logic. At runtime explicitly record and enforce the resolved source-cell ID, both target-cell IDs, group ordering, worker ID, one-based batch ID, global row interval and all row IDs. Do not assume that the source/group can be identified just from numeric indices; fail on a mismatch. One group means exactly 4 components (two visible + two complement), **not** all source/target groups in the selected batch. This is the **planned** locus, not authorization to execute it.

B8 tail `[832,840)` is a separate later comparison and must use the whole original FFT path. Full-width mixed Fast/GEMM routing is a separate bounded CUDA requirement; do not fabricate inputs to force mixed routing and do not claim a mixed real-data result if no such chunk pattern occurs.

## 3. Exact static implementation scope (new files only)

A future **CPU/static implementation** may add exactly:

1. `scripts/gen5_v10_bounded_worker_export_candidate.py` — new, isolated, versioned bounded orchestration and GPU-input-capture *code*, unit-tested using CPU fakes; no scientific execution in this phase.
2. `scripts/gen5_v10_e0e3_v2_bundle_verifier_candidate.py` — independent strict validator for a *new* versioned bundle schema; no reuse of the V1 draft export as if it proved runtime lineage.
3. `tests/test_gen5_v10_bounded_worker_export_candidate.py` — stubbed seam and negative tests, no model, no real CUDA.
4. `tests/test_gen5_v10_e0e3_v2_bundle_verifier_candidate.py` — strict schema/hash/guard/diagnostic/failure-path tests on synthetic records only.
5. This report as `reports/native_recurrent_generator_anchor_pair_localization_v10_stage_b_bounded_worker_v2_scope.md`.

No modifications to any earlier tracked file; no generated evidence or runtime data staged. New worker CLI is `--static-contract-check` **only** during this phase. A future `--run-bounded` mode must remain disabled or fail closed until a separately committed bounded GPU execution gate, fresh full execution HEAD, authenticated Kaggle environment and an explicit user decision to enable that mode. Never expose existing frozen `--run`, `--run-worker` or `--merge-only` as aliases. No training/evaluation/dataset regeneration or full 840-row run.

## 4. Frozen worker seam and implementation strategy

The frozen worker constructs `batch_features`, authentic `role_masks` and `fine_masks`, a common `context`, `pair_log_a`, `pair_survival`, `raw_by_cell`, frozen gradients/projector, and then `combined = torch.cat((visible, complement), dim=0)` for exactly two targets. The new bounded implementation must reproduce only this single selected batch/source/group by calling the original frozen helper functions rather than altering globals or modifying the frozen source. Independently review the narrow worker-control-flow transcription against the exact frozen script. Never call frozen `run_full()` or `run_worker()` and expect it to use v10, and never monkeypatch `_fine_pair_gap_interference_fft`.

- For the frozen engine, call the genuine full frozen `_component_fine_pair_stats` on the exact `combined` inputs so all original coarse and fine replay guards execute.
- For v10, use the unchanged `hybrid_fine_pair_gap` with the *same* original input tensors; use the original `coarse._component_region_pair_stats` and a **new GPU-capable, independently line-reviewed** copy of the frozen post-reducer reconstruction/record section, calling the genuine `coarse._replay_vector` without changing normalization or tolerances. Do not pass CUDA tensors to the CPU-only paired adapter or disable its CPU guard.
- Capture both original per-engine diagnostic extrema without copying, clipping or replacing them. Preserve existing failed legacy full-record comparisons as `FAIL`.
- Preserve original raw energy reconstruction and cross-component replay, projectors and original restrictions on gradient accumulation, checkpoint mutation and data inspection. Ensure all rows and group selectors are bounded before any runtime work is performed.

## 5. Input byte identity, memory, and provenance limits

Stage A's `manifest_for_cpu_inputs` can hash *provided CPU values* only; its `device_at_capture='cpu'` is not evidence the original tensors arose on a GPU or from a particular checkpoint. The new worker must establish source lineage separately: authenticated commit/blobs/clean status; fixed model/tokenizer/checkpoint revisions and hashes; 840-row ordering and active-encoding hashes; worker/row/batch/source/target IDs and row IDs; actual physical T4 identity and CUDA mapping; torch/CUDA/cuBLAS/kernel versions and numerical settings, including `torch.backends.cuda.matmul.allow_tf32=False`.

At the exact pre-reducer seam, **synchronize and capture the same logical 17 tensor values**, in Stage A's canonical little-endian C-order encoding. For CUDA, design a bounded **sequential chunk-to-CPU digest** path that never materializes all input bytes or additional full-size tensors simultaneously on the GPU. Preserve logical element order, including noncontiguous tensors, without silently changing dtypes or data. Compute hashes immediately before frozen, after frozen and after v10; require all three value identities to agree, otherwise fail closed. Record source device, capture phase, synchronization status, input tensor shape/dtype/strides and stage-A-compatible manifest identity independently of the value-only CPU manifest. A hash of metadata, tensor object IDs or filenames alone is inadequate.

For K4/B32/S128/W24576, the three float32 tensors alone occupy ~2.25 GiB of *logical* input bytes (component 1.50 GiB, log_a 0.375 GiB, survival 0.375 GiB); do not make two full GPU clones. The 17-key CPU digest does not by itself allow an offline reviewer to reconstruct input bytes. Until either a lossless bounded input export or independently authenticated deterministic input replay has been verified, `V10_E0_PROVENANCE` cannot be promoted merely from self-asserted manifest hashes. Any lossless raw-byte export must have an explicit disk/transfer bound and checksum/identity review, not be silently added.

## 6. New paired bundle v2 and fail-closed verification

Proposed schema family (names fixed by this scope, fields need strict unit-tested definitions prior to use):

- `GEN5_V10_BOUNDED_PAIRED_EXPORT_V2`: one selected locus, exact immutable source/runtime metadata, canonical input manifest plus pre/post identical digest evidence, independent original and v10 raw K4 records, engine-specific dispatch, full coarse/fine replay evidence, per-stage failures and timings, bundled file-list/hash manifest.
- `GEN5_V10_BOUNDED_PAIRED_VERIFICATION_V2`: read-only verifier report with explicit code and bundle SHA256, tensor byte/source-authentication coverage, E0/E1/E2/E3 separately, retained legacy full-record verdict, exact bounded scope, missing-evidence reason codes.

Run v2 validation against the actual schema and frozen replay *independently*. E0 = `NOT_EVALUATED` unless source/runtime and actual input bytes are authenticated; E1 = `NOT_EVALUATED` unless all original relevant guards have attributable successful evidence; E2 = `NOT_EVALUATED` absent complete authenticated bounded payload (a mismatched actual payload is `FAIL`); E3 = `NOT_EVALUATED` absent independently recomputed true diagnostics and evidence explaining any numerical discrepancy. Legacy full-record `FAIL` must remain `FAIL`. No bounded `PASS` entails whole-840 or scientific claim promotion. The V1 CPU verifier must remain untouched and must not be silently fed the V2 schema.

Any write must be under a unique, non-existing, **outside-repository** artifact root, atomically written with sidecar SHA256; protect against overwrite, path traversal/symlinks, partial outputs and report editing. Preserve partial failure evidence without claiming successful execution. Offline v2 verification must not require GPU or an active model session.

## 7. Required CPU/static tests and stop conditions

Before candidate freeze: exact-HEAD/blob/branch/clean check; CLI rejects all execution flags; mocked frozen worker seam tests assert **one** batch, source, group and no other rows; input hashes include content/sequence/masks/shape/dtype and reject tampering; pre/post engine drift blocks; v2 schema omission, injection, int/ID mismatch, nonfinite/NaN, wrong row identities, incomplete raw payload, missing replay, forged diagnostics, mismatched result/manifest hash and mixed engine records all block; current legacy full-record failure retained; no E0/E1/E2/E3 false PASS. Run only CPU unittest, compile/AST/static checks and `git diff --check` under this stage.

**STOP** on unclear worker seam, source provenance mismatch, altered original tolerances or aggregation semantics, memory-unsafe CUDA captures, accidental GPU/model entrypoint, schema drift, dirty worktree or any attempt to run Kaggle. Semantic verifier of the worker-control-flow transplant is mandatory before an independent bounded GPU execution gate. The unresolved historical signed upstream replay failure remains a blocker for claims of arbitrary-input robustness.

**Disposition:** `STAGE_A_FREEZE=CONFIRMED`; `NEXT=STAGE_B_CPU_STATIC_IMPLEMENTATION_REVIEW`; `CUDA=OFF`; `CODEX=NO`; `COMMIT_PUSH=NO_UNTIL_SEPARATE_REVIEW`; `SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`; `FULL_840_RUN=BLOCKED`.
