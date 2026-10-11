# Gen5 v10 — Bounded Input Provenance and Worker/Export Static Design v1

**Status:** `STATIC_DESIGN_CANDIDATE / NO_RUNNER_IMPLEMENTATION / NO_GPU / NO_SCIENTIFIC_EXECUTION`

**Decision basis:** A clean, pushed code-only freeze on `gen5-causal-role-state-ownership`, commit `24705c6b3ce6d76ddd8fe4328ef4a1497b468468` (2026-10-11 Asia/Seoul). This report defines the smallest auditable *next implementation boundary*, not a run or commit/push authorization.

## 1. Authority, identity and immutable files

- Current frozen candidate HEAD: `24705c6b3ce6d76ddd8fe4328ef4a1497b468468`.
- Frozen original audit `scripts/audit_native_recurrent_generator_anchor_pair_localization.py`: Git blob `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`.
- Frozen original tests: `tests/test_native_recurrent_generator_anchor_pair_localization.py`: Git blob `26c90ada9a74a2ac2716c880e58d15df5bd01704`.
- Hybrid reducer: `scripts/gen5_v10_hybrid_fine_pair_candidate.py`: blob `56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0`.
- Shared GEMM kernel: `scripts/gen5_v10_shared_dyadic_gemm_candidate.py`: blob `9003fb9bd23b2a6680ef210a51845058622badd2`.
- CPU record verifier: `scripts/gen5_v10_e0e3_record_verifier_candidate.py`: blob `7f547adc42a6e113dd179cb0f4988336ee476b17`.
- CPU paired adapter: `scripts/gen5_v10_bounded_pair_adapter_candidate.py`: blob `d2f1535fe19b07483911d3293be5d0be9d400732`.
- Paired adapter tests: blob `98add0ce1caec00c8a15a534d58f9e67774f379e`.
- Approved E2/E3 split: `reports/native_recurrent_generator_anchor_pair_localization_v10_numeric_contract_and_candidate_freeze_spec.md`, blob `661c627f09034c34f9830b18b7e9938c72c05925`.

Do not change any of the files above, original gate, upstream guards, replay criteria, frozen reference artifacts, model/checkpoint/data/split/tokenizer or output schema V1. No Codex. No training. No Kaggle/GPU until independently authorized.

## 2. Exact worker seam and architectural restriction

The original audit `run_worker()` authenticates the repository, loads the fixed runtime via `transport._runtime_inputs(args)`, makes one batch's `batch_features`, `batch_mask`, original `role_masks`, `fine_masks`, `context`, `pair_log_a`, `pair_survival` and frozen source-gradient/projector outputs. For each source and `TARGETS_PER_TRANSPORT=2` targets it constructs:

```python
combined = torch.cat((visible, complement), dim=0)  # K=4, B=32 or last B=8, S=128, W=24576
```

The original call at that seam is `_component_fine_pair_stats(component=combined, log_a=pair_log_a, survival=pair_survival, attention_mask=batch_mask, role_masks=role_masks, fine_masks=fine_masks, shape=wrapper.correction.shape)`.

The existing original `run_full()` uses `_spawn_worker()` pointing at its own `__file__` and its `--run-worker`; it cannot transparently launch a v10 process. **Never monkeypatch globals**, invoke the frozen `run_full()` assuming substitution, or edit the frozen script. A subsequent versioned worker must explicitly construct this *same* source/target-group input once, invoke each engine as an independent reducer in the same process, and preserve the genuine frozen coarse and parent reconstruction checks. Avoid copying/reimplementing frozen model, gradient, projector or split logic; delegate to identified original helper calls. Review any small necessary loop transcription against the frozen worker line-for-line.

The current `gen5_v10_bounded_pair_adapter_candidate.py` **rejects GPU tensors** and is only a CPU injected-input validation utility. It is not a CUDA execution adapter. A future CUDA-capable paired worker therefore needs **new** code with its own identity and tests, not silent use of the CPU adapter or removal of its CPU guard.

## 3. Proposed two-stage new implementation (neither currently authorized for execution)

**Stage A — CPU/static source-byte contract and manifest-only tooling** (first implementation):

1. Add `scripts/gen5_v10_bounded_tensor_provenance_candidate.py`, supporting *only* canonical tensor manifests, streaming SHA256 of **provided CPU tensors**, strict schema checks, record-identity comparison and a static-plan CLI. No model import/load, CUDA use, process spawning, input data loading, output into repository, or experiment.
2. Add `tests/test_gen5_v10_bounded_tensor_provenance_candidate.py` with deterministic CPU fixtures, content/metadata tamper negatives, key omission/injection, order/device/dtype/shape mismatch, noncontiguous canonicalization, boolean encoding and digest repeatability.
3. Add this document to `reports/native_recurrent_generator_anchor_pair_localization_v10_bounded_worker_export_static_design_v1.md` as the exact implementation-scope report, after confirming its reviewed bytes.

Stage A freezes only after 1) independent test and static review, 2) exact three-file scope review and 3) separate manual commit/push. **Do not build a GPU/production runner in Stage A.**

**Stage B — separately versioned bounded worker/exporter and new v2 verifier** (requires independent implementation authority and high-risk review *after Stage A*):

- New `scripts/gen5_v10_bounded_worker_export_candidate.py`: an explicit isolated bounded worker process with a future dedicated `--run-bounded` option; cannot inherit the old `--run`, `--run-worker`, `--merge-only`, `--preflight` semantics accidentally. Its parser, code HEAD, workload selector, output root and runtime identities must be explicitly authenticated; no execution mode is enabled under this static design.
- New v2 verifier implementing an exact, separately versioned input/record schema. The frozen `GEN5_SHARED_GEMM_CPU_EQUIVALENCE_DRAFT_V1` record verifier is **not** silently re-used for a new artifact schema; keep its existing `NOT_EVALUATED` claims unchanged.
- One selected **existing** worker shard / batch / source / target pair-group, not all 840 rows. B32 and B8 are distinct routes; any full-width mixed-route claim requires observing actual fast/GEMM chunks, not synthesizing or altering scientific inputs to force a route.

## 4. Proposed canonical input-manifest contract (Stage A)

Use a **new** schema identifier `GEN5_V10_BOUNDED_INPUT_MANIFEST_V1`; do not repurpose existing frozen worker or report schemas. Stage A computes tensor digests from user-provided CPU tensors only. For the later GPU stage, snapshot/synchronize/copy from the actual runtime tensors without in-place mutation, then use the exact same CPU canonicalizer. Record copy time and device mapping; do not interpret hashes as proof of how the tensors were generated.

Mandatory logical tensor keys (fixed sorted set, no omissions/extras): `component`, `log_a`, `survival`, `attention_mask`, `role_masks.CLAIM`, `role_masks.EVIDENCE`, `role_masks.EOS`, `fine_masks.CLAIM.<5 frozen classes>`, `fine_masks.EVIDENCE.<5 frozen classes>`. Include `shape.intermediate_size` and `shape.state_size`, and exact K/B/S/W geometry. For each tensor: `key`, `shape` (integer vector), `dtype` (canonical `float32` or `bool` only; do not silently cast), `layout` (`C_CONTIGUOUS_CANONICAL_V1`), `numel`, `byte_count`, `content_sha256` and `device_at_capture`. Boolean data bytes must be exactly 0 or 1, in row-major logical order. Floats must be IEEE-754 little-endian float32 in row-major logical order; nonfinite values block. Noncontiguous CPU inputs are canonicalized into a fresh contiguous copy with identical logical values, *not* via a storage-order byte dump. Encode metadata as canonical UTF-8 JSON with sorted keys and no NaN; form a domain-separated hash over schema version, tensor key, dtype, shape, and content hash. Reject schema mismatch, inconsistent byte counts, duplicate/missing keys, and non-string/non-integer identities.

A future paired GPU worker must prove *both reducers consumed the same inputs*: snapshot a single source-group bundle immediately before either reducer; hash its exact semantic tensor values; then verify no input tensor value changed after each reducer, recording any transient copy separately. Same shape, object identity, or matching source labels **alone** is insufficient. Keep per-engine `engine`, `code_blob`, `dispatch`, `runtime`, `input_manifest_sha256`, output record SHA256, and independent frozen replay evidence. A digest authenticates byte equality of the captured tensors, **not** frozen dataset/checkpoint/runtime provenance by itself.

Future runtime identity (not claimed by Stage A): pinned code HEAD, Git blobs, expected branch, clean worktree; exact tokenizer/model revision, seed180 checkpoint SHA256, dev order/encoding SHA256, worker/shard/start-stop/batch ID, source cell, target-group IDs and order, K4 and B32/B8, S128/W24576, chunk width 256, all 75 fine cells; active CUDA visibility, actual physical device, Torch/CUDA/cuBLAS versions, TF32 and deterministic settings. Missing values fail closed; do not invent unavailable hashes or hardware fields.

## 5. Paired output and E0–E3 verdict boundary (future Stage B)

Retain both complete raw K4 engine outputs, genuine coarse upstream checks, original per-gap parent replay results and genuine unmodified `fine_parent_reconstruction_abs_max` and `fine_parent_reconstruction_rel_max`. Capture the v10 routing record including `fast_chunks`, `gemm_chunks`, `frozen_tail_calls`, `total_chunks`, `max_half_span`. Numeric comparison obeys the frozen `abs <= 2e-5 OR relative <= 2e-5` test with its original denominator. All nonnumeric IDs, source/batch/shape/order and masks must match exactly. Distinguish the independent E0 input/code provenance, E1 all-original-guard coverage, E2 complete *authenticated bounded* science payload, and E3 separately recomputed authentic diagnostic discrepancies. Always preserve `LEGACY_FULL_RECORD=FAIL` if any original all-field diagnostic comparison fails. Bounded verification does **not** confer population-level PASS.

The future exporter must emit a new bounded bundle schema, a canonical manifest and sidecar hashes **outside the repository** under a non-existing dedicated output root; no overwrite, no mutable evidence replacement, no partial artifact promotion. Output includes both success/failure status and exact missing evidence fields. If execution or replay fails, preserve the failure and available provenance; do not overwrite guard failures or silently continue to the next source.

## 6. Validation and stop conditions

Stage A: CPU-only unit tests, static import/plan, `git diff --check`, exact branch/full HEAD and frozen Git blobs. Reject any changed mask or tensor key, dtype, signed/nonfinite floating values where prohibited, altered tensor bytes, changed sequence order, unexpected cast, inconsistent manifest digest, or embedded runtime/E0 PASS. Stage A must emit `GPU=OFF; MODEL_LOAD=NO; SCIENTIFIC_EXECUTION=NO` and must never generate experimental output.

Stage B (not presently authorized): independent semantic verifier for lossless worker seam, memory headroom and guards; bounded exact-input GPU route checks only after new signed-off execution gate; unique run name and pinned code commit; no population merge. GPU output failure blocks promotion.

**Current disposition:** `24705c6_CODE_FREEZE=CONFIRMED`; `NEXT=CPU_STATIC_PROVENANCE_IMPLEMENTATION`; `GPU=OFF`; `CODEX=NO`; `TRAINING=NO`; `SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`; `COMMIT_PUSH=NO`.
