# Gen5 v10 Stage E2 — Local Source Byte Verification (CPU-only candidate)

**Status:** `E2_CPU_LOCAL_BYTE_VERIFICATION_CANDIDATE / LIVE_PROVENANCE_NOT_ESTABLISHED / NO_GPU_GATE`
**Frozen base:** `9b6fcc8150c45e717ff4c0d8f1d7f39731566f9a` (`gen5-causal-role-state-ownership`, `C:\g5-state`). Code-only predecessor E1 was committed/pushed and its exact three-file scope independently checked against remote GitHub.

## Objective and boundary

E1 bounded an injected CPU helper sequence; it does **not** authenticate the bytes actually used to build the original model or the GPU runtime. Stage E2 changes the evidence type: it **computes hashes over actual provided local files and supplied CPU tensors**, using exact frozen canonical digest definitions, not self-asserted SHA strings. This is still not a live runtime capture or source-origin attestation: a caller can supply correct files/CPU tensors and later execute a different model. No model weights are loaded, no tokenizer is invoked, no re-encoding is performed, no inference/training is performed, and no CUDA API is called.

## Exact three new files

1. `reports/native_recurrent_generator_anchor_pair_localization_v10_stage_e2_local_source_byte_scope.md` (this document)
2. `scripts/gen5_v10_stage_e2_cpu_source_byte_verifier_candidate.py` (actual-byte digest functions, exact canonical dev-order/encoding verification, static/Git CLI only)
3. `tests/test_gen5_v10_stage_e2_cpu_source_byte_verifier_candidate.py` (CPU fixture/negative tests; synthetic test reference pins patched **only in tests**, never evidence)

**Immutable:** frozen worker/original numerical paths, Stage A/B/C, Stage D1–D4, E0/E1, checkpoint/model/tokenizer/static datasets and all earlier reports and artifacts. No Codex, runtime execution, GPU, Kaggle, training, evaluation, artifact write, or commit/push in installation phase.

## Source pin algorithms and physical bytes

`CHECKPOINT_SHA256=1ff3fcf2ebd754ab6f9483d6a9982b9b04b3357f9f8cdbe2b30399e7d2f` is the **raw parent checkpoint file SHA256**; E2 streams the actual path in at most 8 MiB host blocks and refuses checkpoint symlinks, changed size/mtime during read and mismatch. This does not invoke `torch.load`.

The exact model revision directory `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37` must contain the four frozen pinned files: `config.json` (895 bytes), `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`. Files are read and hashed, never imported or instantiated. Ordinary HF snapshot file symlinks are allowed because content hashes refer to resolved bytes, not names.

`DEV_ORDER_SHA256=b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25` is the SHA256 of exact `p3a.row_order_sha256` canonical JSON over **840 ordered dev rows**, not the hash of an arbitrary raw JSONL file. E2 reconstructs this layout from the six exact fields and separately checks 840 unique `row_id` strings.

`DEV_ENCODING_SHA256=e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51` is the digest of **already-encoded 840-row CPU tensors** using frozen `p3a.encoded_bundle_sha256`: named lists, dtype/shape prefixes, contiguous tensor bytes, and line delimiters in frozen order. E2 checks `input_ids`, attention/claim/evidence masks, final labels, stressor flags, targets, exact row-ID/pair-ID/cell-ID bindings, EOS=0 serialization, and frozen active-token total 60,094. It then computes the pinned digest. **No tokenizer call**. The bounded 32 `row_ids` `[800:832]` are required to match between dev rows and encoding.

Frozen Git blob identities also pin Phase3a serializer `45e333128c4fa31f4f502c5fdcf324243c1084fd`, Phase2 snapshot resolver `8f711776c6b8cab90fbdafcda166780f3295e4ae`, transport runtime source `984319fecb5ed2c103b92da0ac868ac7746770d8`, original worker `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635` and Stage E1 `5cd4e67b94dcb5c4612613f8fdb9eda600be03fd`.

## Evidence classification

Successful `audit_local_source_bytes(...)` indicates only the hash identities of **provided CPU-side inputs at the instant inspected**. It must return `local_source_bytes_match_pins=True`, but *always* `checkpoint_bytes_loaded_into_model_authenticated=False`, `active_gpu_tensor_lineage_authenticated=False`, `runtime_module_identity_authenticated=False`, `physical_gpu_memory_measured=False`, `GPU_EXECUTION=SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`, `E0=E1=E2=E3=NOT_EVALUATED`, `PRODUCTION=BLOCKED`.

No external source file paths are accepted by CLI in this phase; the byte auditor is an **unexecuted callable for later independently authorized staging**. No synthetic test fixture output is scientific evidence. Peak CUDA memory with concurrent frozen FFT, Hybrid GEMM and input copying remains unmeasured. The live model checkpoint identity, exact tokenizer invocation path and dynamic CPU→CUDA tensor lineage still require process-bound capture and independently authorized runtime execution. The E2 source-byte candidate does not resolve these execution blockers.

## Validation

Only run `TestE2LocalSourceBytes` CPU tests (33, no skip); `--static-contract-check`; `--inspect-local-git` at exact base HEAD (read-only Git metadata and frozen source blobs). Tests may patch pins only for synthetic CPU fixture cases; never use these patched pins to claim real provenance. No model load, tokenizer encode, GPU visibility, raw checkpoint deserialization, Kaggle or output artifact.

Stop if a test fails/skips, original Git blobs drift, a previously tracked file changes, actual local-source checks are promoted to real runtime E0–E3 PASS, or any runnable CUDA execution mode appears. **Commit/Push: NO pending operator validation and review**.
