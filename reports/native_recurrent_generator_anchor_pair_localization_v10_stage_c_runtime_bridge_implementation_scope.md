# Gen5 v10 — Stage C Bounded Runtime Bridge / V3 Offline Verifier Implementation Scope v1

**Status:** `STAGE_C_IMPLEMENTATION_SCOPE_CANDIDATE / CPU_STATIC_ONLY / NO_GPU_EXECUTION_GATE`
**Date:** 2026-10-11 (Asia/Seoul)
**Branch / worktree:** `gen5-causal-role-state-ownership` / `C:\g5-state`
**Frozen predecessor commit:** `6683d4c0dc9e742f5ac503fe4851e1c5218b99d8` (five-file Stage B code-only freeze).
**Phase:** new implementation candidate with CPU tests and independent semantic audit; no Kaggle, CUDA experiment, model forward, training, evaluation, artifact collection, or population execution.

## 1. Motivation and hard phase boundary

Stage B introduced `GEN5_V10_BOUNDED_PAIRED_EXPORT_V2` and `GEN5_V10_BOUNDED_PAIRED_VERIFICATION_V2` as strictly **CPU synthetic-only** contracts. The frozen V2 verifier explicitly requires `runtime_claims.source_kind=CPU_SYNTHETIC_ONLY`, `execution=NO`, `physical_device=CPU`, a synthetic input manifest, and `code_head=f8e60f173d3a72aa17779449ec9b931c7fa4d043`. It cannot legitimately verify an actual CUDA bundle. Do not relax, repurpose, or edit V2 to accept actual runtime records. Likewise, the Stage A manifest attests only provided CPU tensor values; and the paired adapter rejects CUDA tensors.

The Stage C objective is **to implement and independently review a bounded runtime seam as callable, testable code** and an independently versioned **V3 offline record contract**. No executable GPU experiment is authorized merely by having code that accepts tensors originating from CUDA. During this phase, the public CLI must expose only a static plan / contract check, not a model-load, `--run-bounded`, `--run-worker`, `--run-full`, `--merge-only`, or preflight-execution mode. No Kaggle notebook cells shall be run.

## 2. Immutable references and exact target locus

Never modify these tracked files or their ancestors/guards:

- Original frozen audit `scripts/audit_native_recurrent_generator_anchor_pair_localization.py`: blob `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`.
- Original frozen audit tests: blob `26c90ada9a74a2ac2716c880e58d15df5bd01704`.
- Hybrid reducer `scripts/gen5_v10_hybrid_fine_pair_candidate.py`: blob `56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0`.
- Shared GEMM `scripts/gen5_v10_shared_dyadic_gemm_candidate.py`: blob `9003fb9bd23b2a6680ef210a51845058622badd2`.
- CPU V1 verifier `scripts/gen5_v10_e0e3_record_verifier_candidate.py`: blob `7f547adc42a6e113dd179cb0f4988336ee476b17`.
- CPU adapter `scripts/gen5_v10_bounded_pair_adapter_candidate.py`: blob `d2f1535fe19b07483911d3293be5d0be9d400732`.
- Stage A provenance `scripts/gen5_v10_bounded_tensor_provenance_candidate.py`: blob `703e0f6faf15ee499ec6a97f1aa8edc8d2a93333`.
- Stage B static worker `scripts/gen5_v10_bounded_worker_export_candidate.py`: blob `cbfc6004c6ab177eea8e7def1cc001049ec4bc4e`.
- Stage B static V2 verifier `scripts/gen5_v10_e0e3_v2_bundle_verifier_candidate.py`: blob `674f8a22b0b9ddaa51f60ff126c0f152c0b49e31`.
- Approved E2/E3 numeric contract: `reports/native_recurrent_generator_anchor_pair_localization_v10_numeric_contract_and_candidate_freeze_spec.md`, blob `661c627f09034c34f9830b18b7e9938c72c05925`.

Bounded target: worker `1`; one-based worker batch `13`; **global dev rows `[800,832)`**; K=4, B=32, S=128, W=24576; source `(6201,6201)`; target group `(6201,6202)`, `(6201,6203)` in that exact order, both `CONTROL_R`. These identities are resolved from the original `FULL_FACTORIAL_CELLS`, `all_orientations()`, and `TARGETS_PER_TRANSPORT=2`, not inferred from a custom reordered list. Frozen B8 tail `[832,840)` and all other source groups remain out of scope.

## 3. Authorized candidate additions — five files, no edits to earlier ones

1. `scripts/gen5_v10_bounded_runtime_bridge_candidate.py`: a **new explicitly versioned**, bounded callable runtime bridge that delegates to frozen runtime preparation, anchor-mask, model/gradient/projector, coarse and replay functions rather than modifying or monkeypatching originals. In this implementation phase, the CLI supports only `--static-contract-check`; the bridge must not launch or load an actual model from the CLI.
2. `scripts/gen5_v10_e0e3_v3_runtime_verifier_candidate.py`: an **independent offline V3** schema validator for proposed real-source bounded records, with strict identity, hashes, geometry, lineage fields and guard/failure evidence. Its CLI remains CPU/static only for Stage C; reading synthetic test objects is permitted. Never silently pass V2 bundles through V3, and do not mutate V1/V2.
3. `tests/test_gen5_v10_bounded_runtime_bridge_candidate.py`: deterministic CPU mocked-seam tests, including positive selected locus, frozen call order/count, forbidden alternative source/group, input mutation, error/failure propagation, no entrypoint and no unintended source-side effects.
4. `tests/test_gen5_v10_e0e3_v3_runtime_verifier_candidate.py`: CPU-only independently constructed synthetic V3 records, strict negative cases for source/row/order/byte hashes, incomplete upstream guards, omitted fields, diagnostic forgery, finite/normalization failures, source claims, schema misuse and artificial E0–E3 promotion.
5. `reports/native_recurrent_generator_anchor_pair_localization_v10_stage_c_runtime_bridge_implementation_scope.md`: this exact document.

**Whitelisted delta:** these five newly added files only. No changes to original audit, frozen hybrid and adapters, existing V1/V2 schemas, original execution gate, models/tokenizer/checkpoint, evidence, data, tests, or prior reports. No staging, commit or push until independent code-review and explicit manual instruction. Codex is not used.

## 4. Mandatory semantic equivalence and failure preservation

Review the frozen worker control flow at the exact `combined = torch.cat((visible, complement), dim=0)` seam. For one selected existing batch/source/two-target group only, preserve *all preceding authentic calls* to build batch features/masks, `context`, `pair_log_a`, `pair_survival`, raw writes, source gradients and the unchanged projector. Do not run the 36 orientations or all 840 rows as an implementation test. Do not invoke the frozen `run_full()` or `run_worker()` expecting it to dispatch v10: original `_spawn_worker()` launches its own `__file__`.

A real future paired call must send **one and the same `combined`, log_a, survival, attention_mask, role_masks, fine_masks and shape** first through frozen `_component_fine_pair_stats` and then independently through Hybrid plus the strictly matched *post-reducer* parent reconstruction. Replay must call the authentic `coarse._replay_vector` and preserve `abs_error <= 2e-5 OR relative_error <= 2e-5` and `ENERGY_EPSILON=1e-30`. No copying or replacing engine diagnostics. Execute and attribute all original raw-component, projector, coarse, source-target and fine-parent guards; a trace of only fine-parent replay **does not** prove complete E1. Missing guards or any raised frozen exception block, without forging completed artifacts.

Explicitly validate selector bounds **before** any possible model/gradient work. Record exact row IDs and original target order. Do not assume CPU-only adapter accepts GPU tensors. Do not hide a GPU launch behind an import, `--static-contract-check`, tests or an unguarded callable invoked by a CLI.

## 5. GPU input bytes, E0 and bounded memory

The 17 logical input tensors are those of the frozen Stage A manifest (three float32 tensors plus 14 masks). For K4/B32/S128/W24576, float inputs amount to about 2.25 GiB; therefore no full extra GPU clone, no whole-input GPU-to-CPU materialization, and no concurrent full copies. Use the Stage B `capture_value_digest` chunk semantics (C-order float32 little-endian / bool 0–1), independently confirming stage-A consistency with CPU fixtures. Plan synchronization and sequential digest captures at PRE_FROZEN, POST_FROZEN, POST_V10; compare all 17 per-tensor content digests and their bindings, shape/dtype/order before calling the record equivalent. Fail closed on any mutation. Record device mapping and strides separately.

Digests of self-supplied tensors prove only value identity. Real E0 additionally requires actual tensor source authentication: pinned full code HEAD/blobs and clean worktree, exact model/tokenizer revision, checkpoint SHA256 `1ff3fcf2ebd754ab6f9483d6a9982b9b04b3357f9f8cdbe2b30399e7d2f`, dev order SHA256 `b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25`, active encoding SHA256 `e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51`, shard/row/batch/target identity, physical GPU, torch/CUDA/cuBLAS version, TF32/deterministic flags and validated authentic upstream chain. Never assert E0 PASS based only on a JSON hash, tensor object identity or path name. The CPU/static phase cannot authenticate a live CUDA source.

## 6. V3 record schema and verdict floor

Use a **new** strict schema family, e.g. `GEN5_V10_BOUNDED_RUNTIME_EXPORT_V3` and `GEN5_V10_BOUNDED_RUNTIME_VERIFICATION_V3`, not the CPU-only V2 identifier. Require exact selector, versioned code identity and runtime/source-evidence status, three-phase 17-tensor captures with byte counts, independently identifiable frozen/v10 K4 raw/coarse/fine records, their own SHA256, actual frozen guard coverage/failures, v10 dispatch with full chunk coverage, and independent bundle/sidecar hashes. Preserve genuine `fine_parent_reconstruction_abs_max` and `fine_parent_reconstruction_rel_max`, actual `LEGACY_FULL_RECORD=FAIL` where applicable, complete H-by-gap (38,100 values) and H-window (1,500 values) bounded comparisons. E3 `NONCOMPARABLE` requires independently supported numeric accumulation-order sensitivity, never simply a different diagnostic value.

Fail-closed verdict floor in Stage C: `V10_E0_PROVENANCE=NOT_EVALUATED`, `V10_E1_FROZEN_GUARDS=NOT_EVALUATED`, `V10_E2_PAYLOAD=NOT_EVALUATED` unless a payload mismatch demands `FAIL`, `V10_E3_DIAGNOSTICS=NOT_EVALUATED`, `V10_PAYLOAD_EQUIVALENCE=NOT_EVALUATED` unless `FAIL`; `PRODUCTION=BLOCKED`; `SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`. **CPU tests cannot yield formal E0/E1/E2/E3 PASS.** A new future execution authority, independent semantic audit, artifact writer and provenance-backed runtime verification will be needed for any live bounded CUDA run.

## 7. Independent audit and go/no-go criteria

Review the relevant original `run_worker` statements line-by-line; check actual role/fine mask source, single batch/source/group, target ordering, projector-gradient ownership, raw reconstruction and logging with a separate verifier. Execute only CPU synthetic unit tests, static CLI and Python compile/AST checks, `git diff --check`, plus narrow earlier-stage CPU regressions when affected. New tests must fail for a missing guard trace, omitted blob, unsupported head, extra schema field, tampered tensor digest, replay forged diagnostics, wrong source/target order, CPU/GPU source claim conflation, and a spurious formal PASS. Inspect exact new-file paths and SHA256. Do not run model forward, train, execute CUDA or Kaggle; do not create a runnable GPU entrypoint, commit, push or stage unreviewed files.

**Stop** on any incompatible prior schema, unknown actual code/branch head, required modification to a frozen file, missing original upstream guard coverage, noncanonical digest order, excessive GPU scratch budget, undocumented CUDA side effect, broken source-identity attribution or scientific promotion without evidence.

**Disposition:** `STAGE_B_CPU_STATIC_FREEZE=CONFIRMED`; `NEXT=STAGE_C_CPU_STATIC_RUNTIME_BRIDGE_AND_V3_IMPLEMENTATION`; `GPU=OFF`; `CODEX=NO`; `TRAINING=NO`; `SCIENTIFIC_EXECUTION=NOT_AUTHORIZED`; `COMMIT_PUSH=NO`.
