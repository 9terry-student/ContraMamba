# Gen5 v10 — CPU-only E0–E3 Record Verifier Implementation Scope v1

Status: `IMPLEMENTATION_SCOPE_FOR_CPU_RECORD_CANDIDATE_ONLY / NO_RUNNER / NO_GPU`

## Authority and versioning

- Frozen code-only candidate commit: `14608d234f6c0e1e8d7cd446b3e384cf77fc86da` on `gen5-causal-role-state-ownership`.
- Versioned E2/E3 policy: `reports/native_recurrent_generator_anchor_pair_localization_v10_numeric_contract_and_candidate_freeze_spec.md` at that commit (blob `661c627f09034c34f9830b18b7e9938c72c05925`).
- Frozen scientific audit script and tests remain byte-identical: blobs `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`, `26c90ada9a74a2ac2716c880e58d15df5bd01704`.

## Explicit authorized phase and scope

Phase: **new independent CPU-only verifier candidate**, not existing scientific execution. Add only:

1. `scripts/gen5_v10_e0e3_record_verifier_candidate.py`
2. `tests/test_gen5_v10_e0e3_record_verifier_candidate.py`
3. This scope document, `reports/native_recurrent_generator_anchor_pair_localization_v10_e0e3_record_verifier_implementation_scope.md`.

Purpose: take previously exported frozen-vs-v10 K4 raw batch records and strictly check all known fields, original per-gap fine-parent replay, recomputed diagnostic truth, and derived normalized fine H-by-gap / five frozen H windows; reject forged/missing/nonfinite payload and retain legacy full-record failures. Show bounded comparisons **only**.

Accepts only exact legacy bounded-export schema `GEN5_SHARED_GEMM_CPU_EQUIVALENCE_DRAFT_V1`; the exporter declares frozen source HEAD `e145d7d...`. Do not confuse that evidence source HEAD with the candidate-code HEAD `14608d2...`. Authenticate the clean repository with an explicitly supplied **full expected future verifier-code HEAD** (never infer a moving HEAD), verify ancestry from the 14608d2 frozen candidate base, and verify six known frozen/candidate/contract Git blobs separately; those code checks do not authenticate exported tensors. Never hardcode an unknown future freeze commit.

### Mandatory fail-closed limits

- `V10_E0_PROVENANCE=NOT_EVALUATED` whenever original tensor bytes and hardware/runtime lineage cannot be independently authenticated by the verifier.
- `V10_E1_FROZEN_GUARDS=NOT_EVALUATED` because exported reconstructed fine-parent replay does not prove all upstream guards executed. Guard failures that can be checked from exported values must block.
- `V10_E2_PAYLOAD=NOT_EVALUATED` even if all available bounded numeric payload values compare successfully, because unexported source-matched orientation and population aggregation are unknown. A real bounded mismatch yields `FAIL`.
- `V10_E3_DIAGNOSTICS=NOT_EVALUATED` if numerical diagnostic differences are present but the engine-sensitive cause has not been independently proven; raw engine values and recomputation must still be reported. Preserve `LEGACY_FULL_RECORD=FAIL` when observed.
- `V10_PAYLOAD_EQUIVALENCE` cannot be PASS in this stage.
- Never infer GPU experiment/840-row scientific equivalence from exported records.

Tests: positive bounded synthetic and actual pre-existing exported ZIP as *regression* only; negatives for tampered parent, forged diagnostic, omitted field/cell, nonfinite, wrong K/row shapes, payload mismatch, normalized H mismatch, missing provenance. Do not edit, overwrite, or back-fill user artifacts.

**Do not change:** frozen script/tests/gate, existing v10 candidate blobs, data, checkpoint, tolerance, schemas or runtime. Do not run Kaggle, GPU, model inference, training or evaluation; do not add a production runner or execution authority. Commit/push: **NO** until independent static review and explicit manual command. Codex: **NO**.

After verifier code review, the subsequent stage is a new versioned v10 runner integration authority, followed by an independently authenticated bounded mixed-route CUDA preflight; neither is authorized here.

Implementation-specific decision: E3 comparison of unauthenticated raw records is presented in `E3_RAW_NUMERIC_COMPARISON`; formal `V10_E3_DIAGNOSTICS` remains `NOT_EVALUATED` even when these raw values agree.
