# PRE-URP HANDOFF

MARKER: PRE_URP_FINAL_HANDOFF
EVIDENCE_BOUNDARY: NOT_SCIENTIFIC_EVIDENCE
FORMAL_EXECUTION: NOT_STARTED
A0_AUTHORITY: NOT_CREATED_BY_DESIGN
P4L_GATE: BLOCKING_A0

## 1. Identity

- Current infrastructure HEAD: `a82b10caff1d7b5238a7faf6a605c7c989e18f5e`
- Active trainer: `scripts/train_controlled_v6b_minimal.py`
- Formal scientific execution status: `NOT_STARTED`
- Historical research-authority HEAD context: `346583b794ec5de9afae2b14a6af4dbdd9bbf90e`; this remains historical authority context and is not overwritten or reinterpreted by the current infrastructure HEAD.

## 2. Scientific Boundary

- P3-W7-A0 has not been executed.
- A0/A1/A2/A3 results are not established.
- Reason-router effectiveness and first-blocker effectiveness are not established.
- Seeds 180/181/182 have not been executed under formal P3-W7-A0 authority.
- Infrastructure validation is `NOT_SCIENTIFIC_EVIDENCE`.

## 3. Closed Pre-URP Infrastructure

The following milestones are `CLOSED` and should not be reopened absent a real failure:

- Research operations, state, and artifact documentation.
- Research-status, preflight, artifact-validator, and handoff-validator tooling.
- Kaggle environment verifier, persistent wheelhouse, and 10-package offline restore.
- Offline Hugging Face model cache and one-command Kaggle bootstrap.
- Mamba fast-path and CUDA smoke validation.
- Resume checkpoint infrastructure and immutable Kaggle resume store.
- Post-restart resume persistence, active trainer opt-in resume integration, and synthetic interruption/resume validation.
- Existing P4-X regression: 26 passed on Kaggle/Linux.
- HF cache integrity manifest, immediate verification, and post-restart verification.
- Persistent venv decision: `KEEP_CURRENT_BOOTSTRAP_ONLY`.

## 4. Canonical Data

Canonical controlled dataset path:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

- Physical SHA256: `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- Semantic SHA256: `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

The dataset must not be modified.

## 5. P4-L Blocking State

Expected sidecar path:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`

- Expected sidecar physical SHA256: `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- Expected sidecar semantic SHA256: `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

Expected provenance file:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`

- Expected provenance physical SHA256: `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`

The required bytes are absent locally. The exact external source location is `NOT_ESTABLISHED`; no external Kaggle path is asserted. Do not reconstruct, manually edit, normalize, or silently overwrite these files. Exact-byte provisioning with physical and semantic hash validation is required before formal A0 execution. This is `BLOCKING_A0`.

## 6. Formal Start Gate

At formal start, perform this order:

1. Resolve and provision the exact P4-L bytes.
2. Verify expected physical, semantic, and provenance identities.
3. Create an independent P3-W7-A0 execution authority.
4. Freeze the exact execution commit and commands.
5. Run local preflight.
6. Run Kaggle bootstrap.
7. Verify registration and provenance gates.
8. Execute authorized A0 runs only after all preceding gates pass.

No A0 execution is permitted before all blocking gates pass.

## 7. A0 Role

Established planning intent only:

- A0 is the P3-W7 baseline arm.
- Router: `explicit_product`.
- Ownership: joint.
- Primary reason CE: inactive; effective reason loss weight `0.0`.
- A0 alone is not evidence that first-blocker routing, reason supervision, or explicit-local ownership works.

This section is not execution authority.

## 8. Resume / Recovery Contract

- Latest resume is operational state, not the best scientific checkpoint.
- Resume is supported only at completed-epoch boundaries: `RESUME_POINT = AFTER_COMPLETED_EPOCH`.
- Restored completed epoch `E` continues at `E+1`.
- `continuation_index` tracks process/restart segments.
- Parent SHA chains to the immediately preceding persisted resume object.
- `DATA_ORDER_NOT_ESTABLISHED` remains the default unless separately established.
- Persistence failures fail closed.

## 9. Kaggle Recovery Contract

- Use the current bootstrap-only architecture.
- Do not introduce persistent venv/site-packages without new evidence.
- Persistent wheelhouse and model cache are the recovery sources.
- Bootstrap verifies live Python, torch/CUDA, extensions, and Mamba fast-path availability.
- HF cache integrity validates bytes and layout only.
- Model usability remains a separate environment-verifier concern.

## 10. Do-Not-Redo List

Do not redo without a real failure:

- Bootstrap implementation.
- Resume store.
- Trainer resume integration.
- Persistent venv feasibility.
- HF cache integrity.
- Synthetic resume validation.
- P4-X regression validation.

## 11. Immediate Next Action

Resolve and hash-validate P4-L exact-byte provisioning at formal start, then create/freeze the P3-W7-A0 execution authority.

No new infrastructure implementation before that.
