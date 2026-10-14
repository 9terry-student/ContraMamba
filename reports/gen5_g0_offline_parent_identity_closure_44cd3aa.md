# Gen5 G0 — Offline Parent and Snapshot Identity Closure

**Status:** `G0_LOCAL_BYTE_IDENTITY_PASS / PRETRAINED_RUNTIME_NOT_TESTED`
**Audit repository HEAD:** `44cd3aa3f897b9a5a70a4ddb9c8407d63c95ba17`
**Branch:** `gen5-causal-role-state-ownership`
**Date:** 2026-10-14 (Asia/Seoul)

## Evidence and scope

- User-executed offline audit JSON: `gen5_g0_pretrained_parent_identity_44cd3aa.json`, SHA256 `7ebc453639885a73701c5bcfd313bf7fc53904bb9ea02dea4b6b5e25d822433e`.
- Executed offline audit script: `gen5_g0_pretrained_parent_offline_identity_audit_44cd3aa.py`, SHA256 `7ddf243042840b56199cd50938454525a550ec55bfeffe13a97a922efe0de545`.
- This report intentionally excludes local absolute paths and unsuccessful checkpoint-candidate paths from the original JSON. The original audit output remains outside the repository.
- Offline execution: no checkpoint deserialization, model construction, network access, model forward, training, task evaluation, or GPU use.

## Verified input identities

| Object | Frozen identity | Result |
|---|---|---|
| Model | `state-spaces/mamba-130m-hf` | Named lineage matched |
| Snapshot revision | `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37` | Local revision path matched |
| Model config | SHA256 `784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a` | PASS |
| `tokenizer.json` | SHA256 `b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf` | PASS |
| `tokenizer_config.json` | SHA256 `9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb` | PASS |
| `special_tokens_map.json` | SHA256 `57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8` | PASS |
| Historical `G3-GROUP-D-HALF` seed-180 parent checkpoint | SHA256 `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f` | PASS |
| Frozen Phase3A `P0`, seed 6201 | SHA256 `157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf` | Provenance PASS |
| Frozen Phase3A `P0`, seed 6202 | SHA256 `1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214` | Provenance PASS |
| Frozen Phase3A `P0`, seed 6203 | SHA256 `c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770` | Provenance PASS |

All three P0 sidecars passed audit, identify historical parent checkpoint SHA256 `1ff3fcf2...`, and record matching parent signature `b0b62100691686acf2597159add2a9e252a757f63f61ef52115136d9cf33a3c9`. G1 evidence linkage passed.

## Static loader and integration boundary

At the audited HEAD, the historical model constructor in `scripts/reason_router_gen4_six_cell_tier2_scientific_inference.py` authenticates the frozen config, constructs `MambaModel(config)`, and subsequently uses strict state-dict loading of the separately authenticated historical parent checkpoint via `scripts/train_reason_router_gen5_phase2_state_update_ownership.py`. Therefore the four authenticated snapshot metadata/tokenizer files do not themselves imply that a pretrained model-weight file is present in that snapshot. G0 did not validate model construction or strict load at runtime.

`src/contramamba/gen5_shared_bypass_g1_installation_diagnostic.py` remains synthetic/CPU-only, rejects nested wrappers, and uses `g1_companion_active_mask`; the existing Phase2/3A helper requires `Phase2Layer22MixerWrapper` instead. An isolated diagnostic-only integration path is still required before any genuine pretrained-backbone execution. No modification to the legacy runner is justified by G0.

## Decision

`G0_OFFLINE_PARENT_SNAPSHOT_AND_FROZEN_P0_IDENTITY=PASS` for the pinned metadata and historical checkpoint bytes. `PRETRAINED_CONSTRUCTION`, `STRICT_RUNTIME_LOAD`, `REAL_LAYER22_FORWARD`, `TASK_EVALUATION`, and `SHARED_FUNCTIONAL_CORE` remain **NOT TESTED**. The next authorized step is a static integration-contract review only; G0 PASS does not authorize model forward, evaluation, training, or Kaggle/GPU work.
