# Gen5 G1 frozen-P0 CPU installation — validated execution evidence

**Status:** `PASS_G1_FROZEN_FACTORS_SYNTHETIC_PARENT_NOT_PRETRAINED`
**Date:** 2026-10-14 (Asia/Seoul)
**Branch:** `gen5-causal-role-state-ownership`
**Execution HEAD:** `7e19ecbd61c1108282b1ad2602c8d6dfc94c7469`
**Historical Phase3A execution HEAD:** `d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`

## Inputs and provenance

- Execution script: `gen5_g1_frozen_checkpoint_installation_cpu_7e19ecb.py`, SHA256 `efa6e5873934eed54169e2f3fae71da83997a2a68434f98174ad94653f6d6cc3`.
- Output JSON: `gen5_g1_frozen_checkpoint_installation_cpu_7e19ecb.json`, SHA256 `b106d4ed2c6f338b145967c435bd6614273ceefc961795a7f20a370d8a341466` (SHA256 independently rechecked against the uploaded bytes).
- Preflight coefficient-contract JSON SHA256: `82f93e692a84c6890c36179fe522f1ee30e61ccf002f07836cfbe93f26a46641`.
- Historical frozen P0 correction checkpoints, respectively seeds 6201 / 6202 / 6203, SHA256:
  - `157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf`
  - `1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214`
  - `c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770`
- Historical parent checkpoint *identity declared by the P0 checkpoints*: `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`. The parent checkpoint itself was **not loaded or independently authenticated** in this run.

## Observations

The user-reported run and the uploaded JSON agree on 3 seeds x 6 diagnostic arms = **18 installed-and-restored conditions**, on a synthetic, pretrained-shaped layer-22 parent running on CPU. The G1 companion uses the genuine repository correction streaming backend and the three frozen P0 correction A/B tensors, while the parent native mixer and hidden input are synthetic.

- All 18 conditions installed and restored without changing parent parameters/buffers or the frozen A/B factor bytes.
- Duplicate installation and the legacy Phase2 mask helper were rejected; controlled exception rollback succeeded.
- The original FULL-wrapper comparison against the shared direct correction reference recorded zero discrepancy for each seed. This is a **shared-backend consistency check**, not an independent pretrained implementation comparison.
- All 18 arm output comparisons against their sum-of-donor correction reference recorded zero discrepancy. Correction-only comparisons subtract the synthetic native baseline, so FP32 subtraction/roundoff is relevant.

| Held-out seed | Maximum correction-only relative Frobenius error | FULL reconstruction relative error (total-output denominator) |
| --- | ---: | ---: |
| 6201 | 2.7724989595e-4 | 3.5597520872e-8 |
| 6202 | 3.6474792716e-4 | 3.5367279606e-8 |
| 6203 | 3.5620089839e-4 | 3.6016270051e-8 |

The maximum correction-only relative error over all 18 arms is **3.6474792716e-4** (seed 6202, `RESIDUAL_ONLY`), with observed componentwise maximum absolute discrepancy for that arm approximately **2.94e-8**. Do not confuse a total-output-relative reconstruction discrepancy with correction-only relative accuracy.

## Interpretation and boundary

**Validated:** CPU synthetic-parent installation, single native-mixer invocation in the diagnostic implementation, masking/rollback, frozen-factor identities and static arm coefficients, and frozen-source runtime insertion with the existing correction backend.

**Not validated:** loading/inserting into the historical pretrained Mamba-130M parent; authentic fast-kernel runtime behavior; real model forward, loss, or task metrics; shared-component functional preservation; prospective unseen-seed generalization; R22 ownership contention. No GPU, training or task evaluation was performed. This evidence does not grant permission for those operations.

**Next bounded action:** read-only G0 audit of the historical pretrained parent checkpoint and snapshot file identities/availability, without model construction or forward. Do not modify historical Phase2/3A code.
