# Gen5 Shared Bypass — Frozen-Checkpoint Synthetic Runtime Equivalence: Static Scientific Interpretation

**Status:** `FROZEN_CORRECTION_SYNTHETIC_RUNTIME_PASS / PRETRAINED_RUNTIME_NOT_TESTED / FUNCTIONAL_CORE_NOT_TESTED`
**Date:** 2026-10-14 (Asia/Seoul)
**Repository:** `9terry-student/ContraMamba`
**Branch:** `gen5-causal-role-state-ownership`
**Analysis HEAD:** `429a770b869ccbaad8bf843af4a3fb7e074d8b34`
**Historical Phase3A checkpoint execution HEAD:** `d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`
**Execution type:** `READ_ONLY_CPU / SYNTHETIC_NATIVE_MIXER / ACTUAL_FROZEN_P0_A_B_FACTORS`
**No:** real pretrained model forward, research-data forward, task evaluation, training, backward, GPU, Kaggle, inferential p-values, or Codex.

## 1. Reviewed evidence and source identity

The user executed the bounded Python program and supplied its result JSON. The reviewer independently verified the JSON file SHA256, checked its linkage to the previously verified arm preflight, checked all reported seed/arm accounting and tensor identities against that preflight, and recomputed derived error normalizations. The reviewer **did not** independently rerun the PyTorch experiment on the actual checkpoint files in this review.

- Result JSON: `gen5_frozen_checkpoint_synthetic_runtime_equivalence_429a770.json`
  SHA256 `96fd56e3a0bacd90b3ac939ec3c910bfcd3503bd2458bb5c47b5afd0769cb042`.
- Executed Python program: `gen5_frozen_checkpoint_synthetic_runtime_equivalence_429a770.py`
  SHA256 `ce89364f9be17894d7d01480bfc06980d4452c87b84e4e5b011556bdb613d78f`.
- Frozen six-arm preflight JSON SHA256: `82f93e692a84c6890c36179fe522f1ee30e61ccf002f07836cfbe93f26a46641`.
- Checkpoint source: three `G5-C0 / P0` final-step A/B factor checkpoints at seeds 6201, 6202, 6203, from the same Phase3A run and train/dev lineage.

| Seed | Exact checkpoint SHA256 |
|---|---|
| 6201 | `157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf` |
| 6202 | `1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214` |
| 6203 | `c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770` |

The program used `torch.load(..., map_location="cpu", weights_only=True)` after checkpoint-file SHA validation; validated source tensor types/shapes/exact hashes; placed the frozen factors into the existing `StateWriteCorrection`; reused the repository's original rank-2 reference and streaming correction machinery; and instantiated a **synthetic**, fixed layer-22-shaped mixer and synthetic input, not the original Mamba-130M backbone. It reported source and synthetic-mixer identity unchanged.

## 2. Bounded validations and their independence

All `3 seeds × 6 arms = 18` frozen-coefficient conditions were evaluated. The six arm names were `ZERO`, `FULL`, `SHARED_ONLY`, `RESIDUAL_ONLY`, `SHARED_NORMMATCH`, and `RESIDUAL_NORMMATCH`. Arm coefficient vectors were checked against the exact frozen preflight JSON.

| Verification | Observed result | What it does not establish |
|---|---|---|
| Historical reference recurrence vs streaming accelerated recurrence, source correction | Zero reported discrepancy for all 3 seeds on the synthetic fixture | Full pretrained-model behavioral equivalence |
| Historical Phase2 `FULL` wrapper vs new companion `FULL` | Zero reported discrepancy for all 3 seeds | Independence of the two paths: they reuse the same accelerated correction backend |
| All 18 companion arm outputs vs expected weighted sums of source streaming contributions | Zero reported discrepancy | Independent reference evaluation for all weighted arms: the expected values are assembled using the same source streaming contributions |
| `SHARED_ONLY + RESIDUAL_ONLY - ZERO` vs `FULL` | Relative full-output discrepancy 5.81e-8–6.19e-8 | Machine-precision correction-only equivalence or task sufficiency |
| All-zero active mask vs native synthetic output | Zero reported discrepancy | Correct behavior on actual tokenized parent-model data |
| Frozen source and synthetic-mixer parameters | Unchanged in run report | Identity of a pretrained backbone: none was loaded |

`REFERENCE_REL=0` and `OLD_FULL_REL=0` in the console are correct bounded observations for the synthetic fixture; they are not evidence that two independently implemented pretrained models were compared. The native fixture is synthetic and the companion explicitly calls the historical streaming implementation.

## 3. Reconstruction error and denominator audit

The supplied JSON reports full-output-relative reconstruction errors. Because the synthetic native output norm is about 10.8565 while the source correction norm is only about 0.0008–0.0020, full-output normalization substantially suppresses the apparent relative discrepancy.

Define `E_s = SHARED_ONLY_s + RESIDUAL_ONLY_s - ZERO_s - FULL_s`. With the JSON's reported full-output norm and full-output relative error, the Frobenius error norm is reconstructible as `||E_s|| = reported_full_relative_error × ||FULL_s||`. Dividing by the *isolated source correction* norm (from the reference-versus-streaming entry) yields a more relevant diagnostic scale:

| Seed | Isolated source correction norm | Full-output-relative `||E||/||FULL||` | Derived `||E||/||source correction||` | Max absolute entrywise error |
|---|---:|---:|---:|---:|
| 6201 | 0.0013923134 | 6.1870752e-8 | **4.8243837e-4** | 1.1920929e-7 |
| 6202 | 0.0008088064 | 6.0545534e-8 | **8.1269792e-4** | 1.1920929e-7 |
| 6203 | 0.0019686611 | 5.8142743e-8 | **3.2064000e-4** | 1.1920929e-7 |

The derived correction-relative discrepancy spans **0.0321%–0.0813%**. This is small in the tested float32 synthetic recurrence, and consistent with accumulated arithmetic/rounding, but it should **not** be described as `~1e-8` correction-only relative error. These numbers are derived from recorded norms, not from newly executed tensor forwards.

The earlier operator-level, float64 CPU preflight established `FULL = SHARED + RESIDUAL` to approximately `1e-16` on canonical synthetic inputs; that is a different numerical contract from repeated float32 native-output additions in the present test. Both results are compatible.

## 4. Retained scientific boundaries

**Code correctness:** the narrow frozen-source loading, coefficient selection, real historical correction-backend invocation, historical FULL compatibility, active masking, and arithmetic reconstruction checks are supported **for the tested synthetic mixer**.

**Execution success:** CPU synthetic runtime completed for all 18 arm conditions. No training, backward, optimizer, scientific data evaluation, or GPU execution occurred.

**Artifact/provenance:** the delivered result JSON SHA256, prior-preflight SHA256, checkpoint/tensor identity declarations, arm cardinality, and unchanged-parameter flag reconcile. This review independently authenticated the delivered JSON and its reference preflight, but did not reread the original `.pt` checkpoint bytes from the user's machine.

**Scientific conclusion:** no task-functional shared core, cross-seed generalization, R22 contention, native-causal role ownership, or pretrained Mamba-130M runtime claim is established. The existing Phase2 ownership negative result and Phase3A contention-manipulation failure remain unchanged. The three seeds share the same Phase3A dataset lineage and are not independent task-level replications.

The earlier geometry result remains descriptive: mean actual donor-operator span coverage approximately `0.280930895061`, with `0.719069104939` unexplained operator energy. This does not imply any proportion of task gain without an authorized functional experiment.

## 5. Next stage and non-authorizations

The frozen-factor/synthetic-mixer equivalence contract is **complete at its declared scope**. Repeating more random synthetic fixtures or adding unrelated CPU numerical audits is not needed to promote its existing conclusion.

A subsequent **prospective, static-only pretrained-insertion contract review** may specify how to compare the original frozen parent layer-22 site with the companion under the same attention mask, native forward semantics, numerical tolerance, native correction-only denominator, model checkpoint identity, and six-arm coefficient identity. Such a review is a design/feasibility step, not execution authority.

**Do not run** pretrained Mamba forward, new model evaluation, confirmatory task data, training, GPU, or Kaggle on the authority of this report. No code modification is required by the present result. Any prospective functional-core experiment must separately freeze its dataset, comparison conditions, endpoints, intervention controls, evaluation mode, interpretation/falsification criteria, and execution authorization before forward passes.

## 6. Disposition

- `SOURCE_FROZEN_FACTOR_AUTHENTICATION=PASS_REPORTED_IN_EXECUTION`
- `SYNTHETIC_MIXER_REFERENCE_STREAMING_EQUIVALENCE=PASS`
- `COMPANION_HISTORICAL_FULL_COMPATIBILITY=PASS`
- `SIX_ARM_FROZEN_FACTOR_RUNTIME_CONSTRUCTION=PASS_SYNTHETIC_SCOPE`
- `CORRECTION_RELATIVE_RECON_MAX=0.000812697923874009`
- `PRETRAINED_RUNTIME_EQUIVALENCE=NOT_TESTED`
- `SHARED_FUNCTIONAL_CORE=NOT_TESTED`
- `R22_OWNERSHIP_CONTENTION=NOT_ESTABLISHED`
- `TRAINING=NO; TASK_EVALUATION=NO; GPU=NO; KAGGLE=NO; CODEX=NO`
- `NEXT=STATIC_PRETRAINED_INSERTION_CONTRACT_REVIEW_ONLY`

This is a documentation-only interpretation artifact. It is not an execution authority, numeric promotion of the native pretrained model, or a scientific performance claim. The two helper scripts and off-repository JSONs are not automatically archived by committing this report.
