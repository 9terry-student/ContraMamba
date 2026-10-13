# Gen5 v10 two-locus FFT–GEMM bounded evidence analysis (candidate)

**Status:** `BOUNDED_IMPORTED_EVIDENCE / STATIC_NUMERIC_AUDITED / FORMAL_E0_E3_NOT_PROMOTED`  
**Report date:** 2026-10-13 (Asia/Seoul)  
**Scientific decision:** `LEGACY_FULL_RECORD=FAIL`; `V10_E2_BOUNDED_NUMERIC=PASS_PROVISIONAL`; `V10_E3_BOUNDED_RAW=DIVERGENCE_UNRESOLVED`. No E0–E3 formal promotion and no population-level equivalence claim.

## 1. Identity and scope

- Repository/branch: `9terry-student/ContraMamba`, `gen5-causal-role-state-ownership`.
- Pinned run commit: `7f74a84ed19e49727e3adf69c016af9e4339fb86` (two-locus bounded CUDA runner).
- Run name: `gen5-v10-two-locus-7f74a84-r1`.
- Pinned command SHA256: `9d245a161ac1dc1d7cfd6bd41fbbdb0498de696f0bc2b31b38c5a31a3afc9e89`.
- Run UTC: `2026-10-13T13:15:05Z` to `2026-10-13T13:19:07Z`, exit code 0; both workers exit 0.
- Handoff ZIP SHA256: `0c47a22b83dfbb16c9e87d7e020a88f40edb50209be5e12788073c39168cb16b`. Collect PASS (8 files), Import PASS (`VALIDATED=8; COPIED=8`); import audit performed at the pinned head with a clean worktree. The ZIP identity and file validation establish collection/import integrity, **not** formal independent E0 model-runtime lineage attestation.
- Frozen model/tokenizer revision: `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`; selected seed180 checkpoint SHA256 `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f`.
- Kaggle provisioning demonstrated Python 3.12.13, PyTorch 2.10.0+cu128, Transformers 5.0.0, tokenizers 0.22.2, kernels 0.10.2, with exact Mamba and causal-conv1d binary SHA256; CUDA preflight independently observed two Tesla T4s with sufficient memory. These are operator-provided execution/provision logs, not third-party attestation of the complete execution lineage.

| Locus | GPU | Frozen worker / rows | Source | First two frozen targets | Measured routing |
|---|---:|---|---|---|---|
| `gpu0` | 0 | worker 0, `[0,32)` | `(6202,6203)` | `(6201,6203)`, `(6202,6201)` | `FAST=0, GEMM=96` |
| `gpu1` | 1 | worker 1, `[416,448)` | `(6203,6202)` | `(6201,6202)`, `(6202,6202)` | `FAST=0, GEMM=96` |

Each locus: `K=4`, `B=32`, `S=128`, `W=24576`, one frozen full-width FFT reducer and one v10 chunkwise hybrid reducer on the same captured inputs. Two independent masked single-GPU processes; no DDP, optimizer or training. **All 96 v10 chunks at each locus used GEMM.** This is not a mixed fast-FFT/GEMM validation and is not an 840-row population audit.

## 2. Validated observations and comparison accounting

The read-only imported static audit reported:

| Metric | Two-locus count or outcome |
|---|---:|
| Frozen records / v10 records | 8 / 8 |
| Raw numeric field comparisons (including E3 scalar) | 82,408 |
| Scientific payload tolerance failures (excluding the divergent relative-max diagnostic) | 0 |
| Normalized H-by-gap comparisons | 76,200; 0 failures |
| H window comparisons | 3,000; 0 failures |
| `fine_parent_reconstruction_rel_max` comparisons | 8; **8 failures** |
| Other full-record failures reported | 0 |
| Local imported result/schema/input-capture checks | PASS |

The numerical comparisons retain the original `abs_error <= 2e-5 OR relative_error <= 2e-5` rule, including its frozen epsilon definition. Field omission, numeric adjustment, diagnostic removal, and tolerance changes were not used. Therefore the all-field legacy record verdict remains **FAIL**, notwithstanding the provisional E2 result.

## 3. Matched parent-gap E3 localization

The second read-only static pass evaluated `2 loci × 4 components × 3 parents × 127 gaps = 3,048` matched parent-gap pairs. Its maxima are:

| Locus | k | Frozen `fine_parent_reconstruction_rel_max` | v10 `fine_parent_reconstruction_rel_max` | Frozen worst parent:gap | v10 worst parent:gap | Max normalized coarse engine delta | Max fine-sum engine delta |
|---|---:|---:|---:|---|---|---:|---:|
| gpu0 | 0 | 41.35847453 | 1.0 | EVIDENCE→EVIDENCE:66 | CLAIM→CLAIM:39 | 0 | 3.36288949e-08 |
| gpu0 | 1 | 103.90377394 | 1.0 | CLAIM→EVIDENCE:103 | CLAIM→CLAIM:39 | 0 | 4.05129991e-08 |
| gpu0 | 2 | 2.82810996 | 1.0 | EVIDENCE→EVIDENCE:47 | CLAIM→CLAIM:39 | 0 | 2.76999700e-09 |
| gpu0 | 3 | 22.96441039 | 1.0 | EVIDENCE→EVIDENCE:47 | CLAIM→CLAIM:39 | 0 | 8.48622404e-09 |
| gpu1 | 0 | 95.80810006 | 1.0 | EVIDENCE→EVIDENCE:82 | CLAIM→CLAIM:35 | 0 | 2.81541870e-08 |
| gpu1 | 1 | 99.34299056 | 1.0 | EVIDENCE→EVIDENCE:82 | CLAIM→CLAIM:35 | 0 | 2.87354578e-08 |
| gpu1 | 2 | 6.30587073 | 1.0 | EVIDENCE→EVIDENCE:47 | CLAIM→CLAIM:35 | 0 | 3.15247506e-09 |
| gpu1 | 3 | 4.10359357 | 1.0 | EVIDENCE→EVIDENCE:125 | CLAIM→CLAIM:35 | 0 | 3.86616869e-09 |

Across matched positions, normalized coarse engine deltas were 0; maximum fine-sum engine delta was `4.05129991e-08`. Under the original 2e-5 absolute-or-relative threshold these paired numeric differences do not invalidate bounded E2. In contrast, the scalar **maximum relative reconstruction residual** is dominated by specific denominator-sensitive positions; the maximizing parent/gap differs by engine.

Illustrative genuine frozen near-zero denominator: GPU0/k1, `CLAIM→EVIDENCE`, gap 103, normalized coarse ≈ `−3.36515e-14`, fine sum ≈ `−3.53016e-12`, absolute residual ≈ `3.49651e-12`, relative residual ≈ `103.90377`. The tiny absolute residual is within the unchanged per-gap replay tolerance; the relative maximum is large because of the small denominator. The raw discrepancy must **not** be discarded or backfilled with the v10 value.

**Important limits:** `max_coarse_delta=0` and small fine-sum deltas demonstrate numerical proximity at these matched parent-gaps, not identical reconstruction diagnostics. Neither pure floating-point accumulation-order causation nor zero combinatorial opportunity was independently established. In particular, `fine_sum=0` does not prove zero opportunity. The correct E3 description is `DIVERGENCE_UNRESOLVED`, **not** `NONCOMPARABLE` promoted by assumption.

## 4. Formal gate separation

| Gate | Current disposition | Reason |
|---|---|---|
| Code validation | CPU static checks PASS, original runner regressions 22/22 PASS | Does not prove scientific equivalence |
| GPU execution | Two bounded runs completed, worker RC=0 | Execution success only |
| Artifact transport | Collect/Import PASS; 8/8 files validated | Provenance transport integrity only |
| `V10_E0_PROVENANCE` | `NOT_EVALUATED` / **not promoted** | Formal independent end-to-end execution/source attestation not established by this report |
| `V10_E1_FROZEN_GUARDS` | `NOT_EVALUATED` / **not promoted** | In-process guard success and static record replays do not replace independent formal E1 |
| `V10_E2_PAYLOAD` | Formal `NOT_EVALUATED`; bounded numeric `PASS_PROVISIONAL` | Two loci have 0 reported scientific-payload tolerance failures; not full required E0/E1 verified scope |
| `V10_E3_DIAGNOSTICS` | Formal `NOT_EVALUATED`; raw `DIVERGENCE_UNRESOLVED` | Eight authentic relative-max differences; numerical cause and opportunity coverage not proven |
| `LEGACY_FULL_RECORD` | **FAIL** | Eight `fine_parent_reconstruction_rel_max` differences must be retained |
| `V10_PAYLOAD_EQUIVALENCE` | `NOT_EVALUATED` | Do not promote without formal E0/E1/E2/E3 coverage |
| Population, mixed-route and production | `NOT_EVALUATED` / `BLOCKED` | Only two B32 all-GEMM loci inspected; 840-row and mixed fast/GEMM not tested |

## 5. Reproducible evidence inventory

Expected source root (after `cm import`):

`outputs/gen5_v10_two_locus_runs/gen5-v10-two-locus-7f74a84-r1/`

The eight imported transport files are `execution_summary.json`, `execution_summary.json.sha256`, `gpu0/bounded_result.json`, `gpu0/bounded_result.json.sha256`, `gpu0/worker.log`, `gpu1/bounded_result.json`, `gpu1/bounded_result.json.sha256`, `gpu1/worker.log`. SHA256 sidecars accompany the three JSON result/summary files.

Two separately generated CPU-only analysis documents:

- `gen5-v10-two-locus-7f74a84-r1_static_science_audit.json`
- `gen5-v10-two-locus-7f74a84-r1_e3_paired_gap_diagnostic.md`

For result archival, preserve **all eight** imported files and **both** analysis documents, without altering contents, under `reports/gen5_v10_two_locus_imports/gen5-v10-two-locus-7f74a84-r1/`; this report lives at `reports/gen5_v10_two_locus_bounded_evidence_analysis_report_candidate.md`. An archive commit is a reproducibility/result record, **not** a scientific promotion or a fresh execution authorization. Refer to the run wrapper/collector manifests and the original ZIP SHA above for external handoff provenance.

## 6. Research decision

1. **Close this bounded all-GEMM numeric comparison with a provisional result**: E2 tolerance-conformant at two loci; legacy record FAIL; E3 divergence unresolved. This represents two bounded repetitions at different row/source locations, not a statistically powered replication or population claim.
2. **Do not rerun Kaggle/GPU or change the frozen diagnostic/tolerance to make it pass.** The matched-gap static analysis has identified the discrepancy and documented the missing causal attestation. More analysis from the same aggregates alone cannot prove opportunity or accumulation-order causation.
3. **Freeze the exact result evidence and report**, review the explicit file list, then manually commit/push if `cm ship` reports a clean scoped change. A new mixed-route or full-population experiment, if ever pursued, requires its own separately justified objective and authorization.

`TRAINING=NO` · `NEW_GPU_EXECUTION=NO` · `CODEX=NO` · `ORIGINAL_FROZEN_IMPLEMENTATION=UNCHANGED` · `FORMAL_SCIENTIFIC_PROMOTION=NO`
