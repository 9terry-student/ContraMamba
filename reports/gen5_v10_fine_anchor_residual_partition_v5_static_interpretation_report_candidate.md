# Gen5 v10 — Frozen fine-anchor residual-partition static interpretation (candidate)

**Status:** `DESCRIPTIVE_PROVISIONAL_NOT_VALIDATED / STRICT_ZERO_OPPORTUNITY_FAIL / NO_FORMAL_SCIENCE_PROMOTION`
**Date:** 2026-10-14 (Asia/Seoul)
**Scientific disposition:** The four pre-specified complement comparisons show a consistent *aggregate-mass* dominance by fine cells involving `RESIDUAL`, but they do **not** establish a universal per-opportunity residual enhancement, a unique causal mechanism, or formal FFT–GEMM equivalence.

## 1. Fixed inputs, provenance, and analysis boundary

- Repository: `9terry-student/ContraMamba`; branch: `gen5-causal-role-state-ownership`.
- Read-only analysis HEAD: `531bd0deeba5de43841b6ed9c148a8390c0cc481` (reported clean at analysis time).
- Historical full-hybrid execution HEAD: `9135810af6c607cc61a7c6b4197e4cebf2eaa059`; run `gen5-v10-full-hybrid-9135810-r1`.
- Exact archived artifact root: `reports/gen5_v10_full_hybrid_imports/gen5-v10-full-hybrid-9135810-r1/artifacts/`.
- Worker 0: dev rows `[0,416)`, 29,765 valid tokens, `234 B32 GEMM`, no B8 tail; `worker_result.json` SHA256 `c87412e111b70da6c4d4ede92235ccd3fa15b727d13852c5e9c6fabb70a2d1d4`.
- Worker 1: dev rows `[416,840)`, 30,329 valid tokens, `234 B32 GEMM + 18 B8 frozen FFT`; SHA256 `64a9c8300fdfc8cd260e4661557cab8b5772424f6af692c9e4848b4b0286864d`.
- Both JSON/sidecar identities passed in the reported diagnostics. The full-hybrid direct import had previously passed artifact-integrity checks but is **not** an official `cm import` or independent E0–E3 attestation.
- Read-only analysis scripts, executed externally to the repository: v3 SHA256 `bef627df4593f2798530c03ec7cf176be4d46e6971f87a6fb82141f2a9d096c2`; v5 SHA256 `cbf88f3f2aafffd6a190d386c7da59702377b824ee2b7546a0e23e2f4ed0df4c`.
- The attached v5 console transcript was user-supplied and reports `GEN5_V10_V5_PER_SOURCE_PARTITION_COMPLETE_STRICT_FAIL`. The interpretation below is an audit of that transcript and the reviewed static analysis logic; it is **not** an independently executed rerun on the author's computer.
- No model forward, backward, GPU execution, training, new tokenizer, new seed, or mutation of archived numeric values occurred during v3/v5 analysis.

## 2. Pre-specified estimand and partition

The analyzed object is the additive normalized fine pair-gap quantity `H = C / S`, at the exact three frozen parent classes, five frozen gap windows, and 25 fine cells per parent (75 in total). The 5 frozen token classes are `A_TITLE`, `A_NAME`, `A_ROLE`, `A_PREDICATE`, and `RESIDUAL`.

The analysis partitions the 25 fine cells of each parent into disjoint sets:

- **Anchor-only (16 cells):** both endpoints belong to the four named anchor classes.
- **RESIDUAL-involving (9 cells):** at least one endpoint is `RESIDUAL`.

For each of the nine frozen source cells and worker/pooled scope, `ΔH = mean_PRIMARY_A(H) − mean_CONTROL_R(H)`, with two target orientations in each group. At **each source** the additive identity is `ΔH_total = ΔH_anchor-only + ΔH_RESIDUAL-involving`. Medians across sources need not obey this addition.

The four comparisons were fixed before inspecting these v5 source-partition results:

1. `CLAIM->CLAIM`, W1 gaps 1–8;
2. `EVIDENCE->EVIDENCE`, W1 gaps 1–8;
3. `CLAIM->EVIDENCE`, W3 gaps 17–32;
4. `CLAIM->EVIDENCE`, W4 gaps 33–64.

Pooling adds the two workers' `C` and `S` before normalization; it does **not** average worker-specific `ΔH`. V3 previously evaluated 20,250 fine-cell/source/window/component/scope values. V5 uses the four pre-specified comparisons for the more detailed two-set, per-source analysis.

## 3. Complement-side pooled findings

Each figure below is reported from the nine frozen source cells, not an inferential estimate. The *RESIDUAL magnitude share* is the **median of the nine per-source** values `|ΔH_residual|/(|ΔH_anchor|+|ΔH_residual|)`. It is **not** a causal share of `L_interference`. The opportunity share is the fraction of actual counted eligible pairs in the RESIDUAL-involving group.

| Parent/window | Median total ΔH | Median anchor-only ΔH | Median RESIDUAL ΔH | Median RESIDUAL magnitude share | RESIDUAL opportunity share | Per-opportunity `|RESIDUAL|/|anchor|` density ratio |
|---|---:|---:|---:|---:|---:|---:|
| CLAIM→CLAIM, W1 | −1.56731298 | −0.235152358 | −1.33523035 | 85.1923% | 85.2366% | 0.983× |
| EVIDENCE→EVIDENCE, W1 | −1.39796895 | −0.204394797 | −1.19002207 | 85.1293% | 85.7858% | 0.965× |
| CLAIM→EVIDENCE, W3 | −0.550323408 | −0.0298738586 | −0.52044955 | 94.5726% | 93.1845% | 1.274× |
| CLAIM→EVIDENCE, W4 | −0.433199824 | −0.0532099007 | −0.379989923 | 87.6648% | 90.2436% | 0.772× |

The density ratio is computed from the displayed pooled **median ΔH divided by pooled opportunity counts** for each disjoint set; it is a descriptive group-density contrast, not a per-pair causal effect or a statistical estimate. Paired source medians and aggregation denominators must not be silently interchanged.

**Interpretation by distance regime:**

- **Within-region W1:** RESIDUAL-involving cells hold a large share of additive signal, but that share closely tracks their eligible-pair counts. The frozen named-anchor-only cells are consistently negative too. Neither universal residual-pair strength nor exclusive anchor localization is established.
- **Cross-region W3:** The RESIDUAL-involving group has 94.57% median signed-magnitude share versus 93.18% opportunity share. Its descriptive pair-density magnitude is approximately 1.27 times anchor-only. This is the strongest *relative RESIDUAL enrichment candidate* of the four comparisons, not proof of a specific token-level cause.
- **Cross-region W4:** Although RESIDUAL-involving cells dominate summed magnitude, they account for **less** of that magnitude than their opportunity share, and the anchor-only group has stronger density by this metric. The per-opportunity structure therefore changes with gap regime; a universal RESIDUAL-specific strengthening hypothesis is not supported.

### Per-source/shard controls

For **all four complement parent/windows**, both workers and pooled have `9/9` negative RESIDUAL ΔH, `9/9` negative anchor-only ΔH, `9/9` negative total ΔH, and `9/9` absolute RESIDUAL-over-anchor dominance. Each group's per-source dominance is also `9/9` robust to the specifically bounded removal of *observed zero-opportunity* numeric values. These are **two nonoverlapping partitions of one frozen 840-row population**, not independently seeded replicates.

The pooled per-source RESIDUAL magnitude-share ranges were:

| Comparison | Min | Median | Max |
|---|---:|---:|---:|
| CLAIM→CLAIM W1 | 83.1238% | 85.1923% | 85.4665% |
| EVIDENCE→EVIDENCE W1 | 83.2235% | 85.1293% | 85.7093% |
| CLAIM→EVIDENCE W3 | 94.4020% | 94.5726% | 95.0912% |
| CLAIM→EVIDENCE W4 | 86.9287% | 87.6648% | 88.0130% |

The preceding validated coarse region analysis and the new full-hybrid pooled summaries are numerically close (median absolute differences in the four complement comparisons approximately `6.5e−9` to `1.8e−8`), but that descriptive agreement does **not** establish formal E2 equivalence.

## 4. Visible-side control is not null

The pooled visible-side median total `ΔH` and source-level sign patterns were:

| Parent/window | Visible median total ΔH | Positive source count | RESIDUAL magnitude dominance |
|---|---:|---:|---:|
| CLAIM→CLAIM W1 | +1.12498045 | 9/9 | 9/9 |
| EVIDENCE→EVIDENCE W1 | −1.10151579 | 0/9 | 9/9 |
| CLAIM→EVIDENCE W3 | +0.0537379594 | 8/9 | 9/9 |
| CLAIM→EVIDENCE W4 | +0.102454706 | 8/9 | 9/9 |

The visible component sometimes agrees in sign with complement (EVIDENCE→EVIDENCE W1), and sometimes differs sharply (CLAIM→CLAIM W1 and both CLAIM→EVIDENCE windows). This supports a heterogeneous region/component structure rather than a universal global decrease. RESIDUAL magnitude dominance is **also seen in visible**; therefore RESIDUAL dominance alone is not a complement-exclusive causal signature.

## 5. Strict validation failure and bounded sensitivity

**Retained validation status:** `STRICT_ZERO_OPPORTUNITY=FAIL`.

The prior global diagnostic observed `8,350` fine-cell/gap positions with pooled pair-opportunity count zero. For those positions, worker 0 yielded exactly zero unsupported raw entries while worker 1 yielded `218,520` nonzero entries across all orientations/components. Maximum worker-1 normalized `|H|` over these unsupported positions was approximately `4.3670655215e−10`.

The execution-route difference is consistent with small residual terms from worker 1's `18 B8 frozen FFT` calls, but neither the computational cause nor the correctness of all nonzero-opportunity fine-cell values has been independently proven. The original nonzero values, raw hashes, replay tolerances, and the strict validation failure are preserved without zeroing, imputing, or relabeling them.

For the four pooled complement comparisons, the maximum source-level absolute `ΔH` change bounded **only when hypothetically deleting observed entries at worker-specific zero-opportunity positions** is:

| Parent/window | Max source-level bound |
|---|---:|
| CLAIM→CLAIM W1 | `2.52756155552e−10` |
| EVIDENCE→EVIDENCE W1 | `1.76189617515e−10` |
| CLAIM→EVIDENCE W3 | `8.24149125928e−11` |
| CLAIM→EVIDENCE W4 | `1.85628983254e−10` |

Under precisely this conditional perturbation, all nine source-level complement RESIDUAL signs and RESIDUAL absolute dominance outcomes remain unchanged in each focus window. These are **sensitivity statements only**. They do not relax the strict count contract, bound numerical error where opportunity is positive, identify the cause of the leakage, or authorize E3 `NONCOMPARABLE` promotion.

## 6. Scientific decision and stop conditions

**Supported, provisional observational claim:** In this frozen, one-seed 840-row cohort, the component-specific short/mid-gap coherence deficit is consistently represented by both named-anchor and RESIDUAL-involving fine-pair groups. RESIDUAL-involving cells dominate **aggregate magnitude** in both shards and at every matched source, but their relative **opportunity-normalized density** is gap-dependent. W3 offers a modest, descriptive enrichment candidate, while W1 does not and W4 reverses its ordering.

**Not supported:** A unique `RESIDUAL` causal mechanism, absence of meaningful named-anchor interactions, universal pair-level RESIDUAL strength, independent-seed replication, new statistical significance, matched full-840-row frozen-FFT↔v10 equivalence, or formal E0/E1/E2/E3 promotion. In particular, the nonlinear `L_interference` cannot be apportioned causally using additive `ΔH` shares.

**Frozen status labels:**

- `ORIGINAL_FULL_HYBRID_ARTIFACT_INTEGRITY=VERIFIED_PREVIOUSLY`
- `V5_SOURCE_PARTITION_STATIC_REPLAY=COMPLETE_OPERATOR_REPORTED`
- `SHARD_RESIDUAL_AGGREGATE_MAGNITUDE_DOMINANCE=9_OF_9_ALL_FOUR_COMPLEMENT_WINDOWS_DESCRIPTIVE`
- `STRICT_ZERO_OPPORTUNITY=FAIL`
- `ZERO_OPPORTUNITY_ROOT_CAUSE=NOT_PROVEN`
- `FINE_ANCHOR_SCIENTIFIC_VALIDATION=PROVISIONAL_DESCRIPTIVE_ONLY_NOT_VALIDATED`
- `LEGACY_FULL_RECORD=FAIL` (separate two-locus FFT↔GEMM comparison; unchanged)
- `V10_E2_BOUNDED_NUMERIC=PASS_PROVISIONAL` (separate bounded comparison; unchanged)
- `V10_E3_BOUNDED_RAW=DIVERGENCE_UNRESOLVED` (unchanged)
- `FORMAL_E0_E3=NOT_PROMOTED`
- `TRAINING=NO; GPU=OFF; CODEX=NO; NEW_EVALUATION=NO`

**Research disposition:** Close the current read-only source-partition interpretation with these stated limitations. Further attempts to label the observed zero-count leakage as harmless or to upgrade formal numerical equivalence from these same aggregate records would not be justified. No additional scripts, experiments, or commits are authorized by this report alone. The candidate report can be preserved after a clean, one-file staging review; preserving it would not promote any scientific gate.
