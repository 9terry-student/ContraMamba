# ContraMamba Gen5 G2-H1 — Full-population layer-22 output observation: independent static analysis

**Classification:** Exploratory, descriptive static analysis of an already executed read-only observation. **Not** a causal mediation result, training authorization, Phase3B promotion, or a new confirmatory cohort.

**Source execution identity:** `40a279a28ac088b8728f828038127343f598fc9e` (Kaggle checkout), `Tesla T4`, Python `3.12.13`, PyTorch `2.10.0+cu128`, frozen Phase3A P0 dev 840 rows / 120 complete seven-cell factual pairs, three frozen seeds `6201,6202,6203`, arms `ZERO,FULL`.

**Direct-run source:** `external_runner/gen5_g2_h1_full_direct_runner_eaeb4a39.py`, SHA256 `eaeb4a39d32cfd17004e4954f5471262a8beba4f548234ec617fab3472194945`, saved in imported `provenance/`. The separately staged repository script `scripts/gen5_g2_h1_full_population_readonly_observation.py` is **not byte-identical** to the executed runner: the external version uses an explicit repository root, out-of-tree execution and SHA256 pin instead of Git blob identity, and records the runner SHA in results; the experiment's scientific calculations are otherwise the same in the inspected diff. Preserve the exact executed runner, not just the staged script.

**Input/handoff identity:** manual import of `g2h1-full-observation-direct-40a279a-r1-unregistered-handoff.zip` with SHA256 `bac4c8d61a192d4f0530d6e074554e0529c26c10f00826db40144e4f980d87c7`; handoff manifest `G2H1_DIRECT_UNREGISTERED_HANDOFF_V1`; summary SHA256 `92e418f9f63c9bd0c5b8807758b50a935bf830e7ad28641bcbd0e82ab5f29202`. **Not a `cm run`/`cm collect v4`/`cm import`-registered result.** Manual hash validation and static replay do not grant the formal run-registry/provenance status.

## 1. Independent integrity/identity replay

Using the exact attached handoff ZIP, independently checked:

- SHA256 and byte size of every one of the eight manifest-listed files: one external Python runner, one summary, six seed/arm observation JSONs. ZIP and summary hashes also match the supplied values.
- Each condition contains **840** unique row IDs and **120** unique complete pairs with C0–C6 exactly once each. All six conditions share the same ordered row/pair/cell identities.
- Per-condition expected and recorded backbone forwards equal **54**; total **324**. This verifies artifact *records*, not an independent reenactment of GPU calls.
- Recomputed `D_RS` from each supplied 3-class logit vector and `(C0,C6)` row identity; all **720** pair-level values match the stored contrasts within floating-point tolerance. Recomputed all supplied pair `C6−C0` region summaries from per-row values (native/wrapped/contribution × valid/claim/evidence/other × mean/mean-abs/RMS), including explicit missing region behavior.
- `ZERO` contribution region statistics are exactly zero for every populated region in all three seeds. The per-condition records report `reference_max_abs_diff=0.0` to frozen G2-P0 logits and the runner code enforces a 1e-5 maximum; this source-level numerical match **was not independently GPU-rerun here**. No model reloading, backpropagation or new evaluation occurred during this analysis.
- Reported setup identities include frozen model revision `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`, parent checkpoint SHA256 `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f` and exact-kernel transport status.

## 2. Frozen final-logit contrast: reproduced, not newly discovered

Define `D_RS(p,arm) = (z_REFUTE-z_SUPPORT)[C6,p,arm] - (z_REFUTE-z_SUPPORT)[C0,p,arm]`. A pair, not a row or seed×pair, is the basic identity unit. Exact means replayed from the six condition files:

| Heldout seed | mean D_RS ZERO | mean D_RS FULL | mean (FULL − ZERO) | Sign counts |
|---|---:|---:|---:|---|
| 6201 | +2.759805 | −0.058027 | −2.817832 | ZERO>0: 120/120; FULL<0: 120/120 |
| 6202 | +2.759805 | −0.029036 | −2.788841 | ZERO>0: 120/120; FULL<0: 120/120 |
| 6203 | +2.759805 | −0.048808 | −2.808613 | ZERO>0: 120/120; FULL<0: 120/120 |

This is the previously frozen G2-P0 within-pair inversion, now repeated in an observer-equipped run. It is not fresh out-of-distribution evidence, nor a new heldout population.

## 3. Newly measured layer-22 output statistics

The actual measured `contribution = wrapped_mixer_output − native_mixer_output`; it is **not** the native recurrent raw write or a measured causal state mediator. For `FULL`, compute within-pair `C6_RMS − C0_RMS` separately by declared token region. All values below are descriptive, averaged over 120 pairs **within each seed**.

| Seed | Claim RMS difference | Evidence RMS difference | Valid-token RMS difference | Evidence C6>C0 | Valid C6>C0 |
|---|---:|---:|---:|---:|---:|
| 6201 | ≈0 | +0.070560 | +0.050587 | 119/120 | 120/120 |
| 6202 | ≈0 | +0.093140 | +0.064343 | 119/120 | 120/120 |
| 6203 | ≈0 | +0.076117 | +0.054277 | 119/120 | 120/120 |

Claim differences are numerical roundoff scale (mean between −3.9e−9 and −4.7e−9); the same claim prefix/tokens are shared across the seven cell variants. Zero claim-region differences therefore must not be interpreted as an independently discovered localization mechanism. The FULL evidence RMS mean rises from 2.430→2.501, 2.691→2.784, and 2.469→2.545 (C0→C6), approximately 2.9%, 3.5%, and 3.1% relative changes.

### Essential selectivity and length controls

**Token-length confound:** The `C6−C0` token-count difference is **+5** for `evidence` and `valid` in **all 120 pairs, all three seeds**, but exactly **0** in `claim` and `other`. Current statistics are token-wise averages over differently sized regions; this is *not* an equal-length matched-token comparison. Semantic selectivity cannot be attributed apart from text length/content variation.

**Alternative-cell control:** When comparing the FULL evidence contribution RMS for C6 against each of C1–C5 within the same pair, C6 is the largest in only **35/120 (6201), 50/120 (6202), 38/120 (6203)**. With evidence `mean_abs` instead of RMS, the analogous counts rise to **91/120, 110/120, 99/120**. This metric-dependent ranking is a limitation, not strong unique C6 RMS specificity. The C1–C5 controls cannot be omitted in an H1 selectivity claim.

**Correlation with decision change:** Across 120 pairs within a seed, Pearson correlation between evidence-region contribution RMS `(C6−C0)` and the final-logit change `D_RS(FULL)−D_RS(ZERO)` is approximately **r=+0.223 (6201), +0.100 (6202), +0.258 (6203)**. The signs are positive (larger RMS contrast is associated with a *less* negative final-logit change), not uniformly large in strength, and not evidence for a monotone suppression mechanism. Exploratory Pearson relationships for valid RMS are +0.224, +0.087, +0.255. Pair rankings are strongly related across seeds but are measured on **the same 120 factual pairs**, not three statistically independent datasets. No confirmatory p-values, mediation, independent generalization, or multiple-comparison claims are justified.

## 4. Evidence tiers and H1 disposition

1. **Code-path numerical integrity:** A frozen-parent 4-forward first-pair preflight passed; the full run reports 324 read-only forwards with observer-installed outputs exactly matching unhooked outputs and frozen logits under the runner's checks. Artifacts/runner hashes and static consistency have been independently replayed. A separate GPU rerun has not been performed.
2. **Scientific observational evidence:** A nonzero, reproducible C6−C0 magnitude difference exists in the layer-22 **mixer correction output** for FULL, mainly on evidence/valid regions. This coexists with the frozen final-logit inversion. ZERO has zero contribution.
3. **H1 selective-mechanism claim:** **NOT ESTABLISHED.** Alternative-cell RMS maxima, token-count mismatch, weak/seed-varying association with decision change, and absence of raw recurrent write or causal intervention rule out asserting a selective causal write mediator. These observations neither prove suppression's layer-22 necessity nor its sufficiency.
4. **Scientific conclusion / gates:** Treat as **exploratory descriptive support for an output-level C6–C0 contrast**, with **causal mediation unresolved**. `PHASE3B=BLOCKED_UNCHANGED`. No training, rerun or automatic promotion authorized by this report.

## 5. Next scientific question (not an execution instruction)

The smallest diagnostic to separate semantic and length effects would match token positions and length while holding the pretrained model and correction fixed, and compare C6 to all C1–C5 under prespecified region/statistic criteria. To test a *mediating* write would additionally require a separately specified intervention and matched numerical/equivalence gates, not another observer-only run. **This report itself authorizes neither execution.**

**Repository disposition:** Preserve the original 7 observation JSONs, the exact executed external runner, and handoff manifest. Include this interpretive report alongside them. The independently staged in-repo prototype and the executed out-of-tree runner must remain distinguishable in provenance; neither source is silently relabeled a formally `cm`-registered execution. Commit/push, if undertaken, must be explicitly staged by the operator after `cm ship` review.
