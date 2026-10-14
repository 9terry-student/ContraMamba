# Gen5 v10 — Gap-standardized fine-pair density contrast: static scientific interpretation

**Status:** `STATIC_JSON_REVIEW_COMPLETE / V5_ZERO_SCOPE_RECONCILED / UNIVERSAL_9_OF_9_REVERSAL_FALSIFIED_ON_COMMON_SUPPORT / STRICT_ZERO_OPPORTUNITY_FAIL_RETAINED / PROVISIONAL_DESCRIPTIVE_ONLY`
**Date:** 2026-10-14 (Asia/Seoul)
**Repository/branch:** `9terry-student/ContraMamba` / `gen5-causal-role-state-ownership`
**Read-only analysis HEAD:** `782a3fff554697df269ef2ba0b7a5f81df284cc5`
**Historical execution HEAD:** `9135810af6c607cc61a7c6b4197e4cebf2eaa059`

## 1. Evidence and review boundary

This report interprets the **user-executed** CPU-only static audit output, not a new independent execution of the original two large worker files. The uploaded result JSON was read and its full SHA256 verified:

- Result: `gen5_gap_standardized_density_result_782a3ff.json`, SHA256 `43b20fdd195f0e861b8172c547980ccccaf08483c0442b32f51b45ef954f55ad`.
- Static analysis script inspected: `gen5_gap_standardized_density_static_audit_782a3ff.py`, SHA256 `8ba797eead52e3d9ff701cc7a9ab29aa5c6f20cf311d32004691143045e1ac9f`.
- Frozen worker 0 SHA256: `c87412e111b70da6c4d4ede92235ccd3fa15b727d13852c5e9c6fabb70a2d1d4`; frozen worker 1 SHA256: `64a9c8300fdfc8cd260e4661557cab8b5772424f6af692c9e4848b4b0286864d` (these were the identities **reported by the audit**, not independently rehashed here).
- Frozen validated coarse summary SHA256: `24006b8c650582348bb6b9754b4568f30c5bf079329eb1a02f7e0f1b536d78fa`; coarse orientation JSONL SHA256: `a2eae4c48bbbcb3d677f44fc8189f344201d3e7b3991a4509906959d13d4eb13`.
- The user reported the current worktree clean, upstream synchronized at the exact read-only HEAD. No project file, model, tokenization, training, GPU, or Git mutation occurred in this review.

**Reported run guards:** `V5_FOUR_WINDOW_REPLAY=PASS`, `FROZEN_COARSE_COUNTS_REPLAY=PASS`, `COARSE_H_ABS_GUARD=PASS`. The JSON reports maximum conservative coarse-H parent reconstruction absolute error `1.688810218336556e-7`. The audit code used a `2e-5` absolute static guard: this is **not** an independent formal E0–E3 attestation. Internal result checks in this review independently reproduced the nine-source medians/sign counts and verified source-level `ΔH_total ≈ ΔH_A + ΔH_R` up to `2.22e-16` within the uploaded JSON.

## 2. Frozen classes and the new estimand

Under `CLAIM→EVIDENCE`, `A` is the 16 named-anchor-only ordered fine pairs; `R` is the nine `RESIDUAL`-involving pairs. `RESIDUAL` is a mechanical complement, **not a semantic class**. Each source has two PRIMARY_A and two CONTROL_R orientations; the two nonoverlapping workers cover **one seed180 dev population of 840 rows / 60,094 valid tokens**, not independent seeds.

For each component `c`, source `s`, gap `d`, and group `g`, `ΔH_c(s,g,d)` is the source-matched PRIMARY_A-minus-CONTROL_R difference of normalized additive pair interference. For pooled data, raw worker `C` and self-energy `S` were summed **before** calculating `H=C/S`. With opportunity `N_g(d)`, the new signed contrast is:

\[
T_c(s,W)=\sum_{d\in G_W}\omega_W(d)\left(\frac{\Delta H_c(s,R,d)}{N_R(d)}-\frac{\Delta H_c(s,A,d)}{N_A(d)}\right),\quad
\omega_W(d)=\frac{\min(N_R(d),N_A(d))}{\sum_{u\in G_W}\min(N_R(u),N_A(u))}.
\]

`G_W` includes only gaps with **both** group opportunity counts positive. Negative `T` means relatively more negative `R` signed density; positive `T` means relatively more negative `A` signed density. This is a *descriptive, gap-standardized signed contrast*, **not** a causal per-pair effect or decomposition of nonlinear `L_interference`.

## 3. Primary result: universal 9/9 reversal falsified on common support

| Scope | W3 complement `T<0` | W4 complement `T>0` | W3 visible `T<0` | W4 visible `T>0` | W3 common gaps | W4 common gaps |
|---|---:|---:|---:|---:|---|---|
| Worker 0 | 9/9 | 3/9 | 0/9 | 8/9 | 22–32 (11/16) | 33–61 (29/32) |
| Worker 1 | 9/9 | 5/9 | 0/9 | 9/9 | 22–32 (11/16) | 33–59 (27/32) |
| Pooled, primary | **9/9** | **3/9** | **0/9** | **8/9** | 22–32 (11/16) | 33–61 (29/32) |

**Pooled T medians:** complement W3 `−6.221378551564386e-8`, W4 `−1.121737466336609e-8`; visible W3 `+1.896762225581824e-7`, W4 `+1.248560165982953e-7`.

**Decision:** The pre-specified conjunction `T_complement(W3)<0` **and** `T_complement(W4)>0` for all nine source cells does not hold. The **strong universal 9/9 gap-standardized reversal hypothesis is falsified within the observed common support**. It does **not** follow that the original, distinct full-window V5 estimand was invalid or that all sources lack a W3/W4 transition.

## 4. Common-support exclusions matter

| Scope | Window | Common gaps | Excluded gaps | Excluded `A` opportunities | Excluded `R` opportunities | Excluded `R` opportunity fraction |
|---|---|---|---|---:|---:|---:|
| Pooled | W3 (17–32) | 22–32 | 17–21 | 0 | 75,600 | **25.6869%** |
| Pooled | W4 (33–64) | 33–61 | 62–64 | 0 | 21,522 | **3.9434%** |
| Worker 0 | W3 | 22–32 | 17–21 | 0 | 37,440 | 25.7120% |
| Worker 0 | W4 | 33–61 | 62–64 | 0 | 10,671 | 3.9529% |
| Worker 1 | W3 | 22–32 | 17–21 | 0 | 38,160 | 25.6622% |
| Worker 1 | W4 | 33–59 | 60–64 | 0 | 20,205 | 7.3253% |

At excluded gaps, `A` has zero opportunities but `R` has positive opportunities; the pair-density difference is therefore **undefined**, not zero. In particular, the W3 common-support estimand excludes over one-quarter of `R` pair opportunities. One cannot infer a full-window standardized contrast without inventing a counterfactual or imputation. Differences between original window-level V5 density ratios and the new `T` involve **both** reweighting and exclusion; this audit does **not** causally identify their separate contributions.

## 5. Source-specific heterogeneity: exploratory only

Pooled complement `T`, displayed in units of `1e-8`:

| Frozen source | W3 | W4 |
|---|---:|---:|
| A6201-R6201 | −4.7846 | +7.6160 |
| A6201-R6202 | −1.6548 | +7.1439 |
| A6201-R6203 | −2.4355 | +7.6825 |
| A6202-R6201 | −8.0889 | −3.1663 |
| A6202-R6202 | −7.5530 | −2.6965 |
| A6202-R6203 | −6.2214 | −1.4863 |
| A6203-R6201 | −4.1543 | −0.7990 |
| A6203-R6202 | −6.3250 | −1.1217 |
| A6203-R6203 | −9.0055 | −4.3108 |

All three pooled W4 positives have first-axis source `A6201`; all six `A6202`/`A6203` pooled values remain negative. Worker 0 matches this partition. Worker 1 has two small positive `A6203` values (about `+6.39e-9` and `+1.06e-9`), so **the exact first-axis split is not stable in both shards**. This is an exploratory stratification of nine related source cells, not independent replication, significance, or attribution to a source-family causal mechanism. No new threshold or class is introduced.

The visible control has W3 `T>0` in all nine sources for both shards and pooled. This is evidence of a component-sensitive *signed descriptive pattern*; it does not establish complement-exclusive causal ownership.

## 6. Exact zero-opportunity scope reconciliation (completed)

A subsequent **user-executed, read-only** reconciliation inspected the same historical worker JSON artifacts under the exact analysis HEAD, checked the two frozen worker SHA256 identities and source/worker identities, and reported:

`GEN5_ZERO_OPPORTUNITY_SCOPE_RECONCILE_COMPLETE_READ_ONLY` / `V5_SCOPE_COUNT_RECONCILIATION=PASS`.

- Reconciliation script: `gen5_zero_opportunity_scope_reconcile_782a3ff.py`, SHA256 `dae468634e3cda2717fe748615358b2c168c90510e515dfd83700b26e0ab8b7a`.
- Analysis HEAD: `782a3fff554697df269ef2ba0b7a5f81df284cc5`.
- Pooled-zero fine-parent-gap **positions**: **8,350** (neither worker has a pair opportunity).
- Worker1-only-zero fine-parent-gap **positions**: **44** (`N_worker1=0`, `N_worker0>0`).
- Worker0 nonzero entries at worker0-zero opportunity: **0**.
- Worker1 nonzero entries at worker1-zero opportunity: **221,616**.
- Worker1 nonzero entries at pooled-zero opportunity: **218,520** (the original V5 count).
- Worker1 nonzero entries at worker1-only-zero opportunity: **3,096**, partitioned as CLAIM→CLAIM **360**, CLAIM→EVIDENCE **2,376**, and EVIDENCE→EVIDENCE **360**.

The arithmetic identity is exact: **221,616 = 218,520 + 3,096**. The 44-position difference is a count of fine-parent-gap locations, whereas the 3,096 difference counts nonzero values over orientation/component entries; they are **not interchangeable units**. Thus the historical V5 and present audit counts are **reconciled by their denominator/scope definitions**. Neither value is revised or replaced. This scope reconciliation was observed in user-provided console output; the analyst did not independently re-run the original worker files.

**The strict numeric failure remains unresolved.** Reconciliation establishes bookkeeping consistency, **not** legitimacy of nonzero `H` at zero counted opportunity, its computational cause, E3 diagnostic comparability, or verified numeric behavior when opportunity is positive. Retain all simultaneous verdicts:

- `V5_ZERO_SCOPE_RECONCILIATION=PASS`;
- `STRICT_ZERO_OPPORTUNITY=FAIL_RETAINED`;
- `LEGACY_FULL_RECORD=FAIL`;
- `V10_E2_BOUNDED_NUMERIC=PASS_PROVISIONAL_NOT_FORMAL`;
- `V10_E3_BOUNDED_RAW=DIVERGENCE_UNRESOLVED`;
- `FORMAL_E0_E3=NOT_PROMOTED`;
- `INDEPENDENT_REPLICATION=NOT_ESTABLISHED`;
- `CAUSAL_MECHANISM=NOT_ESTABLISHED`.

Conditional sensitivity to *deleting observed worker-zero-opportunity numeric terms*, **not** to arbitrary computation error: pooled maximum absolute `T` change is `7.590440093223098e-16` (W3 complement) and `8.378811014565145e-16` (W4 complement). Under this narrow hypothetical deletion, the observed source-level signs do not flip. This does not clear the strict failure, missing common support, positive-opportunity numeric correctness, or E3 divergence.

## 7. Final scientific decision and archival boundary

1. **Reject** the strong hypothesis of a universal `T_complement(W3)<0`, `T_complement(W4)>0` transition across all nine source cells **on the shared supported gaps**. The W3 relative-negative pattern remains 9/9; W4 is 3/9 positive pooled. This result is conditional on the reported historical frozen artifacts and remains descriptive/provisional.
2. Preserve original V5 full-window density ratios as a **different estimand**, without treating the new common-support rejection as an invalidation of their calculations. In W3, 25.6869% of pooled RESIDUAL pair opportunities fall outside common support; extrapolation to the full W3 window is not identified by this analysis.
3. Treat the A6201 W4 pooled sign concentration as **exploratory only**, not a discovered source mechanism. One seed180 dev population and two nonindependent row shards are insufficient for causal attribution, independent replication, or generalizability claims.
4. **Close** zero-opportunity count-scope accounting as reconciled, while preserving the original strict failure and all formal equivalence blockers. Do not run additional GPU, tokenizer, model, training or evaluation jobs to resolve this bookkeeping question.
5. The report may be archived as **one documentation-only candidate** after an explicit, clean `cm ship` review. The source audit JSON and helper scripts remain outside the repository; no promotion to E0–E3, validated v10 science, or production authority is granted by archival. Do not stage anything implicitly or commit/push automatically.

**Overall:** `STATIC_SCOPE_RECONCILIATION_COMPLETE / STRONG_HYPOTHESIS_FALSIFIED_ON_COMMON_SUPPORT / SOURCE_HETEROGENEITY_EXPLORATORY / STRICT_NUMERIC_FAIL_RETAINED / SCIENCE_PROMOTION=NO`.
