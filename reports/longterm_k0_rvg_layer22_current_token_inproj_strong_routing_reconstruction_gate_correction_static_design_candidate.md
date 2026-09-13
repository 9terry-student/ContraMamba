# ContraMamba K0-RVG Layer-22 Current-Token In-Projection Strong-Routing
## Reconstruction-Gate Correction Static Design Candidate

## 1. Status

**Phase:** bounded correction to the frozen static design after runtime-preflight failure.

This document corrects one numerical-validation semantic defect in:

`reports/longterm_k0_rvg_layer22_current_token_inproj_strong_routing_static_design_candidate.md`

Frozen design commit:

`d4b3b2fb47c92d2e20b65764178ada4f82c2697b`

Frozen implementation commit:

`cc539ff5a3b9d692fa946c0d8e470685ea8ac6e3`

Parent validated evidence freeze:

`bfa626261aba575ab7316877bc69ca6e5df38157`

No scientific evidence was emitted by the failed runtime preflight.

No full execution is authorized until the corrected implementation is frozen and the bounded runtime preflight passes.

---

## 2. Runtime-preflight blocker

The first bounded runtime preflight stopped at:

`0:corr_H_RECON_FAILURE:1.5891726920633263e-06`

The implementation required:

`||W_H ΔX_t - ΔH_t|| / ||ΔH_t|| <= 1e-6`.

The observed value exceeded that threshold.

This is a validation-gate failure.

It is not a scientific result.

---

## 3. Root cause

The frozen upstream U-path implementation authenticates the hidden in-projection **branchwise**.

For matched branch `m`:

`H_m ≈ W_H X_m`.

For swapped branch `s`:

`H_s ≈ W_H X_s`.

Its frozen `1e-6` gate is applied independently to each branch reconstruction relative to the corresponding branch output norm.

The new strong-routing design incorrectly transferred that same relative tolerance to the differenced quantity:

`ΔH = H_m - H_s`

versus:

`W_H ΔX = W_H(X_m - X_s)`.

These are algebraically equal in exact arithmetic, but the validation residual after finite-precision branch evaluation and subtraction is:

`W_H ΔX - ΔH = e_m - e_s`

where:

`e_m = W_H X_m - H_m`

and:

`e_s = W_H X_s - H_s`.

A bound on:

`||e_m|| / ||H_m||`

and:

`||e_s|| / ||H_s||`

does not imply the same relative bound on:

`||e_m-e_s|| / ||H_m-H_s||`.

The denominator can be much smaller because matched and swapped branch activations are close.

Therefore applying the branch-relative `1e-6` tolerance directly to the difference-relative residual is not the frozen parent gate and is numerically ill-conditioned.

---

## 4. Correction principle

Do **not** widen the failed threshold.

The correction is semantic:

1. preserve the frozen parent branch-level reconstruction gate exactly;
2. define the scientific algebraic difference in float64;
3. explicitly account for the runtime-observed difference through the branch-error difference identity;
4. retain the already-preregistered parent strong-energy bridge;
5. keep all scientific channel-routing identities unchanged.

No scientific threshold is tuned from the failed value.

---

## 5. Corrected branch-level reconstruction gate

For every exercised matched/swapped branch and current-token target, preserve:

`H_b ≈ W_H X_b`

for:

`b ∈ {matched, swapped}`.

The existing frozen U-path capture already enforces its branch-level hidden in-projection reconstruction tolerance:

`1e-6`.

The corrected runner must continue to call that authenticated capture unchanged.

If the parent branch-level gate fails, the new stage is blocked.

The new runner may additionally recompute and record branch residuals, but it may not weaken the parent gate.

---

## 6. Canonical algebraic difference

After read-only float32 capture, convert the captured current-token inputs and frozen weight to float64.

Define:

`x = X_m - X_s`.

Define the canonical algebraic hidden difference:

`h_alg = W_H x`.

This is the vector used for the exact row-gain × directional-alignment decomposition:

`e_j = h_alg,j² / ||x||²`

and:

`e_j = ||w_j||² cos²(w_j,x)`.

All exact channel-routing identities remain defined on `h_alg`.

---

## 7. Runtime-observed difference bridge

Separately define:

`h_obs = H_m - H_s`

from the captured runtime branch outputs.

Define float64 branch reconstruction errors using the same captured vectors:

`e_m = W_H X_m - H_m`

`e_s = W_H X_s - H_s`.

Then require the exact error-difference identity:

`h_alg - h_obs = e_m - e_s`.

The corrected implementation must report an absolute numerical closure residual for this identity and gate it with the pre-runtime algebraic tolerance:

`5e-12`.

The quantity:

`||h_alg - h_obs|| / ||h_obs||`

may be recorded as a diagnostic.

It is **not** gated at `1e-6`.

No new result-dependent relative threshold is introduced.

---

## 8. Parent current-token metric reproduction

The parent U-path evidence must still be reproduced from the runtime-observed difference:

- `||ΔX_t||`;
- `||ΔH_t||`.

The frozen parent scalar matching tolerances remain:

- relative:
  `1e-13`;
- absolute:
  `1e-13`.

This verifies that the same current-token boundary is being read.

---

## 9. Parent strong-energy reproduction

The frozen lag-0 parent strong/weak energy quantities are defined on the runtime-observed `ΔH_t`.

Therefore the corrected runner must reproduce parent itemwise:

- `P_S,obs`;
- `P_W,obs`.

The exact frozen strong/weak channel partition remains:

- strong:
  `240`;
- weak:
  `1296`;
- equal:
  `0`.

Lag-0 kernel RMS remains:

`0.24383223809052498`.

---

## 10. Algebraic-to-observed strong-energy bridge

The implementation had already fixed, before scientific runtime execution, the bridge tolerance:

`2e-6`

for:

`|P_S,alg - P_S,obs|`.

This tolerance was not chosen from the failed runtime-preflight value.

It remains unchanged.

The corrected runner must report the maximum itemwise algebraic↔observed strong-energy bridge residual.

If this bridge fails, execution remains blocked.

The weak-energy bridge follows from the complete strong/weak partition but may also be reported directly.

---

## 11. Scientific routing identities remain unchanged

For each output row:

`h_alg,j = w_j^T x`.

Define:

`r_j² = ||w_j||²`.

Define:

`A_j(x) = cos²(w_j,x)`.

Define:

`e_j = h_alg,j² / ||x||²`.

Require:

`e_j = r_j² A_j(x)`.

For aligned corr/ctrl items define:

`D_j = r_j²(A_corr,j - A_ctrl,j)`.

Require:

`Σ_j D_j = T_H,corr² - T_H,ctrl²`.

For the frozen strong partition:

`Σ_(j∈S) D_j = T_S,corr² - T_S,ctrl²`.

For the frozen weak partition:

`Σ_(j∈W) D_j = T_W,corr² - T_W,ctrl²`.

The algebraic absolute tolerance remains:

`5e-12`.

---

## 12. Enrichment identities remain unchanged

For defined items:

`G_H = log(T_H,corr/T_H,ctrl)`.

`G_S = log(T_S,corr/T_S,ctrl)`.

`G_W = log(T_W,corr/T_W,ctrl)`.

Require:

`log(P_S,alg,corr/P_S,alg,ctrl) = 2(G_S-G_H)`.

Require:

`log(P_W,alg,corr/P_W,alg,ctrl) = 2(G_W-G_H)`.

Tolerance remains:

`5e-12`.

---

## 13. Required implementation delta

The correction must be minimal.

The runner must:

1. stop treating difference-relative `H_RECON_REL_TOL=1e-6` as a gate;
2. preserve the parent branch-level `1e-6` reconstruction gate through the frozen parent capture;
3. compute `h_alg` and `h_obs` separately;
4. compute branch error vectors `e_m` and `e_s`;
5. gate:
   `(h_alg-h_obs) - (e_m-e_s)`
   with absolute tolerance `5e-12`;
6. retain the difference-relative residual only as a diagnostic;
7. keep parent current-token metric reproduction unchanged;
8. keep parent `P_S/P_W` reproduction unchanged;
9. keep the preregistered `2e-6` algebraic↔observed `P_S` bridge unchanged;
10. keep all downstream routing identities and artifact policy unchanged.

No other scientific definition may change.

---

## 14. Scope that remains frozen

Unchanged:

- layer:
  `22`;
- coordinate:
  `k=2`;
- current token only;
- common DDSSSSS cohort:
  `330`;
- corr/ctrl aligned roles;
- same handoff/checkpoint/runtime lineage;
- same `W_H`;
- same lag-0 kernel;
- same strong/weak partition;
- all 1536 H channels preregistered;
- no raw-vector persistence;
- no PCA/SVD/whitening/probe/learned geometry;
- no intervention;
- no training/evaluation;
- no tokenizer;
- no logits/task heads;
- no post-hoc layer/lag/channel/window search;
- no K1 transition.

---

## 15. Interpretation boundary

Passing the corrected runtime bridge establishes only that:

- the frozen parent branch-level in-projection reconstruction remains valid;
- the algebraic difference `W_H ΔX_t` is connected to the runtime-observed `ΔH_t` through the explicitly measured finite-precision branch errors;
- the strong-routing decomposition is numerically well-defined.

It does not establish a scientific corr/ctrl claim by itself.

Scientific interpretation remains forbidden until full execution artifacts pass provenance and algebraic validation.

---

## 16. Stop conditions

Stop if:

- either parent branch-level in-projection reconstruction fails;
- parent current-token scalar reproduction fails;
- parent `P_S/P_W` reproduction fails;
- strong/weak partition changes;
- lag-0 kernel RMS changes;
- algebraic↔observed `P_S` bridge exceeds the already-fixed `2e-6`;
- error-difference identity closure exceeds `5e-12`;
- any existing exact routing identity fails;
- unrelated tracked files change;
- either unrelated K1 file changes state.

---

## 17. Correction conclusion

The failed preflight does not justify relaxing the frozen parent reconstruction tolerance.

The correct validation semantics are:

**preserve `1e-6` on each authenticated branch reconstruction, and validate the differenced algebra through the exact branch-error difference identity rather than reusing the branch-relative tolerance on a cancellation-sensitive difference denominator.**

This correction changes validation semantics only.

It does not change the scientific question, population, channel partition, routing factorization, or interpretation boundary.
