# Stage41-A Paper-Ready Result Freeze: Structured Coverage Owner and Opt-In Final Composer

## 1. Executive Summary

- **Stage33-F** establishes a structured coverage shadow owner, showing strong potential on external Stage31 coverage entailment diagnostics (shadow macro-F1 0.9083 vs. current 0.1818).
- **Stage35** reveals unsafe adversarial leakage in the Stage33-F shadow owner alone — overclaim, exception, and location-scope cases leak into SUPPORT (`scope_safety = unsafe`, `reverse_handling = unsafe`).
- **Stage36** blocks this unsafe SUPPORT leakage (all three leakage counters driven to 0) without reducing existing SUPPORT recovery (`support_shadow_support` held at 91), reaching `scope_safety = safe`.
- **Stage37** restores safe SUPPORT recovery on top of the Stage36 blockers, lifting `support_shadow_support` from 91 to 153 while keeping all unsafe leakage counters at 0.
- **Stage38** validates the shadow owner (Stage33-F + Stage36 + Stage37) through an integrated regression audit against dev, with zero dev prediction mismatches and all adversarial safety/recovery checks passing (`STAGE38A_INTEGRATED_REGRESSION_PASS`).
- **Stage39-C** upgrades the validated shadow owner into an explicit opt-in final composer (`safe_structured_v2`), producing large macro-F1 gains on Stage34 (0.2069 → 0.9703) and Stage35 (0.2832 → 0.7202) with zero dev changes and zero introduced unsafe SUPPORT / REFUTE→SUPPORT / SUPPORT→REFUTE errors (`STAGE39C_SAFE_STRUCTURED_V2_PASS`).
- **Stage40-A** confirms the integrated final-composer regression audit across dev, Stage34, and Stage35, verifying the composer remains default-off and introduces no unsafe errors in any evaluated profile (`STAGE40A_FINAL_COMPOSER_REGRESSION_PASS`).

## 2. Main Result Table

| Stage | Mode | Split/Probe | Original/Current Macro-F1 | Shadow/Composed Macro-F1 | Key Recovery | Unsafe SUPPORT / Introduced Unsafe | Decision | Role in Paper |
|---|---|---|---|---|---|---|---|---|
| Stage33-F2 | shadow (diagnostic only) | external Stage31 coverage | 0.1818 | 0.9083 (Δ +0.7265) | whole_to_part_SUPPORT 19/20; part_to_whole_NE 20/20 | overclaim_to_SUPPORT 0; refute_to_SUPPORT 0 | Promising shadow owner; final composer still DENY | Establishes structured coverage shadow owner |
| Stage34-C | shadow (diagnostic only) | held-out generalization | 0.2069 | 0.9652 (Δ +0.7583) | whole_to_part_support_recovered 120; part_to_whole_ne_preserved 120 | not assessed for leakage safety at this stage | Held-out generalization strong; composer not enabled | Shows held-out support-oriented generalization |
| Stage35-A | shadow (diagnostic only) | adversarial stress test | 0.2832 | 0.5832 | — | overclaim_to_SUPPORT 58; exception_to_SUPPORT_error 8; location_scope_to_SUPPORT_error 25 | unsafe (scope_safety=unsafe, reverse_handling=unsafe) | Exposes unsafe adversarial leakage, motivates Stage36 |
| Stage36-A | shadow safety blockers | adversarial stress test | 0.2832 | 0.639 (Δ +0.3558) | support_shadow_support held at 91 | post-blocker: overclaim_to_SUPPORT 0; exception_to_SUPPORT_error 0; location_scope_to_SUPPORT_error 0 | scope_safety=safe, reverse_handling=mixed | Blocks unsafe SUPPORT leakage without regressing recovery |
| Stage37-A | shadow safe recovery | adversarial stress test | 0.2832 | 0.7363 (Δ +0.4531) | support_shadow_support 91→153; subset_support_recovered 60→104; exception_support_recovered 6→24; numeric_support_recovered 0→24 | overclaim_to_SUPPORT 0; exception_to_SUPPORT_error 0; location_scope_to_SUPPORT_error 0; refute_to_SUPPORT 0 | scope_safety=safe, reverse_handling=mixed | Restores safe SUPPORT recovery atop Stage36 blockers |
| Stage38-A | shadow (diagnostic only) | integrated regression audit (dev) | n/a | n/a | dev mismatch count 0; dev mismatch rate 0.0 | adversarial_safety_pass=true; adversarial_recovery_pass=true | STAGE38A_INTEGRATED_REGRESSION_PASS | Validates shadow owner is safe as diagnostic owner |
| Stage39-A | final composer (support_only) | dev / Stage34 / Stage35 | dev 0.3617; Stage34 0.2069; Stage35 0.2832 | dev 0.3617; Stage34 0.6063; Stage35 0.5789 | Stage34 changed_to_SUPPORT 164; Stage35 changed_to_SUPPORT 150 | 0 unsafe SUPPORT counters | safe but too weak | Shows SUPPORT-only composition is useful but insufficient |
| Stage39-B | final composer (safe_structured) | dev / Stage34 / Stage35 | dev 0.3617; Stage34 0.2069; Stage35 0.2832 | dev 0.3617; Stage34 0.8431; Stage35 0.7202 | Stage34 changed_to_SUPPORT 164, changed_to_REFUTE 20; Stage35 changed_to_SUPPORT 150, changed_to_REFUTE 19 | 0 unsafe SUPPORT counters | strong improvement, Stage34 still misses some REFUTE coverage | Intermediate composer step toward safe_structured_v2 |
| Stage39-C | final composer (safe_structured_v2) | dev / Stage34 / Stage35 | dev 0.3617; Stage34 0.2069; Stage35 0.2832 | dev 0.3617; Stage34 0.9703; Stage35 0.7202 | Stage34 changed_to_SUPPORT 164, changed_to_REFUTE 40; Stage35 changed_to_SUPPORT 150, changed_to_REFUTE 19 | 0 introduced unsafe SUPPORT / REFUTE→SUPPORT / SUPPORT→REFUTE | STAGE39C_SAFE_STRUCTURED_V2_PASS | Explicit opt-in final composer, still off by default |
| Stage40-A | final composer integrated audit | dev + Stage34 + Stage35 | dev 0.3617; Stage34 0.2069; Stage35 0.2832 | dev 0.3617; Stage34 0.9703; Stage35 0.7202 | Stage34 changed_to_SUPPORT 164, changed_to_REFUTE 40; Stage35 changed_to_SUPPORT 150, changed_to_REFUTE 19 | aggregate_introduced_unsafe_SUPPORT_total 0; aggregate_introduced_refute_to_SUPPORT 0; aggregate_introduced_support_to_REFUTE 0 | STAGE40A_FINAL_COMPOSER_REGRESSION_PASS | Confirms integrated regression safety of the final composer |

## 3. Final Composer Table

See [stage41a_final_composer_table.md](stage41a_final_composer_table.md) for the standalone version of this table.

| Policy/Audit | Dev Changed Rows | Stage34 Macro-F1 | Stage34 Changed SUPPORT/REFUTE | Stage35 Macro-F1 | Stage35 Changed SUPPORT/REFUTE | Introduced Unsafe SUPPORT | Introduced REFUTE→SUPPORT | Introduced SUPPORT→REFUTE | Final Status |
|---|---|---|---|---|---|---|---|---|---|
| support_only (Stage39-A) | 0 | 0.2069 → 0.6063 | 164 / 0 | 0.2832 → 0.5789 | 150 / 0 | 0 | 0 | 0 | safe but too weak |
| safe_structured (Stage39-B) | 0 | 0.2069 → 0.8431 | 164 / 20 | 0.2832 → 0.7202 | 150 / 19 | 0 | 0 | 0 | strong improvement, Stage34 REFUTE coverage gap |
| safe_structured_v2 (Stage39-C) | 0 | 0.2069 → 0.9703 | 164 / 40 | 0.2832 → 0.7202 | 150 / 19 | 0 | 0 | 0 | STAGE39C_SAFE_STRUCTURED_V2_PASS |
| Stage40 integrated audit | 0 | 0.9703 (confirmed) | 164 / 40 | 0.7202 (confirmed) | 150 / 19 | 0 (aggregate) | 0 (aggregate) | 0 (aggregate) | STAGE40A_FINAL_COMPOSER_REGRESSION_PASS |

## 4. Paper-Ready Prose

Building on a structured coverage shadow owner (Stage33-F/F2) that was hardened through an adversarial safety pass (Stage35→Stage36) and a safe recovery pass (Stage37), and validated via an integrated regression audit against the dev set (Stage38), we introduce an explicit opt-in final composer, `safe_structured_v2` (Stage39-C). Under this opt-in composer, held-out generalization macro-F1 improves from 0.2069 to 0.9703 on the Stage34 held-out probe, and adversarial-stress macro-F1 improves from 0.2832 to 0.7202 on the Stage35 probe, while the dev set remains completely unchanged (0 changed rows). An integrated regression audit (Stage40-A) confirms that across dev, Stage34, and Stage35, the aggregate counts of composer-introduced unsafe SUPPORT predictions, REFUTE→SUPPORT flips, and SUPPORT→REFUTE flips are all zero. These results support using `safe_structured_v2` as an explicit, opt-in final-prediction replacement under the evaluated profiles; the composer remains disabled by default pending further external and naturalistic validation.

## 5. Allowed Claims

- The structured owner improves held-out synthetic coverage entailment diagnostics.
- Stage36/37 separate safety blocking from safe support recovery.
- Stage39-C can be used as an explicit opt-in final-prediction replacement under the evaluated profiles.
- Stage40-A confirms no Stage39-C-introduced unsafe SUPPORT / REFUTE-to-SUPPORT / SUPPORT-to-REFUTE errors across dev/Stage34/Stage35.
- Default behavior remains off.

## 6. Forbidden Claims

- Do not claim production robustness.
- Do not claim broad naturalistic NLI robustness.
- Do not claim the composer should be enabled by default.
- Do not claim learned neural understanding of coverage semantics.
- Do not claim Stage34/35 are independent real-world benchmarks.
- Do not claim zero total model errors, only zero introduced unsafe errors by the composer.

## 7. Limitations

- Stage34/35 are synthetic/probe-style evaluations, not naturalistic benchmarks.
- Stage35 residual `support_to_refute` remains 7 after composition — inherited from the underlying model, not introduced by Stage39-C.
- Dev residual `support_to_refute` remains 46 — inherited from the underlying model, not introduced by Stage39-C.
- Stage39-C's gains are rule/composer-mediated, not proof of neural semantic generalization.
- Default-off behavior is policy-confirmed (the composer ships disabled and requires explicit opt-in), not established via an explicit default-off prediction diff inside Stage40.
- External/naturalistic validation remains required before any broader robustness claim.

## 8. Final Frozen Interpretation

`Stage39-C safe_structured_v2, audited by Stage40-A, is approved as an explicit opt-in final composer under the evaluated profiles, while remaining disabled by default.`
