# Stage41-A Final Composer Comparison Table

| Policy/Audit | Dev Changed Rows | Stage34 Macro-F1 | Stage34 Changed SUPPORT/REFUTE | Stage35 Macro-F1 | Stage35 Changed SUPPORT/REFUTE | Introduced Unsafe SUPPORT | Introduced REFUTE→SUPPORT | Introduced SUPPORT→REFUTE | Final Status |
|---|---|---|---|---|---|---|---|---|---|
| support_only (Stage39-A) | 0 | 0.2069 → 0.6063 | 164 / 0 | 0.2832 → 0.5789 | 150 / 0 | 0 | 0 | 0 | safe but too weak |
| safe_structured (Stage39-B) | 0 | 0.2069 → 0.8431 | 164 / 20 | 0.2832 → 0.7202 | 150 / 19 | 0 | 0 | 0 | strong improvement, Stage34 REFUTE coverage gap |
| safe_structured_v2 (Stage39-C) | 0 | 0.2069 → 0.9703 | 164 / 40 | 0.2832 → 0.7202 | 150 / 19 | 0 | 0 | 0 | STAGE39C_SAFE_STRUCTURED_V2_PASS |
| Stage40 integrated audit | 0 | 0.9703 (confirmed) | 164 / 40 | 0.7202 (confirmed) | 150 / 19 | 0 (aggregate) | 0 (aggregate) | 0 (aggregate) | STAGE40A_FINAL_COMPOSER_REGRESSION_PASS |

## Note: Residual vs. Introduced Errors

"Introduced" errors are unsafe transitions (unsafe SUPPORT, REFUTE→SUPPORT, SUPPORT→REFUTE) that the composer itself creates when replacing the model's original prediction. All three policies audited here — `support_only`, `safe_structured`, and `safe_structured_v2` — introduce zero such errors on dev, Stage34, and Stage35, and this is independently confirmed in aggregate by the Stage40-A integrated regression audit.

"Residual" errors are pre-existing errors from the underlying model that the composer does not touch or does not fully correct. These are not composer defects: Stage35 retains a residual `support_to_refute` count of 7, and dev retains a residual `support_to_refute` count of 46, both inherited from the original model's predictions rather than caused by Stage39-C's composition logic. The final composer is safe with respect to what it changes; it is not a claim that all underlying model errors are eliminated.
