# Stage42-B Source-Recovered Paper-Claim Consistency Audit

## 1. Overall Decision

**`STAGE42B_SOURCE_RECOVERED_CLAIM_CONSISTENCY_PASS`**

All eight source artifacts that Stage42-A flagged as missing (the Stage39-C dev/Stage34/Stage35 reports and the Stage40-A JSON/MD audit, plus their .md companions) are now present in `reports/`. Every Stage41-A number checked against these recovered source reports matches exactly, no forbidden claims appear, all boundary distinctions hold, and the final composer table is consistent with source data. Stage41-A's claims are now independently source-verified rather than resting on audit-supplied reference values.

## 2. Source Recovery Status

| File | Status |
|---|---|
| `reports/stage39c_safe_structured_v2_dev_report.json` | **present** |
| `reports/stage39c_safe_structured_v2_dev_report.md` | **present** |
| `reports/stage39c_safe_structured_v2_stage34_report.json` | **present** |
| `reports/stage39c_safe_structured_v2_stage34_report.md` | **present** |
| `reports/stage39c_safe_structured_v2_stage35_report.json` | **present** |
| `reports/stage39c_safe_structured_v2_stage35_report.md` | **present** |
| `reports/stage40a_final_composer_regression_audit.json` | **present** |
| `reports/stage40a_final_composer_regression_audit.md` | **present** |

All previously-missing files from Stage42-A's `missing_files` list are resolved. No files remain missing.

## 3. Files Checked

- `reports/stage41a_paper_ready_result_freeze.md`
- `reports/stage41a_paper_ready_result_freeze.json`
- `reports/stage41a_final_composer_table.md`
- `reports/stage39c_safe_structured_v2_dev_report.json` / `.md`
- `reports/stage39c_safe_structured_v2_stage34_report.json` / `.md`
- `reports/stage39c_safe_structured_v2_stage35_report.json` / `.md`
- `reports/stage40a_final_composer_regression_audit.json` / `.md`
- `reports/stage42a_paper_claim_consistency_audit.json` / `.md` (compared for delta, not modified)

## 4. Numeric Consistency Table

All values below are read directly from the recovered on-disk source reports and compared against Stage41-A's `.md`/`.json` figures.

| Metric | Source Report Value | Stage41-A `.md` | Stage41-A `.json` | Consistent? |
|---|---|---|---|---|
| dev original acc | 0.5611 | 0.5611 | 0.5611 | yes (source-verified) |
| dev composed acc | 0.5611 | 0.5611 | 0.5611 | yes (source-verified) |
| dev original macro-F1 | 0.3617 | 0.3617 | 0.3617 | yes (source-verified) |
| dev composed macro-F1 | 0.3617 | 0.3617 | 0.3617 | yes (source-verified) |
| dev changed rows | 0 | 0 | 0 | yes (source-verified) |
| dev introduced unsafe SUPPORT | 0 | 0 | 0 | yes (source-verified) |
| dev introduced REFUTE→SUPPORT | 0 | 0 | 0 | yes (source-verified) |
| dev introduced SUPPORT→REFUTE | 0 | 0 | 0 | yes (source-verified) |
| dev residual support_to_refute (informational) | 46 | 46 | n/a (not in main_metrics) | yes (source-verified) |
| Stage34 decision | `STAGE39C_SAFE_STRUCTURED_V2_PASS` | present | present | yes (source-verified) |
| Stage34 original acc | 0.4500 | 0.45 | 0.45 | yes (source-verified) |
| Stage34 composed acc | 0.9600 | 0.96 | 0.96 | yes (source-verified) |
| Stage34 original macro-F1 | 0.2069 | 0.2069 | 0.2069 | yes (source-verified) |
| Stage34 composed macro-F1 | 0.9703 | 0.9703 | 0.9703 | yes (source-verified) |
| Stage34 changed rows | 204 | 204 | 204 | yes (source-verified) |
| Stage34 changed_to_SUPPORT | 164 | 164 | 164 | yes (source-verified) |
| Stage34 changed_to_REFUTE | 40 | 40 | 40 | yes (source-verified) |
| Stage34 introduced unsafe SUPPORT/REFUTE→SUPPORT/SUPPORT→REFUTE | 0/0/0 | 0/0/0 | 0/0/0 | yes (source-verified) |
| Stage35 decision | `STAGE39C_SAFE_STRUCTURED_V2_PASS` | present | present | yes (source-verified) |
| Stage35 original acc | 0.5033 | 0.5033 | 0.5033 | yes (source-verified) |
| Stage35 composed acc | 0.7850 | 0.785 | 0.785 | yes (source-verified) |
| Stage35 original macro-F1 | 0.2832 | 0.2832 | 0.2832 | yes (source-verified) |
| Stage35 composed macro-F1 | 0.7202 | 0.7202 | 0.7202 | yes (source-verified) |
| Stage35 changed rows | 169 | 169 | 169 | yes (source-verified) |
| Stage35 changed_to_SUPPORT | 150 | 150 | 150 | yes (source-verified) |
| Stage35 changed_to_REFUTE | 19 | 19 | 19 | yes (source-verified) |
| Stage35 residual support_to_refute | 7 | 7 | 7 | yes (source-verified) |
| Stage35 introduced unsafe SUPPORT/REFUTE→SUPPORT/SUPPORT→REFUTE | 0/0/0 | 0/0/0 | 0/0/0 | yes (source-verified) |
| Stage40-A decision | `STAGE40A_FINAL_COMPOSER_REGRESSION_PASS` | present | present | yes (source-verified) |
| Stage40-A pass flags (6 flags) | all `true` | all true | all true | yes (source-verified) |
| Stage40-A aggregate introduced unsafe SUPPORT/REFUTE→SUPPORT/SUPPORT→REFUTE | 0/0/0 | 0/0/0 | 0/0/0 | yes (source-verified) |

**No numeric contradictions found.** Every number in the task's "Required source values" checklist matches the recovered Stage39-C and Stage40-A source reports exactly, and every corresponding Stage41-A figure matches those source values exactly.

## 5. Claim Consistency Audit

Stage41-A's "Allowed Claims" section (§5 of `stage41a_paper_ready_result_freeze.md`) against the required allowed-claims list:

| Allowed claim | Present in Stage41-A? | Source-supported? |
|---|---|---|
| Stage33-F established a structured coverage shadow owner | yes | yes — Stage41-A executive summary; not contradicted by any recovered report |
| Stage36/37 separate safety blocking from safe recovery | yes | yes — consistent with Stage41-A's stage36/stage37 metric summaries |
| Stage39-C `safe_structured_v2` is safe as an explicit opt-in final-prediction replacement under the evaluated profiles | yes | yes — `stage39c_safe_structured_v2_stage34_report.json` and `stage35_report.json` recommendation fields state this verbatim |
| Stage40-A confirms zero Stage39-C-introduced unsafe SUPPORT / REFUTE-to-SUPPORT / SUPPORT-to-REFUTE errors across dev/Stage34/Stage35 | yes | yes — `stage40a_final_composer_regression_audit.json` `introduced_safety_summary.all_zero = true` |
| Default behavior remains off | yes | yes — `stage40a_...json` `sections.default_off_policy_check.policy_confirmed = true` |
| Stage34/35 are diagnostic/probe-style evaluations | yes | yes — Stage41-A §7 Limitations; consistent with `leakage_policy` fields in all recovered Stage39-C/Stage40-A reports stating "Diagnostic-only" / "Diagnostic/aggregation-only" |

Forbidden-claim scan of `stage41a_paper_ready_result_freeze.md`, `.json`, and `stage41a_final_composer_table.md`:

| Forbidden claim | Found in Stage41-A? |
|---|---|
| production robustness | no — explicitly disclaimed |
| broad naturalistic NLI robustness | no — explicitly disclaimed |
| composer should be enabled by default | no — explicitly disclaimed |
| learned neural understanding of coverage semantics | no — explicitly disclaimed |
| Stage34/35 as independent real-world benchmarks | no — explicitly disclaimed |
| zero total model errors | no — explicitly disclaimed (distinguishes "zero introduced" from "zero total") |
| zero residual errors | no — Stage41-A states residual `support_to_refute` = 7 (Stage35) and 46 (dev), both confirmed against source reports |
| zero hallucination | no — not claimed anywhere |
| general solution to entailment/coverage reasoning | no — not claimed anywhere |

**Result: no forbidden claims found; all allowed claims present and now source-supported.**

## 6. Boundary Audit

| Boundary | Clearly distinguished in Stage41-A? | Evidence |
|---|---|---|
| Shadow diagnostic owner vs. explicit opt-in final composer | yes | Executive summary and main result table separate Stage33/36/37/38 ("shadow", "diagnostic only") from Stage39/40 ("final composer") |
| Explicit opt-in vs. default-off production behavior | yes | §5 allowed claims and §8 final interpretation state the composer "remains disabled by default"; confirmed by the recovered `stage40a_...json` `default_off_policy_check.status = "policy_confirmed"` |
| Residual total errors vs. Stage39-introduced errors | yes | §7 Limitations and the composer-table note separate "residual" (inherited: dev=46, Stage35=7) from "introduced" (composer-caused: 0 in all cases); both figures confirmed against the recovered source reports' `safety_counters` / `residual_errors` sections |
| Synthetic/probe validation vs. naturalistic external validation | yes | §7 Limitations states Stage34/35 are "synthetic/probe-style evaluations"; §9/recommendation calls for "external and naturalistic validation"; the recovered `stage40a_...md` recommendation independently repeats "Further external/naturalistic validation is still required before claiming broad production robustness" |

**Result: no boundary violations found.**

## 7. Table Audit

`reports/stage41a_final_composer_table.md` checked against the recovered source reports:

- Contains all four required rows: `support_only` (Stage39-A), `safe_structured` (Stage39-B), `safe_structured_v2` (Stage39-C), and the Stage40 integrated audit row. **Correct.**
- Stage34/Stage35 macro-F1 values, changed_to_SUPPORT/REFUTE counts, and introduced-error counts for the `safe_structured_v2` and `Stage40 integrated audit` rows match the recovered `stage39c_safe_structured_v2_stage34_report.json`, `stage39c_safe_structured_v2_stage35_report.json`, and `stage40a_final_composer_regression_audit.json` exactly (0.9703/0.7202 macro-F1, 164/40 and 150/19 changed counts, 0/0/0 introduced errors). **No discrepancy.**
- The accompanying "Note: Residual vs. Introduced Errors" correctly separates the two error categories; the residual `support_to_refute` figures it cites (Stage35=7, dev=46) match `safety_counters.support_to_refute` in the recovered source reports exactly. **No confusion found.**
- The table and note contain no statement implying default-on behavior; default status is not asserted in the table file itself, and the referring document explicitly states default-off, consistent with the recovered Stage40-A `default_off_policy_check`. **No default-on implication found.**

**Result: table audit passes.**

## 8. Artifact Index

See [stage42b_artifact_index.md](stage42b_artifact_index.md) for the full artifact index with purpose, evidence classification, citability, and leakage warnings.

## 9. Issues and Warnings

### Issues

None. All required source files are present and all required numbers/claims are consistent.

### Warnings

1. Stage41-A's dev residual `support_to_refute = 46` figure is not part of the task's "required source values" checklist, but it is independently confirmed against `reports/stage39c_safe_structured_v2_dev_report.json` `safety_counters.support_to_refute = 46`, matching Stage41-A's limitations-section figure exactly. This is noted for completeness, not as a discrepancy.

## 10. Final Recommendation

- Stage42-A's incomplete provenance gap has been resolved by committing the Stage39-C and Stage40-A source reports.
- Stage41-A's frozen result claims are now traceable to committed source artifacts.
- Proceed to external/naturalistic validation before making broader model-level claims.

## 11. Leakage Policy

This is reporting/audit only. Do not use Stage34/35 probes or derived reports for training, calibration, threshold selection, checkpoint selection, or loss design. Stage41/42 documents are derived summaries, not new experimental evidence.
