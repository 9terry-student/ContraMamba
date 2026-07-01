# Stage42-A Paper-Claim Consistency Audit

## 1. Overall Decision

**`STAGE42A_PAPER_CLAIM_CONSISTENCY_INCOMPLETE`**

Five of the eight required source/summary files for this audit do not exist anywhere in the repository. The Stage41-A freeze documents are internally self-consistent and consistent with the "required frozen numbers" reference values supplied for this audit, and contain no forbidden claims or boundary violations. However, the audit cannot independently verify Stage41-A's numbers against the underlying Stage39-C and Stage40-A *source* report artifacts, because those artifacts were never written to disk — only their generating scripts were committed. This is a completeness failure, not a numeric contradiction, so the decision is `INCOMPLETE` rather than `FAIL` or `PASS`.

## 2. Files Checked

| File | Status |
|---|---|
| `reports/stage41a_paper_ready_result_freeze.md` | found |
| `reports/stage41a_paper_ready_result_freeze.json` | found |
| `reports/stage41a_final_composer_table.md` | found |
| `reports/stage39c_safe_structured_v2_dev_report.json` | **missing** |
| `reports/stage39c_safe_structured_v2_stage34_report.json` | **missing** |
| `reports/stage39c_safe_structured_v2_stage35_report.json` | **missing** |
| `reports/stage40a_final_composer_regression_audit.json` | **missing** |
| `reports/stage40a_final_composer_regression_audit.md` | **missing** |

Verification method: `ls`/`find` over `reports/` and a full-repository filename search for `*stage39c*` and `*stage40a*` (also checked `git show --stat` on the commits titled "Add Stage39 final composer opt-in validation", "Add Stage39C safe structured v2 composer audit", and "Add Stage40 final composer regression audit" — those commits added only `scripts/evaluate_stage39_final_composer.py` and the Stage38/Stage40 evaluator scripts, no report artifacts). No files were run to regenerate the missing reports, per audit scope.

## 3. Numeric Consistency Table

Because the five source files above are missing, the numbers below are checked for **internal consistency between the Stage41-A documents and the "required frozen numbers" reference values supplied for this audit** — not independently re-derived from Stage39-C/Stage40-A source artifacts. This distinction is called out explicitly in the Issues section.

| Metric | Required Frozen Value | Stage41-A `.md` | Stage41-A `.json` | Consistent? |
|---|---|---|---|---|
| dev original acc | 0.5611 | 0.5611 | 0.5611 | yes (vs. reference; source unverifiable) |
| dev composed acc | 0.5611 | 0.5611 | 0.5611 | yes (vs. reference; source unverifiable) |
| dev original macro-F1 | 0.3617 | 0.3617 | 0.3617 | yes (vs. reference; source unverifiable) |
| dev composed macro-F1 | 0.3617 | 0.3617 | 0.3617 | yes (vs. reference; source unverifiable) |
| dev changed rows | 0 | 0 | 0 | yes (vs. reference; source unverifiable) |
| dev introduced unsafe SUPPORT | 0 | 0 | 0 | yes (vs. reference; source unverifiable) |
| dev introduced REFUTE→SUPPORT | 0 | 0 | 0 | yes (vs. reference; source unverifiable) |
| dev introduced SUPPORT→REFUTE | 0 | 0 | 0 | yes (vs. reference; source unverifiable) |
| Stage34 original acc | 0.4500 | 0.45 | 0.45 | yes (vs. reference; source unverifiable) |
| Stage34 composed acc | 0.9600 | 0.96 | 0.96 | yes (vs. reference; source unverifiable) |
| Stage34 original macro-F1 | 0.2069 | 0.2069 | 0.2069 | yes (vs. reference; source unverifiable) |
| Stage34 composed macro-F1 | 0.9703 | 0.9703 | 0.9703 | yes (vs. reference; source unverifiable) |
| Stage34 changed rows | 204 | 204 | 204 | yes (vs. reference; source unverifiable) |
| Stage34 changed_to_SUPPORT | 164 | 164 | 164 | yes (vs. reference; source unverifiable) |
| Stage34 changed_to_REFUTE | 40 | 40 | 40 | yes (vs. reference; source unverifiable) |
| Stage34 introduced unsafe SUPPORT/REFUTE→SUPPORT/SUPPORT→REFUTE | 0/0/0 | 0/0/0 | 0/0/0 | yes (vs. reference; source unverifiable) |
| Stage34 decision | `STAGE39C_SAFE_STRUCTURED_V2_PASS` | present | present | yes (vs. reference; source unverifiable) |
| Stage35 original acc | 0.5033 | 0.5033 | 0.5033 | yes (vs. reference; source unverifiable) |
| Stage35 composed acc | 0.7850 | 0.785 | 0.785 | yes (vs. reference; source unverifiable) |
| Stage35 original macro-F1 | 0.2832 | 0.2832 | 0.2832 | yes (vs. reference; source unverifiable) |
| Stage35 composed macro-F1 | 0.7202 | 0.7202 | 0.7202 | yes (vs. reference; source unverifiable) |
| Stage35 changed rows | 169 | 169 | 169 | yes (vs. reference; source unverifiable) |
| Stage35 changed_to_SUPPORT | 150 | 150 | 150 | yes (vs. reference; source unverifiable) |
| Stage35 changed_to_REFUTE | 19 | 19 | 19 | yes (vs. reference; source unverifiable) |
| Stage35 residual support_to_refute | 7 | 7 | 7 | yes (vs. reference; source unverifiable) |
| Stage35 introduced unsafe SUPPORT/REFUTE→SUPPORT/SUPPORT→REFUTE | 0/0/0 | 0/0/0 | 0/0/0 | yes (vs. reference; source unverifiable) |
| Stage35 decision | `STAGE39C_SAFE_STRUCTURED_V2_PASS` | present | present | yes (vs. reference; source unverifiable) |
| Stage40-A decision | `STAGE40A_FINAL_COMPOSER_REGRESSION_PASS` | present | present | yes (vs. reference; source unverifiable) |
| Stage40-A pass flags (6 flags) | all `true` | all true | all true | yes (vs. reference; source unverifiable) |
| Stage40-A aggregate introduced unsafe SUPPORT/REFUTE→SUPPORT/SUPPORT→REFUTE | 0/0/0 | 0/0/0 | 0/0/0 | yes (vs. reference; source unverifiable) |

No numeric contradictions were found between Stage41-A and the supplied reference numbers. Also note: Stage41-A's dev residual `support_to_refute` limitation figure of 46 is stated only in Stage41-A itself (not part of the "required frozen numbers" list) and likewise cannot be cross-checked against a Stage39-C dev source report, since none exists.

## 4. Claim Consistency Audit

Stage41-A's own "Allowed Claims" section matches the allowed-claims list for this audit:

| Allowed claim | Present in Stage41-A? |
|---|---|
| Stage33-F established a structured coverage shadow owner | yes |
| Stage36/37 separate safety blocking from safe recovery | yes |
| Stage39-C `safe_structured_v2` is safe as an explicit opt-in final-prediction replacement under the evaluated profiles | yes |
| Stage40-A confirms zero Stage39-C-introduced unsafe SUPPORT / REFUTE-to-SUPPORT / SUPPORT-to-REFUTE errors across dev/Stage34/Stage35 | yes |
| Default behavior remains off | yes |
| Stage34/35 are diagnostic/probe-style evaluations | yes |

Forbidden-claim scan of `stage41a_paper_ready_result_freeze.md`, `.json`, and `stage41a_final_composer_table.md`:

| Forbidden claim | Found in Stage41-A? |
|---|---|
| production robustness | no — explicitly disclaimed |
| broad naturalistic NLI robustness | no — explicitly disclaimed |
| composer should be enabled by default | no — explicitly disclaimed |
| learned neural understanding of coverage semantics | no — explicitly disclaimed |
| Stage34/35 as independent real-world benchmarks | no — explicitly disclaimed |
| zero total model errors | no — explicitly disclaimed (Stage41-A distinguishes "zero introduced" from "zero total") |
| zero residual errors | no — Stage41-A states residual `support_to_refute` = 7 (Stage35) and 46 (dev) |
| zero hallucination | no — not claimed anywhere |
| general solution to entailment/coverage reasoning | no — not claimed anywhere |

**Result: no forbidden claims found.**

## 5. Boundary Audit

| Boundary | Clearly distinguished in Stage41-A? | Evidence |
|---|---|---|
| Shadow diagnostic owner vs. explicit opt-in final composer | yes | Executive summary and main result table separate Stage33/36/37/38 ("shadow", "diagnostic only") from Stage39/40 ("final composer") |
| Explicit opt-in vs. default-off production behavior | yes | §5 allowed claims and §8 final interpretation both state the composer "remains disabled by default" |
| Residual total errors vs. Stage39-introduced errors | yes | §7 Limitations and the composer-table note explicitly separate "residual" (inherited from the model: Stage35=7, dev=46) from "introduced" (composer-caused: 0 in all cases) |
| Synthetic/probe validation vs. naturalistic external validation | yes | §7 Limitations states Stage34/35 are "synthetic/probe-style evaluations," and §9 calls for "external and naturalistic validation" before broader claims |

**Result: no boundary violations found.**

## 6. Table Audit

`reports/stage41a_final_composer_table.md` was checked against the required comparison:

- Contains all four required rows: `support_only` (Stage39-A), `safe_structured` (Stage39-B), `safe_structured_v2` (Stage39-C), and the Stage40 integrated audit row. **Correct.**
- The accompanying "Note: Residual vs. Introduced Errors" explicitly separates the two error categories, stating the composer introduces zero unsafe transitions while residual model errors (Stage35 support_to_refute=7, dev support_to_refute=46) are pre-existing and untouched by the composer. **No confusion found.**
- The table and note contain no statement implying default-on behavior; default status is not asserted in the table file itself, and the referring document (`stage41a_paper_ready_result_freeze.md`) explicitly states default-off. **No default-on implication found.**

**Result: table audit passes.**

## 7. Artifact Index

See [stage42a_artifact_index.md](stage42a_artifact_index.md) for the full artifact index with purpose, evidence classification, citability, and leakage warnings.

## 8. Issues and Warnings

### Issues (block full PASS)

1. **Missing source artifacts.** `reports/stage39c_safe_structured_v2_dev_report.json`, `reports/stage39c_safe_structured_v2_stage34_report.json`, `reports/stage39c_safe_structured_v2_stage35_report.json`, `reports/stage40a_final_composer_regression_audit.json`, and `reports/stage40a_final_composer_regression_audit.md` do not exist in the repository. Only the generating scripts (`scripts/evaluate_stage39_final_composer.py`, Stage38/Stage40 evaluator scripts) were committed; their output reports were not saved. This means Stage41-A's numbers trace back to numbers supplied directly in the Stage41-A task prompt, not to a verifiable on-disk source report. Independent numeric verification against source artifacts is currently impossible.

### Warnings (wording/format only, no numeric contradiction)

1. Stage41-A's dev residual `support_to_refute = 46` figure is asserted only in the Stage41-A limitations section; it has no corresponding entry in the "required frozen numbers" list for this audit and, like all other figures, cannot be cross-checked against a missing Stage39-C dev source report.
2. Because the Stage39-C/Stage40-A source reports are absent, any future regeneration of those reports carries latent risk of numeric drift from the currently frozen Stage41-A values; this should be re-audited once the source reports are produced and committed.

## 9. Final Recommendation

Treat Stage41-A's numbers as **provisionally frozen but not yet independently source-verified**. Before citing Stage41-A/Stage39-C/Stage40-A results in a paper or external-facing report, the missing Stage39-C and Stage40-A report artifacts should be generated (by running the already-committed evaluator scripts) and saved to `reports/`, then re-audited against this same numeric-consistency checklist. Until that happens, downstream citation should note that Stage41-A's numbers currently rest on the audit-supplied reference values rather than a re-verifiable on-disk source report. No claim, boundary, or table issue was found — the only blocking issue is source-artifact completeness.

## 10. Leakage Policy

This document is reporting/audit only. Do not use Stage34/35 probes or any derived reports (Stage39, Stage40, Stage41, Stage42) for training, calibration, threshold selection, checkpoint selection, or loss design. Stage41/Stage42 documents are derived summaries, not new experimental evidence.
