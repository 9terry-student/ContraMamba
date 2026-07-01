# Stage43-A External/Naturalistic Validation Manifest

Preparation/inventory only. No model training, evaluation, or Kaggle/local model execution was performed to produce this manifest.

## 1. Overall Decision

**Decision:** `STAGE43A_EXTERNAL_VALIDATION_MANIFEST_AMBIGUOUS`

- Candidates scanned: 34
- Adaptable: 0
- Ambiguous: 10
- Not adaptable: 24
- Naturalistic adaptable (excludes known synthetic probes): 0

## 2. Candidate File Inventory

| Path | Type | Rows (est.) | Schema status | Mapping status | Claim field(s) | Evidence field(s) | Label field(s) |
|---|---|---|---|---|---|---|---|
| `data/stage10a_number_swap_probe.jsonl` | jsonl | 120 | ambiguous | missing_label | claim | evidence | - |
| `data/stage15_slot_sensitivity_probe.jsonl` | jsonl | 540 | ambiguous | missing_label | claim | evidence | - |
| `data/stage31_coverage_entailment_probe.jsonl` | jsonl | 200 | ambiguous | ambiguous | claim | evidence | label, gold |
| `data/stage34a_heldout_coverage_probe.jsonl` | jsonl | 400 | ambiguous | ambiguous | claim | evidence | label, gold_label |
| `reports/stage27_h3_final_evidence_package.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage29d_external_probe_evaluation_report.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage31_coverage_entailment_probe_report.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage34a_heldout_coverage_probe_report.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage34b_metadata_preservation_report.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage34c_evaluator_summary_fix_report.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage35a_adversarial_coverage_probe_report.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage39c_safe_structured_v2_stage34_report.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage39c_safe_structured_v2_stage35_report.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage42a_paper_claim_consistency_audit.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `reports/stage42b_source_recovered_claim_consistency_audit.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `results/stage10a_matched_controlled_seed1_preds.json` | json | 780 | ambiguous | missing_label | claim | evidence | - |
| `results/stage10a_matched_controlled_seed2_preds.json` | json | 780 | ambiguous | missing_label | claim | evidence | - |
| `results/stage10a_matched_controlled_seed3_preds.json` | json | 780 | ambiguous | missing_label | claim | evidence | - |
| `results/stage10a_number_swap_seed1.csv` | csv | 2 | not_adaptable | missing_label | - | - | - |
| `results/stage10a_number_swap_seed1_preds.json` | json | 120 | ambiguous | missing_label | claim | evidence | - |
| `results/stage10a_number_swap_seed2.csv` | csv | 2 | not_adaptable | missing_label | - | - | - |
| `results/stage10a_number_swap_seed2_preds.json` | json | 120 | ambiguous | missing_label | claim | evidence | - |
| `results/stage10a_number_swap_seed3.csv` | csv | 2 | not_adaptable | missing_label | - | - | - |
| `results/stage10a_number_swap_seed3_preds.json` | json | 120 | ambiguous | missing_label | claim | evidence | - |
| `results/stage21_e3_bestdev_v5_vs_v6b_ood_3seed_mean.csv` | csv | 16 | not_adaptable | missing_label | - | - | - |
| `results/stage21_e3_bestdev_v5_vs_v6b_ood_3seed_summary.csv` | csv | 48 | not_adaptable | missing_label | - | - | - |
| `results/stage21_f0_ood_tradeoff_table.csv` | csv | 25 | not_adaptable | missing_label | - | - | - |
| `results/stage21_f1_v6b_ood_ablation_3seed_summary.csv` | csv | 80 | not_adaptable | missing_label | - | - | - |
| `results/stage22_G2_devcal_ne_shift_ood_seed1.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `results/stage22_G2_devcal_ne_shift_ood_seed2.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `results/stage22_G2_devcal_ne_shift_ood_seed3.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `results/stage22_G3_train_dev_calib_ood_seed1.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `results/stage22_G3_train_dev_calib_ood_seed2.json` | json | None | not_adaptable | not_mapped | - | - | - |
| `results/stage22_G3_train_dev_calib_ood_seed3.json` | json | None | not_adaptable | not_mapped | - | - | - |

## 3. Adaptable Files

No files with `schema_status = adaptable` were found.

## 4. Ambiguous / Non-Adaptable Files

### Ambiguous
- `data/stage10a_number_swap_probe.jsonl` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records.
- `data/stage15_slot_sensitivity_probe.jsonl` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records.
- `data/stage31_coverage_entailment_probe.jsonl` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. Some sampled label values could not be confidently mapped; conservative/manual review required.
- `data/stage34a_heldout_coverage_probe.jsonl` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. Some sampled label values could not be confidently mapped; conservative/manual review required.
- `results/stage10a_matched_controlled_seed1_preds.json` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records.
- `results/stage10a_matched_controlled_seed2_preds.json` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records.
- `results/stage10a_matched_controlled_seed3_preds.json` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records.
- `results/stage10a_number_swap_seed1_preds.json` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records.
- `results/stage10a_number_swap_seed2_preds.json` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records.
- `results/stage10a_number_swap_seed3_preds.json` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records.

### Not adaptable
- `reports/stage27_h3_final_evidence_package.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage29d_external_probe_evaluation_report.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage31_coverage_entailment_probe_report.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage34a_heldout_coverage_probe_report.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage34b_metadata_preservation_report.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage34c_evaluator_summary_fix_report.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage35a_adversarial_coverage_probe_report.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage39c_safe_structured_v2_stage34_report.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage39c_safe_structured_v2_stage35_report.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage42a_paper_claim_consistency_audit.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `reports/stage42b_source_recovered_claim_consistency_audit.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `results/stage10a_number_swap_seed1.csv` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records. No recognized claim-like field found in sampled records. No recognized evidence-like field found in sampled records.
- `results/stage10a_number_swap_seed2.csv` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records. No recognized claim-like field found in sampled records. No recognized evidence-like field found in sampled records.
- `results/stage10a_number_swap_seed3.csv` -- Filename matches a known synthetic/probe-style controlled artifact from a prior stage (Stage10/13/14/15/31/34/35 or controlled_v5); not naturalistic external evidence even if schema is adaptable. No recognized label-like field found in sampled records. No recognized claim-like field found in sampled records. No recognized evidence-like field found in sampled records.
- `results/stage21_e3_bestdev_v5_vs_v6b_ood_3seed_mean.csv` -- No recognized label-like field found in sampled records. No recognized claim-like field found in sampled records. No recognized evidence-like field found in sampled records.
- `results/stage21_e3_bestdev_v5_vs_v6b_ood_3seed_summary.csv` -- No recognized label-like field found in sampled records. No recognized claim-like field found in sampled records. No recognized evidence-like field found in sampled records.
- `results/stage21_f0_ood_tradeoff_table.csv` -- No recognized label-like field found in sampled records. No recognized claim-like field found in sampled records. No recognized evidence-like field found in sampled records.
- `results/stage21_f1_v6b_ood_ablation_3seed_summary.csv` -- No recognized label-like field found in sampled records. No recognized claim-like field found in sampled records. No recognized evidence-like field found in sampled records.
- `results/stage22_G2_devcal_ne_shift_ood_seed1.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `results/stage22_G2_devcal_ne_shift_ood_seed2.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `results/stage22_G2_devcal_ne_shift_ood_seed3.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `results/stage22_G3_train_dev_calib_ood_seed1.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `results/stage22_G3_train_dev_calib_ood_seed2.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).
- `results/stage22_G3_train_dev_calib_ood_seed3.json` -- File could not be parsed as row-oriented data (unreadable or not a records list).

## 5. Recommended Stage43-B Validation Plan

- Only synthetic/controlled probes were found (Stage10/13/14/15/31/34/35, controlled_v5). Recommend collecting or adding a small external naturalistic claim-evidence set before attempting Stage43-B.

Missing expected external sources:

- No file with 'vitaminc' in its path was found under data, experiments, reports, results, docs, outputs.
- No file with 'fever' in its path was found under data, experiments, reports, results, docs, outputs.
- No file with 'rte' in its path was found under data, experiments, reports, results, docs, outputs.
- No file with 'mnli' in its path was found under data, experiments, reports, results, docs, outputs.
- No file with 'snli' in its path was found under data, experiments, reports, results, docs, outputs.
- data/stage35a_adversarial_coverage_probe.jsonl (Stage35-A default probe output referenced by scripts/build_stage35_adversarial_coverage_probe.py) is not present as a persisted file; Stage39-C's Stage35 evaluation likely regenerated it on demand.

## 6. Label Mapping Table

| Path | Label field(s) | Sampled values | Mapping status | Recommended mapping |
|---|---|---|---|---|
| `data/stage31_coverage_entailment_probe.jsonl` | label, gold | SUPPORT, 2 | ambiguous | `{"SUPPORT": "SUPPORT"}` |
| `data/stage34a_heldout_coverage_probe.jsonl` | label, gold_label | 2, SUPPORT | ambiguous | `{"SUPPORT": "SUPPORT"}` |

## 7. Risks

- All currently discoverable candidate data files are synthetic or controlled probe-style artifacts generated for prior ContraMamba stages (Stage10/13/14/15/31/34/35, controlled_v5); none constitute naturalistic external evidence.
- No VitaminC, FEVER, RTE, MNLI, or SNLI style file currently exists in this repository.
- Label field aliasing is heuristic (string/lowercase match against a fixed alias and value list); any 'mapped' status should still be manually spot-checked before use in Stage43-B.
- Treating any file flagged here as naturalistic (when it is in fact a synthetic probe) would overstate external validation readiness for publication claims.
- This manifest inspects only up to the first 5 records per file; field/label distributions beyond the sample are not verified.

## 8. Leakage Policy

This manifest is inventory/preparation only. None of the listed candidate files, sample records, or mappings may be used for training, calibration, threshold selection, checkpoint selection, or loss design. If Stage43-B is executed, it must remain external/evaluation-only, consistent with the eval-only protocol already used for Stage29-D, Stage34-A, and Stage35-A.

## 9. Recommendation

Stage43-A inventory is complete. No naturalistic external claim-evidence dataset (e.g. VitaminC/FEVER/MNLI/SNLI/RTE) currently exists in the repository; all adaptable-schema candidates found are synthetic or controlled probe artifacts from earlier stages. Before any Stage43-B external evaluation can proceed, a genuinely external/naturalistic claim-evidence source should be added to the repository under a dedicated path (e.g. data/external/). Do not use Stage34/35 probes as a substitute for naturalistic external validation, and do not use this manifest or any candidate file for training, calibration, threshold selection, or checkpoint selection.
