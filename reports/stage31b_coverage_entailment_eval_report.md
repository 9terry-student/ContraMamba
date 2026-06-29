# Stage31-B Coverage/Entailment Probe Evaluation

## Purpose
Diagnostic evaluation of the ContraMamba proxy stack on directional Coverage/Entailment cases. Tests whether the current architecture treats quantifier-scope failures as NOT_ENTITLED or incorrectly resolves them as SUPPORT due to frame/predicate/polarity compatibility.

## Probe
- **File:** `data/stage31_coverage_entailment_probe.jsonl`
- **Row count:** 200
- **Mode:** Dry-run (no predictions)
- **Run name:** `stage31b_coverage_entailment_eval`

## Label Distribution
| Label | Count |
|---|---|
| REFUTE | 40 |
| NOT_ENTITLED | 80 |
| SUPPORT | 80 |

## Group Counts
| Group | Count |
|---|---|
| all_to_some_support | 20 |
| also_to_only_not_entitled | 20 |
| general_to_specific_not_entitled | 20 |
| none_to_some_refute | 20 |
| only_to_base_support | 20 |
| part_to_whole_not_entitled | 20 |
| some_to_all_not_entitled | 20 |
| some_to_none_refute | 20 |
| specific_to_general_support | 20 |
| whole_to_part_support | 20 |

## Interpretation
No model predictions were evaluated (dry-run mode). Run with --predictions-file to obtain a Coverage/Entailment diagnostic.

## Observed Stage31-B Diagnostic Pattern
Current observed result: total_accuracy=0.445, macro_f1=0.3607, coverage_failure_predicted_support=6, support_entailment_predicted_ne=61, refute_case_predicted_support=5, refute_case_predicted_ne=27.
The current proxy stack is conservative but under-structured. It can often suppress over-claims, but it cannot reliably preserve valid SUPPORT under weakening/generalization/part-inclusion entailment.
Stage31-C should add a directional Coverage/Entailment owner, not merely another cap.

## Leakage Policy
This probe is **diagnostic-only**. It must not be used for training, fine-tuning, calibration, threshold selection, checkpoint selection, model selection, or any form of model optimisation.

## Next-Step Recommendation
Run with --predictions-file to determine next steps. If predictions reveal systematic failure, proceed to Stage31-C.
