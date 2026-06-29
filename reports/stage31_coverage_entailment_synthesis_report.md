# Stage31 Coverage/Entailment Synthesis Lock Report

## Purpose
This report locks the Stage31 Coverage/Entailment diagnostic findings before moving to structural redesign. Stage31 tested whether the current ContraMamba v7 proxy stack can handle directional Coverage/Entailment ownership through the existing final stack or through diagnostic auxiliary heads.

## Comparative Results
| Stage | Variant | Total Accuracy | Macro-F1 | Direction Alignment | Support Recovered | Overclaim Detected | Refute Detected | Critical Misreads | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Stage31-A/B | current proxy stack | 0.445 | 0.3607 | n/a | n/a | n/a | n/a | support_entailment_predicted_ne=61; coverage_failure_predicted_support=6; refute_case_predicted_support=5; refute_case_predicted_ne=27 | Conservative but under-structured; suppresses some overclaims but fails to preserve valid SUPPORT under weakening/generalization/part-inclusion. |
| Stage31-C | 4-class diagnostic head | n/a | n/a | 0.445 | 39 / 80 | 44 / 80 | 6 / 40 | CONTRADICTS_REFUTE->ENTAILS_SUPPORT=20 / 40; OVERCLAIM_NOT_ENTITLED->ENTAILS_SUPPORT=27 / 80 | Partial directional signal exists, but the head is unsafe for composer integration. |
| Stage31-C2 | 3-class hard-contrast auxiliary redesign | 0.450 | 0.3535 | 0.365 | 42 / 80 | 14 / 80 | 17 / 40 | overclaim_misread_as_entails_support=32; refute_misread_as_entails_support=17 | Hard 3-class supervision did not solve the problem; refute improved slightly, but overclaim ownership collapsed. |
| Stage31-C3 | raw_pair representation access | 0.420 | 0.3096 | 0.400 | 46 / 80 | 18 / 80 | 16 / 40 | refute_to_entails=15; overclaim_to_entails=34; support_to_overclaim=19; support_to_refute=15 | Less-compressed pair access does not stabilize the diagnostic head. |
| Stage31-C3 | hybrid representation access | 0.460 | 0.3743 | 0.340 | 31 / 80 | 26 / 80 | 11 / 40 | refute_to_entails=17; overclaim_to_entails=23; support_to_overclaim=40; support_to_refute=9 | Hybrid access improves some final-label metrics but worsens direction alignment; it is still unsafe. |

## Locked Conclusion
Directional Coverage/Entailment cannot be reliably recovered as a small post-hoc auxiliary neural readout over the current proxy stack. The repeated failure across Stage31-B, Stage31-C, Stage31-C2, and Stage31-C3 shows that this is not merely a 4-class-vs-3-class issue, not merely a hard-contrast data issue, and not merely compressed representation access.

## Final Decision
| Decision | Status |
|---|---|
| Stage31-D composer integration | DENY |
| Coverage/Entailment cap/boost/head wiring | DENY |
| Additional small head-only patching | NOT RECOMMENDED |
| Proceed to Stage32 structural redesign | ALLOW |

## Architectural Implication
Coverage/Entailment must become a first-class owner or layer in the next architecture, not a post-hoc diagnostic head. It should be placed after Hard Core validity and before Residual Adjudication / ANI diagnostic.

The next architecture should explicitly model directional relation types, including:

| Directional Relation | Intended Outcome |
|---|---|
| all -> some | SUPPORT |
| some -> all | NOT_ENTITLED |
| specific -> general | SUPPORT |
| general -> specific | NOT_ENTITLED |
| whole -> part | SUPPORT |
| part -> whole | NOT_ENTITLED |
| only -> base | SUPPORT |
| also -> only | NOT_ENTITLED |
| none -> some | REFUTE |

This may require explicit symbolic/scope features or structured contrastive operators rather than an MLP over existing representations.

## Leakage Policy
The Stage31 probe remains diagnostic-only. It must not be used for training, fine-tuning, calibration, threshold selection, checkpoint selection, model selection, or auxiliary loss construction.

## Remaining Risk
The Stage31 probe is templated and diagnostic, not a broad benchmark. However, the repeated failure across head/input variants is strong enough to justify structural redesign.

## Lock
Stage31-D is denied. Stage32 structural redesign is the next approved step.
