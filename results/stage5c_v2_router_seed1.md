## ROUTER_CLASSIFICATION_SUMMARY

| system | accuracy | macro-F1 | NOT_ENTITLED F1 | REFUTE F1 | SUPPORT F1 | prediction distribution |
|---|---:|---:|---:|---:|---:|---|
| classifier_only | 0.908 | 0.876 | 0.937 | 1.000 | 0.692 | `{"NOT_ENTITLED": 182, "REFUTE": 30, "SUPPORT": 48}` |
| balanced_only | 0.869 | 0.843 | 0.908 | 0.984 | 0.637 | `{"NOT_ENTITLED": 168, "REFUTE": 31, "SUPPORT": 61}` |
| strict_only | 0.858 | 0.831 | 0.900 | 1.000 | 0.593 | `{"NOT_ENTITLED": 169, "REFUTE": 30, "SUPPORT": 61}` |
| conservative_balanced_router | 0.915 | 0.882 | 0.943 | 1.000 | 0.703 | `{"NOT_ENTITLED": 186, "REFUTE": 30, "SUPPORT": 44}` |
| conservative_strict_router | 0.908 | 0.871 | 0.938 | 1.000 | 0.676 | `{"NOT_ENTITLED": 186, "REFUTE": 30, "SUPPORT": 44}` |
| dual_auditor_router | 0.912 | 0.875 | 0.941 | 1.000 | 0.685 | `{"NOT_ENTITLED": 187, "REFUTE": 30, "SUPPORT": 43}` |

## ROUTER_PAIRWISE_SUMMARY

| system | paraphrase | predicate-pair | polarity-flip | deletion | truncation | entity | event |
|---|---:|---:|---:|---:|---:|---:|---:|
| classifier_only | 0.850 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 0.950 |
| balanced_only | 0.950 | 0.900 | 0.950 | 1.000 | 1.000 | 0.900 | 0.850 |
| strict_only | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 0.950 | 0.850 |
| conservative_balanced_router | 0.800 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| conservative_strict_router | 0.750 | 0.800 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| dual_auditor_router | 0.750 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |

## ROUTER_INTERNAL_FAITHFULNESS_SUMMARY

| system | entitled gate violation rate | entitled violations/count | polarity output ok | polarity internal ok | output-internal gap | output-ok/internal-bad count |
|---|---:|---:|---:|---:|---:|---:|
| classifier_only | 0.295 | 23/78 | 0.850 | 0.450 | 0.400 | 8 |
| balanced_only | 0.130 | 12/92 | 0.950 | 0.900 | 0.050 | 1 |
| strict_only | 0.077 | 7/91 | 0.900 | 0.900 | 0.000 | 0 |
| conservative_balanced_router | 0.000 | 0/74 | 0.850 | 0.850 | 0.000 | 0 |
| conservative_strict_router | 0.000 | 0/74 | 0.850 | 0.850 | 0.000 | 0 |
| dual_auditor_router | 0.000 | 0/73 | 0.850 | 0.850 | 0.000 | 0 |

## ROUTER_KEY_CONTRAST

| system | macro-F1 | predicate-pair | polarity-flip | paraphrase |
|---|---:|---:|---:|---:|
| classifier_only | 0.876 | 0.800 | 0.850 | 0.850 |
| balanced_only | 0.843 | 0.900 | 0.950 | 0.950 |
| strict_only | 0.831 | 0.800 | 0.900 | 0.850 |
| conservative_balanced_router | 0.882 | 0.850 | 0.850 | 0.800 |
| conservative_strict_router | 0.871 | 0.800 | 0.850 | 0.750 |
| dual_auditor_router | 0.875 | 0.850 | 0.850 | 0.750 |

## INTERPRETATION

Best final-label macro-F1: **conservative_balanced_router**. Best predicate-pair consistency: **balanced_only**. Best polarity-flip consistency: **balanced_only**.
