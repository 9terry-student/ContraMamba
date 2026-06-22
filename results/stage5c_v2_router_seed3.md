## ROUTER_CLASSIFICATION_SUMMARY

| system | accuracy | macro-F1 | NOT_ENTITLED F1 | REFUTE F1 | SUPPORT F1 | prediction distribution |
|---|---:|---:|---:|---:|---:|---|
| classifier_only | 0.900 | 0.869 | 0.931 | 1.000 | 0.675 | `{"NOT_ENTITLED": 176, "REFUTE": 32, "SUPPORT": 52}` |
| balanced_only | 0.904 | 0.870 | 0.934 | 1.000 | 0.675 | `{"NOT_ENTITLED": 179, "REFUTE": 32, "SUPPORT": 49}` |
| strict_only | 0.831 | 0.803 | 0.878 | 1.000 | 0.532 | `{"NOT_ENTITLED": 162, "REFUTE": 32, "SUPPORT": 66}` |
| conservative_balanced_router | 0.923 | 0.885 | 0.948 | 1.000 | 0.706 | `{"NOT_ENTITLED": 188, "REFUTE": 32, "SUPPORT": 40}` |
| conservative_strict_router | 0.908 | 0.869 | 0.938 | 0.984 | 0.685 | `{"NOT_ENTITLED": 184, "REFUTE": 31, "SUPPORT": 45}` |
| dual_auditor_router | 0.919 | 0.879 | 0.946 | 0.984 | 0.706 | `{"NOT_ENTITLED": 189, "REFUTE": 31, "SUPPORT": 40}` |

## ROUTER_PAIRWISE_SUMMARY

| system | paraphrase | predicate-pair | polarity-flip | deletion | truncation | entity | event |
|---|---:|---:|---:|---:|---:|---:|---:|
| classifier_only | 0.950 | 0.850 | 0.950 | 1.000 | 1.000 | 0.950 | 1.000 |
| balanced_only | 0.900 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| strict_only | 0.850 | 0.800 | 0.900 | 1.000 | 1.000 | 1.000 | 0.900 |
| conservative_balanced_router | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |
| conservative_strict_router | 0.850 | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 |
| dual_auditor_router | 0.900 | 0.850 | 0.850 | 1.000 | 1.000 | 1.000 | 1.000 |

## ROUTER_INTERNAL_FAITHFULNESS_SUMMARY

| system | entitled gate violation rate | entitled violations/count | polarity output ok | polarity internal ok | output-internal gap | output-ok/internal-bad count |
|---|---:|---:|---:|---:|---:|---:|
| classifier_only | 0.167 | 14/84 | 0.950 | 0.850 | 0.100 | 2 |
| balanced_only | 0.111 | 9/81 | 0.950 | 0.850 | 0.100 | 2 |
| strict_only | 0.184 | 18/98 | 0.900 | 0.900 | 0.000 | 0 |
| conservative_balanced_router | 0.000 | 0/72 | 0.850 | 0.850 | 0.000 | 0 |
| conservative_strict_router | 0.000 | 0/76 | 0.900 | 0.900 | 0.000 | 0 |
| dual_auditor_router | 0.000 | 0/71 | 0.850 | 0.850 | 0.000 | 0 |

## ROUTER_KEY_CONTRAST

| system | macro-F1 | predicate-pair | polarity-flip | paraphrase |
|---|---:|---:|---:|---:|
| classifier_only | 0.869 | 0.850 | 0.950 | 0.950 |
| balanced_only | 0.870 | 0.950 | 0.950 | 0.900 |
| strict_only | 0.803 | 0.800 | 0.900 | 0.850 |
| conservative_balanced_router | 0.885 | 0.850 | 0.850 | 0.900 |
| conservative_strict_router | 0.869 | 0.850 | 0.900 | 0.850 |
| dual_auditor_router | 0.879 | 0.850 | 0.850 | 0.900 |

## INTERPRETATION

Best final-label macro-F1: **conservative_balanced_router**. Best predicate-pair consistency: **balanced_only**. Best polarity-flip consistency: **classifier_only**.
