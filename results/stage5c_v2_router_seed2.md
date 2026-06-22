## ROUTER_CLASSIFICATION_SUMMARY

| system | accuracy | macro-F1 | NOT_ENTITLED F1 | REFUTE F1 | SUPPORT F1 | prediction distribution |
|---|---:|---:|---:|---:|---:|---|
| classifier_only | 0.885 | 0.864 | 0.919 | 1.000 | 0.674 | `{"NOT_ENTITLED": 172, "REFUTE": 28, "SUPPORT": 60}` |
| balanced_only | 0.854 | 0.839 | 0.896 | 1.000 | 0.620 | `{"NOT_ENTITLED": 164, "REFUTE": 28, "SUPPORT": 68}` |
| strict_only | 0.842 | 0.825 | 0.888 | 1.000 | 0.586 | `{"NOT_ENTITLED": 165, "REFUTE": 28, "SUPPORT": 67}` |
| conservative_balanced_router | 0.896 | 0.868 | 0.929 | 1.000 | 0.675 | `{"NOT_ENTITLED": 181, "REFUTE": 28, "SUPPORT": 51}` |
| conservative_strict_router | 0.885 | 0.852 | 0.921 | 1.000 | 0.634 | `{"NOT_ENTITLED": 182, "REFUTE": 28, "SUPPORT": 50}` |
| dual_auditor_router | 0.892 | 0.859 | 0.927 | 1.000 | 0.650 | `{"NOT_ENTITLED": 184, "REFUTE": 28, "SUPPORT": 48}` |

## ROUTER_PAIRWISE_SUMMARY

| system | paraphrase | predicate-pair | polarity-flip | deletion | truncation | entity | event |
|---|---:|---:|---:|---:|---:|---:|---:|
| classifier_only | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 0.950 | 1.000 |
| balanced_only | 0.950 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.800 |
| strict_only | 0.850 | 0.900 | 1.000 | 1.000 | 1.000 | 0.950 | 0.850 |
| conservative_balanced_router | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| conservative_strict_router | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |
| dual_auditor_router | 0.800 | 0.950 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 |

## ROUTER_INTERNAL_FAITHFULNESS_SUMMARY

| system | entitled gate violation rate | entitled violations/count | polarity output ok | polarity internal ok | output-internal gap | output-ok/internal-bad count |
|---|---:|---:|---:|---:|---:|---:|
| classifier_only | 0.205 | 18/88 | 1.000 | 0.750 | 0.250 | 5 |
| balanced_only | 0.135 | 13/96 | 1.000 | 0.950 | 0.050 | 1 |
| strict_only | 0.116 | 11/95 | 1.000 | 0.950 | 0.050 | 1 |
| conservative_balanced_router | 0.000 | 0/79 | 0.950 | 0.950 | 0.000 | 0 |
| conservative_strict_router | 0.000 | 0/78 | 0.950 | 0.950 | 0.000 | 0 |
| dual_auditor_router | 0.000 | 0/76 | 0.950 | 0.950 | 0.000 | 0 |

## ROUTER_KEY_CONTRAST

| system | macro-F1 | predicate-pair | polarity-flip | paraphrase |
|---|---:|---:|---:|---:|
| classifier_only | 0.864 | 0.950 | 1.000 | 0.950 |
| balanced_only | 0.839 | 0.900 | 1.000 | 0.950 |
| strict_only | 0.825 | 0.900 | 1.000 | 0.850 |
| conservative_balanced_router | 0.868 | 0.950 | 0.950 | 0.800 |
| conservative_strict_router | 0.852 | 0.950 | 0.950 | 0.800 |
| dual_auditor_router | 0.859 | 0.950 | 0.950 | 0.800 |

## INTERPRETATION

Best final-label macro-F1: **conservative_balanced_router**. Best predicate-pair consistency: **classifier_only**. Best polarity-flip consistency: **classifier_only**.
