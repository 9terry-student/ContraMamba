# Stage 5B — Controlled V2 Main Comparison

## Classification summary

| Configuration | Accuracy | Macro-F1 | Frame | Predicate | Polarity | Sufficiency |
|---|---:|---:|---:|---:|---:|---:|
| `v2_full4e` | 0.844 ± 0.014 | 0.820 ± 0.014 | 0.869 ± 0.010 | 0.923 ± 0.035 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `v2_no_intervention` | 0.897 ± 0.012 | 0.870 ± 0.006 | 0.905 ± 0.006 | 0.949 ± 0.002 | 1.000 ± 0.000 | 0.999 ± 0.002 |

## Pairwise consistency summary

| Configuration | Paraphrase | Predicate-pair | Polarity-flip | Deletion | Truncation | Entity | Event |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v2_full4e` | 0.733 ± 0.126 | 0.950 ± 0.050 | 0.883 ± 0.076 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `v2_no_intervention` | 0.583 ± 0.161 | 0.750 ± 0.100 | 0.217 ± 0.202 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.983 ± 0.029 | 1.000 ± 0.000 |

## Key contrast

`v2_no_intervention` is the stronger final-label classifier: macro-F1 **0.870 ± 0.006** versus **0.820 ± 0.014**.

`v2_full4e` is the stronger intervention-consistency model:

- Polarity-flip consistency: **0.883 ± 0.076** versus **0.217 ± 0.202**.
- Predicate-pair consistency: **0.950 ± 0.050** versus **0.750 ± 0.100**.

## Main interpretation

Final-label accuracy and intervention-consistent entitlement behavior are empirically separable.

