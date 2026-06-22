## CLASSIFICATION_SUMMARY

| config | accuracy | macro-F1 | frame | predicate | polarity | sufficiency |
| --- | --- | --- | --- | --- | --- | --- |
| v2_full4e | 0.844 ± 0.014 | 0.820 ± 0.014 | 0.869 ± 0.010 | 0.923 ± 0.035 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| v2_no_intervention | 0.897 ± 0.012 | 0.870 ± 0.006 | 0.905 ± 0.006 | 0.949 ± 0.002 | 1.000 ± 0.000 | 0.999 ± 0.002 |
| v2_no_predicate_contrast | 0.794 ± 0.022 | 0.784 ± 0.010 | 0.877 ± 0.020 | 0.846 ± 0.027 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| v2_no_polarity_flip | 0.876 ± 0.026 | 0.850 ± 0.017 | 0.894 ± 0.014 | 0.945 ± 0.012 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## PAIRWISE_CONSISTENCY_SUMMARY

| config | paraphrase | predicate-pair | polarity-flip | deletion | truncation | entity | event |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2_full4e | 0.733 ± 0.126 | 0.950 ± 0.050 | 0.883 ± 0.076 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| v2_no_intervention | 0.583 ± 0.161 | 0.750 ± 0.100 | 0.217 ± 0.202 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.983 ± 0.029 | 1.000 ± 0.000 |
| v2_no_predicate_contrast | 0.750 ± 0.050 | 0.817 ± 0.144 | 0.900 ± 0.087 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| v2_no_polarity_flip | 0.783 ± 0.076 | 0.917 ± 0.076 | 0.867 ± 0.058 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## KEY_CONTRAST_SUMMARY

| config | macro-F1 | polarity accuracy | polarity-flip | predicate-pair |
| --- | --- | --- | --- | --- |
| v2_full4e | 0.820 ± 0.014 | 1.000 ± 0.000 | 0.883 ± 0.076 | 0.950 ± 0.050 |
| v2_no_intervention | 0.870 ± 0.006 | 1.000 ± 0.000 | 0.217 ± 0.202 | 0.750 ± 0.100 |
| v2_no_predicate_contrast | 0.784 ± 0.010 | 1.000 ± 0.000 | 0.900 ± 0.087 | 0.817 ± 0.144 |
| v2_no_polarity_flip | 0.850 ± 0.017 | 1.000 ± 0.000 | 0.867 ± 0.058 | 0.917 ± 0.076 |

## INTERPRETATION

- Best final-label classifier: `v2_no_intervention` (macro-F1 0.870).
- Best predicate-pair consistency: `v2_full4e` (0.950).
- Best polarity-flip consistency: `v2_no_predicate_contrast` (0.900).
- Final-label classification and intervention consistency diverge across configurations.
- Removing predicate contrast weakens predicate behavior (0.817 vs 0.950).
- Removing polarity-flip loss does not collapse polarity-flip behavior (0.867 vs 0.883).
