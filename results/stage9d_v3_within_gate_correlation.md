# Stage 9D Within-Stratum Gate Correlation

## Global, between-intervention, and residualized correlation

| scope | pearson_mean_abs_off_diagonal | pearson_max_abs_off_diagonal | spearman_mean_abs_off_diagonal | spearman_max_abs_off_diagonal |
|---|---:|---:|---:|---:|
| global | 0.3538 +/- 0.0056 | 0.5910 +/- 0.0034 | 0.4144 +/- 0.0196 | 0.7744 +/- 0.0168 |
| between_intervention_means | 0.4000 +/- 0.0068 | 0.7463 +/- 0.0309 | 0.4899 +/- 0.0022 | 0.8407 +/- 0.0440 |
| residualized | 0.3721 +/- 0.0684 | 0.9372 +/- 0.0408 | 0.3017 +/- 0.0257 | 0.8339 +/- 0.0463 |

## Within-intervention correlation

| intervention_type | pearson_mean_abs_off_diagonal | pearson_max_abs_off_diagonal | spearman_mean_abs_off_diagonal | spearman_max_abs_off_diagonal |
|---|---:|---:|---:|---:|
| entity_swap | 0.5678 +/- 0.2435 | 0.9966 +/- 0.0027 | 0.3597 +/- 0.0726 | 0.8934 +/- 0.0327 |
| event_swap | 0.8830 +/- 0.0712 | 0.9975 +/- 0.0019 | 0.4332 +/- 0.1765 | 0.9560 +/- 0.0229 |
| evidence_deletion | 0.5697 +/- 0.0633 | 0.9996 +/- 0.0007 | 0.6262 +/- 0.0555 | 0.9984 +/- 0.0023 |
| evidence_truncation | 0.3348 +/- 0.0416 | 0.9938 +/- 0.0106 | 0.4269 +/- 0.1191 | 0.9811 +/- 0.0293 |
| irrelevant_evidence | 0.4506 +/- 0.1319 | 0.8321 +/- 0.1473 | 0.4969 +/- 0.1589 | 0.8486 +/- 0.1280 |
| location_swap | 0.4191 +/- 0.0445 | 0.9994 +/- 0.0004 | 0.5838 +/- 0.0361 | 0.9803 +/- 0.0268 |
| none | 0.6169 +/- 0.0130 | 0.9715 +/- 0.0220 | 0.5734 +/- 0.0899 | 0.9903 +/- 0.0071 |
| paraphrase | 0.5753 +/- 0.1048 | 0.9817 +/- 0.0239 | 0.5313 +/- 0.0890 | 0.9652 +/- 0.0413 |
| polarity_flip | 0.4174 +/- 0.1186 | 0.9972 +/- 0.0025 | 0.4261 +/- 0.0940 | 0.9908 +/- 0.0073 |
| predicate_swap | 0.5329 +/- 0.0633 | 0.9995 +/- 0.0005 | 0.4281 +/- 0.0240 | 0.9946 +/- 0.0068 |
| role_swap | 0.4002 +/- 0.0758 | 0.9769 +/- 0.0284 | 0.4230 +/- 0.0145 | 0.9686 +/- 0.0472 |
| time_swap | 0.6023 +/- 0.2031 | 0.9710 +/- 0.0249 | 0.3904 +/- 0.0595 | 0.9977 +/- 0.0019 |
| title_name_swap | 0.4970 +/- 0.2477 | 0.9702 +/- 0.0084 | 0.3820 +/- 0.0999 | 0.8139 +/- 0.0578 |

## Interpretation

If global correlation is moderate while residualized and within-intervention correlations are low, the global pattern may be driven by between-intervention difficulty rather than a shared gate mechanism. If within-intervention correlations remain high, the heads move together even within the same perturbation type, supporting partial redundancy or a shared mechanism.

The gate heads show moderate global correlation, but within-stratum analysis is needed to distinguish shared mechanism from between-intervention difficulty.

Do not claim four independent gates or one effective gate from global correlations alone. The conclusion must follow the within-intervention and residualized results.

## Link to Stage 10A

Stage 9D does not determine whether time_swap is temporal-specific or a broader same-type low-surface-change substitution failure. The number_swap probe tests that distinction directly.
