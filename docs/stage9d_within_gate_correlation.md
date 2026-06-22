# Stage 9D Within-Stratum Gate Correlation

## Purpose

Stage 9C reports global correlations among frame, predicate, sufficiency, entitlement, and polarity outputs. Global correlation alone cannot establish that the gate heads share one mechanism: the association may be induced by differences in average difficulty across intervention types.

Stage 9D decomposes the correlation into four views:

1. global correlation across all examples;
2. within-intervention correlation for each intervention type;
3. between-intervention correlation using intervention means;
4. residualized correlation after subtracting each intervention's mean score.

Both Pearson and Spearman coefficients are reported for every pair, along with mean and maximum absolute off-diagonal correlation.

## Interpretation

- If global correlation is moderate but residualized and within-intervention correlation is low, the global association may reflect between-intervention difficulty rather than a shared gate mechanism.
- If within-intervention correlations also remain high, the heads move together for examples with the same perturbation type, supporting partial redundancy or a shared mechanism.
- Mixed results should be described conservatively: "The gate heads show moderate global correlation, but within-stratum analysis is needed to distinguish shared mechanism from between-intervention difficulty."

Do not claim either "four independent gates" or "one effective gate" unless the within-intervention and residualized evidence supports it.

## Relationship to Stage 10A

Stage 9D concerns dependence among gate outputs. It does not determine why `time_swap` receives high entitlement scores. Stage 10A adds a number-swap probe to distinguish a temporal-specific failure from a broader same-type, low-surface-change slot-value comparison failure.
