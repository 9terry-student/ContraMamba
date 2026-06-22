# Stage 9B Stratified and Paired Analysis

## Why Stage 9B is needed

Stage 9A shows that ContraMamba-CAR rarely changes the raw classifier output in aggregate. At threshold 0.5, the mean downgrade rate is 0.006 +/- 0.007. The central reviewer-facing question is therefore not whether routing changes many examples, but whether its limited interventions concentrate in the controlled failure types that the auditor is intended to detect.

Stage 9B evaluates this question in three ways:

1. It stratifies pre-routing gate failures, downgrades, recall, precision, and final-label changes by `intervention_type`.
2. It compares raw-classifier and routed pairwise consistency for applicable interventions.
3. It uses paired statistical analyses because the systems are evaluated on the same examples and `pair_id` groups.

No Stage 9B result should be stated before the existing v3 prediction files have been analyzed.

## Stratified analysis

For each seed, routing threshold, and intervention type, the analysis reports raw counts and safe-divided rates. Counts accompany every rate whose denominator may be absent in a stratum. A zero denominator produces numeric `0.0`, not a missing value or `NaN`.

The principal quantities are:

- pre-router classifier-entitled candidates and auditor-gate failures;
- post-router entitled predictions and downgrades;
- `SUPPORT` and `REFUTE` recall and precision before and after routing;
- removed false support and false refute predictions;
- raw-versus-routed accuracy and macro-F1 changes; and
- pairwise success for paraphrase, predicate swap, polarity flip, deletion, truncation, entity swap, and event swap.

## Paired tests

### Exact McNemar test

Final-label correctness is compared between `raw_classifier_only` and `conservative_balanced_router` at threshold 0.5. Tests are reported globally and separately by intervention type. The implementation uses the exact two-sided binomial probability for discordant pairs and reports the full contingency table. The continuity-corrected McNemar statistic is included as a descriptive statistic.

### Pair-ID bootstrap

The bootstrap resamples complete `pair_id` groups rather than individual rows. This preserves dependence between each original example and its interventions. With 1,000 samples, it reports percentile 95% intervals for accuracy delta, macro-F1 delta, `SUPPORT` precision gain, `SUPPORT` recall drop, downgrade rate, and pre-router candidate gate-fail rate.

## Evidence that would support the CAR interpretation

- Aggregate downgrade rate remains low.
- Downgrades or gate failures concentrate in targeted interventions such as `predicate_swap`, `polarity_flip`, `evidence_deletion`, `evidence_truncation`, `entity_swap`, and `event_swap`.
- Downgrades preferentially remove false `SUPPORT` or `REFUTE` predictions.
- Paired intervention-specific behavior improves, or at minimum does not produce substantial recall harm.

## Evidence that would weaken the CAR interpretation

- Downgrades remain uniformly near zero across all intervention types.
- Pairwise analysis shows no intervention-specific improvement.
- The auditor fails to catch false entitled predictions in strata where evidence entitlement should fail.
- Apparent effects depend on very small raw counts or vary sharply across seeds.

If the auditor is nearly inactive across all intervention types, the paper should not claim that CAR improves verification. It should instead present CAR as a diagnostic architecture and report the result as negative or limited.

## Recommended claim wording

Use:

> Final-label prediction and evidence-entitlement auditing can diverge under targeted controlled interventions.

Avoid:

> Final-label prediction and evidence-entitlement auditing are broadly separable functions.

Stage 9B remains a controlled intervention analysis. It does not establish state-of-the-art performance, real-world hallucination reduction, or deployed RAG reliability.

## Execution modes

- **Per seed:** load the v3 dataset and one classifier/balanced/strict prediction triplet; write stratified CSV and Markdown files.
- **Aggregate:** read all three per-seed stratified CSVs; compute mean +/- sample standard deviation by threshold, intervention, and metric.
- **Paired tests:** load all three prediction triplets; write seed-specific exact McNemar and pair-ID bootstrap results into one paired report.
