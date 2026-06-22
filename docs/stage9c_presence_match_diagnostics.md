# Stage 9C Presence-vs-Match Diagnostics

## Motivation

Stage 9B shows that thresholded router downgrades are sparse. A zero downgrade count, however, has several possible explanations: the classifier may already be correct, the intervention may not be a `NOT_ENTITLED` target, the gate may be uninformative near the threshold, or the gate may confidently point in the wrong direction. Stage 9C separates these cases by analyzing balanced-auditor gate probabilities before routing.

The central mechanism question is:

> Do the structured gate heads detect the presence of evidence, or do they detect whether the evidence matches the claim?

## Opportunity and gate-state analysis

For each seed and intervention type, Stage 9C reports gold and predicted label counts, classifier errors, false entitled predictions, and mean gate probabilities. Each frame, predicate, sufficiency, and entitlement head is then classified on gold-`NOT_ENTITLED` cases as:

- `no_opportunity`: no gold `NOT_ENTITLED` cases or no classifier-entitled predictions;
- `correct_rejection`: mean pass probability below 0.2;
- `uncertain_no_signal`: mean pass probability from 0.4 through 0.6;
- `confidently_inverted`: mean pass probability above 0.8;
- `mixed`: all other cases.

This classification is descriptive and threshold-based. It is paired with the threshold-free expected-direction analysis below.

## Presence-vs-match contrast

The analysis groups interventions into:

- **presence removed:** evidence deletion, evidence truncation, and irrelevant evidence;
- **match perturbed:** time, entity, event, location, role, title/name, and predicate swaps;
- **controls:** none, paraphrase, and polarity flip.

If sufficiency and entitlement probabilities collapse for deletion and truncation but remain high for time swaps, the gate heads are sensitive to evidence presence but fail to reject a claim-evidence mismatch. A lack of router downgrades for deletion or truncation is not evidence of a dead sufficiency head when the classifier already predicts `NOT_ENTITLED` and no downgrade opportunity exists.

## Expected-direction signal analysis

Failure scores are defined as one minus the corresponding pass probability. Their expected direction is fixed in advance: higher failure scores should rank classifier errors and false entitled predictions above correct cases. The report preserves raw AUC:

- `expected_positive`: raw AUC above 0.6;
- `uninformative`: raw AUC from 0.4 through 0.6, or a single-class target;
- `confidently_inverted`: raw AUC below 0.4 while positive cases receive mean pass probability above 0.8;
- `inverted_low_confidence`: other raw AUC values below 0.4.

The analysis also reports `1 - raw_auc` as `inverted_auc`, but never uses `max(AUC, 1-AUC)` as the main score. A raw AUC near zero remains a negative result under the expected direction.

## Polarity interpretation

`polarity_flip` is not automatically a `NOT_ENTITLED` intervention. When its gold labels are `SUPPORT` and `REFUTE`, the relevant question is whether the model reverses polarity correctly while remaining entitled. Stage 9C therefore reports its gold and predicted support/refute distributions rather than treating the absence of downgrades as gate failure.

## Gate-correlation analysis

Pearson and Spearman correlations are computed globally and within each intervention type for frame, predicate, sufficiency, entitlement, and polarity-margin outputs. The report includes every pairwise correlation plus mean and maximum absolute off-diagonal correlation.

High correlations would weaken any claim that the four named gate heads implement independent mechanisms. In that case they should be described as correlated gate heads over a shared representation.

## Paper implication

If the aggregate analysis confirms the preliminary contrast, the appropriate mechanism-level claim is:

> Structured entitlement heads detect evidence presence but fail to reliably detect claim-evidence match, with `time_swap` exposing a confidently inverted entitlement judgment.

This wording is conditional on replication across all three seeds. Stage 9C does not establish temporal mismatch as the root cause, four independent mechanisms, broad entitlement checking, or external validity.
