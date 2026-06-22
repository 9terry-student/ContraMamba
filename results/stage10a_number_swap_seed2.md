# Stage 10A Number-Swap Single-Axis Probe

| probe | n | gold NOT_ENTITLED | pred SUPPORT | pred REFUTE | pred NOT_ENTITLED | errors | false-entitled rate | frame | predicate | sufficiency | entitlement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| number_swap | 60 | 60 | 1 | 0 | 59 | 1 | 0.0167 | 0.0531 | 0.5847 | 0.2698 | 0.0068 |
| time_swap | 60 | 60 | 60 | 0 | 0 | 60 | 1.0000 | 0.9081 | 0.9953 | 0.9898 | 0.8945 |

## Temporal-specific vs same-type-substitution decision

Diagnostic classification: `temporal_specific_failure`.

If number_swap and time_swap both show high false-entitled rates with high pass probabilities, the evidence supports a broader same-type low-surface-change or slot-value comparison failure. If number_swap is rejected while time_swap fails, the evidence favors a temporal-specific failure. If the ontology labels numeric mismatch as REFUTE, it must be analyzed separately as a value contradiction.

## Interpretation constraints

Do not generalize the time-swap result into broad presence-vs-match blindness: entity, event, and predicate swaps are already mostly rejected in v3.

Do not infer shared gate mechanisms from global correlation alone. Stage 9D within-stratum and residualized correlations are the relevant diagnostic.
