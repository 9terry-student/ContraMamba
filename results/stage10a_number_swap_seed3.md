# Stage 10A Number-Swap Single-Axis Probe

| probe | n | gold NOT_ENTITLED | pred SUPPORT | pred REFUTE | pred NOT_ENTITLED | errors | false-entitled rate | frame | predicate | sufficiency | entitlement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| number_swap | 60 | 60 | 0 | 0 | 60 | 0 | 0.0000 | 0.0112 | 0.5818 | 0.0042 | 0.0002 |
| time_swap | 60 | 60 | 59 | 0 | 1 | 59 | 0.9833 | 0.9054 | 0.9816 | 0.9912 | 0.8827 |

## Temporal-specific vs same-type-substitution decision

Diagnostic classification: `temporal_specific_failure`.

If number_swap and time_swap both show high false-entitled rates with high pass probabilities, the evidence supports a broader same-type low-surface-change or slot-value comparison failure. If number_swap is rejected while time_swap fails, the evidence favors a temporal-specific failure. If the ontology labels numeric mismatch as REFUTE, it must be analyzed separately as a value contradiction.

## Interpretation constraints

Do not generalize the time-swap result into broad presence-vs-match blindness: entity, event, and predicate swaps are already mostly rejected in v3.

Do not infer shared gate mechanisms from global correlation alone. Stage 9D within-stratum and residualized correlations are the relevant diagnostic.
