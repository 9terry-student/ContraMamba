# Paper-Ready ContraMamba-CAR Result Tables

Values are means +/- sample standard deviations across three seeds. All results are from the controlled intervention evaluations recorded under `results/`.

## Table 1. Main ContraMamba-CAR result

| Dataset/version | System | Threshold | Final accuracy | Macro-F1 | SUPPORT F1 | Gate violation rate | Output/internal polarity gap |
|---|---|---:|---:|---:|---:|---:|---:|
| `controlled_v5_v2` | `conservative_balanced_router` | 0.5 | 0.912 +/- 0.014 | 0.878 +/- 0.009 | 0.694 +/- 0.017 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| `controlled_v5_v3` | **ContraMamba-CAR (`conservative_balanced_router`)** | **0.5** | **0.929 +/- 0.003** | **0.906 +/- 0.005** | **0.765 +/- 0.011** | **0.000 +/- 0.000** | **0.000 +/- 0.000** |

The v3 row is the main result. ContraMamba-CAR combines the `v3_no_intervention` classifier with the `v3_no_polarity_flip` balanced auditor.

## Table 2. Threshold stability

ContraMamba-CAR (`conservative_balanced_router`) on `controlled_v5_v3`:

| Threshold | Final accuracy | Macro-F1 | SUPPORT F1 | Gate violation rate | Output/internal polarity gap |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 0.762 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| 0.4 | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 0.762 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| **0.5** | **0.929 +/- 0.003** | **0.906 +/- 0.005** | **0.765 +/- 0.011** | **0.000 +/- 0.000** | **0.000 +/- 0.000** |
| 0.6 | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 0.763 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| 0.7 | 0.930 +/- 0.002 | 0.906 +/- 0.003 | 0.765 +/- 0.008 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

Macro-F1 remains within 0.905-0.906 across the sweep, while both internal-faithfulness diagnostics remain zero.

## Table 3. Self-routing ablation

| System | Threshold | Macro-F1 | Gate violation rate | Polarity output ok | Output/internal gap | Paraphrase preserved | Predicate disentanglement |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw_classifier_only` | 0.4 | 0.903 +/- 0.005 | 0.131 +/- 0.124 | 0.994 +/- 0.010 | 0.156 +/- 0.241 | 0.983 +/- 0.017 | 0.978 +/- 0.019 |
| `self_routed_classifier` | 0.4 | **0.912 +/- 0.018** | 0.000 +/- 0.000 | 0.839 +/- 0.251 | 0.000 +/- 0.000 | 0.894 +/- 0.142 | 0.894 +/- 0.142 |
| `raw_balanced_only` | 0.5 | 0.880 +/- 0.012 | 0.023 +/- 0.014 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.989 +/- 0.010 |
| `self_routed_balanced` | 0.5 | 0.888 +/- 0.008 | 0.000 +/- 0.000 | **1.000 +/- 0.000** | 0.000 +/- 0.000 | **1.000 +/- 0.000** | **0.989 +/- 0.010** |

`self_routed_classifier` is the highest-macro-F1 ablation but is weaker and more variable on controlled intervention consistency. `self_routed_balanced` is more intervention-faithful but has lower macro-F1 than ContraMamba-CAR.

## Recommended paper framing

- The main system is a multi-layer classifier-auditor routing architecture: ContraMamba-CAR.
- The work should not be framed as a single-model compression project.
- The classifier supplies final-label strength.
- The balanced auditor supplies evidence-entitlement checking through frame, predicate, sufficiency, and entitlement gates.
- The router retains supported `SUPPORT` or `REFUTE` classifier outputs and downgrades unsupported entitled outputs to `NOT_ENTITLED`.
- `self_routed_classifier` is a high-macro-F1 ablation, but its intervention consistency is weaker and more variable.
- ContraMamba-CAR is the main system because it balances final-label performance with controlled internal and intervention-level faithfulness.

## Numbers to avoid overclaiming

- Do not make a state-of-the-art claim.
- Do not claim real-world hallucination reduction or elimination.
- Do not claim validation in a deployed RAG system.
- Do not claim a general solution to factuality.
- Do not claim that single-model compression is impossible or unnecessary in all settings.

## Sources

- `docs/stage7_v3_results_narrative.md`
- `results/stage7c_v3_router_aggregate.csv`
- `results/stage7c_v3_self_routing_aggregate.csv`
- `results/stage5c_v2_router_aggregate.csv`
