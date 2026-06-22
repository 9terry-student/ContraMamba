# Stage 7 v3 Results: ContraMamba-CAR

## Dataset expansion

Stage 7 evaluates ContraMamba on `controlled_v5_v3`, an expansion to 300 claim-evidence pair groups and 3,900 examples. Each group contains the original example and the same 12 controlled interventions used in v2. Train and development partitions are split by `pair_id`, so an original example and its interventions cannot cross partitions. Results below are means and sample standard deviations over three seeds.

## Main system

The main system is **ContraMamba-CAR**, the ContraMamba Classifier-Auditor Router. It separates final-label prediction from evidence-entitlement auditing:

- the `v3_no_intervention` classifier supplies the candidate `REFUTE`, `NOT_ENTITLED`, or `SUPPORT` label;
- the `v3_no_polarity_flip` balanced auditor supplies frame, predicate, sufficiency, and entitlement gates;
- an entitled classifier output is retained only when all auditor gates meet the routing threshold; otherwise it is downgraded to `NOT_ENTITLED`.

The primary configuration is `conservative_balanced_router` at threshold 0.5. An optional strict auditor was evaluated, but it is not required for the main system. ContraMamba-CAR is deliberately a multi-layer classifier-auditor architecture; the experiments do not assume that the final system must be compressed into one model.

## Main result

| Threshold | Accuracy | Macro-F1 | NOT_ENTITLED F1 | REFUTE F1 | SUPPORT F1 | Gate violation rate | Output/internal polarity gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 0.951 +/- 0.002 | 1.000 +/- 0.000 | 0.762 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| 0.4 | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 0.951 +/- 0.002 | 1.000 +/- 0.000 | 0.762 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| **0.5** | **0.929 +/- 0.003** | **0.906 +/- 0.005** | **0.952 +/- 0.002** | **1.000 +/- 0.000** | **0.765 +/- 0.011** | **0.000 +/- 0.000** | **0.000 +/- 0.000** |
| 0.6 | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 0.952 +/- 0.002 | 1.000 +/- 0.000 | 0.763 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| 0.7 | 0.930 +/- 0.002 | 0.906 +/- 0.003 | 0.952 +/- 0.001 | 1.000 +/- 0.000 | 0.765 +/- 0.008 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

Macro-F1 remains within 0.905-0.906 across thresholds 0.3-0.7. The measured entitled-output gate violation rate and output/internal polarity gap remain zero throughout the sweep. Threshold 0.5 is therefore a conventional operating point rather than a value selected around an isolated peak.

## Self-routing ablation

| System | Threshold | Accuracy | Macro-F1 | Gate violation rate | Polarity output ok | Output/internal gap | Paraphrase | Predicate disentanglement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_classifier_only` | 0.4 | 0.928 +/- 0.004 | 0.903 +/- 0.005 | 0.131 +/- 0.124 | 0.994 +/- 0.010 | 0.156 +/- 0.241 | 0.983 +/- 0.017 | 0.978 +/- 0.019 |
| `self_routed_classifier` | 0.4 | **0.942 +/- 0.004** | **0.912 +/- 0.018** | 0.000 +/- 0.000 | 0.839 +/- 0.251 | 0.000 +/- 0.000 | 0.894 +/- 0.142 | 0.894 +/- 0.142 |
| `raw_balanced_only` | 0.5 | 0.904 +/- 0.012 | 0.880 +/- 0.012 | 0.023 +/- 0.014 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.989 +/- 0.010 |
| `self_routed_balanced` | 0.5 | 0.912 +/- 0.007 | 0.888 +/- 0.008 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.989 +/- 0.010 |

`self_routed_classifier` has the highest final-label macro-F1, but its polarity-flip, paraphrase, and predicate consistency are weaker and more variable than ContraMamba-CAR. It is therefore a high-accuracy ablation, not a replacement for the main architecture.

`self_routed_balanced` is the strongest faithful single-model ablation: it eliminates measured gate violations and preserves polarity flips and paraphrases. Its macro-F1, 0.888 +/- 0.008, is lower than CAR's 0.906 +/- 0.005. This result shows that self-routing is feasible, but does not establish single-model compression as the preferred endpoint.

## Interpretation

ContraMamba-CAR provides the strongest balance of final-label performance and intervention-level faithfulness. The classifier contributes label strength, while the balanced auditor verifies entitlement to an asserted support or refute decision. Their separation preserves macro-F1 near the strongest classifier variants while enforcing zero measured gate violations and zero output/internal polarity gap.

The ablations sharpen the central finding. Final-label prediction and evidence-entitlement auditing are separable functions: optimizing one does not guarantee the other. A self-routed classifier can improve aggregate macro-F1 while weakening controlled polarity, paraphrase, and predicate behavior. Conversely, a balanced model can be highly intervention-faithful while giving up final-label performance. Fixed hybrid router rules evaluated in Stage 6C did not improve on the conservative balanced-router baseline, so additional router complexity was not needed for the current controlled result.

## Claims we can make

- In this controlled intervention setting, classifier-only final-label performance and intervention-level faithfulness diverge.
- ContraMamba-CAR preserves high final-label performance while eliminating measured entitled-output gate violations.
- The CAR result is stable across routing thresholds from 0.3 through 0.7.
- A balanced auditor can enforce evidence-entitlement constraints without requiring the strict model in the main configuration.
- The more complex fixed hybrid routers tested in Stage 6C were not needed to obtain the main controlled result.

## Claims we cannot make

- We do not claim state-of-the-art performance.
- We do not claim general hallucination reduction or elimination.
- We do not claim validation in real-world RAG or deployed systems.
- We do not claim that ContraMamba solves factuality.
- We do not claim that single-model compression or distillation is necessary.
- The evidence does not extend beyond the controlled intervention data evaluated here.
