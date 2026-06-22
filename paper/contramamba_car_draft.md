# ContraMamba-CAR: Separating Label Prediction from Evidence-Entitlement Auditing in Controlled Fact Verification

## 1. Abstract

Fact-verification systems are commonly evaluated by whether they assign the correct final label, but label correctness does not establish that a decision is internally licensed by the supplied evidence. We study this distinction through controlled fact verification, where targeted interventions alter entities, events, predicates, evidence sufficiency, or polarity while preserving other properties of a claim-evidence pair. We introduce ContraMamba-CAR, a classifier-auditor routing architecture that separates final-label prediction from evidence-entitlement auditing. A classifier proposes `SUPPORT`, `REFUTE`, or `NOT_ENTITLED`; a balanced auditor independently checks frame compatibility, predicate coverage, sufficiency, and polarity consistency; and a conservative router retains an entitled prediction only when the auditor gates pass. On `controlled_v5_v3`, containing 300 pair groups and 3,900 examples, ContraMamba-CAR reaches 0.929 +/- 0.003 accuracy and 0.906 +/- 0.005 macro-F1 across three seeds, with a measured entitled-output gate violation rate of 0.000 +/- 0.000 and output/internal polarity gap of 0.000 +/- 0.000. Performance is stable from routing thresholds 0.3 to 0.7. Self-routing ablations reveal that higher macro-F1 can coexist with weaker intervention consistency. These results support separating prediction and entitlement auditing within this controlled setting; they do not establish general factuality or real-world hallucination reduction.

## 2. Introduction

Fact verification is often formulated as final-label classification: given a claim and evidence, a model predicts whether the evidence supports the claim, refutes it, or does not justify either conclusion. This formulation is useful but incomplete. A correct label can result from correlations that do not track whether the supplied evidence actually licenses the decision. Aggregate accuracy therefore cannot, by itself, distinguish evidence-grounded behavior from a correct output with an internally inconsistent rationale.

Controlled interventions make this limitation observable. A paraphrase should preserve the decision, whereas an entity or event substitution should disrupt frame compatibility, a predicate substitution should disrupt predicate coverage, evidence deletion should disrupt sufficiency, and a polarity flip should reverse support and refute while preserving entitlement. A system may score well on individual labels yet respond incorrectly or inconsistently across these paired transformations.

We call the relevant property **evidence entitlement**. A `SUPPORT` or `REFUTE` output is entitled only when the evidence matches the claim frame, covers the relevant predicate, is sufficient for a decision, and has the appropriate polarity. `NOT_ENTITLED` is consequently not a synonym for low confidence: it states that the available evidence does not license an asserted support or refute decision.

We introduce **ContraMamba-CAR**, the ContraMamba Classifier-Auditor Router. The architecture assigns final-label prediction and entitlement verification to separately optimized model roles. A label-focused classifier proposes a decision, while a balanced auditor evaluates structured entitlement gates. A conservative router retains an asserted `SUPPORT` or `REFUTE` label only when the auditor passes it; unsupported entitled predictions are downgraded to `NOT_ENTITLED`.

We evaluate this architecture on `controlled_v5_v3`, a deterministic benchmark of 300 pair groups and 3,900 examples, using pair-level splits and three random seeds. At the default threshold of 0.5, ContraMamba-CAR reaches 0.906 +/- 0.005 macro-F1 while producing zero measured entitled-output gate violations and zero measured output/internal polarity gap. These properties remain stable across thresholds from 0.3 to 0.7.

Our contributions are exactly threefold:

- A formulation of controlled fact verification as separating final-label prediction from evidence-entitlement auditing.
- A classifier-auditor router architecture, ContraMamba-CAR, that combines a final-label classifier with structured entitlement gates.
- A controlled intervention evaluation showing high macro-F1 with zero measured gate violations and zero output/internal polarity gap, plus ablations showing that single-model self-routing is not the preferred main architecture.

## 3. Problem Framing: Evidence-Entitlement Verification

Let a verification input be a claim-evidence pair (x = (c, e)). The output space is:

- `SUPPORT`: the evidence licenses the claim.
- `REFUTE`: the evidence licenses the negation of the claim.
- `NOT_ENTITLED`: the evidence does not license either asserted polarity.

For an entitled output, four conditions must hold:

1. **Frame compatibility:** the claim and evidence concern the same relevant entities, event, roles, time, and location.
2. **Predicate coverage:** the evidence addresses the predicate asserted by the claim.
3. **Sufficiency:** enough evidence is present to make an entitled decision.
4. **Polarity consistency:** the evidence polarity agrees with the proposed support or refute label.

If (g_f), (g_p), and (g_s) denote frame, predicate, and sufficiency gate values, and (g_e) denotes the combined polarity/entitlement condition, an asserted `SUPPORT` or `REFUTE` decision is retained only when the relevant conditions pass the operating threshold. This criterion differs from scalar uncertainty estimation. It asks which evidence relation failed rather than whether a single predictor is globally confident.

`NOT_ENTITLED` is a first-class decision. A model may be confident that the evidence is irrelevant, incomplete, or mismatched; conversely, a high-confidence `SUPPORT` output may still be disallowed when predicate coverage or sufficiency fails. The distinction is thus between confidence in a prediction and entitlement to assert it from the supplied evidence.

## 4. Controlled Intervention Benchmark

`controlled_v5_v3` contains 300 unique `pair_id` groups and 3,900 records. Each group contains one unmodified example and the same 12 controlled variants, yielding 13 records per group. The interventions cover paraphrase, entity, event, time, location, role, title/name, predicate, evidence deletion, evidence truncation, irrelevant evidence, and polarity flip.

Each record includes the claim, evidence, final label, pair identifier, intervention type, and pair-level supervision for frame compatibility, predicate coverage, sufficiency, and polarity. Final labels follow the fixed order `REFUTE`, `NOT_ENTITLED`, and `SUPPORT`. Splits are performed by `pair_id`, preventing an original pair and its interventions from appearing in different partitions.

The benchmark is designed to isolate expected invariances and contrasts. Paraphrases should preserve frame, predicate, sufficiency, and label behavior. Frame substitutions should reduce frame compatibility. Predicate substitutions should preserve the broader frame while reducing predicate coverage. Evidence deletion and truncation should reduce sufficiency. Polarity flips should preserve entitlement while reversing support/refute direction.

This benchmark is controlled and template-generated. Its purpose is diagnostic: it tests whether model outputs and internal gates follow specified intervention semantics. It is not intended as a substitute for naturally occurring fact-verification corpora.

## 5. Method: ContraMamba-CAR

### 5.1 Sequence architecture

Both model roles use the ContraMamba v5 sequence architecture. A Mamba encoder produces token-level representations that feed a `FrameGate`, `PredicateCoverageHead`, `SufficiencyGate`, `PolarityEnergyHead`, and `FinalEntitlementDecisionHead`. The v5 model preserves token-level states through the frame and predicate stages rather than collapsing the sequence before these checks.

The main method figure is documented in [`docs/architecture_diagram.md`](../docs/architecture_diagram.md), with reusable Mermaid source in [`docs/figures/contramamba_car_architecture.mmd`](../docs/figures/contramamba_car_architecture.mmd).

**Figure 1 caption.** ContraMamba-CAR separates final-label prediction from evidence-entitlement auditing. A classifier proposes a `SUPPORT`/`REFUTE`/`NOT_ENTITLED` decision, while a balanced auditor checks frame compatibility, predicate coverage, sufficiency, and polarity consistency. The conservative router retains entitled `SUPPORT`/`REFUTE` predictions and downgrades unsupported entitled predictions to `NOT_ENTITLED`.

### 5.2 Classifier path

The classifier is the `v3_no_intervention` configuration. It is selected for final-label strength and produces the candidate decision. The classifier is not asked to be the sole authority on whether an asserted label is supported by all entitlement conditions.

### 5.3 Auditor path

The auditor is the `v3_no_polarity_flip` balanced configuration. It supplies structured scores for frame compatibility, predicate coverage, sufficiency, and entitlement/polarity behavior. Its role is not to replace the classifier's output universally, but to audit asserted support or refute decisions against the supplied evidence.

### 5.4 Conservative routing

The main operating point is (	au = 0.5). The decision rule is:

```text
if classifier_label == NOT_ENTITLED:
    final_label = NOT_ENTITLED
elif all auditor gates pass:
    final_label = classifier_label
else:
    final_label = NOT_ENTITLED
```

Operationally, all required frame, predicate, sufficiency, and entitlement conditions must pass. The router never promotes a classifier `NOT_ENTITLED` output in the main configuration. It only retains or downgrades an asserted entitled label.

This design is not scalar confidence thresholding. The candidate label and the entitlement audit come from separately optimized model instances, and the audit decomposes evidence support into structured conditions. The architecture can therefore reject a confident classifier decision when a specific evidence relation fails.

## 6. Experimental Setup

We report development-set results over three seeds. The main experiments use a frozen Mamba encoder and train the v5 heads under the controlled supervision defined by each configuration. The classifier is `v3_no_intervention`; the balanced auditor is `v3_no_polarity_flip`. An optional strict auditor was evaluated but is not required by the main CAR configuration.

We evaluate final accuracy, macro-F1, and per-label F1. Pairwise metrics test paraphrase preservation, predicate disentanglement, polarity-flip reversal, evidence-deletion and truncation behavior, and entity/event frame contrasts. Internal-faithfulness metrics count entitled outputs that fail required gates and compare output-level polarity reversals with gate-valid, sign-consistent internal reversals.

The primary threshold is 0.5. We additionally sweep thresholds 0.3, 0.4, 0.6, and 0.7 without changing router rules. Single-model self-routing ablations apply each model's own gates to its own prediction. Fixed hybrid expert routers were also evaluated as diagnostics; their rules were defined before reporting results.

## 7. Results

### 7.1 Main ContraMamba-CAR result

| Dataset/version | System | Threshold | Final accuracy | Macro-F1 | SUPPORT F1 | Gate violation rate | Output/internal polarity gap |
|---|---|---:|---:|---:|---:|---:|---:|
| `controlled_v5_v2` | `conservative_balanced_router` | 0.5 | 0.912 +/- 0.014 | 0.878 +/- 0.009 | 0.694 +/- 0.017 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| `controlled_v5_v3` | **ContraMamba-CAR** | **0.5** | **0.929 +/- 0.003** | **0.906 +/- 0.005** | **0.765 +/- 0.011** | **0.000 +/- 0.000** | **0.000 +/- 0.000** |

On v3, ContraMamba-CAR reaches 0.929 +/- 0.003 accuracy and 0.906 +/- 0.005 macro-F1. The measured entitled-output gate violation rate and output/internal polarity gap are both 0.000 +/- 0.000. These metrics should be read jointly: classification measures whether the final decision is correct, while the latter diagnostics measure whether retained entitled decisions satisfy the configured internal criteria.

### 7.2 Threshold stability

| Threshold | Final accuracy | Macro-F1 | SUPPORT F1 | Gate violation rate | Output/internal polarity gap |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 0.762 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| 0.4 | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 0.762 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| **0.5** | **0.929 +/- 0.003** | **0.906 +/- 0.005** | **0.765 +/- 0.011** | **0.000 +/- 0.000** | **0.000 +/- 0.000** |
| 0.6 | 0.929 +/- 0.003 | 0.905 +/- 0.004 | 0.763 +/- 0.011 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| 0.7 | 0.930 +/- 0.002 | 0.906 +/- 0.003 | 0.765 +/- 0.008 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

Macro-F1 remains between 0.905 and 0.906 across the sweep. Gate violations and the output/internal polarity gap remain zero at every threshold. Threshold 0.5 is therefore an operating point, not a cherry-picked isolated optimum.

## 8. Ablations

### 8.1 Single-model self-routing

| System | Threshold | Macro-F1 | Gate violation rate | Polarity output ok | Output/internal gap | Paraphrase preserved | Predicate disentanglement |
|---|---:|---:|---:|---:|---:|---:|---:|
| `raw_classifier_only` | 0.4 | 0.903 +/- 0.005 | 0.131 +/- 0.124 | 0.994 +/- 0.010 | 0.156 +/- 0.241 | 0.983 +/- 0.017 | 0.978 +/- 0.019 |
| `self_routed_classifier` | 0.4 | **0.912 +/- 0.018** | 0.000 +/- 0.000 | 0.839 +/- 0.251 | 0.000 +/- 0.000 | 0.894 +/- 0.142 | 0.894 +/- 0.142 |
| `raw_balanced_only` | 0.5 | 0.880 +/- 0.012 | 0.023 +/- 0.014 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.989 +/- 0.010 |
| `self_routed_balanced` | 0.5 | 0.888 +/- 0.008 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1.000 +/- 0.000 | 0.989 +/- 0.010 |

`self_routed_classifier` obtains the highest macro-F1, 0.912 +/- 0.018, but its polarity output consistency, paraphrase preservation, and predicate disentanglement are weaker and more variable than those of ContraMamba-CAR. We therefore treat it as a high-accuracy ablation rather than the main architecture.

`self_routed_balanced` eliminates measured gate violations and preserves polarity flips and paraphrases, but reaches 0.888 +/- 0.008 macro-F1, below CAR's 0.906 +/- 0.005. It is a faithful single-model ablation, not evidence that the classifier-auditor separation should be removed.

### 8.2 Fixed hybrid routers

Stage 6C evaluated fixed agreement, override, strict-veto, majority, and cautious-promotion rules. None improved on the conservative balanced-router baseline in that controlled comparison. This result argues against adding rule complexity without additional evidence, but it does not establish that all learned or alternative routers would fail.

## 9. Discussion

The results support the paper's central thesis within the controlled benchmark: final-label prediction and evidence-entitlement auditing are related but separable functions. The self-routed classifier illustrates the distinction most directly. It improves macro-F1 while reducing intervention-level consistency, showing that aggregate classification and controlled faithfulness need not move together.

ContraMamba-CAR uses this separation constructively. The classifier contributes final-label strength, while the balanced auditor constrains when a support or refute assertion may survive. The resulting system does not maximize every individual metric, but it provides a stable joint profile: high macro-F1, zero measured gate violations, zero output/internal polarity gap, and strong controlled intervention behavior.

The zero gate-violation rate must be interpreted carefully. Conservative routing enforces this property by construction for retained entitled outputs. It demonstrates compliance with the specified entitlement rule, not calibrated probability, causal reliance on every gate, or correctness beyond the benchmark. Reporting final-label metrics, entitled-output counts, pairwise intervention checks, and internal diagnostics together is therefore essential.

The current evidence does not require a single-model endpoint. Separate classifier and auditor roles are a legitimate architecture when they encode distinct objectives. Self-routing remains relevant as an efficiency or compression direction, but its desirability depends on whether it can preserve the intervention behavior supplied by the balanced auditor.

## 10. Limitations

- **Controlled benchmark only.** The experiments use a deterministic, template-generated intervention dataset.
- **Synthetic interventions.** The transformations isolate intended failure modes but may not capture the diversity of naturally occurring evidence errors.
- **No state-of-the-art comparison.** The current study does not provide a comprehensive comparison with external fact-verification systems.
- **No external dataset validation.** Generalization to natural fact-verification benchmarks, domain shift, long evidence, and annotation noise remains untested.
- **No deployment claim.** The results do not establish real-world hallucination reduction, RAG reliability, or deployed factuality performance.
- **No impossibility result for single models.** The ablations do not prove that one model cannot learn both label prediction and entitlement auditing.
- **Rule-dependent faithfulness.** Zero measured gate violations partly follows from the conservative routing rule and the selected gate definitions.
- **Frozen encoder.** The main controlled experiments use a frozen Mamba encoder; broader fine-tuning behavior is not evaluated here.

## 11. Conclusion

We presented ContraMamba-CAR, a classifier-auditor architecture for controlled fact verification. The method separates the task of proposing a final label from the task of determining whether the supplied evidence licenses an asserted support or refute decision. On `controlled_v5_v3`, ContraMamba-CAR reaches 0.906 +/- 0.005 macro-F1 while producing zero measured entitled-output gate violations and zero output/internal polarity gap, with stable behavior across thresholds 0.3-0.7.

Single-model ablations reinforce the motivation for this separation: the strongest self-routed classifier improves macro-F1 but weakens controlled intervention consistency, while the balanced self-routed model is more faithful but less accurate. These findings justify ContraMamba-CAR as the main architecture for the present controlled study. Future work must test external datasets, natural evidence perturbations, calibration, and whether more efficient implementations can retain the same entitlement behavior.

## Drafting Notes for Later Refinement

- Add related-work coverage and citations for fact verification, selective prediction, abstention, rationale faithfulness, counterfactual evaluation, and state-space sequence models.
- Specify all training hyperparameters, split sizes, compute details, and model parameter counts in a reproducibility appendix.
- Convert the Mermaid architecture into publication-quality vector artwork.
- Add statistical testing or confidence intervals appropriate to the final experimental design.
- Add external baselines only after running matched evaluations; do not infer comparative claims from the current controlled results.
