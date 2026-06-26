# ContraMamba: Evidence-Entitlement Modeling for Claim-Evidence Verification

ContraMamba studies whether a claim-evidence verifier is not only correct at the final-label level, but also internally entitled to make that decision from the supplied evidence.

The central distinction is:

```text
A model can predict the correct final label
without having an internally faithful evidence-entitlement path for that label.
```

ContraMamba therefore separates final judgment into intermediate epistemic signals: frame compatibility, predicate coverage, evidence sufficiency, entitlement, and polarity. The current system predicts:

```text
REFUTE / NOT_ENTITLED / SUPPORT
```

while exposing internal channel diagnostics for controlled intervention analysis.

This is not a generic uncertainty-estimation project. Confidence, entropy, or softmax calibration are not treated as sufficient evidence of epistemic entitlement. ContraMamba directly evaluates whether the model's internal decision path respects the evidence structure.

---

## Current status

The current mainline is **Stage26-H1**, a repaired v7 hierarchical model using a real Mamba backbone.

Stage26-H1 preserves the v7 hierarchical channel architecture but replaces the unstable raw-additive final decision with a v6B-style softplus/multiplicative final-decision geometry.

The key finding so far is:

```text
The v7 hierarchy itself was not the main failure.
The failure came from the final-decision geometry.
Treating entitlement as an additive logit feature caused SUPPORT collapse.
Restoring entitlement as a gate over polarity energies recovered 3-way judgment behavior.
```

---

## Architecture

The current v7-H1 pipeline is:

```text
claim + evidence
      |
Mamba encoder
      |
Frame channel
      |
Predicate channel
      |
Sufficiency / temporal channel
      |
Entitlement gate
      |
Polarity channel
      |
H1 final decision geometry
      |
REFUTE / NOT_ENTITLED / SUPPORT
```

The H1 final decision uses nonnegative polarity energies and entitlement-gated final logits:

```text
positive_energy = softplus(support_polarity_logit)
negative_energy = softplus(refute_polarity_logit)

support_score = entitlement_for_decision * positive_energy
refute_score  = entitlement_for_decision * negative_energy
ne_score      = ne_bias + alpha * (1 - entitlement_for_decision)

final_logits = [refute_score, ne_score, support_score]
```

This preserves the epistemic role of entitlement: SUPPORT and REFUTE should only be available when the evidence entitles a polarity decision.

---

## Model components

* **Mamba encoder:** produces contextual sequence states for the claim-evidence pair.
* **Frame channel:** estimates whether the evidence and claim refer to a compatible event/entity/frame.
* **Predicate channel:** estimates whether the evidence covers the claim predicate.
* **Sufficiency channel:** estimates whether the available evidence is enough for an entitled decision.
* **Entitlement gate:** estimates whether SUPPORT/REFUTE judgment should be permitted.
* **Polarity channel:** estimates support-vs-refute polarity evidence.
* **H1 final decision head:** converts entitlement and polarity into REFUTE / NOT_ENTITLED / SUPPORT.

Earlier versions explored classifier-auditor routing, prototype memory, geometric classifiers, and auxiliary diagnostic heads. The current mainline focuses on the v7 hierarchical architecture plus H1 final-decision repair.

---

## Controlled intervention evaluation

The current controlled clean dataset is:

```text
data/controlled_v5_v3_without_time_swap.jsonl
```

It removes the earlier corrupted `time_swap` family and contains:

```text
12 intervention types
300 pair groups
3,600 examples
```

The label distribution is:

```text
NOT_ENTITLED: 2700
REFUTE:        450
SUPPORT:       450
```

The controlled intervention families include:

* `none`
* `paraphrase`
* `entity_swap`
* `event_swap`
* `location_swap`
* `role_swap`
* `title_name_swap`
* `predicate_swap`
* `evidence_deletion`
* `evidence_truncation`
* `irrelevant_evidence`
* `polarity_flip`

Splits are performed by `pair_id`, preventing an original pair and its interventions from crossing train/development partitions.

---

## Stage26-H1 results

Stage26-H1 was evaluated with a real Mamba backbone on the clean no-time-swap controlled development set.

```text
Stage26-H1 real Mamba, clean controlled dev, 3 seeds

Macro-F1: 0.900 ± 0.062
Accuracy: 0.917 ± 0.067
```

Seed-level results:

| Seed | Best epoch | Accuracy | Macro-F1 | NOT_ENTITLED recall | REFUTE recall | SUPPORT recall | SUPPORT precision |
| ---: | ---------: | -------: | -------: | ------------------: | ------------: | -------------: | ----------------: |
|    1 |         50 |    0.840 |    0.829 |               0.787 |         1.000 |          1.000 |             0.441 |
|    2 |         41 |    0.960 |    0.938 |               0.967 |         1.000 |          0.879 |             0.860 |
|    3 |         47 |    0.951 |    0.934 |               0.939 |         1.000 |          0.978 |             0.727 |

The main improvement is not only higher score, but removal of the prior v7 SUPPORT-collapse failure mode. Earlier v7 variants frequently collapsed to zero SUPPORT predictions. H1 restored active REFUTE / NOT_ENTITLED / SUPPORT behavior.

---

## Ablation summary

Auxiliary losses were tested on top of H1.

Seed-2 comparison:

| Configuration                | Accuracy | Macro-F1 | NOT_ENTITLED recall | SUPPORT recall | SUPPORT precision |
| ---------------------------- | -------: | -------: | ------------------: | -------------: | ----------------: |
| H1 baseline                  |    0.960 |    0.938 |               0.967 |          0.879 |             0.860 |
| H1 + entitlement BCE 0.5     |    0.944 |    0.925 |               0.926 |          1.000 |             0.711 |
| H1 + entitlement BCE 0.3     |    0.929 |    0.909 |               0.906 |          1.000 |             0.650 |
| H1 + BCE + class-balanced CE |    0.878 |    0.861 |               0.837 |          1.000 |             0.511 |
| H1 + constrained selection   |    0.878 |    0.858 |               0.839 |          0.989 |             0.511 |
| H1 + class-balanced CE 0.3   |    0.825 |    0.816 |               0.767 |          1.000 |             0.425 |

These results suggest that the repair is not primarily a loss-trick effect. The main improvement comes from the final-decision geometry: entitlement must function as a gate rather than as an additive feature.

---

## Intervention-level diagnosis

The H1 model handles several intervention families strongly:

```text
evidence_deletion:     SUPPORT 0 / 180
evidence_truncation:   SUPPORT 0 / 180
irrelevant_evidence:   SUPPORT 0 / 180
predicate_swap:        SUPPORT 10 / 180
entity_swap:           SUPPORT 8 / 180
event_swap:            SUPPORT 11 / 180
title_name_swap:       SUPPORT 4 / 180
```

The main residual false-entitlement failures concentrate in controlled location/role swaps:

```text
location_swap: SUPPORT 81 / 180
role_swap:     SUPPORT 46 / 180
```

This does not yet establish a broad natural-language location/role reasoning weakness. It shows that, in the current controlled intervention family, false SUPPORT decisions are concentrated in semantic near-mismatch cases.

A key diagnostic pattern is that the frame channel often detects low frame compatibility, but the learned entitlement scalar remains too weak or too constant to fully block SUPPORT. This motivates the next H2 direction: explicit compositional entitlement gates.

---

## H2 direction

Post-hoc counterfactual analysis suggests that explicit channel-compositional gating is promising.

Candidate H2 entitlement decision signals:

```text
learned:
    entitlement_for_decision = learned_entitlement_prob

product:
    entitlement_for_decision = frame_prob * predicate_coverage_prob * sufficiency_prob

min:
    entitlement_for_decision = min(frame_prob, predicate_coverage_prob, sufficiency_prob)

frame_predicate_min:
    entitlement_for_decision = min(frame_prob, predicate_coverage_prob)

frame_predicate_product:
    entitlement_for_decision = frame_prob * predicate_coverage_prob
```

A post-hoc `min_gate` counterfactual at threshold 0.30 achieved:

```text
Macro-F1:           0.955 ± 0.006
Accuracy:           0.970
SUPPORT precision:  0.870
SUPPORT recall:     0.916
NOT_ENTITLED recall: 0.975
REFUTE recall:      1.000
```

This is not yet a trained H2 model. It is a diagnostic result showing that explicit channel-compositional gating may correct residual false entitlement more effectively than the current learned entitlement scalar.

---

## Historical Stage7 result

Earlier Stage7 experiments used a classifier-auditor router over the v5 pipeline.

The main Stage7 system was ContraMamba-CAR at threshold 0.5, using `v3_no_intervention` as the classifier and `v3_no_polarity_flip` as the balanced entitlement auditor.

|      Accuracy |      Macro-F1 | NOT_ENTITLED F1 |     REFUTE F1 |    SUPPORT F1 | Gate violation rate | Output/internal gap |
| ------------: | ------------: | --------------: | ------------: | ------------: | ------------------: | ------------------: |
| 0.929 ± 0.003 | 0.906 ± 0.005 |   0.952 ± 0.002 | 1.000 ± 0.000 | 0.765 ± 0.011 |       0.000 ± 0.000 |       0.000 ± 0.000 |

Stage7 remains an important historical result, but the current mainline has shifted toward the v7 hierarchical architecture and H1 final-decision repair.

---

## Repository structure

| Path               | Purpose                                                                      |
| ------------------ | ---------------------------------------------------------------------------- |
| `src/contramamba/` | ContraMamba models, heads, labels, and losses                                |
| `scripts/`         | Controlled-data builders, training utilities, evaluators, and report writers |
| `data/`            | Controlled intervention datasets                                             |
| `experiments/`     | Stage plans and experiment notes                                             |
| `results/`         | Seed-level and aggregate reports                                             |
| `docs/`            | Architecture and paper-oriented results documentation                        |
| `tests/`           | Unit, validation, training-smoke, and reporting tests                        |

---

## Reproducibility

Basic local validation:

```bash
pip install -e .
pytest
```

The main current training entry point is:

```text
scripts/train_controlled_v6b_minimal.py
```

Current mainline configuration:

```text
architecture: v7_hierarchical
backbone: mamba
data: data/controlled_v5_v3_without_time_swap.jsonl
final decision: H1 v6B-style softplus/multiplicative bridge
```

The project uses controlled clean data for Stage26-H1. OOD and broader benchmark validation remain future work.

---

## Limitations

* Current evidence is from a controlled synthetic/intervention dataset.
* Stage26-H1 has not yet been validated as a real-world RAG component or deployed hallucination detector.
* The clean controlled dataset removes the corrupted `time_swap` family; temporal robustness still requires separate validation.
* The learned entitlement scalar is not yet sufficiently identified as a faithful semantic gate.
* Current results support controlled evidence-entitlement behavior, not broad real-world hallucination elimination.
* Post-hoc H2 gate results are diagnostic only until implemented and trained as part of the model.

The current contribution is therefore a controlled study of evidence-entitlement structure in claim-evidence verification, not a claim of general-purpose hallucination prevention.

---

## Current research claim

The strongest current claim is:

```text
Hierarchical epistemic decomposition alone is insufficient.
The final decision geometry must preserve the role of entitlement as a gate.
When entitlement is treated as an additive logit feature, v7 collapses.
When entitlement gates polarity energies, v7 recovers stable 3-way judgment behavior.
```

ContraMamba's current direction is to make that entitlement gate more faithful by explicitly composing frame compatibility, predicate coverage, and evidence sufficiency.
