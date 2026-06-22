# ContraMamba: Evidence-Entitlement Modeling for Claim-Evidence Verification

ContraMamba studies whether final-label correctness and internal evidence-entitlement faithfulness can diverge in claim-evidence verification. The current Stage 7 system predicts `REFUTE`, `NOT_ENTITLED`, or `SUPPORT` while exposing intermediate signals for frame compatibility, predicate coverage, evidence sufficiency, and polarity.

This is not a generic uncertainty-estimation project. A model can be confident or produce the correct label without being internally entitled to that decision by the supplied evidence. ContraMamba evaluates that distinction directly through controlled interventions and classifier-auditor routing.

## Architecture

The v5 pipeline preserves token-level encoder states until the frame and predicate stages:

```text
claim + evidence
      |
Mamba encoder
      |
FrameGate
      |
PredicateCoverageHead
      |
SufficiencyGate
      |
PolarityEnergyHead
      |
FinalEntitlementDecisionHead
```

- **Mamba encoder:** produces contextual token states for the claim-evidence pair.
- **FrameGate:** estimates category-free pairwise frame compatibility.
- **PredicateCoverageHead:** measures whether the evidence covers the claim predicate.
- **SufficiencyGate:** estimates whether the available evidence is sufficient for an entitled decision.
- **PolarityEnergyHead:** represents support and refute polarity evidence.
- **FinalEntitlementDecisionHead:** combines entitlement and polarity into the three final labels.
- **ContraMamba-CAR:** the ContraMamba Classifier-Auditor Router retains an entitled classifier output only when the auditor's frame, predicate, sufficiency, and entitlement gates pass; otherwise it returns `NOT_ENTITLED`.

The v5 core does not use the v4 geometric classifier, prototype memory, routing memory, FAISS, GAT, novelty, ambiguity, or RAG components.

## Controlled intervention evaluation

Stage 7 uses `controlled_v5_v3`, containing 300 pair groups and 3,900 examples. Each original claim-evidence pair is compared with targeted interventions including:

- `paraphrase`
- `entity_swap`
- `event_swap`
- `time_swap`
- `location_swap`
- `role_swap`
- `title_name_swap`
- `predicate_swap`
- `evidence_deletion`
- `evidence_truncation`
- `polarity_flip`

Splits are performed by `pair_id`, preventing an original pair and its interventions from crossing train and development partitions. Metrics cover final-label classification, output-level pairwise consistency, and internal gate faithfulness.

## Stage 7 v3 results

The main system is **ContraMamba-CAR** at threshold 0.5. It uses `v3_no_intervention` as the classifier and `v3_no_polarity_flip` as the balanced entitlement auditor. The optional strict model is not required for the main configuration.

| Accuracy | Macro-F1 | NOT_ENTITLED F1 | REFUTE F1 | SUPPORT F1 | Gate violation rate | Output/internal gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.929 +/- 0.003 | **0.906 +/- 0.005** | 0.952 +/- 0.002 | 1.000 +/- 0.000 | 0.765 +/- 0.011 | **0.000 +/- 0.000** | **0.000 +/- 0.000** |

CAR macro-F1 remains 0.905-0.906 across thresholds 0.3-0.7, while gate violations and the output/internal polarity gap remain zero. Threshold 0.5 is therefore not an isolated selected optimum.

The `self_routed_classifier` ablation reaches a higher macro-F1 of 0.912 +/- 0.018 at threshold 0.4, but has weaker polarity-flip, paraphrase, and predicate consistency. The faithful `self_routed_balanced` ablation reaches 0.888 +/- 0.008 macro-F1. These are informative single-model ablations, not replacements for the main multi-layer classifier-auditor architecture.

The current result supports the central finding that final-label prediction and evidence-entitlement auditing are separable functions under controlled interventions. CAR combines a strong classifier with a balanced auditor to preserve high final-label performance while enforcing the measured entitlement criteria.

For the full ablation and router analyses, see:

- [Stage 7 v3 results narrative](docs/stage7_v3_results_narrative.md)
- [Stage 5/6 results narrative](docs/stage6_results_narrative.md)
- [Stage 5B ablation comparison](results/stage5b_v2_ablation_comparison.md)
- [Stage 7 v3 router aggregate](results/stage7c_v3_router_aggregate.md)
- [Stage 7 v3 self-routing aggregate](results/stage7c_v3_self_routing_aggregate.md)

## Repository structure

| Path | Purpose |
|---|---|
| `src/contramamba/` | ContraMamba-v5 model, heads, labels, and losses |
| `scripts/` | Controlled-data builders, training utilities, evaluators, and report writers |
| `data/` | Controlled intervention datasets |
| `results/` | Seed-level and aggregate Stage 5-7 reports |
| `tests/` | Unit, data-validation, training-smoke, and reporting tests |
| `docs/` | Architecture and paper-oriented results documentation |

## Reproducibility

Install the project and run the test suite:

```bash
pip install -e .
pytest
```

Regenerate the Stage 7 v3 router aggregate report from three router result CSVs:

```bash
python scripts/write_stage6a_threshold_sweep_aggregate.py \
  --input results/stage7c_v3_router_seed1.csv \
  --input results/stage7c_v3_router_seed2.csv \
  --input results/stage7c_v3_router_seed3.csv \
  --output-md results/stage7c_v3_router_aggregate.md \
  --output-csv results/stage7c_v3_router_aggregate.csv
```

## Limitations

- The current evidence comes from a controlled synthetic/intervention dataset.
- Evaluation is small-scale relative to benchmark or deployment settings.
- The main controlled experiments use a frozen Mamba encoder.
- ContraMamba has not yet been validated as a real-world RAG component or deployed hallucination detector.
- These results demonstrate controlled evidence-entitlement behavior; they do not establish general hallucination elimination or reduction.

The current contribution is therefore a controlled evaluation of whether a prediction is internally supported by the supplied evidence, not a claim of broad real-world reliability.
