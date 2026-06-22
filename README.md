# ContraMamba: Evidence-Entitlement Modeling for Claim-Evidence Verification

ContraMamba studies whether final-label correctness and internal evidence-entitlement faithfulness can diverge in claim-evidence verification. The current Stage 5 system predicts `REFUTE`, `NOT_ENTITLED`, or `SUPPORT` while exposing intermediate signals for frame compatibility, predicate coverage, evidence sufficiency, and polarity.

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
- **Classifier-auditor router:** retains an entitled classifier output only when the auditor's frame, predicate, sufficiency, and entitlement gates pass; otherwise it returns `NOT_ENTITLED`.

The v5 core does not use the v4 geometric classifier, prototype memory, routing memory, FAISS, GAT, novelty, ambiguity, or RAG components.

## Controlled intervention evaluation

Stage 5 uses pair-grouped controlled examples so that an original claim-evidence pair can be compared with targeted interventions. The evaluated interventions include:

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

## Stage 5C results

Results are means and sample standard deviations across three seeds on the controlled v2 dataset.

| System | Macro-F1 | Entitled-output gate violation rate | Output/internal polarity gap |
|---|---:|---:|---:|
| `classifier_only` | 0.870 +/- 0.006 | 0.222 +/- 0.066 | 0.250 +/- 0.150 |
| `conservative_balanced_router` | **0.878 +/- 0.009** | **0.000 +/- 0.000** | **0.000 +/- 0.000** |

The conservative balanced-auditor router improves macro-F1 over the classifier-only baseline while eliminating entitled-output gate violations and closing the measured output/internal polarity-consistency gap.

This result supports the project's central finding: final-label accuracy and evidence-entitlement faithfulness are empirically separable under controlled interventions. Output-level correctness can conceal internal gate or polarity inconsistencies, and a conservative auditor can expose those cases without necessarily reducing aggregate classification quality.

For the full ablation and router analyses, see:

- [Stage 5/6 results narrative](docs/stage6_results_narrative.md)
- [Stage 5B ablation comparison](results/stage5b_v2_ablation_comparison.md)
- [Stage 5C router aggregate](results/stage5c_v2_router_aggregate.md)

## Repository structure

| Path | Purpose |
|---|---|
| `src/contramamba/` | ContraMamba-v5 model, heads, labels, and losses |
| `scripts/` | Controlled-data builders, training utilities, evaluators, and report writers |
| `data/` | Controlled intervention datasets |
| `results/` | Seed-level and aggregate Stage 5 reports |
| `tests/` | Unit, data-validation, training-smoke, and reporting tests |
| `docs/` | Architecture and paper-oriented results documentation |

## Reproducibility

Install the project and run the test suite:

```bash
pip install -e .
pytest
```

Regenerate the Stage 5C aggregate report from three router result CSVs:

```bash
python scripts/write_stage5c_router_aggregate.py \
  --input results/stage5c_v2_router_seed1.csv \
  --input results/stage5c_v2_router_seed2.csv \
  --input results/stage5c_v2_router_seed3.csv \
  --output-md results/stage5c_v2_router_aggregate.md \
  --output-csv results/stage5c_v2_router_aggregate.csv
```

## Limitations

- The current evidence comes from a controlled synthetic/intervention dataset.
- Evaluation is small-scale relative to benchmark or deployment settings.
- The main Stage 5 experiments use a frozen Mamba encoder.
- ContraMamba has not yet been validated as a real-world RAG component or deployed hallucination detector.
- These results demonstrate controlled evidence-entitlement behavior; they do not establish general hallucination elimination or reduction.

The current contribution is therefore a controlled evaluation of whether a prediction is internally supported by the supplied evidence, not a claim of broad real-world reliability.
