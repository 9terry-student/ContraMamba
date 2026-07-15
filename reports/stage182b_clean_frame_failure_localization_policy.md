# Stage182-B clean native-frame failure-localization policy

## Scientific target

Stage182-B localizes failure of the native frame head on the 14 deterministic-clean Stage182-A candidates. It does not analyze final-classifier errors and does not infer a causal mechanism. A candidate must satisfy all of the following:

```text
item_role == hard
native_frame_prediction != native_frame_label
final_diagnostic_class == CLEAN_MODEL_FAILURE_CANDIDATE
grammar_valid == true
contract_exact_match == true
canonical_control_valid == true
schema_resolved == true
```

The fixed comparison cohorts are the 14 candidates, their 14 matched clean controls, seven `CLEAN_HARD_NATIVE_FRAME_CORRECT` rows, and all 30 clean controls. Contaminated rows are provenance only and must not enter a localization comparison.

## Artifact boundary and normalization

Only frozen CSV/JSON artifacts may be read. Native scalars are selected in this order: Stage180-A unblinded Pass-2 packet, Stage179-A hard-39 attribution, then Stage176-A row transitions. `row_id` is the canonical identity; Stage180 review IDs must resolve through the hidden key. Aliases are normalized only at the input boundary. Multiple populated aliases for one canonical field must agree exactly (within `1e-9` for numeric values), or the run is blocked.

The canonical fields are `frame_label`, `frame_prediction`, `frame_logit`, `frame_probability`, `frame_head_projection`, `representation_movement_from_none`, `centroid_prediction`, and `centroid_correct`. Final-classifier logits and predictions are never localization evidence.

## Allowed analysis

Allowed operations are deterministic joins, fixed descriptive statistics, paired differences, two-sided exact sign tests, fixed-seed bootstrap confidence intervals, descriptive Fisher exact tests, Benjamini-Hochberg correction for family-specific tests, and reuse of existing Stage179 centroid/projection diagnostics.

The Stage179 centroid is a gold-conditioned, leave-one-row-out, transductive diagnostic. It is not a deployable classifier or a fitted Stage182 probe. Representation movement is a scalar magnitude; without stored vectors it cannot establish direction, cosine similarity, a new centroid, or nearest neighbors.

Projection/bias claims require artifact-level verification of `frame_logit = frame_head_projection + frame_classifier_bias`: inferred bias must be effectively constant and maximum reconstruction error must be at most `1e-5`. If that relationship is not verified, bias-specific gates are unavailable rather than guessed.

## Fixed gates

- Representation mislocalization: candidate centroid-wrong rate at least 0.65, matched-control centroid-correct rate at least 0.75, paired-logit bootstrap upper bound below zero, and candidate centroid coverage at least 0.80.
- Readout/projection misalignment: candidate centroid-correct rate at least 0.65, matched-control centroid-correct rate at least 0.75, paired-logit bootstrap upper bound below zero, and either projection-negative rate at least 0.70 or the verified bias-dominant gate.
- Compatible-positive margin collapse: compatible false negatives comprise at least 0.75 of candidates; their count meets `--minimum-candidates`; median logit is negative; and the paired-logit bootstrap upper bound is below zero.
- Polarity-conditioned association: `polarity_flip` support is at least five, the absolute polarity-versus-`none` gap difference is at least half the pooled scale, the fixed-seed bootstrap interval excludes zero, and pooled scale is nonzero.

The decision priority is representation, readout, compatible-positive margin, polarity-conditioned association, mixed, then insufficient artifacts. A narrow polarity association cannot replace a passing general representation gate. Compatible-margin evidence remains eligible when centroid/projection artifacts are unavailable, provided scalar coverage is sufficient.

## Safety and claims

Scalar margin, centroid/representation, linear-readout, and family-conditioned results are localization evidence only. None is a causal mechanism finding. No annotation, model import, `torch` import, checkpoint load, forward pass, embedding recomputation, learned probe, classifier fitting, calibration, threshold optimization, relabeling, dataset mutation, generator modification, or training is authorized.
