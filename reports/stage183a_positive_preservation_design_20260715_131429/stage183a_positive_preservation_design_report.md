# Stage183-A positive-preservation design report

## Decision

`STAGE183A_CONTROLLED_TRAIN_INTEGRITY_MASK_REQUIRED_FIRST`

Authorized next route: `STAGE184_CONTROLLED_TRAIN_INTEGRITY_MASK_SPEC`.

The compatible-positive absolute-margin hinge is the contingent best scientific fit, but implementation is not authorized because the current main JSONL cannot construct a complete contamination-safe clean-compatible mask. No nonzero target margin or weight is selected.

## Current objective

The native frame loss is row-mean `BCEWithLogits(frame_logit, frame_compatible_label)` with no `pos_weight`; it enters the five-term native objective with implicit weight 1.0. `frame_prob` is `sigmoid(frame_logit)`, and the linear frame classifier has a trainable bias. Default checkpoint selection maximizes clean-dev final macro-F1, not frame-positive recall or margin.

## Label topology

- Full rows/pairs: 3600 / 300
- Train rows/pairs: 2880 / 240
- Train compatible/incompatible: 1440 / 1440
- Dev rows/pairs: 720 / 60

The frozen pair topology is balanced, so positive-class reweighting has no imbalance-based justification.

## Integrity gate

Complete mask constructible: `false`.
Missing authoritative integrity fields: `grammar_valid, intervention_contract_exact, polarity_contamination_absent, schema_resolved, canonical_row_valid`.

Stage182-A showed that generator equality is not cleanliness and found 22 deterministic contaminated review items. Missing grammar, contract, polarity-contamination, or canonical-valid metadata is fail-closed; it may not be inferred from family names or text heuristics.

## Candidate selection

Candidate B adds an absolute compatible-positive logit floor that native BCE and Stage177 pairwise ordering do not guarantee, and it targets a different head from Stage175's final SUPPORT anchor. It requires no teacher or counterpart and adds no direct negative-row loss. It remains contingent until the integrity mask exists and fixed no-sweep hyperparameters are justified.

Candidates A, C, D, E, and F are not selected: respectively they lack an imbalance rationale, reopen reference preservation, lack bias evidence, introduce a selection/leakage trade-off, or overfit noncausal family composition.

## Safety

Static audit only. No model/Torch execution, checkpoint load, forward, training, smoke, dataset mutation, relabeling, threshold fitting, calibration, external evaluation, `time_swap`, multi-seed run, or hyperparameter sweep is authorized.
