# Stage183-A controlled-train integrity-mask-required closure

## Decision

`STAGE183A_CONTROLLED_TRAIN_INTEGRITY_MASK_REQUIRED_FIRST`

The authoritative main dataset is `data/controlled_v5_v3_without_time_swap.jsonl`. Under the frozen seed-174, 80/20 pair split it contains 3,600 rows / 300 pairs, with 2,880 train rows / 240 train pairs and 720 dev rows / 60 dev pairs. Train frame labels are balanced: 1,440 compatible and 1,440 incompatible.

## Objective audit closure

The native frame objective is row-mean `BCEWithLogits(frame_logit, frame_compatible_label)`, with no positive-class weight and implicit native frame-loss weight 1.0. Default checkpoint selection maximizes clean-dev final macro-F1; frame-positive recall and compatible-positive margin are not direct selection metrics.

The balanced 1,440/1,440 topology supplies no class-imbalance rationale for positive class reweighting. The contingent best scientific fit is the compatible-positive absolute-margin hinge because it targets the native frame head directly. It is distinct from Stage175's final SUPPORT reference anchor and Stage177's relative compatible-over-incompatible ordering objective.

## Integrity blocker

The main JSONL does not carry authoritative row-level `grammar_valid`, `intervention_contract_exact`, `polarity_contamination_absent`, `schema_resolved`, or `canonical_row_valid` fields. Stage182-A found 21 non-polarity polarity changes and one `did not` plus already-inflected-predicate defect in its reviewed subset. Exact reproduction of generator output establishes provenance, not semantic cleanliness.

Consequently no complete contamination-safe positive-loss mask exists. No target margin or nonzero loss weight is selected. Hinge implementation, smoke, training, tuning, and dataset mutation remain unauthorized.

Authorized next stage:

`STAGE184_CONTROLLED_TRAIN_INTEGRITY_MASK_SPEC`
