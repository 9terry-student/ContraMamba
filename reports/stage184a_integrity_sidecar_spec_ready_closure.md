# Stage184-A integrity-sidecar specification closure

## Decision

`STAGE184A_DETERMINISTIC_FAIL_CLOSED_INTEGRITY_SIDECAR_SPEC_READY`

The authoritative dataset is `data/controlled_v5_v3_without_time_swap.jsonl` with SHA-256 `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`, 3,600 rows, 300 pairs, and 12 intervention families. The frozen seed-174 split contains 2,880/720 train/dev rows and 240/60 train/dev pairs; train frame labels are balanced at 1,440 compatible and 1,440 incompatible. `time_swap` count is zero.

Exact row identity and a one-to-one external sidecar join are available. Every required integrity criterion has a deterministic `PASS`, `FAIL`, or `UNRESOLVED` route, and missing evidence is fail-closed. A complete sidecar means complete decision coverage, not that every row is clean. General grammar may remain unresolved. Generator equality establishes provenance only.

The exact train-compatible positive eligibility count remains unknown until Stage185 materializes the sidecar. Stage185 is authorized to build and statically audit the sidecar only. Positive-margin loss implementation, target-margin selection, nonzero weight, checkpoint-selection changes, and training remain unauthorized.

Authorized next stage:

`STAGE185_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILDER`
