# Stage185-A integrity sidecar materialized closure

## Decision

`STAGE185A_INTEGRITY_SIDECAR_BUILT_AND_POSITIVE_ELIGIBILITY_MATERIALIZED`

The authoritative input is `data/controlled_v5_v3_without_time_swap.jsonl`, SHA-256 `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`, with 3,600 rows and 300 pairs. The authoritative successful Stage185-A folder is `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914`; the failed `..._141721` folder is not an input or fallback. The sidecar semantic SHA-256 is `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`.

The row-ID one-to-one join passed, blocked invariants were zero, and the Stage182 overlap regression passed with all 22 deterministic contaminations recovered. Among 1,440 train-compatible rows, exactly 605 are eligible (`0.4201388888888889`). They cover 121 pairs and five families. Every eligible pair contributes five rows and every eligible family contributes 121 rows, so the largest family share is 0.2 and no family-concentration warning applies.

Coverage remains pair-concentrated: only 121 of 240 train pairs contribute. This cohort is an integrity-filtered usable subset, not evidence that the complete controlled dataset is clean or representative.

Loss implementation, target execution, checkpoint-selection changes, and training remain unauthorized. The only authorized next step is:

`STAGE186_COMPATIBLE_POSITIVE_MARGIN_FIXED_SPEC_AUDIT`
