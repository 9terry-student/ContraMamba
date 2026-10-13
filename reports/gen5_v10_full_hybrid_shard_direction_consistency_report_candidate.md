# Gen5 v10 Full-Hybrid Shard Direction Consistency

Status: STATIC_SHARD_REPLAY_PASS / DESCRIPTIVE_ONLY / NO_FORMAL_SCIENCE_PROMOTION

## Source identity

- Repository: 9terry-student/ContraMamba
- Analysis HEAD: b8c7e5b512625ce59225eef4683ad0d9ac10da1d
- Execution HEAD: 9135810af6c607cc61a7c6b4197e4cebf2eaa059
- Execution: gen5-v10-full-hybrid-9135810-r1
- Worker 0: 416 rows, 29,765 valid tokens
- Worker 1: 424 rows, 30,329 valid tokens
- Total: 840 rows, 60,094 valid tokens
- Method: CPU-only static analysis of SHA256-verified original worker artifacts
- No new model, tokenizer, CUDA, training, or evaluation execution

## Definition

For each source and worker, compute orientation-specific:

L = log[(E_complement / S_complement) / (E_visible / S_visible)]

Then:

DeltaL = mean(L for PRIMARY_A) - mean(L for CONTROL_R)

Each source has exactly two PRIMARY_A and two CONTROL_R orientations.

For the pooled replay, sum E and S across workers before calculating L. Do not substitute a simple average of worker-level DeltaL values.

## Observed source-wise results

| Source | Worker 0 | Worker 1 | Pooled |
|---|---:|---:|---:|
| A6201-R6201 | -1.345239 | -1.354666 | -1.349906 |
| A6201-R6202 | -1.192483 | -1.196814 | -1.194651 |
| A6201-R6203 | -1.222125 | -1.229468 | -1.225752 |
| A6202-R6201 | -0.801634 | -0.812574 | -0.807187 |
| A6202-R6202 | -0.784772 | -0.791424 | -0.788124 |
| A6202-R6203 | -0.784385 | -0.794282 | -0.789404 |
| A6203-R6201 | -1.007191 | -1.016172 | -1.011708 |
| A6203-R6202 | -1.033251 | -1.041582 | -1.037444 |
| A6203-R6203 | -1.067226 | -1.073955 | -1.070608 |

## Validation results

- Worker 0 negative sources: 9/9
- Worker 1 negative sources: 9/9
- Cross-shard sign agreement: 9/9
- Sign disagreements: 0
- Worker 0 mean DeltaL: approximately -1.026478
- Worker 1 mean DeltaL: approximately -1.034549
- Worker 1 minus Worker 0: negative for all nine sources
- Mean worker difference: approximately -0.008070
- Pooled values reproduce the previous report to 1e-5
- POOLED_REPORT_REPLAY=PASS

## Scientific interpretation and limits

The negative PRIMARY_A-minus-CONTROL_R interference direction is present in both nonoverlapping dev shards for all nine source cells.

This rules out a pooled sign reversal caused by one shard having an opposite source-level direction. It does not prove seed-level replication, independence, statistical significance, causal ownership, or generalization to another dataset.

Worker 0 and Worker 1 are partitions of one frozen dev population, not independent experimental replications.

The full-hybrid B32 path used GEMM for all 96 chunks; this analysis is not a full-population frozen FFT versus v10 paired equivalence test.

The separate bounded FFT-versus-GEMM evidence remains:
LEGACY_FULL_RECORD=FAIL
V10_E2_BOUNDED_NUMERIC=PASS_PROVISIONAL
V10_E3_BOUNDED_RAW=DIVERGENCE_UNRESOLVED
FORMAL_E0_E3=NOT_PROMOTED

No numerical tolerance was changed and no formal scientific claim is promoted.

## Disposition

SHARD_DIRECTION_CONSISTENCY=PASS_DESCRIPTIVE
SOURCE_LEVEL_SIGN_AGREEMENT=9_OF_9
INDEPENDENT_REPLICATION=NOT_ESTABLISHED
CAUSAL_CLAIM=NOT_ESTABLISHED
ADDITIONAL_GPU_EXECUTION=NO
