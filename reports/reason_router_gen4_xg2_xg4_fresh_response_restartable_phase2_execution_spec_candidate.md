# Gen4-K XG2/XG4 Fresh Response Restartable Phase-2 Execution Freeze

## Status

This file authorizes only the restartable Phase-2 alignment-response
scientific execution derived from implementation commit:

`5bddad126ded92b8c2d612c38cb1b96f68388e13`

Runner:

`scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2.py`

Families authorized:

- XG2
- XG4

The execution consumes the already frozen Phase-1 artifacts. It does not
authorize any Phase-1 baseline re-execution.

## Scientific execution boundary

For each family:

- 300 fixed source pairs, pair IDs `301..600`;
- consume frozen Phase-1 baseline PE and `alignment_delta_h`;
- alignment plus and alignment minus only;
- exactly 600 scientific model forwards in this Phase-2 run;
- exactly 0 baseline model forwards in this Phase-2 run;
- persisted Phase-1 baseline forward count = 1200;
- complete protocol forward count = 1800;
- compute `delta_alignment`;
- compute `R_ALIGN = delta_alignment - delta_baseline`.

Across XG2 and XG4:

- exactly 1200 new Phase-2 scientific forwards total;
- exactly 0 new baseline forwards.

Frozen regime counts remain:

- XG2: LARGE = 92, SMALL = 208
- XG4: LARGE = 57, SMALL = 243

Frozen threshold remains:

`0.11228626366380845`

Any baseline forward, pair/order drift, regime-count drift, threshold
re-estimation, or artifact identity mismatch blocks the run.

## Frozen inputs

Phase-1 artifact freeze:

`f4f5aef2025a66e8b7e7ed6d077523eec46eb6f0`

Frozen Phase-1 artifact root:

`reports/reason_router_gen4_xg2_xg4_fresh_response_restartable_phase1_d217bad_r3`

The existing frozen model/tokenizer revision, runtime snapshot, representative
checkpoint, CUDA/T4 backend identity, kernel identities, tokenizer identity,
and file hashes from the completed Phase-1 execution lineage remain unchanged
and authoritative.

No asset substitution, tokenizer regeneration, mutable latest revision,
threshold re-estimation, or Phase-1 baseline regeneration is allowed.

## Prohibited operations and conclusions

This execution does not authorize:

- H1 testing;
- H2 testing;
- p-value computation;
- replication conclusion;
- ICML-level scientific conclusion;
- magnitude intervention;
- training;
- backward;
- task-head execution;
- logit reads.

Successful execution establishes only Phase-2 response observation artifacts.

## Output isolation

Scientific family outputs must be written outside the Git repository during
the two family executions so XG2 completion cannot dirty the repository before
XG4 authentication.

Run name:

`g4k-xg2-xg4-restartable-phase2-5bddad1-r1`

Scientific output root:

`/kaggle/working/g4k_xg2_xg4_restartable_phase2_5bddad1_r1`

Family directories:

- `/kaggle/working/g4k_xg2_xg4_restartable_phase2_5bddad1_r1/xg2`
- `/kaggle/working/g4k_xg2_xg4_restartable_phase2_5bddad1_r1/xg4`

Expected files per family:

- `response_items.jsonl`
- `phase2_summary.json`
- `artifact_manifest.json`
- `SHA256SUMS.txt`

The output root must not exist before formal execution.

After both scientific family runs have completed successfully, byte-preserving
packaging into a collector-visible `reports/` directory is allowed solely for
handoff/collection. Packaging must not perform any model forward or scientific
recomputation, and source/destination hashes must match.

## Formal execution identity

The formal Kaggle run must bootstrap the exact commit that contains this
execution-freeze document.

The runner argument:

`--expected-head`

must equal that exact full bootstrapped HEAD.

Both XG2 and XG4 must execute from that same clean HEAD.

## Acceptance

Each family must report:

`PASS_XG2_XG4_FRESH_RESPONSE_RESTARTABLE_PHASE2_FREEZE`

with:

- `source_pair_count = 300`;
- `baseline_model_forward_count_this_run = 0`;
- `alignment_model_forward_count_this_run = 600`;
- `current_run_scientific_forward_count = 600`;
- `persisted_phase1_baseline_forward_count = 1200`;
- `complete_protocol_forward_count = 1800`;
- `support_gate_pass = true`;
- exact frozen LARGE/SMALL counts;
- `r_align_observed = true`;
- `h1_test_executed = false`;
- `h2_test_executed = false`;
- `training_executed = false`;
- `backward_executed = false`;
- `task_heads_executed = false`;
- `logits_read = false`;
- manifest/checksum validation PASS.

Successful execution alone is not a replication conclusion.

H1/H2 analysis remains a later separate read-only stage after
execution -> collection -> import -> provenance validation -> artifact freeze.

## Stop conditions

Stop before or during execution on any:

- execution HEAD mismatch;
- dirty repository before either family run;
- frozen Phase-1 artifact mismatch;
- manifest/checksum mismatch;
- runtime/snapshot/checkpoint/kernel identity mismatch;
- pair-order mismatch;
- threshold mismatch;
- regime-count mismatch;
- output collision;
- any Phase-2 baseline model forward;
- forward-budget mismatch;
- non-finite response value;
- post-write artifact validation failure.

No automatic fallback or scientific substitution is authorized.
