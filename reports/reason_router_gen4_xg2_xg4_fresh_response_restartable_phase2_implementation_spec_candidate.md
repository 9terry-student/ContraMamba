# XG2/XG4 Fresh Response Restartable Phase-2 Implementation Scope

## Parent state

Restartable Phase-1 artifact freeze:

`f4f5aef2025a66e8b7e7ed6d077523eec46eb6f0`

Protocol correction:

`bc8b0e2ca128eb81271761c9b3f5ed7b0a940455`

Prospective response design:

`2074f52d39bf0fca6d63248016ce47c54bff5e06`

Phase-1 artifacts:

`reports/reason_router_gen4_xg2_xg4_fresh_response_restartable_phase1_d217bad_r3/xg2/`

`reports/reason_router_gen4_xg2_xg4_fresh_response_restartable_phase1_d217bad_r3/xg4/`

Frozen Phase-1 results:

- XG2: 92 LARGE / 208 SMALL
- XG4: 57 LARGE / 243 SMALL
- support gate PASS for both
- `phase2_restartable = true` for both

## Goal

Implement the minimum restartable Phase-2 runner that consumes the frozen
Phase-1 artifacts and executes only the two alignment branches for every
fixed fresh-holdout pair.

No Phase-1 baseline branch may be re-executed.

## Exact implementation scope

Create exactly:

1. `scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2.py`
2. `tests/test_reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2.py`

No existing scientific source file is modified.

## Required frozen inputs

For the selected family, before model construction or scientific forward:

- validate the complete Phase-1 artifact with
  `validate_restartable_artifact`;
- require exactly 300 Phase-1 item rows;
- require one-to-one pair order 301 through 600;
- require unchanged regime counts;
- require support gate PASS;
- require `phase2_restartable = true`;
- require exact baseline PE identity;
- require exact `delta_baseline` identity;
- require exactly 300 losslessly loaded alignment plans;
- require plan dtype/shape identity;
- require manifest/SHA256 validation;
- require the Phase-1 artifact freeze commit to be an ancestor;
- require the frozen Phase-1 artifact tree to be unchanged since
  `f4f5aef2025a66e8b7e7ed6d077523eec46eb6f0`.

Artifact validation/loading performs zero model forwards.

## Frozen scientific mapping

Reuse the existing frozen response semantics unchanged.

Target alignment branches:

- target plus: `C2_NAME`
- target minus: `C0_SHAM`

Intervention:

- layer-17 x branch;
- post-`in_proj`, pre-depthwise-conv;
- own-branch `A_IDENTITY + 2`;
- fixed strong-channel set;
- plus branch applies `+0.5 * alignment_delta_h`;
- minus branch applies `-0.5 * alignment_delta_h`;
- gate half untouched;
- non-strong x channels untouched;
- other tokens untouched.

The persisted `alignment_delta_h` is the intervention plan.
It must not be reconstructed from new baseline forwards.

## Forward budget

Per pair Phase-2 executes exactly:

- one target-plus alignment forward;
- one target-minus alignment forward.

Therefore:

`300 pairs x 2 = 600 model forwards per family`.

Required counters:

- baseline forwards in the Phase-2 run: 0;
- alignment forwards in the Phase-2 run: 600;
- current-run scientific forwards: 600;
- persisted Phase-1 baseline forwards: 1200;
- complete protocol forwards: 1800.

Any baseline model forward in Phase-2 is a hard failure.

## Per-pair response

For each pair preserve the frozen Phase-1 fields needed for provenance,
including:

- family and source-pair identity;
- regime;
- `alignment_shift_abs`;
- threshold;
- frozen baseline plus path efficiency;
- frozen baseline minus path efficiency;
- frozen `delta_baseline`;
- plan index/dtype/shape;
- frozen alignment-plan geometry audit scalars.

Execute the two alignment branches and compute:

`alignment_plus_path_efficiency`

`alignment_minus_path_efficiency`

`delta_alignment =
 alignment_plus_path_efficiency -
 alignment_minus_path_efficiency`

`R_ALIGN =
 delta_alignment -
 delta_baseline`

All response values must be finite.

Use the already-frozen paired intervention audit for the two executed
alignment branches.

## Runtime reuse

Reuse the existing authenticated fast-CUDA path and exact frozen kernel
transport.

Do not change:

- representative checkpoint;
- tokenizer/model snapshot identity;
- backend;
- kernel scientific identity;
- kernel binary SHA256;
- threshold;
- pair population;
- regime assignment;
- intervention layer/site;
- target offset;
- endpoint definition.

## Output contract

Write a new Phase-2 artifact directory containing at minimum:

- per-pair response JSONL;
- Phase-2 summary JSON;
- artifact manifest;
- SHA256SUMS.

The output must be sufficient for later read-only statistical analysis
without any model execution.

The summary must state explicitly:

- Phase-1 artifact freeze commit;
- Phase-1 artifact path/identity;
- 300 source pairs;
- frozen LARGE/SMALL counts;
- baseline forwards this run = 0;
- alignment forwards this run = 600;
- complete protocol forward count = 1800;
- response fields observed = true;
- `R_ALIGN` observed = true;
- training executed = false;
- backward executed = false;
- task heads executed = false;
- logits read = false.

## Statistical boundary

This implementation does not execute H1 or H2 and does not assign a
replication conclusion.

Confirmatory H1/H2 analysis is a separate read-only stage after Phase-2
artifacts have been collected, imported, provenance-validated, and frozen.

No additional model forward is required for that later analysis.

## Tests

Tests must establish at minimum:

- invalid or incomplete Phase-1 artifacts fail closed;
- Phase-1 validation occurs before scientific execution;
- pair order and frozen regime counts are enforced;
- plan index/dtype/shape identity is enforced;
- exactly two alignment branch calls occur per pair;
- no baseline branch call is permitted;
- exact 600-forward family budget is enforced;
- persisted `delta_baseline` is reused rather than recomputed by model execution;
- `delta_alignment` identity is exact;
- `R_ALIGN = delta_alignment - delta_baseline` identity is exact;
- intervention audit is required;
- response output manifest/checksums validate;
- output collision fails closed;
- no H1/H2, training, backward, task-head, or logit path executes.

## Execution boundary

This scope authorizes implementation and local/static validation only.

It does not authorize Kaggle Phase-2 scientific execution.

No training or evaluation is executed while implementing this scope.
