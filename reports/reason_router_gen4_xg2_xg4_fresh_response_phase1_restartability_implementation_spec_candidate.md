# Gen4-K XG2/XG4 Fresh Response Restartable Phase-1 Implementation Scope

## Status

Implementation scope for the restartability correction frozen at:

`bc8b0e2ca128eb81271761c9b3f5ed7b0a940455`

The scientific and persistence contract is defined by:

`reports/reason_router_gen4_xg2_xg4_fresh_response_phase1_restartability_correction_spec_candidate.md`

This scope authorizes implementation and static/unit validation only.

It does not authorize a scientific Phase-1 run, Phase-2 run, training,
evaluation, alignment-response execution, H1/H2 inference, or Kaggle GPU use.

## Exact implementation whitelist

The implementation may add exactly these two files:

1. `scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py`
2. `tests/test_reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py`

No existing production script, test, report, frozen artifact, dataset,
checkpoint, or manifest may be modified.

In particular, do not modify:

- `scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_full_baseline.py`
- `tests/test_reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_full_baseline.py`
- artifacts frozen at `40d2103`
- the restartability correction frozen at `bc8b0e2`

## Required implementation behavior

The new runner must reuse the frozen XG2/XG4 fresh-response population,
tokenizer anchors, representative checkpoint, CUDA-equivalent backend,
kernel bindings, threshold, and regime semantics.

For each family it must preserve:

- exactly 300 pairs;
- exactly four baseline forwards per pair;
- exactly 1200 scientific forwards total;
- no alignment intervention;
- no alignment response;
- no `R_ALIGN`;
- no task-head read;
- no training;
- no backward pass.

For every pair the runner must compute and persist:

- all baseline geometry/regime fields needed by the existing Phase-1 artifact;
- `baseline_plus_path_efficiency`;
- `baseline_minus_path_efficiency`;
- `delta_baseline`;
- the lossless `alignment_delta_h` plan;
- tensor dtype;
- tensor shape;
- explicit source-pair / plan index mapping;
- target plus/minus cells and anchors;
- frozen alignment target/realized geometry audit scalars.

The exact identity must hold:

`delta_baseline =
 baseline_plus_path_efficiency -
 baseline_minus_path_efficiency`

## Alignment-plan persistence

`alignment_delta_h` must be generated from the baseline geometry only.

Generating the plan must not execute an alignment intervention or consume an
additional model forward.

The tensor values must be stored losslessly in a binary tensor artifact:

`alignment_delta_h.pt`

The per-item JSONL must contain an explicit `alignment_plan_index` mapping each
source pair to the first dimension of the persisted tensor, together with the
per-pair tensor dtype and shape.

The binary tensor artifact must be included in the SHA256 manifest.

## Frozen regime reproduction

The corrected Phase-1 implementation must fail closed unless the regime counts
exactly reproduce the already frozen baseline result:

- XG2: LARGE = 92, SMALL = 208
- XG4: LARGE = 57, SMALL = 243

The threshold remains exactly:

`0.11228626366380845`

It must not be re-estimated.

## Restartability validator

The new runner must expose a read-only artifact validator/loader that:

- validates manifest hashes and byte counts;
- validates exactly 300 item rows;
- validates exact pair order `301..600`;
- validates one-to-one plan indices;
- losslessly loads `alignment_delta_h.pt`;
- validates tensor dtype and shape against item metadata;
- validates `delta_baseline` identity;
- validates frozen LARGE/SMALL counts;
- validates support gate PASS;
- performs zero model forwards.

This validator does not authorize Phase-2 execution.

## Required tests

The new test file must cover at minimum:

- exact 1200-forward Phase-1 budget;
- exact correction freeze identity;
- exact threshold;
- XG2/XG4 frozen regime counts;
- required baseline PE fields;
- exact `delta_baseline` identity;
- binary alignment-plan round trip;
- source-pair / plan-index identity;
- tensor dtype/shape identity;
- manifest inclusion and SHA256 validation;
- zero-forward restartability validation;
- no response-field leakage;
- no alignment intervention path;
- no training/backward/task-head path;
- preservation of the existing frozen runner.

## Validation

Before implementation freeze, run:

`python -m py_compile scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py`

`python -m pytest -q tests/test_reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py`

`python -m pytest -q tests/test_reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_full_baseline.py tests/test_reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py`

`git diff --check`

No GPU or scientific execution is part of validation.

## Stop conditions

Stop if implementation would require:

- modifying an existing frozen runner;
- modifying an existing frozen artifact;
- changing pair membership or order;
- changing threshold or group-size semantics;
- changing checkpoint identity;
- changing CUDA backend semantics;
- executing more than four baseline forwards per pair;
- executing an alignment branch;
- reconstructing Phase-2 state through additional model forwards.
