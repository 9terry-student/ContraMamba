# Gen4-K XG2/XG4 Fresh Response Phase-1 Restartability Correction

## Status

Protocol-correction candidate following frozen baseline artifact commit
`40d21032b09560e246b16001c22837e46d5d3c70`.

This correction does not invalidate the Phase-1 support result already frozen at
`40d2103`. It addresses only the persistence boundary required for a separate
Phase-2 alignment-response execution.

No training, alignment intervention, H1/H2 test, or scientific response
evaluation is authorized by this document.

## Frozen facts retained unchanged

The following remain unchanged:

- families: XG2 and XG4;
- 300 fresh source pairs per family;
- pair ordering and frozen tokenizer-anchor eligibility;
- representative checkpoint identity;
- frozen CUDA-equivalent backend and kernels;
- baseline cells C0/C1/C2/C5;
- Phase-1 budget: exactly 1200 model forwards per family;
- alignment-shift threshold:
  `0.11228626366380845`;
- minimum LARGE/SMALL group size: 30;
- regime membership is determined before any alignment-response execution;
- Phase-2, when eligible, is exactly two alignment forwards per pair,
  therefore exactly 600 additional forwards per family;
- successful complete budget remains 1800 forwards per family.

The previous frozen XG2/XG4 Phase-1 results remain valid as
baseline-regime support evidence only.

## Defect

The artifact frozen at `40d2103` persists regime geometry and
`alignment_shift_abs`, but is not sufficient to resume Phase-2 independently.

The causal endpoint is

`R_ALIGN = delta_alignment - delta_baseline`.

Therefore Phase-2 requires the frozen Phase-1 value

`delta_baseline =
 baseline_plus_path_efficiency -
 baseline_minus_path_efficiency`.

The historical same-family prospective implementation also carries an
in-memory alignment runtime plan from Phase-1 into Phase-2. In particular,
Phase-2 consumes `alignment_delta_h` when executing the two alignment
branches.

Consequently, persisting only `delta_baseline` is insufficient for a genuine
+600-only restart. Reconstructing `alignment_delta_h` by rerunning baseline
branches would silently spend additional scientific forwards.

## Correction contract

A replacement Phase-1 run may be executed for each family using exactly the
same frozen inputs and exactly 1200 baseline forwards.

For every source pair, the corrected Phase-1 artifact must persist:

1. all existing frozen regime/geometry fields;
2. `baseline_plus_path_efficiency`;
3. `baseline_minus_path_efficiency`;
4. `delta_baseline`;
5. the exact Phase-2 alignment intervention plan needed to execute the
   alignment branches without any baseline model forward.

The persisted alignment plan must include at minimum:

- exact `alignment_delta_h` tensor values;
- tensor dtype;
- tensor shape;
- source-pair identity;
- target plus/minus cells and anchors;
- alignment target/realized geometry audit scalars required by the frozen
  intervention audit.

`delta_baseline` must satisfy exactly the frozen definition:

`delta_baseline =
 baseline_plus_path_efficiency -
 baseline_minus_path_efficiency`.

All persisted artifacts must be covered by the artifact manifest and SHA256
checksums.

## Persistence format requirement

Scalar scientific fields remain in the per-item JSONL artifact.

`alignment_delta_h` must be persisted losslessly in a binary tensor artifact
with an explicit source-pair index and must be included in the SHA256
manifest. JSON decimal serialization of the intervention tensor is not an
acceptable substitute.

Loading the persisted tensor for Phase-2 must not invoke a model forward.

## Phase-1 boundaries

The corrected Phase-1 run:

- performs exactly 1200 model forwards per family;
- performs no alignment intervention;
- observes no alignment response;
- computes no `R_ALIGN`;
- performs no H1/H2 inference;
- performs no training or backward pass;
- does not re-estimate the frozen threshold;
- does not change regime membership semantics.

## Phase-2 restart gate

Phase-2 is permitted only after a corrected Phase-1 artifact independently
passes all of the following:

- exact family/pair identity;
- 300 persisted item rows;
- unchanged LARGE/SMALL regime counts relative to the frozen baseline result;
- support gate still passes;
- exact baseline PE identity;
- exact `delta_baseline` identity;
- complete one-to-one alignment-plan coverage for all 300 pairs;
- lossless `alignment_delta_h` tensor load;
- manifest and SHA256 validation;
- zero model forwards during artifact validation/loading.

Once that gate passes, Phase-2 may execute exactly:

`300 pairs x 2 alignment branches = 600 model forwards per family`.

Baseline branches must not be executed in Phase-2.

Thus the corrected complete protocol remains:

`1200 Phase-1 + 600 Phase-2 = 1800 forwards per family`.

Any implementation requiring baseline re-execution during Phase-2 is blocked.

## Existing artifact disposition

The artifact frozen at `40d2103` is retained unchanged and remains valid for
its original scope:

`FRESH_BASELINE_REGIME_SUPPORT_ONLY`.

It must not be overwritten, edited, or retroactively promoted to a
restartable Phase-2 input.

A corrected Phase-1 execution must write to a new output directory and use a
new schema identity.
