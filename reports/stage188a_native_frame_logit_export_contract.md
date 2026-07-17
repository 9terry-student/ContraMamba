# Stage188-A native `frame_logit` export contract

## Historical-reference input

Builder option `--stage174d1-reference` takes `reports/stage188a_historical_baseline_recovery_closure.json`. This is a Stage188-A recovery closure object, not original Stage174-D1F runtime provenance. It must contain `decision: STAGE174D1_EXACT_HISTORICAL_RUN_NOT_RECOVERABLE`, `reference_decision: STAGE174D1F_CLEAN_LOCAL_PAIRWISE_OBJECTIVE_DIRECTION_CONFLICT_PATH_CLOSED`, `historical_reference_only: true`, `exact_historical_run_recoverable: false`, `baseline_definition: current_commit_default_off_paired_baseline`, and the verified Stage174-D1F facts in `experimental_scope`. Missing historical argv, Git commit, and resolved configuration are non-blocking; a missing or malformed closure itself fails closed into a Stage188-A blocked report.

## Export surface

When `--stage115-clean-dev-scalar-output-jsonl` is requested, every clean-dev scalar row must add exactly one native field:

```json
{"frame_logit": 0.0}
```

The value is the row's native pre-sigmoid scalar from `output["frame_logit"]`, converted only by detach, CPU transfer, and scalar serialization. Existing scalar fields and their meanings remain unchanged.

The export is additive and evaluation-only. It is not consumed by loss, checkpoint selection, thresholding, calibration, or model decisions. It must behave identically for margin weights `0.0` and `0.05`. Runs that do not request the Stage115 scalar output retain existing behavior.

## Prohibited substitutes

- No `logit(frame_prob)` reconstruction.
- No final-classifier logits.
- No `loss_logits`.
- No probability renamed as a logit.
- No model, head, loss, or training-signal change.

## Fail-closed alignment gates

Before the scalar JSONL is written, all of the following must hold:

- `frame_logit` exists in the selected clean-dev output.
- Flattened native `frame_logit` count equals the clean-dev row count.
- Flattened native `frame_prob`, prediction, source-row, and row-ID counts equal the clean-dev row count; when gold labels are present, their count also equals it.
- Scalar rows retain the exact prediction-export row order.
- Every row ID is non-null and unique.
- Every exported `frame_logit` is a finite numeric scalar.
- Every exported `frame_prob` is a finite numeric scalar.

Any violation blocks the export and therefore blocks Stage188-B analysis.

## Analyzer contract

The Stage188-B analyzer must load `clean_dev_scalars.jsonl` for both arms, require direct finite `frame_logit` values, and join baseline/intervention by exact unique row ID. It must verify identical row-ID sets and matching gold/prediction identity before calculating deltas or joining the exact Stage182-B cohort. Missing or inconsistent native evidence yields `STAGE188B_PAIRED_INTERNAL_MARGIN_EXPERIMENT_BLOCKED`.
