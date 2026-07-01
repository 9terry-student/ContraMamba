# Stage34-B Metadata Preservation Report

## Scope

Stage34-B fixes external prediction export metadata preservation and evaluator fallback diagnostics for the Stage34-A held-out coverage probe.

## Export Metadata Preservation

`scripts/train_controlled_v6b_minimal.py` now preserves the following source-record metadata fields in prediction exports when they are present:

`id`, `pair_id`, `claim`, `evidence`, `group`, `intervention_type`, `normalized_intervention`, `primary_failure_type`, `failure_type`, `source`, `split`, `stage34_family`, `stage34_relation`, `stage34_expected_route`, `stage34_is_heldout`, `final_label`, `gold_label`, `label`.

This preservation is export-only. It does not alter model architecture, final logits, final predictions, H1 composition, Stage33-F owner behavior, training data, or checkpoint selection.

## Evaluator Fallbacks

`scripts/evaluate_stage34_heldout_coverage.py` now resolves missing metadata with explicit fallbacks:

- `group`: `group`, `intervention_type`, `normalized_intervention`, `primary_failure_type`, then `UNKNOWN`.
- `stage34_family`: existing `stage34_family`, otherwise inferred from the resolved group, otherwise `unknown_family`.
- `stage34_relation`: existing `stage34_relation`, otherwise inferred from the resolved group suffix, otherwise `unknown_relation`.
- `stage34_expected_route`: existing `stage34_expected_route`, otherwise inferred from group suffix: support to `ENTAILMENT_PRESERVE`, not-entitled to `OVERCLAIM_NE`, refute to `CONTRADICTION_REFUTE`, otherwise `unknown`.

The evaluator also reports family, relation, and expected-route counts so missing or degraded metadata is visible in the JSON and Markdown reports.

## Corrected Decision Logic

If prediction rows still lack usable metadata, the evaluator returns:

`STAGE34A_METADATA_MISSING_DIAGNOSTIC_INVALID`

In that case, aggregate metrics remain reported, including shadow macro-F1 and delta macro-F1, but group-specific held-out generalization is explicitly not claimed.

The evaluator no longer treats metadata loss alone as symbolic memorization risk. Memorization-risk labels are reserved for actual recoverability or unresolved-pattern failures once metadata is available.

## Diagnostic-Only Policy

Stage34-A remains diagnostic-only. These prediction rows and evaluator outputs must not be used for training, calibration, threshold selection, loss, checkpoint selection, or Kaggle selection.

## Remaining Risks

- Fallback inference from group names is best-effort and cannot recover arbitrary missing metadata if group names are also absent.
- The fix has not been validated by running training, evaluation, Kaggle, smoke tests, or full experiments, by request.
