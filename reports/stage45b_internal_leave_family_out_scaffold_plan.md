# Stage45-B Internal Leave-Family-Out Scaffold Plan

## Decision

`STAGE45B_INTERNAL_LEAVE_FAMILY_OUT_SCAFFOLD_PLAN_READY`

Stage45-B implements an internal-only scaffold for controlled transformation-family robustness diagnostics before any Stage45-C model, loss, curriculum, or data redesign. It creates a reusable family manifest builder and optional leave-family-out split support in the training runner.

## Scope

Stage45-B is internal-only. It uses only `data/controlled_v5_v3_without_time_swap.jsonl`, internal controlled metadata already present in that file, existing internal train/dev mechanics, and existing internal prediction/reporting logic.

It does not use Stage43-B1, VitaminC, Climate-FEVER, external examples, external labels, external prediction distributions, external metrics, or Stage43 external reports for split construction, thresholds, constraints, training, calibration, checkpoint selection, loss design, model selection, data generation, or composer behavior changes.

## Implemented Scaffold

### Family Manifest Builder

`scripts/build_stage45_internal_family_manifest.py` reads an internal controlled JSONL file and writes:

- `reports/stage45b_internal_family_manifest.json`
- `reports/stage45b_internal_family_manifest.md`

The builder infers transformation family from the shared resolver in `scripts/stage45_internal_family_utils.py`. Preferred fields are checked in this order:

- `transformation_family`
- `stage15_probe_type`
- `probe_type`
- `family`
- `source_family`
- `controlled_family`
- `template_family`
- `case_type`
- `metadata.transformation_family`
- `metadata.stage15_probe_type`
- `metadata.probe_type`

If no explicit family is present, the deterministic fallback is `unknown_family`. The manifest reports family counts, label counts by family, eligible holdout families by minimum row count, tiny-family warnings, and time_swap-like warnings.

### Optional Runner Family Holdout

`scripts/train_controlled_v6b_minimal.py` now has an optional Stage45-B family holdout split. When `--stage45-use-family-holdout` is off, the existing `v5.split_by_pair_id(records, dev_ratio=args.dev_ratio, seed=args.seed)` behavior is preserved.

When enabled, the runner uses only the internal `--data` JSONL. Dev/validation rows are records whose resolved family equals `--stage45-holdout-family`; training rows are all other records. The runner fails clearly if the holdout family is missing, too small, or leaves no training rows.

Existing training, loss, model, composer, and checkpoint selection behavior remains unchanged after the split is constructed. Stage44-B2 selection can operate on the current internal holdout dev split, and Stage43 external evaluation remains optional and never implicitly enabled by Stage45-B.

## New CLI Flags

- `--stage45-use-family-holdout`
- `--stage45-family-field`
- `--stage45-holdout-family`
- `--stage45-min-holdout-size`
- `--stage45-family-holdout-report-json`
- `--stage45-family-holdout-report-md`

Manifest builder flags:

- `--data`
- `--output-json`
- `--output-md`
- `--min-family-size`
- `--family-field`

## Report Fields

When family holdout is enabled, the normal output JSON and optional Stage45 report include:

- `stage45b_enabled`
- `stage45b_decision`
- `stage45b_family_field_used`
- `stage45b_holdout_family`
- `stage45b_train_rows`
- `stage45b_holdout_rows`
- `stage45b_train_label_counts`
- `stage45b_holdout_label_counts`
- `stage45b_holdout_metrics`
- `stage45b_leakage_policy`
- `stage45b_recommendation`

The holdout metrics include accuracy, macro-F1, per-label precision/recall/F1, prediction counts, gold counts, NOT_ENTITLED prediction rate, SUPPORT recall, and REFUTE recall.

## Guardrails

- Do not read Stage43-B1 files.
- Do not read Stage43 external reports to set thresholds or families.
- Do not use external examples, metrics, labels, or predictions.
- Do not tune thresholds.
- Do not calibrate.
- Do not change loss behavior.
- Do not change model architecture.
- Do not change composer behavior.
- Do not claim external validation.

## Allowed Claims

- Stage45-B creates an internal family manifest scaffold.
- Stage45-B creates optional internal leave-family-out split support.
- Stage45-B does not use Stage43-B1.
- Stage45-B is designed to diagnose controlled-family robustness before Stage45-C redesign.
- Default runner behavior is unchanged when Stage45 flags are off.

## Disallowed Claims

- Do not claim external PASS.
- Do not claim VitaminC transfer success.
- Do not claim Climate-FEVER robustness.
- Do not claim naturalistic fact-verification generalization.
- Do not claim Stage45-B improves performance.
- Do not use Stage43-B1 for tuning, calibration, thresholds, checkpoint selection, loss selection, data generation, or model selection.

## Remaining Risks

- If the controlled data lacks explicit family metadata, all rows may fall back to `unknown_family`, making leave-family-out diagnostics incomplete until internal metadata is added.
- Tiny families may be unsuitable as holdouts without aggregation or additional internal data.
- Leave-family-out robustness is still an internal proxy, not external validation.

## Next Stage

`Stage45-C: evidence sufficiency contrast curriculum or auxiliary diagnostic, selected only by internal leave-family-out criteria`
