# Stage45D Generalization / Regression Audit

Stage45C compared internal-only SUPPORT entitlement recovery weight settings (baseline support_w=0.0/ne_w=0.0, stable candidate support_w=0.1/ne_w=0.1, and a paraphrase-specialized candidate support_w=0.1/ne_w=0.2) on the `intervention_type=paraphrase` and `primary_failure_type=none` internal family holdouts. This Stage45D audit aggregates every Stage45C/Stage45D-style train report found under the scanned results directory, compares each recovery config against its holdout group's baseline, and checks whether the `recovery_w01_ne01` candidate generalizes across other internal holdout families without catastrophic regression. This is a reporting/aggregation-only pass over existing JSON reports; it does not train or evaluate anything.

Report files scanned: 0. Rows parsed: 0.

## Per-Holdout Comparison Table

_No Stage45C/Stage45D train report JSON files were found under the scanned results directory._

## Delta Table (vs. baseline within each holdout group)

_No non-baseline configs with a matching baseline row were found._

## Overall Selection Summary

| config | holdout_groups_seen | groups_improved_over_baseline | groups_with_support_recovery_gain | groups_with_refute_regression | groups_with_ne_rate_shift_large | groups_with_harmful_ne_rate_shift | groups_with_catastrophic_regression | avg_delta_acc | avg_delta_macro_all3 | avg_delta_support_recall | avg_delta_refute_recall | avg_delta_ne_pred_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |
| recovery_w01_ne01 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |
| recovery_w010_ne020 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |  |  |  |  |

## Recommendation

No Stage45C/Stage45D train report JSON files were found under the scanned results directory. This audit produced no comparison data; run the Stage45C/Stage45D holdout trainings and re-run this script to populate the audit.
