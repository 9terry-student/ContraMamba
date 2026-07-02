# Stage46 Recovery Selection Freeze

Stage46 is a reporting/freeze-only stage. It reads the Stage45D generalization/regression audit and, only when that audit genuinely recommends `recovery_w01_ne01` as the stable global default, freezes the internal SUPPORT entitlement recovery selection and exports final selection artifacts. It performs no training, evaluation, or further experiments.

## Selected Stable Default

- Config name: `recovery_w01_ne01`
- support_w: 0.1
- ne_w: 0.1
- Role: `stable_global_default`

## Diagnostic Runner-Up

- Config name: `recovery_w010_ne020`
- support_w: 0.1
- ne_w: 0.2
- Role: `paraphrase_specialized_diagnostic_runner_up`

## Dropped Settings

- `w0.05_ne0.05`
- `w0.2_ne0.1`

## Stage45D Evidence

- Source summary: `results/stage45d_generalization_summary.json`
- Stage45D decision: `STAGE45D_GENERALIZATION_AUDIT_READY`
- Rows parsed: 10
- Holdout groups: `intervention_type=paraphrase`, `primary_failure_type=none`

| config | holdout_groups_seen | groups_improved_over_baseline | groups_with_support_recovery_gain | groups_with_refute_regression | groups_with_ne_rate_shift_large | groups_with_harmful_ne_rate_shift | groups_with_catastrophic_regression | avg_delta_acc | avg_delta_macro_all3 | avg_delta_support_recall | avg_delta_refute_recall | avg_delta_ne_pred_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recovery_w01_ne01 | 2 | 2 | 2 | 0 | 1 | 0 | 0 | 0.0600 | 0.0308 | 0.1067 | 0.0133 | -0.0633 |
| recovery_w010_ne020 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0.0483 | 0.0270 | 0.0900 | 0.0067 | -0.0450 |

## Caveats / Provenance

- Stage45D used reconstructed/internal Stage45C train-report JSONs where applicable.
- Stage46 is a reporting/freeze stage only.
- No training or evaluation is performed by this script.

## Final Recommendation

Stage46 freezes recovery_w01_ne01 (support_w=0.1, ne_w=0.1) as the stable global recovery setting. recovery_w010_ne020 is retained only as a paraphrase-specialized diagnostic / runner-up and is not selected as the global default.
