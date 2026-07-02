# Stage46 Recovery Selection Freeze

Stage46 is a reporting/freeze-only stage. It reads the Stage45D generalization/regression audit and, only when that audit genuinely recommends `recovery_w01_ne01` as the stable global default, freezes the internal SUPPORT entitlement recovery selection.

## Decision

`STAGE46_RECOVERY_SELECTION_BLOCKED`

No recovery configuration was selected or frozen. Stage46 does not fabricate a selection when the Stage45D evidence does not support one.

## Block Reason

stage45d_no_data: Stage45D found no comparison data (has_any_data is false)

## Source Summary

- Path checked: `results\stage45d_generalization_summary.json`

## Next Step

Re-run the Stage45D generalization audit (`scripts/write_stage45d_generalization_audit.py`) against a results directory that contains real Stage45C/Stage45D holdout train report JSON files, confirm it reports `recommend_stable_default: true` for `recovery_w01_ne01`, and then re-run this script.
