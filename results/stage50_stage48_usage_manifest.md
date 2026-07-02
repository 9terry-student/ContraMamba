# Stage50 Stage48 Usage Manifest

Stage50 records how downstream runs should use the Stage47/48 frozen recovery config. This is a reporting stage only: it performs no training or evaluation.

## Result

`STAGE50_STAGE48_USAGE_MANIFEST_READY`

## How To Use

Use `--use-stage47-selected-recovery-config` for stable default recovery runs. Do not manually retype support_w/ne_w unless intentionally doing an ablation.

This loads `recovery_w01_ne01` (support_w=0.1, ne_w=0.1) from `results/stage47_selected_recovery_config_check.json` and overrides `stage45c_support_recovery_weight=0.1` and `stage45c_entitled_ne_penalty_weight=0.1`.

`recovery_w010_ne020` (support_w=0.1, ne_w=0.2) is diagnostic only, not the global default.

## Dropped Configs

- `w0.05_ne0.05`
- `w0.2_ne0.1`

## Sources

- Stage47 config: `results\stage47_selected_recovery_config_check.json`
- Stage49 report: `results\stage49_stage48_integration_report.json`
