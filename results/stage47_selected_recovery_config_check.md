# Stage47 Selected Recovery Config Check

Stage47 confirms that the Stage46 frozen recovery selection is readable and valid. This is a reporting/helper stage only: it does not train, evaluate, or fabricate any configuration values.

## Result

`STAGE47_SELECTED_RECOVERY_CONFIG_READY`

Use `recovery_w01_ne01` (support_w=0.1, ne_w=0.1) as the stable global default.

Keep `recovery_w010_ne020` (support_w=0.1, ne_w=0.2) only as a paraphrase-specialized diagnostic / runner-up, not as the default.

Do not use dropped configs.

## Dropped Settings

- `w0.05_ne0.05`
- `w0.2_ne0.1`

## Source

- Stage46 file: `results\stage46_selected_recovery_config.json`
