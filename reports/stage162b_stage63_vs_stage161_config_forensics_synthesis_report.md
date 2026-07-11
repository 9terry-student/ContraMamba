# Stage162-B Stage63-vs-Stage161 Config Forensics Synthesis

## 1. Summary decision

Decision: `STAGE162B_STAGE63_NONCOLLAPSE_REQUIRES_BRIDGE_OR_COMPOSER_CONFIG_RECOVERY`

Stage162-A shows that Stage63 achieved a non-collapsed external prediction distribution, while Stage161 proved checkpoint-backed scalar export can work. These two facts do not yet combine into a recovered scalar-bearing, non-collapsed run. The next blocker is recovering the Stage63-equivalent bridge/composer/current-best configuration, not extending epochs.

This is a report-only synthesis. No training code, model code, export script, checkpoint selection policy, or predictions are modified.

## 2. Stage162-A evidence

Stage162-A output:

- Out dir: `reports/stage162a_stage63_vs_stage161_config_forensics_20260711_051300`
- Report JSON: `reports/stage162a_stage63_vs_stage161_config_forensics_20260711_051300/stage162a_stage63_vs_stage161_config_forensics_report.json`
- Report MD: `reports/stage162a_stage63_vs_stage161_config_forensics_20260711_051300/stage162a_stage63_vs_stage161_config_forensics_report.md`
- Decision: `STAGE162A_STAGE63_NONCOLLAPSE_LIKELY_BRIDGE_OR_COMPOSER_CONFIG_DIFFERENCE`

Stage162-A found that Stage63 bridge evidence exists and that the runner includes bridge/external flags. It also found checkpoint/scalar infrastructure flags, including `--save-checkpoint-path` and `--save-checkpoint-mode`.

## 3. Quality comparison

| Run | Accuracy | Macro F1 | REFUTE | NOT_ENTITLED | SUPPORT | Scalar fields | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stage53 frozen external | 0.156 | 0.11671748688294066 | 33 | 944 | 23 | 0 | Collapsed toward `NOT_ENTITLED`; no scalar export. |
| Stage63 bridge external | 0.322 | 0.3153191098829531 | 217 | 492 | 291 | 0 | Non-collapsed distribution, but no scalars. |
| Stage161 checkpoint-backed scalar | 0.14 | 0.08626074131076054 | 13 | 985 | 2 | 10 | Scalar export works, but distribution is collapsed. |

Stage63 is non-collapsed but has no scalars. Stage161 has scalars but is collapsed. Therefore Stage161 cannot be used for recovered-versus-regressed scalar conclusions.

## 4. Config/bridge evidence

Stage162-A reported:

- `stage63_pred_exists`: true
- `stage63_dir_exists`: true
- `n_stage63_related_files`: 24
- `has_stage63_config_artifacts`: true
- `stage63_bridge_evidence`: true
- `runner_has_bridge_flags`: true
- `runner_has_checkpoint_flags`: true

Bridge/external flags present in the runner include Stage57, Stage66, Stage75, and Stage80A bridge train JSONL/mode flags, plus Stage43 external evaluation flags.

## 5. Diagnosis

The scalar export blocker is resolved: Stage161 demonstrates checkpoint-backed scalar export with 10 scalar fields.

The behavior recovery blocker is not resolved: Stage161 does not reproduce the Stage63 non-collapsed external distribution. Its predictions are overwhelmingly `NOT_ENTITLED`, while Stage63 spreads predictions across `REFUTE`, `NOT_ENTITLED`, and `SUPPORT`.

The primary blocker is Stage63 bridge/composer/current-best configuration recovery. More epochs should not be the default response, because the evidence points to missing or mismatched configuration rather than insufficient training duration.

## 6. Recommended Stage163-A

Stage163-A should recover the exact Stage63-equivalent command/config and instantiate a checkpoint-backed scalar export run.

Inputs:

- `reports/stage162a_stage63_vs_stage161_config_forensics_20260711_051300/stage162a_stage63_config_scan.json`
- `reports/stage162a_stage63_vs_stage161_config_forensics_20260711_051300/stage162a_runner_bridge_snippets.json`
- `reports/stage162a_stage63_vs_stage161_config_forensics_20260711_051300/stage162a_runner_relevant_flags.csv`
- `results/stage63_bridge_enabled_vitaminc_external_eval_20260702_060044`

Required actions:

- Extract Stage63 command/config evidence from scan artifacts.
- Identify bridge train JSONL flags and modes used by Stage63.
- Identify composer/recovery-related flags if present.
- Construct a Stage163-A command with recovered Stage63-equivalent settings plus `--save-checkpoint-path`.
- Do not run until the recovered config is explicitly reported.

Stage163-A should extract the actual bridge/composer/training flags before any further training run.

## 7. Safety constraints

- Report-only: true
- Training run executed: false
- External data used for training: false
- External labels used for training: false
- Threshold used for model selection: false
- Checkpoint selection modified: false
- Shadow diagnostics integrated: false
- Final predictions modified by shadow: false

Do not train on external labels, tune thresholds on external labels, use external metrics for checkpoint selection, integrate shadow diagnostics, use collapsed Stage161 scalar predictions for recovered-vs-regressed conclusions, or blindly increase epochs.
