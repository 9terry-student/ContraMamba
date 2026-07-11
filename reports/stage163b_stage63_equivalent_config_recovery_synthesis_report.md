# Stage163-B Stage63-Equivalent Config Recovery Synthesis

## 1. Summary decision

Decision: `STAGE163B_PARTIAL_CONFIG_RECOVERY_INVALID_SKELETON_BRIDGE_CANDIDATES_IDENTIFIED`

Stage163-A partially recovered Stage63-equivalent config evidence, but the generated command skeleton is invalid and must not be executed. The useful outcome is not a runnable command; it is a narrowed set of concrete bridge file candidates, a repeated likely bridge mode, and an architecture clue.

Stage164-A should be audit/command-construction first, not training.

## 2. Stage163-A result

Stage163-A output:

- Out dir: `reports/stage163a_stage63_equivalent_config_recovery_20260711_051651`
- Report JSON: `reports/stage163a_stage63_equivalent_config_recovery_20260711_051651/stage163a_stage63_equivalent_config_recovery_report.json`
- Report MD: `reports/stage163a_stage63_equivalent_config_recovery_20260711_051651/stage163a_stage63_equivalent_config_recovery_report.md`
- Decision: `STAGE163A_STAGE63_EQUIVALENT_CONFIG_PARTIALLY_RECOVERED`

Important Stage163-A result:

- `n_candidate_lines`: 4
- `has_command_like_lines`: false
- `n_bridge_values_recovered`: 8
- `ready_to_run_stage164`: false

Stage163-A recovered evidence, not a valid execution recipe.

## 3. Critical correction: invalid skeleton

The Stage163-A recovered command skeleton is not executable.

The recovered `bridge_flags_with_values` contains invalid garbage values:

- `--stage57-bridge-train-jsonl`: `"is"`
- `--stage66-bridge-train-jsonl`: `"is"`
- `--stage75-bridge-train-jsonl`: `"is"`
- `--stage80a-bridge-train-jsonl`: `"is"`

The `"is"` bridge JSONL values are extraction artifacts caused by loose regex extraction from prose lines. They are not valid JSONL paths. Do not execute the Stage163-A recovered skeleton, and do not use `"is"` as any bridge JSONL value.

## 4. Valid bridge candidates

Valid bridge file candidates were identified from `bridge_path_candidates`, and they should be verified in Stage164-A:

| Stage | Candidate path | Exists repo-relative | Recommended mode |
| --- | --- | --- | --- |
| Stage57 | `data/stage57_nonleaking_external_bridge.jsonl` | true | `append_train_only` |
| Stage66 | `data/stage66_residual_bridge.jsonl` | true | `append_train_only` |
| Stage75 | `data/stage75_targeted_residual_bridge.jsonl` | true | `append_train_only` |
| Stage80a | `data/stage80a_conservative_stage75v2_bridge.jsonl` | true | `append_train_only` |

`append_train_only` was repeatedly recovered as the likely mode for Stage57, Stage66, Stage75, and Stage80a bridge flags. The bridge JSONL paths still require explicit manual resolution from the candidates before any training run.

## 5. Architecture clue

Stage163-A recovered `architecture = v6b_minimal` from Stage63 artifacts.

Stage161-A used `architecture = vnext_minimal` with `vnext_router_mode = learned_x_product` and 30 epochs.

The `v6b_minimal` versus `vnext_minimal` mismatch is likely important. Stage161-A may have collapsed because it did not reproduce the Stage63 architecture/config path.

## 6. Stage63 vs Stage161 comparison

| Run | Accuracy | Macro F1 | REFUTE | NOT_ENTITLED | SUPPORT | Scalar fields | Architecture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stage63 bridge external | 0.322 | 0.3153191098829531 | 217 | 492 | 291 | 0 | `v6b_minimal` recovered from artifacts |
| Stage161-A checkpoint scalar | 0.14 | 0.08626074131076054 | 13 | 985 | 2 | 10 | `vnext_minimal` |

Stage63-equivalent behavior was not recovered yet. Stage63 is the non-collapsed behavior target, while Stage161-A proves checkpoint-backed scalar export can work but remains collapsed.

## 7. Recommended Stage164-A

Goal: validate concrete bridge candidate files and construct a manually resolved Stage63-equivalent checkpoint-backed command.

Actions:

- Verify all concrete bridge JSONL files exist.
- Summarize row counts and label distributions for each bridge file.
- Verify the runner accepts `v6b_minimal` with checkpoint save flags.
- Construct a candidate command using `v6b_minimal` and concrete bridge file paths.
- Do not execute training until command validity is confirmed.

Stage164-A must remain audit/command-construction first unless all bridge paths and modes are verified.

## 8. Safety constraints

- Report-only: true
- Training run executed: false
- External data used for training: false
- External labels used for training: false
- Threshold used for model selection: false
- Checkpoint selection modified: false
- Shadow diagnostics integrated: false
- Final predictions modified by shadow: false

Do not execute the Stage163-A recovered skeleton, use bridge JSONL value `"is"`, train on external labels, tune thresholds on external labels, use external metrics for checkpoint selection, integrate shadow diagnostics, or blindly increase epochs.
