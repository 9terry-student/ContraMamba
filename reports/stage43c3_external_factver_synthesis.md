# Stage43-C3 External Fact-Verification Synthesis

## Decision

`STAGE43C3_EXTERNAL_FACTVER_SYNTHESIS_COMPLETE_BUT_NO_EXTERNAL_PASS`

Stage43-C0/C1/C2 is complete as an external evaluation infrastructure and diagnostic sequence, but it does not support an external validation PASS claim. The composer-path failure was fixed, the Stage43-B1 leakage boundary held, and Stage43-C2 did not introduce unsafe SUPPORT or REFUTE/SUPPORT flips. External performance remains too low and too dominated by NOT_ENTITLED to claim naturalistic fact-verification generalization.

## Background

Stage43-B1 acquired external-evaluation-only VitaminC validation sample1000 and Climate-FEVER test sample1000 datasets. These datasets are external validation artifacts only: they must not be used for training, validation/dev split construction, checkpoint selection, threshold selection, calibration, loss design, model selection, or composer behavior changes.

Stage43-B2 kept the standalone checkpoint evaluator scaffold-only. Stage43-C0 therefore added a post-training, in-memory best-state external evaluation hook so external fact-verification could run after normal best-state restoration without participating in training or selection.

## Stage43-C0 Result

The external evaluation hook ran after best-state restoration with leakage fields preserved:

- `used_for_training`: false
- `used_for_checkpoint_selection`: false
- `used_for_threshold_selection`: false
- `stage43_external_eval_timing`: `post_training_after_best_state_restore`

Both VitaminC and Climate-FEVER were `INCOMPLETE` in C0 because `safe_structured_v2` composer output was unavailable for every evaluated row. Base and composed predictions were identical because the composer had no usable source shadow label.

## Stage43-C1 Diagnostic Result

Stage43-C1 diagnosed the failure without changing predictions or model behavior:

- External prediction path matched the controlled dev path.
- External label mapping matched the controlled dev mapping.
- Token truncation was not excessive: Climate-FEVER was about 5.1%, and VitaminC was about 6.8%.
- Predictions were dominated by NOT_ENTITLED.
- Composer unavailability reason was `missing_source_shadow_label` for every row.
- Required Stage32/36/37/39 intermediate fields were absent.

This ruled out input path, tokenizer/max_length, label mapping, and excessive truncation as the primary explanation. The dominant observed failure mode was external-distribution NOT_ENTITLED collapse, plus missing composer shadow-source fields.

## Stage43-C2 Result

Stage43-C2 enabled/exported the required Stage32/36/37/39 shadow and composer fields for Stage43 external rows using the existing internal export path. Composer availability became complete:

| Dataset | Available rows | Composer changed rows | Missing source shadow label resolved |
|---|---:|---:|---|
| Climate-FEVER test sample1000 | 903/903 | 2 | yes |
| VitaminC validation sample1000 | 1000/1000 | 1 | yes |

No introduced unsafe transitions were observed in C2:

- `introduced_unsafe_SUPPORT_count`: 0
- `introduced_REFUTE_to_SUPPORT_count`: 0
- `introduced_SUPPORT_to_REFUTE_count`: 0

Stage43-C2 is therefore an infrastructure and composer-path success. It is not an external generalization success.

## External Performance

| Dataset | Base macro-F1 | Composed macro-F1 | Delta macro-F1 | Composed prediction collapse |
|---|---:|---:|---:|---|
| Climate-FEVER test sample1000 | 0.165973 | 0.174139 | +0.008166 | NOT_ENTITLED 898/903 |
| VitaminC validation sample1000 | 0.093653 | 0.095031 | +0.001378 | NOT_ENTITLED 979/1000 |

The composed metrics improved slightly after the composer shadow path became available, but the absolute external performance remains poor and still collapsed toward NOT_ENTITLED. These results must remain incomplete/diagnostic, not PASS.

## Interpretation

The current controlled-trained ContraMamba model does not yet generalize to naturalistic external fact-verification datasets. The dominant failure mode is NOT_ENTITLED collapse under external distribution.

Climate-FEVER should be treated as a cross-domain limitation, not a training signal. VitaminC is closer to in-family fact-verification, but it still shows severe collapse and does not justify a transfer-success claim.

## Allowed Claims

- Stage43 external eval hook works after post-best-state restoration.
- Stage43-B1 leakage boundary was preserved.
- Stage43-C2 resolved composer shadow-field availability.
- No introduced unsafe SUPPORT / REFUTE-to-SUPPORT / SUPPORT-to-REFUTE transitions were observed in C2.

## Disallowed Claims

- Do not claim external validation PASS.
- Do not claim naturalistic generalization.
- Do not claim Climate-FEVER robustness.
- Do not claim VitaminC transfer success.
- Do not claim safety improvement from Stage43-B1.

## Recommendation

Freeze Stage43 as an honest negative/incomplete external validation result. Do not tune thresholds or calibration on Stage43-B1. Do not use Stage43-B1 for checkpoint or model selection.

Future improvement should require a new pre-registered training-stage design using only controlled/internal data, followed by one read-only re-evaluation on Stage43-B1. If more external validation is needed before redesign, acquire a separate held-out external set so Stage43-B1 remains protected.
