# Stage45-C Internal SUPPORT Entitlement Recovery Plan

## Decision

`STAGE45C_INTERNAL_SUPPORT_RECOVERY_SCAFFOLD_READY`

Stage45-C implements an internal-only SUPPORT entitlement recovery scaffold: optional diagnostics and an optional small auxiliary training term targeting the SUPPORT under-recall and entitled-to-NOT_ENTITLED over-rejection observed in Stage45-B1 internal family holdouts.

## Stage45-B1 Diagnosis Summary

Stage45-B1 ran internal-only leave-family-out holdouts on `data/controlled_v5_v3_without_time_swap.jsonl` using the recovered `intervention_type` and `primary_failure_type` fields. Key findings:

| Family | Holdout rows | Gold labels | Present-label macro-F1 | NOT_ENTITLED pred rate | SUPPORT recall | REFUTE recall |
|---|---:|---|---:|---:|---:|---:|
| `intervention_type=entity_swap` | 300 | NOT_ENTITLED: 300 | 1.0 | n/a (single class) | n/a | n/a |
| `intervention_type=none` | 300 | REFUTE 150 / SUPPORT 150 | 0.9444 | 0.10 | 0.80 | 1.0 |
| `intervention_type=paraphrase` | 300 | REFUTE 150 / SUPPORT 150 | 0.8293 | 0.2133 | 0.56 | 0.9467 |
| `intervention_type=polarity_flip` | 300 | REFUTE 150 / SUPPORT 150 | 1.0 | 0.0 | 1.0 | 1.0 |
| `primary_failure_type=none` | 600 | REFUTE 300 / SUPPORT 300 | 0.7921 | 0.2483 | 0.4633 | 0.97 |

The `entity_swap` holdout is single-class NOT_ENTITLED and is not informative for SUPPORT/REFUTE robustness.

**Diagnosis:** across every entitled holdout family, REFUTE recall stays at or near 1.0, while SUPPORT recall is consistently lower and drops sharply under `paraphrase` (0.56) and `primary_failure_type=none` (0.4633) holdouts. This pattern — many gold-entitled examples over-rejected as NOT_ENTITLED, with SUPPORT affected far more than REFUTE — is an internal-only analogue of the NOT_ENTITLED collapse seen in external diagnostics, produced entirely from internal controlled data.

## Target Failure Mode

- **Primary:** SUPPORT under-recall.
- **Secondary:** entitled-to-NOT_ENTITLED over-rejection, for both SUPPORT and REFUTE gold rows, most acute for SUPPORT.
- **Not targeted:** REFUTE recall, which is already robust across every internal family holdout observed so far.

## Allowed Inputs

- `data/controlled_v5_v3_without_time_swap.jsonl` (internal controlled data).
- Internal training split `final_labels` and model output `logits`.
- Stage45-B1 recovered family fields (`intervention_type`, `primary_failure_type`, and their composites).
- The internal Stage45-B family holdout dev split, for reporting/selection only — never inside the auxiliary training loss.

## Disallowed Inputs

- Stage43-B1 files or predictions.
- VitaminC data.
- Climate-FEVER data.
- Stage43 external reports.
- Any external examples, labels, metrics, or prediction distributions.
- Dev/holdout labels inside the auxiliary training loss.

## New CLI Flags

- `--stage45c-enable-support-recovery` (default off)
- `--stage45c-support-recovery-weight` (float, default `0.0`)
- `--stage45c-entitled-ne-penalty-weight` (float, default `0.0`)
- `--stage45c-target-label` (default `SUPPORT`)
- `--stage45c-entitled-labels` (default `SUPPORT,REFUTE`)
- `--stage45c-report-json` (optional path)
- `--stage45c-report-md` (optional path)

## Default-Off Behavior

When Stage45-C is disabled, or both weights are `0.0`, no auxiliary loss term is added to `total_loss`, and training behavior is exactly the pre-Stage45-C baseline. Report fields are still emitted with `stage45c_enabled: false` so runs can be audited either way.

## Auxiliary Loss Design

Both terms read only `output["logits"]` and `train_inputs["final_labels"]` from the internal training split for the current batch/epoch. Both degrade to a zero-valued, gradient-connected tensor (`logits.sum() * 0.0`) when no matching gold rows are present, so the run never crashes or produces NaN even under an internal family holdout that removes all rows of a given label from the training split.

**SUPPORT recovery term** (weight: `--stage45c-support-recovery-weight`):

```
support_probs = softmax(logits)[gold_label == target_label][:, target_label_id]
support_recovery_loss = mean(relu(1 - support_probs))
```

Penalizes low predicted probability of `--stage45c-target-label` (default `SUPPORT`) on gold `--stage45c-target-label` training rows.

**Entitled NOT_ENTITLED over-rejection term** (weight: `--stage45c-entitled-ne-penalty-weight`):

```
ne_probs = softmax(logits)[gold_label in entitled_labels][:, NOT_ENTITLED_id]
entitled_ne_penalty_loss = mean(ne_probs)
```

Penalizes high predicted NOT_ENTITLED probability on gold `--stage45c-entitled-labels` (default `SUPPORT,REFUTE`) training rows.

**Integration:**

```
total_loss = total_loss
    + support_recovery_weight * support_recovery_loss
    + entitled_ne_penalty_weight * entitled_ne_penalty_loss
```

added after the existing label/frame/predicate/sufficiency/polarity/intervention losses.

**Constraints:** no pseudo-examples are fabricated, no dev/holdout rows are used, and this scaffold adds no threshold tuning, no calibration, no composer behavior changes, no model architecture changes, and no external hyperparameter tuning.

## Expected Evaluation Protocol

1. Enable Stage45-C alongside a Stage45-B1 family holdout that previously showed SUPPORT brittleness, e.g.:
   `--stage45-use-family-holdout --stage45-family-field intervention_type --stage45-holdout-family paraphrase`
   or
   `--stage45-family-field primary_failure_type --stage45-holdout-family none`
2. Compare `stage45b_holdout_metrics.support_recall` and `not_entitled_prediction_rate` with Stage45-C enabled vs. disabled, holding every other flag fixed.
3. Use only the internal holdout dev split for this comparison; do not introduce Stage43-B1 or any other external data.
4. Existing Stage44-B2 prior-aware selection may still be used for checkpoint selection, restricted to the internal holdout dev split.

## Runner Family Holdout Compatibility

Stage45-C remains compatible with:

- `--stage45-use-family-holdout`
- `--stage45-family-field`
- `--stage45-holdout-family`
- `--stage45-family-holdout-report-json`
- `--stage45-family-holdout-report-md`

Stage45-C reads only `train_inputs["final_labels"]` and `output["logits"]` from whichever train split is active — the family-holdout train split when Stage45-B1 is enabled, or the normal internal train split otherwise. It does not alter how the family holdout split itself is constructed.

## Report Fields

- `stage45c_enabled`
- `stage45c_support_recovery_weight`
- `stage45c_entitled_ne_penalty_weight`
- `stage45c_target_label`
- `stage45c_entitled_labels`
- `stage45c_train_support_count`
- `stage45c_train_refute_count`
- `stage45c_train_not_entitled_count`
- `stage45c_loss_terms_active`
- `stage45c_support_recovery_loss_mean`
- `stage45c_entitled_ne_penalty_loss_mean`
- `stage45c_leakage_policy`
- `stage45c_recommendation`

## Leakage Policy

- Do not read Stage43-B1 files.
- Do not use VitaminC.
- Do not use Climate-FEVER.
- Do not read Stage43 external reports for design, tuning, thresholds, calibration, checkpoint selection, loss design, or model selection.
- Do not use external examples, labels, metrics, or prediction distributions.
- Do not use dev/holdout labels inside the auxiliary loss.

## Allowed Claims

- Stage45-C introduces an internal-only SUPPORT entitlement recovery scaffold.
- Stage45-C targets internal SUPPORT under-recall and entitled-to-NOT_ENTITLED over-rejection.
- Stage45-C does not use external data.
- Stage45-C remains compatible with Stage45-B1 family holdout diagnostics.

## Disallowed Claims

- Do not claim external validation PASS.
- Do not claim VitaminC transfer success.
- Do not claim Climate-FEVER robustness.
- Do not claim naturalistic fact-verification generalization.
- Do not claim Stage45-C solves external collapse before a new held-out external evaluation.
- Do not use Stage43-B1 for tuning or final validation.

## Remaining Risks

- The auxiliary terms are diagnostic/regularization signals derived from training-split labels the model already sees via the main label loss; measured internal-holdout gains may be modest or may trade off against REFUTE/NOT_ENTITLED precision.
- Internal family holdout robustness remains an internal proxy and is not equivalent to external (VitaminC/Climate-FEVER/naturalistic) generalization.
- If `--stage45c-entitled-labels` or `--stage45c-target-label` is misconfigured to a label absent from the internal training split, the corresponding term silently stays inactive (zero-valued) rather than failing — check `stage45c_loss_terms_active` and `stage45c_train_*_count` before interpreting results.
- Combining Stage45-C with `--stage45-use-family-holdout` on a family whose train split has zero SUPPORT or REFUTE rows leaves the corresponding term permanently inactive for that run.

## Next Stage

`Stage45-D: internal-only ablation comparing Stage45-C on/off across multiple Stage45-B1 holdout families, still restricted to internal controlled data, before any new external evaluation is proposed`
