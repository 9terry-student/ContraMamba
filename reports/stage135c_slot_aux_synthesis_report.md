# Stage135-C Slot Auxiliary Synthesis

## 1. Summary decision

Decision: `STAGE135C_SLOT_AUX_INTEGRATED_BUT_CLEAN_SWEEP_UNINFORMATIVE`

Stage135-C records that the best slot mismatch auxiliary path is integrated and audit-visible, but that the available clean-training sweep is not informative for clean performance claims. The slot auxiliary should therefore be treated as a functional diagnostic or candidate risk signal, not as evidence of clean-dev improvement.

The main conclusion is:

- Slot mismatch signal exists robustly in nonlinear pairwise-comparative geometry.
- Stage135 auxiliary integration is functional and visible in reports.
- Stage135-B should not be used as evidence for clean performance improvement or harm because the baseline itself collapsed.
- Final logits are not modified by the Stage135 slot head.
- The Stage128 location-slot guard remains off unless explicitly enabled.

## 2. Stage134-F diagnostic result

Stage134-E/F showed that linear heads were insufficient for recovering the best slot mismatch signal. The robust signal appeared when using pairwise-comparative features with nonlinear capacity:

- Input mode: `pooled_pair_absdiff_product`
- Head type: `mlp`
- Decision: `STAGE134F_SLOT_SIGNAL_ROBUST_USEFUL`

Stage134-F confirmatory sweep results:

- Overall dev AUROC mean: `0.812794`
- Overall dev AUROC min: `0.781250`
- Overall dev AUPRC mean: `0.835267`
- Scenario holdout mean AUROC: `0.833333`
- Slot holdout mean AUROC: `0.803241`
- Random split mean AUROC: `0.801809`

This supports the conclusion that the slot mismatch signal is real and robust when expressed through nonlinear pairwise-comparative geometry, specifically the pooled pair, absolute difference, and product representation.

## 3. Stage135-A/A2 integration result

Stage135-A/A2 added and audited `--stage135-use-best-slot-aux` using the Stage134-F best configuration:

- Input mode: `pooled_pair_absdiff_product`
- Head type: `mlp`
- Slot mismatch head input dimension: `512`
- Loss weight: `0.1`

The Stage135-A2 relaxed audit passed:

- Decision: `STAGE135A2_REPORT_AUDIT_PASSED_RELAXED`
- Enabled: `true`
- Best slot aux enabled: `true`
- Raw slot mismatch loss: `0.692012`
- Weighted slot mismatch loss: `0.069201`
- Final logits modified by slot head: `false`
- Stage128 location-slot guard enabled: `false`

This means the Stage135 auxiliary integration is functional, persistent-report visible, and isolated from final-logit behavior.

## 4. Stage135-B partial sweep limitation

Stage135-B attempted a short controlled clean-training sweep, but the run was stopped early because the baseline itself collapsed.

Completed runs:

- `baseline_no_slot_aux`: seeds `1`, `2`, `3`
- `slot_aux_w005`: seed `1`

Baseline aggregate:

- Accuracy mean: `0.748299`
- Macro F1 mean: `0.285343`
- Prediction behavior: all or near-all `NOT_ENTITLED`

Slot auxiliary `w005` seed 1:

- Accuracy: `0.744898`
- Macro F1: `0.284600`
- Prediction behavior: all `NOT_ENTITLED`
- Weighted slot loss: approximately `0.033486`

Interpretation: there is no evidence of additional clean-dev harm from the slot auxiliary in this partial result, but the sweep is not informative because the baseline itself collapsed. It should not be used as evidence that the slot auxiliary improves clean performance, and it should not be used as strong evidence that the slot auxiliary harms clean performance.

Decision: `STAGE135B_SHORT_SWEEP_UNINFORMATIVE_BASELINE_COLLAPSE`

## 5. Safety and leakage policy

Stage135-C preserves the safety and leakage boundaries established by the prior stages:

- Final logits are not modified by the Stage135 slot head.
- Stage135 does not implicitly enable the Stage128 guard.
- The Stage128 guard remains off unless explicitly enabled.
- No external data is used for training.
- Time-swap data is not used in the main clean data.
- Stage15 is not used.

These constraints keep the slot mismatch auxiliary as an auxiliary training/reporting component rather than a direct routing mechanism for final predictions.

## 6. Recommendation for Stage136

Recommended next stage: `Stage136`

Do not run more lightweight clean-training sweeps under the same collapsed-baseline conditions. If slot risk is used next, evaluate it first as an export-only signal or as a guard/cap candidate with a separate guard evaluation, not as an immediate final-logit modifier.

Stage136 should avoid:

- Large multi-seed short sweeps with collapsed baselines.
- Routing slot mismatch directly into final logits without a separate guard evaluation.
- Enabling the Stage128 guard implicitly.
