# Stage26-F: v7 Epoch Diagnostic and Collapse Diagnostic Fields

**Status:** Implementation-only. No training has been run. No experiment results.

**Files modified:** `scripts/train_controlled_v6b_minimal.py` only.
No model architecture changes. No training behavior changes. No loss changes. No checkpoint selection changes.

---

## 1. Purpose

Stage26-F adds two tiers of diagnostic instrumentation to the output JSON:

1. **Per-epoch dev history** (`v7_epoch_diagnostic_history`) — added in the initial Stage26-F patch.
   Stores per-epoch snapshots of dev accuracy, macro-F1, prediction distribution, and channel
   prob means. Useful for detecting label collapse trajectory and channel saturation over training.

2. **Best-epoch and final-epoch summaries** — added in this extended patch.
   Aggregated logit/prob distributions from the best and final training epochs,
   per-gold-label channel breakdowns, and collapse detection flags. All derived from
   already-computed tensors; no new forward passes.

All fields are reporting-only. They have no effect on training, loss computation, checkpoint
selection, or model weights.

---

## 2. New Root-Level Fields (v7 runs only)

These appear at the root of the output JSON for `--architecture v7_hierarchical` runs.
For v6B runs, these fields are absent (the spread dict is empty).

### 2a. Logit Summaries

**`v7_best_dev_logit_summary`**
Stats (mean/std/min/max) for v7 logit and prob keys from the unconstrained-best dev epoch.

**`v7_final_epoch_logit_summary`**
Stats from the final training epoch's dev output.

Both include:
| Key | Description |
|-----|-------------|
| `v7_entitlement_logit` | EntitlementGate output logit (raw) |
| `v7_entitlement_prob` | sigmoid(entitlement_logit) |
| `v7_polarity_support_logit` | PolarityChannel support logit |
| `v7_polarity_refute_logit` | PolarityChannel refute logit |
| `v7_temporal_prob` | TemporalChannelV2 probability (if temporal channel active) |
| `v7_frame_prob` | FrameGate probability |
| `v7_predicate_prob` | PredicateCoverageHead probability |
| `v7_sufficiency_prob` | SufficiencyGate probability |
| `logits_refute` | Final logit dim 0 (REFUTE) |
| `logits_not_entitled` | Final logit dim 1 (NOT_ENTITLED) |
| `logits_support` | Final logit dim 2 (SUPPORT) |

Each tensor entry has: `{"mean": float, "std": float, "min": float, "max": float}`

Derived scalar fields:
| Key | Formula |
|-----|---------|
| `ne_minus_support_mean` | mean(logits[:, 1] - logits[:, 2]) |
| `ne_minus_refute_mean` | mean(logits[:, 1] - logits[:, 0]) |
| `support_minus_refute_mean` | mean(logits[:, 2] - logits[:, 0]) |
| `entitlement_logit_mean` | mean(v7_entitlement_logit) |
| `entitlement_logit_std` | std(v7_entitlement_logit) |

### 2b. Per-Gold-Label Breakdown

**`v7_best_dev_per_gold_label_summary`**

For each gold label `REFUTE / NOT_ENTITLED / SUPPORT`, includes:

| Field | Description |
|-------|-------------|
| `count` | Number of dev examples with this gold label |
| `prediction_distribution` | Dict of predicted label → count (for this gold subset) |
| `mean_entitlement_prob` | Mean v7_entitlement_prob for this gold class |
| `mean_frame_prob` | Mean v7_frame_prob |
| `mean_predicate_prob` | Mean v7_predicate_prob |
| `mean_sufficiency_prob` | Mean v7_sufficiency_prob |
| `mean_temporal_prob` | Mean v7_temporal_prob (if temporal channel active) |
| `mean_polarity_support_logit` | Mean v7_polarity_support_logit |
| `mean_polarity_refute_logit` | Mean v7_polarity_refute_logit |
| `mean_logit_refute` | Mean logits[:, 0] for this gold class |
| `mean_logit_not_entitled` | Mean logits[:, 1] for this gold class |
| `mean_logit_support` | Mean logits[:, 2] for this gold class |

**Use:** detect channel-level confounders. E.g., if gold=REFUTE examples have
high `mean_entitlement_prob`, the model correctly entitles them but polarity is wrong.

### 2c. Collapse Detection and Recall Fields

| Field | Type | Description |
|-------|------|-------------|
| `v7_predicted_single_class` | bool or null | True if all predictions are the same class |
| `v7_predicted_majority_class` | str or null | Label name of the most-predicted class |
| `v7_predicted_majority_fraction` | float or null | Fraction of predictions in majority class |
| `v7_support_prediction_count` | int or null | Count of SUPPORT predictions |
| `v7_refute_prediction_count` | int or null | Count of REFUTE predictions |
| `v7_ne_prediction_count` | int or null | Count of NOT_ENTITLED predictions |
| `v7_support_recall` | float or null | From best_dev_metrics["per_label"]["SUPPORT"]["recall"] |
| `v7_refute_recall` | float or null | From best_dev_metrics["per_label"]["REFUTE"]["recall"] |
| `v7_ne_recall` | float or null | From best_dev_metrics["per_label"]["NOT_ENTITLED"]["recall"] |

**Collapse detection:** `v7_predicted_single_class=true` indicates full label collapse.
`v7_predicted_majority_fraction > 0.8` with `v7_ne_recall < 0.1` indicates near-collapse
into NOT_ENTITLED (common in early-epoch v7 runs where entitlement_logit has not separated).

---

## 3. Data Sources and Epoch Provenance

| Field group | Source variable | Epoch |
|-------------|----------------|-------|
| `v7_best_dev_logit_summary` | `_best_dev_output_v7` (captured in epoch loop) | Unconstrained best epoch |
| `v7_final_epoch_logit_summary` | `dev_output` (last iteration) | Final training epoch |
| `v7_best_dev_per_gold_label_summary` | `_best_dev_output_v7` + `dev_inputs["final_labels"]` | Unconstrained best epoch |
| Collapse / recall fields | `best_dev_metrics` | After all selection overrides |

**Important:** If TD-constrained or preservation-constrained checkpoint selection overrides
the selected epoch, `best_dev_metrics` reflects the constrained epoch, but the logit/per-gold
summaries are from the unconstrained-best epoch (highest `select_metric`). In practice:
- Collapse/recall fields are always from the actually-selected checkpoint.
- Logit summaries may be from a different epoch when constrained selection applies.
- For most v7 runs (no constrained selection), all fields align to the same epoch.

---

## 4. Implementation Details

### 4a. Module-Level Helpers Added

| Function / Constant | Purpose |
|---------------------|---------|
| `_V7_DIAG_CAPTURE_KEYS` | Tuple of output dict keys to clone per epoch |
| `_v7_capture_dev_output(out)` | Clone v7 keys to CPU for offline diagnostics |
| `_v7_tensor_stats(t)` | Return mean/std/min/max for a tensor |
| `_v7_make_logit_summary(out)` | Build the full logit stats dict |
| `_v7_make_per_gold_summary(out, final_labels)` | Build per-gold-label breakdown |

All helpers are pure-compute: no optimizer, no backward, no model forward.

### 4b. Epoch Loop Change (reporting only)

Inside `if score > best_score:`, after the existing `best_trainable_state` capture:
```python
if args.architecture == "v7_hierarchical":
    _best_dev_output_v7 = _v7_capture_dev_output(dev_output)
```
This clones only the 11 diagnostic keys (tensors + one string) to CPU. The clone happens
only when a new best score is achieved, matching the existing `best_state` capture pattern.

### 4c. Post-Loop Diagnostics Block

Added after all selection overrides (`_tc_constrained_applied`, `_pcs_applied`) and before
`_run_audit_ledger`. Computes `_v7_ext_diagnostics` dict (empty for v6B) from already-computed
variables, then spreads it into `report = {…}`.

### 4d. Zero New Forward Passes

`_best_dev_output_v7` is cloned at the moment `dev_output` is computed (inside the epoch loop),
at no extra cost. `dev_output` for the final epoch is accessed by reading the Python variable
after the for-loop (valid in Python; no re-computation). No new `model(...)` calls.

---

## 5. v6B Compatibility

For `--architecture v6b_minimal` runs:
- `_v7_ext_diagnostics = {}` (empty dict)
- Spreading `**{}` in `report = {…}` adds no keys
- None of the 12 new fields appear in the output JSON
- No v6B code paths are modified

---

## 6. CE / Loss / Training Behavior Invariants

All invariants from Stage26-A through Stage26-D are preserved:

- CE uses `output["logits"]` (unchanged)
- No `loss_logits` introduced
- No `pairwise_logits` introduced
- No `output.get("loss_logits", output["logits"])` pattern
- Stage15 not used for any training/selection/aux-loss/threshold/scale purpose
- `time_swap` not used in main clean train/eval data
- Checkpoint selection unchanged
- Optimizer step unchanged
- Model architecture unchanged

---

## 7. Fields Already Present (Stage26-D, preserved)

The following root-level aliases were added in Stage26-D via `lift_report_aliases()` and
remain unchanged:
```
model_version, architecture, use_v7_hierarchical,
v7_final_logit_composition, v7_channel_output_keys, v7_aux_losses_active,
v7_disable_frame_channel, v7_disable_predicate_channel, v7_disable_sufficiency_channel,
v7_disable_temporal_channel, v7_flat_arbiter, v7_no_entitlement_polarity_conditioning,
v7_no_aux_losses, stage15_used_for_v7_training, stage15_used_for_v7_selection,
stage15_used_for_v7_aux_loss_targets, time_swap_used_in_v7_main_clean_data,
best_dev_acc, best_dev_macro_f1
```

The Stage26-F `v7_epoch_diagnostic_history` field from the initial patch is also preserved.
