# Stage26-G: v7 Polarity/Entitlement Stabilization Options

**Status:** Implementation-only. No training has been run. No experiment results.

**Files modified:**
- `src/contramamba/modeling_v7_hierarchical.py` — `ne_bias` initialization
- `scripts/train_controlled_v6b_minimal.py` — CLI flags, loss computation, audit fields

No data files changed. No v6B paths touched.

---

## 1. Motivation: Stage26-F Diagnosis

Stage26-F (seed1, 5 epochs) observed the following best-checkpoint result:

```
best_epoch = 2
best_dev_acc = 0.75
best_dev_macro_f1 = 0.3244720965309201
v7_predicted_majority_class = NOT_ENTITLED
v7_predicted_majority_fraction = 0.9833333333333333
v7_support_prediction_count = 0
v7_refute_prediction_count = 12
v7_ne_prediction_count = 708
v7_support_recall = 0.0
v7_refute_recall = 0.06666666666666667
v7_ne_recall = 0.9888888888888889
```

Epoch-by-epoch collapse trajectory:
- Epoch 1: {REFUTE: 720}
- Epoch 2 (best): {NE: 708, REFUTE: 12}
- Epoch 3: {NE: 120, SUPPORT: 600}
- Epoch 4: {NE: 183, SUPPORT: 537}
- Epoch 5: {NE: 167, REFUTE: 540, SUPPORT: 13}

The model never stabilizes. It oscillates between single-class collapse modes across epochs.
The 0.75 accuracy at epoch 2 reflects NE dominance matching the majority NOT_ENTITLED class,
not actual entitlement/polarity discrimination.

**Root cause hypothesis:**
1. `entitlement_prob` ≈ 0.56 across all gold classes at best epoch — EntitlementGate is not
   separating titled from untitled examples. Without a clean entitlement signal, the polarity
   gate has no meaningful conditioning to work from.
2. `ne_bias` initialized to 0.0 — in the v7 logit formula `ne_score = -entitlement_logit + ne_bias`,
   a zero bias combined with a near-uniform `entitlement_logit` means `ne_score ≈ -0.5`, which
   is still competitive against the polarity scores in early training.
3. No direct loss pressure on entitlement separation — the EntitlementGate trains only through
   CE backprop, which can be satisfied early by collapsing to the majority class.

---

## 2. Why Full Training Remains Blocked

Stage26-F produced only 1 seed × 5 epoch diagnostic data. Before committing to a full
multi-seed run:

- The collapse oscillation indicates the current setup has not found a stable training regime.
- Running Stage26-G diagnostics (5 epochs, seed1) with each stabilization option independently
  is cheaper than a full multi-seed run and will indicate whether the options work.
- Full training should follow only after at least one diagnostic run shows non-collapsed dev
  predictions and non-trivial recall across all three classes.

---

## 3. Stabilization Options Added

All options are disabled by default (weights 0.0 / flags False). They are activated via CLI
flags. They apply only to `--architecture v7_hierarchical`. v6B behavior is unchanged.

### 3a. `ne_bias` Negative Initialization

**CLI:** `--v7-initial-ne-bias` (default: `-0.5`)

**What it does:** Initializes the learnable `ne_bias` scalar in `ContraMambaV7Hierarchical`
to `-0.5` rather than `0.0`. With the v7 composition formula:
```
ne_score = -entitlement_logit + ne_bias
```
a bias of `-0.5` means the NE score starts at a disadvantage relative to the polarity scores
in early training, reducing the probability that gradient updates collapse to NE from step 0.

**Why not 0.0:** Stage26-F diagnostics showed `entitlement_prob ≈ 0.56` uniformly at epoch 2
(best checkpoint), meaning `entitlement_logit ≈ 0.24`. With `ne_bias=0.0`:
- `ne_score ≈ -0.24 + 0.0 = -0.24`
- `support_score = entitlement_logit + polarity_support ≈ 0.24 + polarity`
- `refute_score = entitlement_logit + polarity_refute ≈ 0.24 + polarity`

The NE score is near the polarity scores and wins when polarity is uncertain.
With `ne_bias=-0.5`: `ne_score ≈ -0.74`, which requires the polarity scores to be unusually
negative for NE to win early in training.

**Constraint compliance:** This changes only initialization, not architecture. CE still uses
`output["logits"]`. No OOD. No Stage15.

---

### 3b. Polarity Margin Auxiliary Loss

**CLI flags:**
- `--v7-use-polarity-margin-loss` (store_true, default: False)
- `--v7-polarity-margin-loss-weight` (float, default: 0.0)
- `--v7-polarity-margin` (float, default: 0.5)

**What it does:** Hinge margin loss on `v7_polarity_support_logit` and `v7_polarity_refute_logit`.
Applied to gold SUPPORT and REFUTE examples only. NOT_ENTITLED examples excluded.

For gold SUPPORT:
```
loss = relu(margin - (support_logit - refute_logit))
```
For gold REFUTE:
```
loss = relu(margin - (refute_logit - support_logit))
```

The loss is 0 if the correct polarity logit already dominates by `margin` or more.
It becomes positive only when the wrong polarity logit is competitive.

**Why this helps:** The EntitlementGate and polarity channel are jointly responsible for
SUPPORT vs REFUTE discrimination. If the entitlement signal is weak (as in Stage26-F),
polarity separation cannot rely on entitlement-gated conditioning. A direct margin loss
on the polarity logits gives the PolarityChannelV7 a training signal that is not diluted
by the entitlement gate's confusion.

**Constraint compliance:**
- CE unchanged: still uses `output["logits"]` (the hierarchical composition).
- No `loss_logits`. No `pairwise_logits`. No `output.get("loss_logits", ...)`.
- NOT_ENTITLED excluded: the loss is 0 for NE examples, not a soft penalty on NE.
- Clean train data only. No Stage15. No OOD. No time_swap.

---

### 3c. Entitlement BCE Auxiliary Loss

**CLI flags:**
- `--v7-use-entitlement-bce-loss` (store_true, default: False)
- `--v7-entitlement-bce-loss-weight` (float, default: 0.0)
- `--v7-entitlement-bce-pos-weight` (float, default: 1.0)

**What it does:** Binary cross entropy on `v7_entitlement_logit` with ground-truth entitled
target derived from clean-data gold labels:
- `entitled=1` for gold SUPPORT (label=2) and gold REFUTE (label=0)
  — these examples require a polarity judgment, so evidence is entitling
- `entitled=0` for gold NOT_ENTITLED (label=1)
  — evidence fails to entitle a polarity judgment

**Why this helps:** Stage26-F shows EntitlementGate is not separating gold classes
(`entitlement_prob ≈ 0.56` for all three). BCE gives the EntitlementGate a direct, clean
signal aligned with the epistemic hierarchy:
- NOT_ENTITLED = evidence insufficient to warrant a polarity judgment
- SUPPORT/REFUTE = evidence sufficient, polarity direction distinguishes them

The `pos_weight` can be increased (e.g., 2.0–3.0) if SUPPORT+REFUTE is underrepresented
relative to NOT_ENTITLED in the training set.

**Constraint compliance:**
- Supervised from clean train labels only. No Stage15. No OOD. No time_swap.
- CE unchanged. The BCE is an auxiliary loss added to `total_loss`.
- `entitlement_logit` is an intermediate computation; BCE on it does not change the
  forward contract for `output["logits"]`.

---

### 3d. Entitled Class-Balanced CE

**CLI flags:**
- `--v7-use-entitled-class-balanced-ce` (store_true, default: False)
- `--v7-entitled-class-balanced-ce-weight` (float, default: 0.0)

**What it does:** Auxiliary CE loss using `v7_polarity_logits` (shape `[B, 2]`, columns
`[refute_logit, support_logit]`) on the subset of examples where gold label is SUPPORT or
REFUTE. Local labels: REFUTE (gold=0) → local 0; SUPPORT (gold=2) → local 1.
NOT_ENTITLED examples are excluded entirely.

**Why this helps:** In the standard CE over `output["logits"]`, the SUPPORT and REFUTE
gradient signal is mixed with the entitlement gate's contribution. By isolating the polarity
logits and computing a 2-class CE over only entitled examples, this loss anchors the polarity
direction signal independently of EntitlementGate confusion.

**Constraint compliance:**
- Uses `v7_polarity_logits`, NOT `output["logits"]`. No `loss_logits` created.
- NOT_ENTITLED examples excluded from this loss.
- Clean train data only. No Stage15. No OOD. No time_swap.

---

## 4. Report/Audit Fields Added

### 4a. Config block (lifted to report root via `_LIFT_CONFIG_KEYS`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `v7_use_polarity_margin_loss` | bool | False | Whether polarity margin loss is enabled |
| `v7_polarity_margin_loss_weight` | float | 0.0 | Weight for polarity margin loss |
| `v7_polarity_margin` | float | 0.5 | Margin threshold for hinge loss |
| `v7_use_entitlement_bce_loss` | bool | False | Whether entitlement BCE is enabled |
| `v7_entitlement_bce_loss_weight` | float | 0.0 | Weight for entitlement BCE loss |
| `v7_entitlement_bce_pos_weight` | float | 1.0 | BCE pos_weight for entitled class |
| `v7_use_entitled_class_balanced_ce` | bool | False | Whether entitled class-balanced CE enabled |
| `v7_entitled_class_balanced_ce_weight` | float | 0.0 | Weight for entitled class-balanced CE |
| `v7_initial_ne_bias` | float | -0.5 | Initial value of learnable ne_bias |

### 4b. Audit ledger (per-epoch raw and weighted):

| Key | Description |
|-----|-------------|
| `v7_polarity_margin_loss` | Raw polarity margin loss (0.0 when disabled) |
| `v7_entitlement_bce_loss` | Raw entitlement BCE loss (0.0 when disabled) |
| `v7_entitled_class_balanced_ce_loss` | Raw entitled class-balanced CE loss (0.0 when disabled) |

Each appears in both `_audit_per_epoch_raw` (raw value) and `_audit_per_epoch_weighted`
(raw × weight). For disabled losses, raw=0.0, weighted=0.0.

### 4c. `v7_aux_losses_active` logic updated:

Now reflects actual activity:
```python
v7_aux_losses_active = (
    architecture == "v7_hierarchical"
    and not v7_no_aux_losses
    and (
        (v7_use_polarity_margin_loss and v7_polarity_margin_loss_weight > 0.0)
        or (v7_use_entitlement_bce_loss and v7_entitlement_bce_loss_weight > 0.0)
        or (v7_use_entitled_class_balanced_ce and v7_entitled_class_balanced_ce_weight > 0.0)
    )
)
```

---

## 5. Safety Constraints (All Preserved)

| Constraint | Preserved by |
|------------|-------------|
| v6B behavior unchanged | All changes guarded by `args.architecture == "v7_hierarchical"` |
| CE uses `output["logits"]` | Aux losses are additive; CE target unchanged |
| No `loss_logits` | Not introduced anywhere |
| No `pairwise_logits` | Not introduced anywhere |
| No OOD for loss/calibration | All loss targets from clean train labels only |
| No Stage15 for any purpose | Hard-wired False in `_active_training_losses` |
| No `time_swap` in clean train/eval | Data pipeline unchanged |
| Checkpoint selection unchanged | No changes to selection logic |
| No post-hoc OOD thresholds | Not introduced |
| Predicate mismatch ≠ frame failure | Architecture unchanged |
| Temporal mismatch ≠ polarity | Architecture unchanged |

---

## 6. Recommended Next Run

**5 epochs, seed 1, with `ne_bias=-0.5` + polarity margin loss + entitlement BCE:**

```
python scripts/train_controlled_v6b_minimal.py \
  --architecture v7_hierarchical \
  --seed 1 \
  --num-epochs 5 \
  --v7-initial-ne-bias -0.5 \
  --v7-use-polarity-margin-loss \
  --v7-polarity-margin-loss-weight 0.3 \
  --v7-polarity-margin 0.5 \
  --v7-use-entitlement-bce-loss \
  --v7-entitlement-bce-loss-weight 0.3 \
  --v7-entitlement-bce-pos-weight 1.0 \
  [... other flags as in Stage26-F ...]
```

**What to look for in v7_epoch_diagnostic_history:**
- `entitlement_prob` should diverge across gold classes by epoch 2
  (gold SUPPORT/REFUTE should show higher values than gold NOT_ENTITLED)
- Epoch-by-epoch prediction distribution should NOT be single-class
- `v7_refute_recall` and `v7_support_recall` should be > 0.0 at best checkpoint

**Escalation path:**
- If collapse persists: increase BCE weight to 0.5–1.0, check `entitlement_logit_std`
- If polarity direction still wrong: increase margin weight or raise margin to 1.0
- If all three classes predicted but macro-F1 < 0.40: proceed to full multi-seed run

---

## 7. Alternative: Entitled Class-Balanced CE

If margin + BCE runs still show NE collapse, add:
```
  --v7-use-entitled-class-balanced-ce \
  --v7-entitled-class-balanced-ce-weight 0.2
```

This provides a third signal path targeting the polarity channel directly.
Run at most one combination at a time to attribute improvement correctly.
