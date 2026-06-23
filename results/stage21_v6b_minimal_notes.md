# Stage21-A v6B-Minimal Implementation Notes

## Overview

**Stage21-A** implements v6B-minimal: a minimal trainable design that integrates learnable slot-level comparator alphas (temporal and predicate) into the v5 forward pass. The key principle is **final logit modulation**: base logits from the decision head are modulated once before loss and prediction, with no dual loss paths or bypass routes.

## Architecture

### Base: V5 Unchanged
- Mamba encoder → frame gate → predicate coverage → sufficiency → polarity → decision head
- Decision head produces base logits: [SUPPORT, NOT_ENTITLED, REFUTE]
- All slot supervision (frame, predicate, sufficiency, polarity) losses unchanged

### Addition: Learnable Comparator Alphas
Two new learnable scalar parameters:
- `alpha_temporal_raw` (Parameter, requires_grad=True)
  - Constrained: `alpha_temporal = softplus(alpha_temporal_raw)`
  - Initialized: raw value such that softplus ≈ 1.25 (empirical from Stage18/20)
- `alpha_predicate_raw` (Parameter, requires_grad=True)
  - Constrained: `alpha_predicate = softplus(alpha_predicate_raw)`
  - Initialized: raw value such that softplus ≈ 1.25

### Logit Modulation (After Decision Head)
```
base_logits = decision_head(...)  shape [batch, 3]
final_logits = base_logits.clone()

if temporal_mismatch_flags active at index i:
  final_logits[i, 0] -= alpha_temporal  # SUPPORT
  final_logits[i, 1] += alpha_temporal  # NOT_ENTITLED
  final_logits[i, 2] -= alpha_temporal  # REFUTE

if predicate_mismatch_flags active at index i:
  final_logits[i, 0] -= alpha_predicate  # SUPPORT
  final_logits[i, 1] += alpha_predicate  # NOT_ENTITLED
  final_logits[i, 2] -= alpha_predicate  # REFUTE
```

### Return Contract
**Critical:** Final logits are the only logits returned as `output["logits"]`.
```python
output = {
    "logits": final_logits,              # ← FINAL (consumed by all losses)
    "base_logits": base_logits,          # ← diagnostic only
    "predictions": final_logits.argmax(), # ← derived from final logits
    "alpha_temporal": model.alpha_temporal(),
    "alpha_predicate": model.alpha_predicate(),
    "temporal_flag_count": count,
    "predicate_flag_count": count,
    "final_logits_used": True,           # ← assertion
    **frame, **predicate, **sufficiency, **polarity,
    "loss": total_loss,
    **losses,  # label_loss, frame_loss, predicate_loss, sufficiency_loss, polarity_loss
}
```

## Implementation Details

**File:** `src/contramamba/modeling_v6b_minimal.py`

**Class:** `ContraMambaV6BMinimal(nn.Module)`

**Constructor Parameters:**
- `use_temporal_comparator: bool` — whether to initialize and apply temporal alpha
- `use_predicate_comparator: bool` — whether to initialize and apply predicate alpha
- `alpha_temporal_init: float = 1.25` — initial softplus target for temporal
- `alpha_predicate_init: float = 1.25` — initial softplus target for predicate
- All other V5 parameters (model_name, frame_size, etc.) unchanged

**Optional Forward Inputs:**
- `temporal_mismatch_flags: Tensor | None` — [batch] binary flags, 1 = apply temporal modulation
- `predicate_mismatch_flags: Tensor | None` — [batch] binary flags, 1 = apply predicate modulation

**Loss Consumption:**
- CE loss: `F.cross_entropy(output["logits"], final_labels)` ← final_logits
- Pairwise losses: `intervention_pairwise_losses(output, ...)` ← uses `output["logits"]`
- Frame/predicate/sufficiency/polarity losses: unchanged from V5

## Smoke Test

**File:** `scripts/smoke_stage21_v6b_minimal_forward.py`

Verifies:
1. ✓ Model import works
2. ✓ Forward pass on synthetic batch
3. ✓ Alpha parameters initialized near 1.25, trainable
4. ✓ Temporal modulation: support-= alpha, not_entitled+= alpha, refute-= alpha
5. ✓ Predicate modulation: same as temporal (slot-agnostic)
6. ✓ Predictions derived from final_logits (not base_logits)
7. ✓ No loss bypass keys (loss_logits, pairwise_logits)
8. ✓ final_logits_used=True flag set

**Run:**
```bash
python -m py_compile src/contramamba/modeling_v6b_minimal.py
python -m py_compile scripts/smoke_stage21_v6b_minimal_forward.py
python scripts/smoke_stage21_v6b_minimal_forward.py
```

## Key Design Decisions

### Why No Dual Loss Paths?
- V6A introduced composer + product_loss (dual logit paths for diagnostics)
- v6B-minimal strips this: single final_logits → single loss path
- Avoids risk of training consuming wrong logits
- Simpler, clearer, safer

### Why Inverse Softplus for Initialization?
- Stage18/20 calibration found optimal alphas near 1.25
- Softplus ensures alpha > 0 always (prevents negative penalties)
- Inverse softplus: `x = log(exp(target) - 1)` such that softplus(x) ≈ target
- Example: `softplus(log(exp(1.25) - 1)) ≈ 1.25`

### Why Both Temporal AND Predicate?
- Stage17-20 showed both comparators independently fix different OOD failures
- Temporal: fixes temporal_mismatch (false-entitled from entailment over time)
- Predicate: fixes predicate_mismatch (false-entitled from conflicting predicates)
- v6B-minimal allows both to be active simultaneously, learning joint optimal alphas

### Why Optional Flags at Forward Time?
- Training script provides flags when available (from probe or external detector)
- At eval on unseen data, flags may not be available → apply best-learned defaults
- Flags can be single-model (static per dataset) or per-example (dynamic)

## Insertion Point Guarantee

**Final logits injection point is safe** because:

1. **Single return path:** `output["logits"] = final_logits` only
2. **All losses consume same logits:**
   - CE: `F.cross_entropy(output["logits"], ...)`
   - Pairwise: `intervention_pairwise_losses(output, ...)` → uses `output["logits"]`
   - Frame/predicate/sufficiency/polarity: unchanged (use gate logits, not final)
3. **Predictions consistent:** `argmax(final_logits)` == `output["predictions"]`
4. **Diagnostics safe:** `base_logits` returned but never used for loss
5. **No bypass keys:** verified by smoke test

## TODO for Stage21-A2 (Training Integration)

1. Create `scripts/train_controlled_v6b_minimal.py` (wrapper like v6a)
   - Import ContraMambaV6BMinimal
   - Add CLI flags: `--use-temporal-comparator`, `--use-predicate-comparator`, `--learnable-comparator-alpha`
   - Provide flags from probe or external detector at training time
   - Track learned alphas in metadata

2. Add temporal/predicate flag extraction
   - From `stage15_slot_sensitivity_probe.jsonl` (stage15_probe_type)
   - From external comparators (Stage17 temporal, Stage20 predicate)

3. Verify all downstream losses consume final_logits correctly
   - Spot-check pairwise_losses() with v6b output
   - Verify metrics computed correctly

4. Multi-seed training run
   - Seed 1, 2, 3 on controlled intervention data
   - Compare learned alphas to Stage18/20 calibrated values
   - Verify OOD accuracy improvements

5. Heldout validation
   - Apply trained v6b_minimal to stage15 probe
   - Measure false-entitled reduction vs. baseline

## Risks

- **Flag availability at eval:** If flags unavailable, modulation is not applied. Mitigate: train with flags, eval with flags.
- **Alpha divergence:** Learned alphas may drift far from 1.25. Mitigate: monitor in training, use L2 regularization if needed.
- **Slot interaction:** Temporal and predicate both modify NOT_ENTITLED. May over-correct. Mitigate: test on multi-slot examples.
- **Generalization:** v6b_minimal trained on controlled data, may not generalize to natural OOD. Expected—Stage21-B will measure this.

## Caveat

v6B-minimal is **trainable post-processing**, not an architectural innovation. If learned alphas converge to 0 or near-random values, it suggests the base model's errors are not principally slot-level and cannot be fixed by logit shifting alone. This would be valuable negative result for Stage21-B analysis.
