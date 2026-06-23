# Stage21-A2 Flag Plumbing Notes

## Overview

**Stage21-A2** establishes how to derive `temporal_mismatch_flags` and `predicate_mismatch_flags` from training records, and plumb them into `ContraMambaV6BMinimal.forward()` during training.

The approach supports two data sources:
1. **stage15_slot_sensitivity_probe.jsonl** (authoritative) — has `stage15_probe_type`
2. **controlled_v5_v3_without_time_swap.jsonl** (fallback) — has `intervention_type` (heuristic only)

## Flag Extraction

### Temporal Mismatch Flags

**Source 1: stage15 probe (preferred)**
```python
temporal_flags = torch.tensor([
  1 if record.get("stage15_probe_type") == "temporal_mismatch" else 0
  for record in records
], dtype=torch.long)
```

**Source 2: controlled data (fallback)**
```python
# CANNOT use time_swap (explicitly forbidden)
# No Stage17 temporal detector integrated into training
# Default: return all zeros
temporal_flags = torch.zeros(len(records), dtype=torch.long)
```

**Why no time_swap fallback:**
- Task constraint: "Do not use `time_swap`"
- time_swap indicates a controlled corruption, not ground truth temporal mismatch
- Proper temporal detection requires Stage17 external comparator (not yet integrated)

### Predicate Mismatch Flags

**Source 1: stage15 probe (preferred)**
```python
predicate_flags = torch.tensor([
  1 if record.get("stage15_probe_type") == "predicate_mismatch" else 0
  for record in records
], dtype=torch.long)
```

**Source 2: controlled data (fallback, heuristic only)**
```python
predicate_flags = torch.tensor([
  1 if record.get("intervention_type") == "predicate_swap" else 0
  for record in records
], dtype=torch.long)
```

⚠️ **WARNING:** Fallback using `intervention_type == "predicate_swap"` is a heuristic approximation.
- Assumes predicate_swap examples exhibit predicate_mismatch errors.
- Does NOT use Stage20 lexical detector (that requires claim/evidence parsing in training loop).
- For production Stage21-B, provide flags from stage15 probe or Stage20 detector.

## Implementation

**File:** `src/contramamba/comparator_flags.py`

**Functions:**
- `temporal_mismatch_flags_from_probe(records, device)` — stage15 extraction
- `temporal_mismatch_flags_none(records, device)` — safe default (all zeros)
- `predicate_mismatch_flags_from_probe(records, device)` — stage15 extraction
- `predicate_mismatch_flags_from_intervention_type(records, device)` — controlled heuristic (documented)

## Training Integration Plan (A3)

When creating `scripts/train_controlled_v6b_minimal.py`:

```python
# In run_training(), before epoch loop:

# Load/derive flags based on record structure
if has_stage15_probe_type(train_records):
    temporal_flags = temporal_mismatch_flags_from_probe(train_records, device)
    predicate_flags = predicate_mismatch_flags_from_probe(train_records, device)
else:
    # Controlled data: no temporal detection, heuristic predicate
    temporal_flags = temporal_mismatch_flags_none(train_records, device)
    predicate_flags = predicate_mismatch_flags_from_intervention_type(train_records, device)
    print("[WARN] Using heuristic predicate flags from intervention_type")

# In epoch loop (line 832 equivalent):
output = model(
    **model_feature_inputs(train_inputs),
    temporal_mismatch_flags=temporal_flags,
    predicate_mismatch_flags=predicate_flags,
)

# Rest of training loop unchanged
```

## Smoke Test Results

**File:** `scripts/smoke_stage21_v6b_flag_plumbing.py`

Verifies:
1. ✓ Flags extractable from stage15 probe (if available)
2. ✓ Flags extractable from controlled data (fallback heuristic)
3. ✓ Flag tensors have correct shape `[batch]` and dtype `torch.long`
4. ✓ v6b_minimal forward accepts flags without error
5. ✓ Flagged rows show logit modulation (final_logits != base_logits)
6. ✓ Unflagged rows unchanged (final_logits == base_logits)
7. ✓ Output contains `temporal_flag_count` and `predicate_flag_count`
8. ✓ No bypass keys in output

## Key Design Decisions

### Why separate helper functions?
- Enables reuse in training script, eval script, and analysis
- Makes fallback strategy explicit and testable
- Allows future integration of Stage17/Stage20 detectors

### Why no Stage17 temporal detector in training?
- Stage17 is a post-processing diagnostic (not integrated into model)
- Would require online temporal expression extraction during training
- Proper integration deferred to Stage21-B full training

### Why keep predicate fallback heuristic?
- Controlled training data has no stage15_probe_type
- Predicate_swap examples are a reasonable heuristic for controlled smoke
- Documented as "heuristic only" and NOT used for evaluation

### Why dtype=torch.long?
- Matches v6b_minimal forward signature (expects binary flags)
- Compatible with `.bool()` casting in model modulation code
- Consistent with final_labels tensor dtype

## Caveats

- **Smoke only:** Tests tiny batches, not full training dynamics
- **No Stage17 integration:** Temporal flags default to zero for controlled data
- **Heuristic predicate:** Fallback uses intervention_type, not true detector
- **Single-stage pipeline:** Flags computed once, not updated during training

## TODO for Stage21-A3 (Training Wrapper)

1. Create `scripts/train_controlled_v6b_minimal.py`
   - Reuse `scripts/train_controlled_v5.py` structure
   - Add flag extraction before `run_training()`
   - Detect data source (stage15 vs controlled) and choose flag strategy
   - Pass flags to model in training loop

2. Handle both dummy and Mamba backbones

3. Extend `prediction_records()` to log learned alphas and flag counts

4. Test on controlled_v5 data with dummy backbone (smoke)

5. Prepare for Stage21-B: multi-seed run with full training dynamics
