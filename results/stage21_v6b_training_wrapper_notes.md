# Stage21-A3 v6B-Minimal Training Wrapper Notes

## Overview

**Stage21-A3** creates `scripts/train_controlled_v6b_minimal.py`, a minimal v6B training wrapper that integrates learnable temporal and predicate comparator alphas into the v5 training pipeline.

## Architecture

The wrapper is a **lightweight extension** of `train_controlled_v5.py`:
- Reuses v5 data loading, encoding, loss utilities
- Swaps `ContraMambaV5` → `ContraMambaV6BMinimal`
- Extracts and plumbs temporal/predicate flags
- Logs learned alphas in metadata

## Training Forward Call Site

**File:** `scripts/train_controlled_v6b_minimal.py`

**Lines 252-257 (training epoch loop):**
```python
# CRITICAL: Pass flags to v6b model forward
output = model(
    **v5.model_feature_inputs(train_inputs),
    temporal_mismatch_flags=train_temporal_flags,
    predicate_mismatch_flags=train_predicate_flags,
)
```

**Key difference from v5:**
- v5 passes only feature inputs: `model(**model_feature_inputs(train_inputs))`
- v6b passes feature inputs + flags: `model(..., temporal_mismatch_flags=..., predicate_mismatch_flags=...)`

**Loss consumption (unchanged from v5):**
```python
# Line 264-272: CE loss
losses = v5.controlled_losses(output, train_inputs, indices, weighted_label_loss)

# Line 274-280: Pairwise loss (if enabled)
pairwise_losses = intervention_pairwise_losses(
    output,  # <- uses output["logits"] (FINAL logits from v6b)
    ...
)

# Both consume output["logits"], which is final_logits from v6b_minimal
```

## How Flags Are Aligned to Records

**Flag Extraction (lines 182-192):**
```python
def extract_flags(records, flag_source, device):
    if flag_source == "stage15_probe_type":
        temporal_flags = temporal_mismatch_flags_from_probe(records, device)
        predicate_flags = predicate_mismatch_flags_from_probe(records, device)
    elif flag_source == "controlled_heuristic":
        temporal_flags = temporal_mismatch_flags_none(records, device)
        predicate_flags = predicate_mismatch_flags_from_intervention_type(records, device)
    ...
    return temporal_flags, predicate_flags
```

**Alignment Strategy (lines 212-220):**
```python
# Extract flags aligned to train/dev records BEFORE training
train_temporal_flags, train_predicate_flags = extract_flags(
    train_records, args.flag_source, device
)
dev_temporal_flags, dev_predicate_flags = extract_flags(
    dev_records, args.flag_source, device
)

# Flags are 1:1 aligned to records by iteration order
# train_records[i] paired with train_temporal_flags[i]
```

**Why alignment is safe:**
1. Records are split deterministically: `split_by_pair_id(records, ...)`
2. Flags extracted from same records: `extract_flags(train_records, ...)`
3. Both use same iteration order → 1:1 pairing
4. Flags passed in epoch loop with same records

## Why Final Logits Are Used for Loss

**v6b_minimal architecture:**
```
base_logits = decision_head(...)  # [batch, 3]
final_logits = base_logits.clone()

if temporal_mismatch_flags[i]:
    final_logits[i, 0] -= alpha_temporal
    final_logits[i, 1] += alpha_temporal
    final_logits[i, 2] -= alpha_temporal

if predicate_mismatch_flags[i]:
    # same modulation
    ...

return {
    "logits": final_logits,  # <- ONLY logits in output
    "base_logits": base_logits,  # diagnostic only
    ...
}
```

**Loss consumption:**
```python
# CE loss
F.cross_entropy(output["logits"], final_labels)
# ↑ final_logits from v6b

# Pairwise loss
intervention_pairwise_losses(output, ...)
# ↑ uses output["logits"] internally (final_logits)

# Both consume FINAL logits, which are modulated for flagged examples
```

**Guarantee:**
- `output["logits"]` = final_logits (only logits path)
- base_logits never used for loss (diagnostic only)
- All downstream losses use final calibrated logits

## CLI Flags

**New flags for v6b:**
- `--use-temporal-comparator` (default True) — enable temporal alpha
- `--use-predicate-comparator` (default True) — enable predicate alpha
- `--flag-source {stage15_probe_type|controlled_heuristic|none}` (default controlled_heuristic)
  - `stage15_probe_type`: extract from stage15_probe_type field
  - `controlled_heuristic`: temporal=zeros, predicate=intervention_type=="predicate_swap"
  - `none`: all zeros (no modulation)
- `--max-train-records` (optional) — limit train size (for debugging)
- `--smoke` (flag) — smoke mode: epochs=2, max_train_records=16

**Inherited from v5:**
- `--backbone {dummy|mamba}`
- `--model-name`, `--max-length`
- `--epochs`, `--lr`, `--head-lr`, `--encoder-lr`
- `--freeze-encoder`, `--freeze-a-log`
- `--use-intervention-loss`, `--ranking-weight`
- `--lambda-*` (loss weights)
- `--output-json`, `--output-predictions-json`
- `--dev-ratio`, `--seed`, `--device`

## Metadata Saved

**In final report (lines 302-315):**
```python
"configuration": {
    "model_version": "v6b_minimal",
    "flag_source": args.flag_source,
    "alpha_temporal": 1.234,  # learned value
    "alpha_predicate": 1.456,  # learned value
    "temporal_flag_count": 47,  # # of temporal examples in train
    "predicate_flag_count": 51,  # # of predicate examples in train
    "final_logits_used": True,  # assertion
    "time_swap_used": False,  # assertion
    ...
}
```

**In prediction export (lines 330-341):**
Same metadata included for reproducibility.

## Constraints Maintained

✓ **No time_swap:** Never used in flag extraction
✓ **No loss_logits:** Model returns only "logits" (final)
✓ **No pairwise_logits:** Model returns only "logits" (final)
✓ **No product_final_loss:** V6B has no composer or product path
✓ **No product-protected loss routing:** Single loss path
✓ **Final logits contract:** output["logits"] = final_logits (only path)
✓ **base_logits diagnostic:** Never used for loss or prediction

## Smoke Mode

**Activation:** `--smoke` flag

**Overrides:**
- epochs = 2
- max_train_records = 16

**Use:** Local testing before Kaggle run

```bash
python scripts/train_controlled_v6b_minimal.py \
  --backbone dummy \
  --epochs 3 \
  --smoke \
  --flag-source controlled_heuristic
```

## Validation Commands for Kaggle

**1. Compile check:**
```bash
python -m py_compile scripts/train_controlled_v6b_minimal.py
```

**2. Smoke run (local or Kaggle):**
```bash
python scripts/train_controlled_v6b_minimal.py \
  --backbone dummy \
  --smoke \
  --flag-source controlled_heuristic \
  --output-json /tmp/smoke_report.json \
  2>&1 | head -100
```

**3. Single-seed training (dummy backbone):**
```bash
python scripts/train_controlled_v6b_minimal.py \
  --backbone dummy \
  --epochs 10 \
  --seed 17 \
  --flag-source controlled_heuristic \
  --output-json results/stage21_v6b_seed17_report.json \
  --output-predictions-json results/stage21_v6b_seed17_preds.json
```

**4. Multi-seed training (use in Stage21-B):**
```bash
for seed in 1 2 3; do
  python scripts/train_controlled_v6b_minimal.py \
    --backbone dummy \
    --epochs 50 \
    --seed $seed \
    --flag-source controlled_heuristic \
    --output-json results/stage21_v6b_seed${seed}_report.json \
    --output-predictions-json results/stage21_v6b_seed${seed}_preds.json
done
```

## Expected Output (Smoke Mode)

```
[SMOKE MODE] epochs=2, max_train_records=16
controlled v6b_minimal | backbone=dummy train=... dev=... flag_source=controlled_heuristic
run=single epoch=001 total=... | train final=... | dev final=...
run=single epoch=002 total=... | train final=... | dev final=...
FINAL_REPORT
{
  "configuration": {
    "model_version": "v6b_minimal",
    "alpha_temporal": 1.25,
    "alpha_predicate": 1.25,
    "temporal_flag_count": 0,
    "predicate_flag_count": X,
    "final_logits_used": true,
    "time_swap_used": false,
    ...
  },
  ...
}
```

## TODO for Stage21-B (Multi-Seed Training)

1. Run validation on Kaggle with `--smoke`
2. Confirm alphas are learned (not stuck at init)
3. Run multi-seed training (seeds 1/2/3)
4. Compare learned alphas to Stage20-C calibrated values (baseline 1.25)
5. Measure OOD accuracy on stage15 probe heldout
6. Analysis: plot alpha convergence, flag coverage, accuracy gains

## Differences from v5

| Aspect | V5 | V6B |
|--------|----|----|
| Model | ContraMambaV5 | ContraMambaV6BMinimal |
| Forward signature | `model(**features)` | `model(**features, temporal_mismatch_flags=..., predicate_mismatch_flags=...)` |
| Learnable params | none (heads only) | alpha_temporal_raw, alpha_predicate_raw |
| Logit path | single | single (but modulated) |
| Loss routing | standard v5 | unchanged (consumes final_logits) |
| Metadata | standard | + alpha values, flag counts |

## Differences from v6A

| Aspect | V6A | V6B |
|--------|-----|-----|
| Model | ContraMambaV6A | ContraMambaV6BMinimal |
| Composer | Yes (10 features, learned) | No |
| Product path | Yes (diagnostic) | No |
| product_final_loss | Yes | No |
| Alpha init | variable | fixed to 1.25 |
| Loss path | dual (final + product) | single (final only) |
| Flag plumbing | no | yes |

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Flag mismatch if records reordered | MEDIUM | Ensure extract_flags() uses exact same records |
| Alphas stuck at init | MEDIUM | Monitor in early epochs; check gradient flow |
| Backward compat with v5 scripts | LOW | Separate file; doesn't modify v5 |
| Loss consuming wrong logits | HIGH | Smoke test verifies no bypass keys |
