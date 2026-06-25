# Stage26-C: v7 Forward-Contract Validation Utility

**Status:** Utility implemented. Static validation only. No forward passes have been run.
No training has been run. No experiment results exist yet.

**Date:** Stage26-C (follows Stage26-B output contract audit)

---

## 1. What Stage26-C Adds

Stage26-C fulfills the deferred action from Stage26-B:

> Stage26-C action: add a single one-shot validation call (not per-step) before the
> training loop begins, guarded by `if args.architecture == "v7_hierarchical"`.

Rather than wiring the contract check into the 5000-line training script at 6 call sites
(risky, per-step overhead), Stage26-C implements a standalone validation utility:

```
tools/validate_v7_forward_contract.py
```

This utility can be run once before any real training to confirm the full output contract
is satisfied on a small clean-data batch with real Mamba weights.

---

## 2. Why This Is Validation-Only, Not Training

The utility explicitly:
- Sets `model.eval()` before the forward pass
- Wraps the forward pass in `torch.no_grad()`
- Passes only `model_feature_inputs(batch)` (4 keys: `input_ids`, `attention_mask`,
  `claim_mask`, `evidence_mask`) — no labels, no optimizer, no loss
- Has no optimizer, no backward(), no scheduler, no checkpoint saving
- Runs exactly one forward pass on at most `--num-records` samples (default: 4)

It validates structure and behavior; it does not update any weights.

---

## 3. What the Utility Checks

### 3a. Key contract
Calls `validate_v7_output_contract(output)` from `modeling_v7_hierarchical.py`.
Raises `KeyError` listing all missing keys if any of `V7_REQUIRED_OUTPUT_KEYS` are absent.

### 3b. Shape checks
- `output["logits"]`: must be `[batch_size, 3]`
- `output["base_logits"]`: must be `[batch_size, 3]`
- `output["predictions"]`: must be `[batch_size]`

### 3c. NaN/Inf check
- `output["logits"]` must be finite (`torch.isfinite(logits).all()`)

### 3d. Probability range checks `[0, 1]`
All channel probabilities must be finite and within `[0.0, 1.0]`:
- `v7_frame_prob`
- `v7_predicate_prob`
- `v7_sufficiency_prob`
- `v7_entitlement_prob`
- `v7_temporal_prob` — checked only when temporal channel is active (not None)

### 3e. base_logits semantics
Checks `torch.equal(logits, base_logits)` and reports whether they are identical.
In Stage26-A/B/C, they should be equal (no separate base projection).

### 3f. Provenance assertions (hardcoded)
The summary includes:
```json
{
  "stage15_used": false,
  "ood_used": false,
  "time_swap_used_in_main_clean_data": false
}
```
These are hardcoded constants in the utility, never derived from data or flags.

### 3g. Label order documentation
Every run prints the verified label order in the summary:
```json
"label_order": {
  "dim_0": "REFUTE (FinalLabel.REFUTE=0)",
  "dim_1": "NOT_ENTITLED (FinalLabel.NOT_ENTITLED=1)",
  "dim_2": "SUPPORT (FinalLabel.SUPPORT=2)"
}
```

---

## 4. Dummy Backbone Policy

Dummy backbone (`--backbone dummy`) is **refused by default**. The utility exits with:
```
ERROR: dummy backbone is plumbing-only and cannot be used as v7 model evidence.
Pass --allow-dummy explicitly if you intentionally want dummy validation.
```

This refusal is intentional and important:
- Dummy backbone produces random hidden states
- Dummy forward results verify the computational graph, not learned behavior
- Dummy results must never be reported as v7 model evidence
- If `--allow-dummy` is passed, the summary includes `"dummy_used": true` and
  `"dummy_evidence_allowed": true` so the output is self-annotating

**Allowed dummy use:** plumbing tests (CI, import chain checks), never performance claims.

---

## 5. Real-Mamba Evidence Boundary

| Property | Real Mamba (`--backbone mamba`) | Dummy (`--backbone dummy`) |
|----------|--------------------------------|---------------------------|
| Model evidence | Yes, after training | No |
| Forward pass validity | Yes | Yes (graph only) |
| Contract check | Yes | Yes (structure only) |
| Performance claims | Yes, from trained checkpoint | Never |
| Comparison to v6B | Valid after training | Invalid |

A PASS result from this utility with real Mamba (`--backbone mamba`) confirms:
- The v7 architecture forward pass computes without errors
- All required output keys are present and shaped correctly
- Channel probabilities are in valid range
- No NaN/Inf in logits

A PASS result does **not** claim:
- The model has been trained
- The model outperforms v6B
- The channel signals are meaningful (requires training)

---

## 6. Label Order

From `src/contramamba/labels.py`, `FinalLabel(IntEnum)`:
```
REFUTE        = 0   → logits[:, 0]  (refute_score)
NOT_ENTITLED  = 1   → logits[:, 1]  (ne_score = -entitlement_logit + ne_bias)
SUPPORT       = 2   → logits[:, 2]  (support_score)
```

v7 final logit stack (confirmed correct in Stage26-B):
```python
final_logits = torch.stack([refute_score, ne_score, support_score], dim=-1)
```

CE loss in the training script uses `output["logits"]` with integer-encoded labels from
`FinalLabel` directly. No remapping is needed.

---

## 7. Required v7 Output Keys

From `V7_REQUIRED_OUTPUT_KEYS` in `src/contramamba/modeling_v7_hierarchical.py`:

```
# Core (inviolable)
logits, base_logits, predictions

# v5.controlled_losses
frame_logit, predicate_coverage_logit, sufficiency_logit,
positive_energy, negative_energy

# v5.compute_metrics / intervention_diagnostics / intervention_objective
# / prediction_records / pairwise_checks
frame_prob, predicate_coverage_prob, sufficiency_prob,
entitlement_prob, polarity_margin

# v7 diagnostic keys (always present when architecture=v7_hierarchical)
v7_frame_logit, v7_frame_prob,
v7_predicate_logit, v7_predicate_prob,
v7_sufficiency_logit, v7_sufficiency_prob,
v7_entitlement_logit, v7_entitlement_prob,
v7_polarity_logits, v7_channel_output_keys
```

**Excluded from contract** (intentionally):
- `v7_temporal_logit` / `v7_temporal_prob` — `None` when `--v7-disable-temporal-channel`
  is active. Checked in the utility only when temporal channel is active.

---

## 8. CLI Reference

```
python tools/validate_v7_forward_contract.py [options]

Options:
  --data PATH           Clean main data JSONL
                        (default: data/controlled_v5_v3_without_time_swap.jsonl)
  --model-name STR      HuggingFace model name for Mamba backbone
                        (default: state-spaces/mamba-130m-hf)
  --device STR          Device: 'cuda', 'cpu', etc.
                        (default: cuda if available, else cpu)
  --max-length INT      Tokenization max length (default: 128)
  --num-records INT     Records to sample for forward pass (default: 4)
  --seed INT            Random seed for record selection (default: 1)
  --backbone {mamba,dummy}
                        Backbone: 'mamba' (default) or 'dummy'
  --architecture {v7_hierarchical}
                        Architecture to validate (default: v7_hierarchical)
  --allow-dummy         Allow dummy backbone (required when --backbone dummy)
```

---

## 9. Summary Output Format

The utility prints a JSON summary:
```json
{
  "status": "PASS | FAIL",
  "architecture": "v7_hierarchical",
  "backbone": "mamba",
  "model_name": "state-spaces/mamba-130m-hf",
  "device": "cuda",
  "data": "data/controlled_v5_v3_without_time_swap.jsonl",
  "num_records": 4,
  "batch_size": 4,
  "logits_shape": [4, 3],
  "base_logits_shape": [4, 3],
  "label_order": {
    "dim_0": "REFUTE (FinalLabel.REFUTE=0)",
    "dim_1": "NOT_ENTITLED (FinalLabel.NOT_ENTITLED=1)",
    "dim_2": "SUPPORT (FinalLabel.SUPPORT=2)"
  },
  "key_contract_passed": true,
  "required_keys_missing": [],
  "shape_errors": [],
  "nan_or_inf_found": false,
  "prob_range_passed": true,
  "base_logits_equals_logits": true,
  "temporal_channel_active": true,
  "v7_final_logit_composition": "hierarchical_additive",
  "dummy_used": false,
  "dummy_evidence_allowed": false,
  "stage15_used": false,
  "ood_used": false,
  "time_swap_used_in_main_clean_data": false
}
```

---

## 10. Remaining Risks Before Stage26-D

1. **Contract check still not in training loop.** The utility is standalone. If a code change
   breaks an output key between a validation run and a training run, it won't be caught until
   the training step that uses the missing key.
   **Action (Stage26-D):** Add a guarded, one-shot pre-loop call in the training script:
   ```python
   if args.architecture == "v7_hierarchical":
       from contramamba.modeling_v7_hierarchical import validate_v7_output_contract
       with torch.no_grad():
           _probe = model(**v5.model_feature_inputs(dev_inputs))
       validate_v7_output_contract(_probe)
   ```

2. **TemporalChannelV2 trains only through CE.** No auxiliary temporal loss in Stage26-A/B/C.
   Signal quality unknown until first real training run.

3. **`ne_bias` initialization.** Learnable scalar initialized to 0.0. May need warm-up.

4. **Real forward pass requires Mamba weights.** The utility will attempt to download
   `state-spaces/mamba-130m-hf` from HuggingFace if not cached locally. Requires network
   access or a pre-cached HuggingFace model directory.

5. **Frame_size mismatch risk.** Dummy path uses `frame_size=32`; real path uses `frame_size=128`.
   A checkpoint trained on real Mamba must always be loaded with `frame_size=128`. The utility
   uses 128 for real Mamba, matching the training script.

6. **Multi-seed training not yet done.** Contract PASS is a structural check. No performance
   comparison to v6B is possible until Stage26-D training runs complete.
