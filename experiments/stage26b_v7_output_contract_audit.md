# Stage26-B: v7 Output Contract, Label Order, and Compatibility Audit

**Status:** Static audit only. No training has been run. No experiment results.

**Date:** Stage26-B (follows Stage26-A implementation)

**Files audited:**
- `src/contramamba/modeling_v7_hierarchical.py`
- `scripts/train_controlled_v6b_minimal.py`
- `scripts/train_controlled_v5.py` (utilities consumed by training script)
- `src/contramamba/labels.py`

---

## 1. Label Order: Verified Correct

**Source of truth:** `src/contramamba/labels.py`, lines 6-9:
```python
class FinalLabel(IntEnum):
    REFUTE = 0
    NOT_ENTITLED = 1
    SUPPORT = 2
```

**v7 final logit stack** (`ContraMambaV7Hierarchical.forward`, `modeling_v7_hierarchical.py`):
```python
final_logits = torch.stack([refute_score, ne_score, support_score], dim=-1)
# dim 0 = refute_score   → REFUTE        = 0  ✓
# dim 1 = ne_score       → NOT_ENTITLED  = 1  ✓
# dim 2 = support_score  → SUPPORT       = 2  ✓
```

**Verdict:** The stack order is correct. CE loss receives `output["logits"]` and integer-encoded
labels from `FinalLabel`. No off-by-one risk.

**Comment added** to forward method near the composition block explicitly calling out the
`FinalLabel` reference and the constraint that the stack order must not be reordered.

---

## 2. v7 Required Output Keys

Keys derived from static inspection of all v5 utility functions in `scripts/train_controlled_v5.py`
that the training script calls on model output.

### 2a. Per-function requirements

| Function | Output keys accessed |
|----------|---------------------|
| `controlled_losses` | `logits`, `frame_logit`, `predicate_coverage_logit`, `sufficiency_logit`, `negative_energy`, `positive_energy` |
| `compute_metrics` | `predictions`, `polarity_margin`, `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob` |
| `intervention_diagnostics` | `logits`, `predictions`, `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, `entitlement_prob`, `polarity_margin` |
| `intervention_objective` | `frame_logit`, `predicate_coverage_logit`, `sufficiency_logit`, `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, `entitlement_prob`, `logits` |
| `prediction_records` | `logits`, `predictions`, `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, `entitlement_prob`, `polarity_margin` |
| `pairwise_checks` | `frame_prob`, `predicate_coverage_prob`, `sufficiency_prob`, `entitlement_prob`, `polarity_margin`, `predictions` |

### 2b. `V7_REQUIRED_OUTPUT_KEYS` constant

Added to `src/contramamba/modeling_v7_hierarchical.py` (before `class TemporalChannelV2`):

```python
V7_REQUIRED_OUTPUT_KEYS: tuple[str, ...] = (
    # Core (inviolable)
    "logits",
    "base_logits",
    "predictions",
    # v5.controlled_losses
    "frame_logit",
    "predicate_coverage_logit",
    "sufficiency_logit",
    "positive_energy",
    "negative_energy",
    # v5.compute_metrics / intervention_diagnostics / intervention_objective
    # / prediction_records / pairwise_checks
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "entitlement_prob",
    "polarity_margin",
    # v7 diagnostic keys (always present when architecture=v7_hierarchical)
    "v7_frame_logit",
    "v7_frame_prob",
    "v7_predicate_logit",
    "v7_predicate_prob",
    "v7_sufficiency_logit",
    "v7_sufficiency_prob",
    "v7_entitlement_logit",
    "v7_entitlement_prob",
    "v7_polarity_logits",
    "v7_channel_output_keys",
)
```

**Intentional exclusions from the contract:**
- `v7_temporal_logit` / `v7_temporal_prob` — these are `None` when `--v7-disable-temporal-channel`
  is active. No v5 utility reads these keys. They are still present in the output dict but
  excluded from the contract check.
- `v7_polarity_support_logit` / `v7_polarity_refute_logit` — accessible via `positive_energy`
  and `negative_energy` aliases; no need to require both names.
- `v7_final_logit_composition` — metadata string, not consumed by any utility function.

### 2c. Contract validator

Added `validate_v7_output_contract(output: dict[str, Any]) -> None` to
`src/contramamba/modeling_v7_hierarchical.py`. Raises `KeyError` listing all missing keys.

**Not wired to the training loop** in Stage26-A/B — see Section 5.

---

## 3. `logits` vs `base_logits` Semantics

### `output["logits"]`
- The final 3-class logit tensor `[B, 3]` passed to CE loss.
- **Inviolable contract**: CE always uses `output["logits"]`.
- Composition: `[refute_score, ne_score, support_score]` with class order per `FinalLabel`.
- In hierarchical mode: `support_score = entitlement_logit + polarity_support_logit`, etc.
- In flat ablation mode (`--v7-no-entitlement-polarity-conditioning`): polarity alone, `ne_bias`.

### `output["base_logits"]`
- In v7 Stage26-A/B: **identical tensor to `output["logits"]`** — `base_logits = final_logits`.
- Present for output contract compatibility only.
- No v5 utility function accesses `base_logits`.
- **Do NOT use as independent model evidence** in v7 Stage26-A/B.
- No separate base projection exists. A future Stage26-C+ may introduce one.

**Comment added** to `ContraMambaV7Hierarchical.forward()` explicitly documenting these semantics.

---

## 4. v7 Final Logit Composition Summary

```
support_score = entitlement_logit + polarity_support_logit
refute_score  = entitlement_logit + polarity_refute_logit
ne_score      = -entitlement_logit + ne_bias
logits[B, 3]  = [refute_score, ne_score, support_score]
             #   dim 0=REFUTE   dim 1=NE   dim 2=SUPPORT
```

**Hierarchical property (default):**
- High `entitlement_logit` → SUPPORT and REFUTE scores rise; NE score falls. Polarity decides.
- Low `entitlement_logit` → SUPPORT and REFUTE scores fall; NE score rises. NOT_ENTITLED wins.
  Polarity is suppressed without explicit masking.

**Flat mode (`--v7-flat-arbiter` + `--v7-no-entitlement-polarity-conditioning`):**
- `EntitlementGateV7` uses explicit product (v6B-like).
- Final composition ignores `entitlement_logit`; polarity alone decides SUPPORT/REFUTE.
- Only for ablation comparison; not the primary Stage26 hypothesis.

---

## 5. Runtime Contract Validation: Deferred to Stage26-C

**Decision: deferred.**

The training script has **6 distinct model forward call sites**:
- Line ~1226 (outer helper)
- Lines ~2828, ~3205, ~3212 (training epoch loop — train output, dev output)
- Lines ~3227+ (temporal diagnostic eval forward)
- Lines ~5131, ~5215 (OOD eval sweep)

All are inside the v6B training loop. Adding `validate_v7_output_contract` to all 6 would:
1. Require importing from v7 code on all runs (including v6B), adding overhead.
2. Risk indentation/logic errors in a 5000+ line script.
3. Run the validator on every forward pass of every epoch (not needed once model is verified).

**Safe approach implemented:**
- `V7_REQUIRED_OUTPUT_KEYS` and `validate_v7_output_contract` are defined in the model file,
  callable from any test or one-shot script.
- A `# Stage26-C TODO` comment is placed near the model build in the training script
  (at `model = model.to(device)`) showing exactly how to add the validation.
- `v7_channel_output_keys` added to both config/audit blocks.

**Stage26-C action:** add a single one-shot validation call (not per-step) before the training
loop begins, guarded by `if args.architecture == "v7_hierarchical"`.

---

## 6. Dummy Backbone Policy

### What "dummy backbone" means
When `--backbone dummy` (or no backbone is specified and the Mamba weight download is skipped),
the model uses `v5.ControlledDummyBackbone(vocab_size, hidden_size, max_length)`.

`build_v7_model(len(vocab), max_length, ...)` is the dummy path. It is called from the
`if model is None:` block — only when the mamba backbone path was skipped.

### Dummy path evidence boundary

- **Dummy v7 results are NOT model evidence.** The dummy backbone produces random hidden states.
  Any output probabilities, logits, or channel values from a dummy v7 run are structural/plumbing
  outputs only — they verify the forward pass computes without crashing, not that the model
  has learned anything.

- **Dummy results MUST NOT be reported as v7 performance.** They cannot be compared to v6B
  checkpoint results or used for any selection decision.

- **Real claims require** `--backbone mamba --architecture v7_hierarchical` with a properly
  trained checkpoint.

- **Static/plumbing validation** (the only use of dummy in Stage26-A/B) is:
  `python -m py_compile` + `--help` check. No forward passes are run in Stage26-A/B.

---

## 7. Audit/Provenance Completeness Check

Fields in the report/config/audit blocks:

| Field | Main config | OOD-sweep config | `_run_audit_ledger` |
|-------|------------|-----------------|-------------------|
| `architecture` | ✓ | ✓ | — |
| `use_v7_hierarchical` | ✓ | ✓ | — |
| `v7_final_logit_composition` | ✓ | — | — |
| `v7_channel_output_keys` | ✓ (added Stage26-B) | ✓ (added Stage26-B) | — |
| `v7_aux_losses_active` | ✓ | ✓ | — |
| `stage15_used_for_v7_training` | ✓ | ✓ | ✓ |
| `stage15_used_for_v7_selection` | ✓ | ✓ | ✓ |
| `stage15_used_for_v7_aux_loss_targets` | ✓ | ✓ | ✓ |
| `time_swap_used_in_v7_main_clean_data` | ✓ | ✓ | ✓ |
| `v7_disable_*` (7 ablation flags) | ✓ | ✓ | — |

**All required provenance fields are present.** The `_run_audit_ledger` carries stage15 and
time_swap provenance. Architecture/ablation fields live in the config/report, which is correct
(they are args, not loss-component measurements).

**Hardcoded provenance values (Stage26-A/B):**
```
stage15_used_for_v7_training        = False
stage15_used_for_v7_selection       = False
stage15_used_for_v7_aux_loss_targets = False
time_swap_used_in_v7_main_clean_data = False
```

These are hardcoded because Stage26-A/B has no mechanism that could set them True.
They become data-driven (verified at runtime) only if Stage26-C adds aux loss wiring.

---

## 8. v5 Compatibility Alias Table

These aliases are required in `output` for the v5 utilities. They are verified present in
`ContraMambaV7Hierarchical.forward()`.

| Alias key | v7 source | Required by |
|-----------|-----------|-------------|
| `positive_energy` | `v7_polarity_support_logit` | `v5.controlled_losses` (polarity CE) |
| `negative_energy` | `v7_polarity_refute_logit` | `v5.controlled_losses` (polarity CE) |
| `polarity_margin` | `support_logit - refute_logit` | `v5.compute_metrics`, `intervention_diagnostics`, `pairwise_checks`, `prediction_records` |
| `entitlement_prob` | `v7_entitlement_prob` | `v5.intervention_diagnostics`, `intervention_objective`, `pairwise_checks`, `prediction_records` |
| `frame_logit` | `frame["frame_logit"]` | `v5.controlled_losses`, `intervention_objective` |
| `frame_prob` | `frame["frame_prob"]` | `v5.compute_metrics`, `intervention_diagnostics`, `intervention_objective`, `pairwise_checks`, `prediction_records` |
| `predicate_coverage_logit` | `predicate["predicate_coverage_logit"]` | `v5.controlled_losses`, `intervention_objective` |
| `predicate_coverage_prob` | `predicate["predicate_coverage_prob"]` | `v5.compute_metrics`, `intervention_diagnostics`, `pairwise_checks`, `prediction_records` |
| `sufficiency_logit` | `sufficiency["sufficiency_logit"]` | `v5.controlled_losses`, `intervention_objective` |
| `sufficiency_prob` | `sufficiency["sufficiency_prob"]` | `v5.compute_metrics`, `intervention_diagnostics`, `pairwise_checks`, `prediction_records` |

**Note on `negative_energy` / `positive_energy`:** In v6B, these were softplus-constrained
positive energies from `PolarityEnergyHead`. In v7, they are raw unconstrained logits from
`PolarityChannelV7`. The polarity CE in `v5.controlled_losses` stacks them as logits:
```python
polarity_logits = torch.stack([zeros, negative_energy, positive_energy], dim=-1)
```
This is correct regardless of constraint — raw logits work fine as CE inputs.

**`PolarityLabel` class order:** `NONE=0, REFUTE=1, SUPPORT=2`. The polarity CE stacks
`[zeros, negative_energy, positive_energy]` matching `[NONE, REFUTE, SUPPORT]`. v7 aliases
are compatible.

---

## 9. Remaining Risks Before Stage26-C

1. **No runtime contract check yet.** The contract is statically documented and the helper exists,
   but `validate_v7_output_contract` is not called in any training or eval path. A bug that
   removes a key from the output dict would not be caught until the training step that uses it.
   **Action:** Stage26-C should add a one-shot pre-loop validation call.

2. **TemporalChannelV2 trains only through CE.** In Stage26-A/B, TemporalChannelV2 receives
   no dedicated auxiliary loss. Its signal quality depends entirely on CE backprop through
   the EntitlementGate composition. The signal may be weak or noisy until verified in a real run.

3. **`ne_bias` initialization at 0.0.** The learnable NE bias is initialized to zero.
   If the training data is imbalanced toward SUPPORT/REFUTE, the model may need several
   epochs before `ne_bias` settles. Monitor `ne_score` distribution in first runs.

4. **EntitlementGateV7 MLP weight initialization.** A 4→16→1 MLP initialized with PyTorch
   defaults. In flat_arbiter mode the gate uses explicit product (stable). In MLP mode, early
   gradients flow through a small network — should be fine but not verified empirically.

5. **`base_logits == logits` in Stage26-A/B.** If any downstream script (outside v5 utilities)
   uses `base_logits` expecting an independent signal, it will silently receive `logits`.
   No such use found in current codebase, but worth noting for Stage26-C.

6. **v7 frame_size=32 in dummy path, 128 in real path.** `build_v7_model` uses `frame_size=32`,
   `build_v7_mamba_model` uses `frame_size=128`. These are intentional (dummy uses tiny dims
   for speed), but frame_size also controls `TemporalChannelV2` input dimension. Confirm
   no frame_size mismatch between checkpoint and inference when loading real checkpoints.

7. **Multi-seed stability not yet verified.** Real v7 runs are needed before claiming
   temporal improvement over v6B baselines.
