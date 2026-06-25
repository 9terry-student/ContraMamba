# Stage26-A: v7 Hierarchical Entitlement Architecture Notes

**Status:** Implementation-only. No training has been run. No experiment results.

**Purpose:** Create a clean v7 architecture path that directly implements the hierarchical
entitlement hypothesis from Stage25 design. Preserves v6B as an unchanged baseline.

---

## 1. Why Stage26 Moves from v6B Patching to a New v7 Class

Stage21–24 produced real temporal mismatch signal (TemporalChannelV1, detached from FrameGate).
But converting that signal into a post-hoc final-logit boost — even when PE-gated — could not
resolve the temporal/preservation trade-off. Stage24 showed:

- `tc_gated_p025`: penalty too weak (FER 0.87 → 0.85).
- `tc_gated_p075`: best temporal improvement (FER → 0.68) but surface_control FNER rose to 0.18.
- `tc_gated_p150`: non-monotonic, unstable.

The bottleneck is crude final arbitration, not signal availability. Stacking another penalty on
v6B would produce the same structural problem. The correct fix is a different arbitration structure.

**Decision:** Preserve v6B exactly as a baseline. Add a separate v7 class that implements the
six-axis hierarchical pipeline directly. Do not patch v6B further.

---

## 2. Historical Framing

ContraMamba originated the six-axis epistemic decomposition:

- Started as -1/0/+1 (false/unknown/true).
- **Ambiguity** and **ignorance** axes added when the three-way geometry could not represent
  underdetermined and evidence-absent cases.
- **Novelty** axis added for cases outside the known epistemic space.

EpistemicBERT was a pragmatic detour/testbed. It operationalized the ContraMamba-originated
six-axis philosophy in a strong-backbone annotation setting. The EpistemicBERT codebook
formalized a judgment order — not as an originator but as a transcription of the decomposition
that ContraMamba produced by experimental necessity.

Stage26 returns the clarified hierarchy to the original ContraMamba architecture. This is a
return, not a derivation from EpistemicBERT.

---

## 3. New Files and Classes

| File | Class | Role |
|------|-------|------|
| `src/contramamba/modeling_v7_hierarchical.py` | `ContraMambaV7Hierarchical` | v7 main model class |
| `src/contramamba/modeling_v7_hierarchical.py` | `TemporalChannelV2` | Temporal invalidity probe |
| `src/contramamba/modeling_v7_hierarchical.py` | `EntitlementGateV7` | Learned channel aggregation |
| `src/contramamba/modeling_v7_hierarchical.py` | `PolarityChannelV7` | Post-entitlement polarity |

**v6B not modified.** `src/contramamba/modeling_v6b_minimal.py` is unchanged.

Training script additions in `scripts/train_controlled_v6b_minimal.py`:
- `build_v7_model(...)` — dummy-backbone v7 constructor
- `build_v7_mamba_model(...)` — real-backbone v7 constructor
- `--architecture` flag with model build branch
- 7 ablation flags + 6 auxiliary loss flags (all off by default)
- 4 v7 provenance fields in audit ledger

---

## 4. v7 Architecture Diagram

```
Input: (input_ids, attention_mask, claim_mask, evidence_mask)
  ↓
Mamba backbone
  ↓ token_states [B, seq_len, hidden_size]
  ↓
FrameGate (reused from v5)
  ├─ claim_frame_state [B, frame_size]    ← pre-pair-projector claim slot state
  ├─ evidence_frame_state [B, frame_size] ← pre-pair-projector evidence slot state
  ├─ frame_pair_repr [B, frame_size]      ← post-pair-projector joint frame repr
  └─ frame_prob [B]                       → FrameChannel signal
  ↓
PredicateCoverageHead (reused from v5, conditioned on frame state)
  ├─ predicate_pair_repr [B, pred_size]
  └─ predicate_coverage_prob [B]          → PredicateChannel signal
  ↓
SufficiencyGate (reused from v5, conditioned on frame + predicate reprs)
  ├─ sufficiency_repr [B, suff_size]
  └─ sufficiency_prob [B]                 → SufficiencyChannel signal

TemporalChannelV2 (NEW — reads cat([claim_frame_state, evidence_frame_state]))
  └─ temporal_prob [B]                    → TemporalChannel signal (1=mismatch)

EntitlementGateV7 (NEW — learned MLP over 4 channel probs)
  Input: [frame_prob, pred_prob, suff_prob, 1-temporal_prob]  [B, 4]
  └─ entitlement_logit [B], entitlement_prob [B]

PolarityChannelV7 (NEW — reads frame+predicate+sufficiency reprs)
  Input: cat([frame_pair_repr, predicate_pair_repr, sufficiency_repr])  [B, combined]
  ├─ polarity_support_logit [B]
  └─ polarity_refute_logit [B]

Final logit composition (REFUTE=0, NOT_ENTITLED=1, SUPPORT=2):
  support_score = entitlement_logit + polarity_support_logit
  refute_score  = entitlement_logit + polarity_refute_logit
  ne_score      = -entitlement_logit + ne_bias
  logits[B, 3]  = [refute_score, ne_score, support_score]
```

---

## 5. Final Logit Composition

```python
support_score = entitlement_logit + polarity_support_logit
refute_score  = entitlement_logit + polarity_refute_logit
ne_score      = -entitlement_logit + ne_bias
final_logits  = stack([refute_score, ne_score, support_score], dim=-1)
```

**Hierarchical property:**
- When `entitlement_logit` is large (positive): SUPPORT and REFUTE scores rise; NE score falls.
  Polarity can decide.
- When `entitlement_logit` is small (negative): SUPPORT and REFUTE scores fall; NE score rises.
  NOT_ENTITLED wins. Polarity is suppressed without explicit masking.

This is the Stage25 architectural correction: temporal mismatch (detected by TemporalChannelV2)
reduces `entitlement_logit` via the EntitlementGate, which then suppresses polarity and raises
NE — not via a direct NE-logit boost.

**`ne_bias`:** learnable scalar parameter (initialized to 0.0). Learned from CE signal.

**Not a post-hoc penalty.** The composition is part of the model graph and trained through
backprop from CE loss. TemporalChannelV2 is trained through this same path in Stage26-A.

---

## 6. Ablation Flags

All flags are False (full hierarchical model) by default. Only active when
`--architecture v7_hierarchical`. Have zero effect on v6B runs.

| Flag | Default | Effect when True |
|------|---------|-----------------|
| `--v7-disable-frame-channel` | False | EntitlementGate sees `frame_prob=1.0`; FrameGate still runs |
| `--v7-disable-predicate-channel` | False | EntitlementGate sees `predicate_prob=1.0` |
| `--v7-disable-sufficiency-channel` | False | EntitlementGate sees `sufficiency_prob=1.0` |
| `--v7-disable-temporal-channel` | False | TemporalChannelV2 not instantiated; gate uses 3-input MLP |
| `--v7-flat-arbiter` | False | EntitlementGate uses explicit product (v6B-like) not learned MLP |
| `--v7-no-entitlement-polarity-conditioning` | False | Final composition ignores `entitlement_logit` |
| `--v7-no-aux-losses` | False | Stage26-A no-op; no v7 aux losses exist yet |

**Disable flag semantics:** When a channel is disabled, its _probability_ fed to EntitlementGate
is overridden with 1.0. The head still runs; `frame_pair_repr`, `predicate_pair_repr`, and
`sufficiency_repr` are still computed and fed to downstream heads and PolarityChannel.
`output["frame_logit"]` and other logit keys still show the actual computed values, so
`v5.controlled_losses` still computes the auxiliary BCE losses.

---

## 7. Audit/Provenance Fields

Added to `_run_audit_ledger` (inside `run_training_v6b`), main config block, OOD-sweep config
block, and the lift copy loop:

```json
{
  "stage15_used_for_v7_training": false,
  "stage15_used_for_v7_selection": false,
  "stage15_used_for_v7_aux_loss_targets": false,
  "time_swap_used_in_v7_main_clean_data": false
}
```

Added to main config block and metadata:

```json
{
  "architecture": "v7_hierarchical",
  "use_v7_hierarchical": true,
  "v7_disable_frame_channel": false,
  "v7_disable_predicate_channel": false,
  "v7_disable_sufficiency_channel": false,
  "v7_disable_temporal_channel": false,
  "v7_flat_arbiter": false,
  "v7_no_entitlement_polarity_conditioning": false,
  "v7_no_aux_losses": false,
  "v7_aux_losses_active": false,
  "v7_final_logit_composition": "hierarchical_additive"
}
```

v6B fields (`active_training_losses.temporal_channel_loss`, comparator alpha fields, etc.)
remain in the audit ledger and are not removed. For v7 runs, they correctly report as disabled.

---

## 8. Tensor Provenance: Channel Inputs

| Channel | Input tensor | Source | Notes |
|---------|-------------|--------|-------|
| FrameChannel | `token_states` | Mamba backbone | Via FrameGate (reused v5 head) |
| PredicateChannel | `token_states` + `frame_pair_repr` + `frame_prob` | Mamba + FrameGate | Via PredicateCoverageHead (reused v5 head) |
| SufficiencyChannel | `frame_pair_repr` + `predicate_pair_repr` + probs | FrameGate + PredicateCoverageHead | Via SufficiencyGate (reused v5 head) |
| TemporalChannel | `cat([claim_frame_state, evidence_frame_state])` | FrameGate `.project()` output | Pre-pair-projector slot states; NOT `frame_pair_repr` |
| EntitlementGate | `[frame_prob, predicate_prob, sufficiency_prob, 1-temporal_prob]` | Channel outputs | Scalar probabilities only; 4-dim or 3-dim depending on temporal flag |
| PolarityChannel | `cat([frame_pair_repr, predicate_pair_repr, sufficiency_repr])` | FrameGate + PredicateCoverageHead + SufficiencyGate | Joint representations |

**No fake tensors.** All inputs are directly available intermediate representations from
existing heads. No OOD data, no Stage15 data, no time_swap used in any channel input.

**TemporalChannel note:** `claim_frame_state` and `evidence_frame_state` are the outputs of
`FrameGate.project(token_states)` pooled over claim/evidence spans. They are NOT `frame_pair_repr`
(which is the output of `FrameGate.pair_projector` applied to the 4-feature concatenation).
Same representation as TemporalChannelV1 in v6B.

---

## 9. What Is Incomplete for Stage26-B

**Temporal channel supervision:** In Stage26-A, TemporalChannelV2 trains only through the CE
loss via the EntitlementGate composition. It has no explicit auxiliary BCE loss. To add dedicated
temporal supervision in Stage26-B, the temporal diagnostic data path (`--temporal-diagnostic-data`)
and a v7 temporal loss flag would need to be wired. The supervision target (`time_swap=1`,
`none/paraphrase=0`) already exists in `data/temporal_diagnostic_v1_from_controlled_v5_v3.jsonl`.

**Channel auxiliary losses:** The `--v7-use-aux-losses`, `--v7-frame-loss-weight`, etc. flags
are registered but inactive in Stage26-A. When aux targets are confirmed clean for each channel,
Stage26-B can wire the BCE losses.

**Aux loss targets that need definition for Stage26-B:**

| Channel | Possible target | Status |
|---------|----------------|--------|
| TemporalChannel | `time_swap=1`; `none/paraphrase=0` (temporal diagnostic JSONL) | Available |
| FrameChannel | `frame_compatible_label` from controlled data | Available |
| PredicateChannel | `predicate_covered_label` from controlled data | Available |
| SufficiencyChannel | `sufficiency_label` from controlled data | Available |
| EntitlementGate | Composite? No clean single target | Needs definition |

Frame/predicate/sufficiency auxiliary targets are already used by `v5.controlled_losses`
(frame_loss, predicate_loss, sufficiency_loss). In Stage26-A, these losses still train through
`v5.controlled_losses` which is called on v7 output, so frame/predicate/sufficiency heads
are already supervised — the `--v7-*-loss-weight` flags are for additional weighting, not
for enabling supervision.

**Multi-seed evaluation:** Not appropriate until at least one seed shows preservation-safe
temporal improvement from the hierarchical structure.

---

## 10. v6B Default Behavior Is Preserved

- `--architecture v6b_minimal` is the default. No v7 code runs in a v6B command.
- The `build_v7_model` / `build_v7_mamba_model` functions are only called when
  `args.architecture == "v7_hierarchical"`.
- The v7 ablation flags default to `False` and are ignored when architecture is v6b_minimal.
- `modeling_v6b_minimal.py` is NOT modified in Stage26-A.
- All existing v6B CLI commands produce identical output to before Stage26-A.
- `output["logits"]` and `output["base_logits"]` contract is preserved in both paths.
- CE loss uses `output["logits"]` in both paths.

---

## 11. Stage15 and time_swap Exclusions Confirmed

- `stage15_used_for_v7_training = False` (hardcoded in audit ledger)
- `stage15_used_for_v7_selection = False` (hardcoded)
- `stage15_used_for_v7_aux_loss_targets = False` (hardcoded)
- `time_swap_used_in_v7_main_clean_data = False` (hardcoded)
- Main clean data remains `data/controlled_v5_v3_without_time_swap.jsonl`
- Temporal diagnostic data (including `time_swap`) remains separate and is NOT loaded for v7
  in Stage26-A (no v7 temporal aux loss active)
- Stage15 remains eval-only; no EntitlementGate threshold, no gate weight, no channel scale,
  no checkpoint is selected using Stage15 results
- v7 ablation flags have no OOD-derived defaults

---

## 12. v7-Specific Output Keys

When `architecture == v7_hierarchical`, the model output dict includes:

```
logits                     # final logits [B, 3] — inviolable; CE uses this
base_logits                # diagnostic alias (= final_logits in v7)
predictions                # argmax [B]
# v5 compatibility keys (required by training script functions):
frame_logit, frame_prob
predicate_coverage_logit, predicate_coverage_prob
sufficiency_logit, sufficiency_prob
entitlement_prob           # from EntitlementGateV7
positive_energy            # alias for v7_polarity_support_logit
negative_energy            # alias for v7_polarity_refute_logit
polarity_margin            # polarity_support_logit - polarity_refute_logit
# v7 diagnostic keys:
v7_frame_logit, v7_frame_prob
v7_predicate_logit, v7_predicate_prob
v7_sufficiency_logit, v7_sufficiency_prob
v7_temporal_logit, v7_temporal_prob   # None when temporal channel disabled
v7_entitlement_logit, v7_entitlement_prob
v7_polarity_support_logit, v7_polarity_refute_logit
v7_polarity_logits         # [refute_logit, support_logit] stacked
v7_channel_output_keys     # list of active v7 diagnostic key names
v7_final_logit_composition # "hierarchical_additive" or "flat"
```

The v5 compatibility keys ensure `v5.controlled_losses`, `v5.compute_metrics`, and
`v5.intervention_diagnostics` work on v7 output without modification.
