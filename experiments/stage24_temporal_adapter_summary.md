# Stage24: Temporal Residual Adapter — Summary and Model-Design Turning Point

## 1. Purpose

Stage24 tested whether a detached temporal residual adapter could absorb temporal diagnostic supervision without corrupting the shared FrameGate / `frame_pair_repr` representation — the failure mechanism identified in Stage23.

This document is a **diagnostic/model-design turning point summary**, not a final performance claim. Stage24 produced the best temporal/OOD frontier seen so far, but no single run is preservation-safe enough to be reported as a final improvement.

---

## 2. Non-Leakage and Data Provenance

All constraints were enforced:

- **Stage15 is eval-only.** It was not used for training, calibration, threshold selection, shift selection, candidate selection, checkpoint selection, adapter construction, or penalty selection at any point in Stage24.
- **`time_swap` was not reintroduced into the main clean train/eval classification data.**
- Main clean training data: `data/controlled_v5_v3_without_time_swap.jsonl` (3600 records, 12 intervention types).
- Temporal diagnostic data: `data/temporal_diagnostic_v1_from_controlled_v5_v3.jsonl`, built from controlled data only:
  - `time_swap` → temporal mismatch, diagnostic label 1
  - `none`, `paraphrase` → temporal control, diagnostic label 0
  - all other intervention types excluded
- Temporal diagnostic data was loaded as a separate file; records were never mixed into main classification CE or pairwise losses.
- Preservation-constrained checkpoint selection used only clean-dev pairwise checks — never Stage15.
- `stage15_used_for_preservation_constrained_selection: false` and `stage15_used_for_temporal_adapter_training: false` are enforced in all run provenance.

---

## 3. Why Stage24 Was Needed After Stage23

Stage23 showed that routing temporal diagnostic (TD) supervision directly through `frame_pair_repr` creates a structural gradient conflict:

- Under sufficient TD weight (`td_005`, weight=0.05), temporal FE dropped from 0.84 to 0.01 — nearly complete suppression.
- But paraphrase preservation collapsed from 0.883 → 0.650, and predicate disentanglement from 0.967 → 0.383.
- The conflict is structural, not a tuning or checkpoint-selection artifact. No epoch within any Stage23 run simultaneously satisfied temporal rejection and preservation thresholds.

**The diagnosis from Stage23:** TD gradients flowing through the shared `frame_pair_repr` shifted FrameGate toward temporal discrimination, corrupting the representation used by preservation and predicate disentanglement checks.

**Stage24 hypothesis:** If the adapter input is detached from the computation graph before the temporal BCE loss, the adapter can learn a temporal probe without perturbing FrameGate. A separate final-logit penalty driven by the detached adapter can then selectively push NOT_ENTITLED logits for temporally mismatched examples, applying temporal correction as a per-example architecture-level intervention rather than a shared-representation distortion.

---

## 4. Detached Temporal Adapter Design Summary

- **Architecture:** `Linear(frame_size, frame_size // 2) → GELU → Linear(frame_size // 2, 1)` — a small 2-layer MLP reading `frame_pair_repr`.
- **Gradient isolation (default):** `adapter_input = frame_pair_repr.detach()`. The adapter BCE loss cannot propagate into FrameGate. This is the critical departure from Stage23's direct TD head.
- **Training signal:** same temporal diagnostic data as Stage23 — separate forward pass on TD batch, separate BCE loss, not mixed into main CE.
- **Final logit penalty (optional):** `penalty = sigmoid(temporal_adapter_logit).detach() * scale`. Applied per-example as NOT_ENTITLED logit boost. `temporal_adapter_logit` is detached again so no gradient flows back into the adapter from this penalty path. When `scale=0.0`, the penalty is a no-op and final logits are unchanged.
- **Preservation entitlement (PE) loss (optional):** Separate BCE on `sufficiency_repr` for preserved-paraphrase examples. Used in combination with the penalty to support preservation.
- **Output contract preserved:** `output["logits"] = final_logits`, `output["base_logits"] = base_logits`.

---

## 5. Baseline and Stage23 Reference

| Run | OOD macro-F1 | Temporal FE | Surface FNE | Paraphrase preserved | Pred disentangled |
|-----|-------------|------------|------------|---------------------|------------------|
| Baseline (seed1) | 0.547 | **0.84** | 0.09 | 0.883 | 0.967 |
| Stage23 `td_005` (direct, weight=0.05) | 0.556 | **0.01** | 0.68 | 0.650 | 0.383 |
| Stage23 `td_pe_005` (direct+PE) | 0.586 | 0.31 | 0.28 | 0.583 | 0.717 |

Stage23 direct TD supervision can nearly eliminate temporal FE but at the cost of severe preservation collapse. The best OOD macro-F1 from Stage23 (`td_pe_005`, 0.586) came with preservation and predicate disentanglement costs that were too large for a clean claim.

---

## 6. Stage24 Main Results

All runs: single seed (seed1), Mamba backbone, frozen encoder, clean-dev `final_macro_f1` as primary checkpoint metric unless noted.

| Run | TA scale | PE weight | OOD acc | OOD macro-F1 | Temporal FE | Surface FNE | Paraphrase preserved | Pred disentangled |
|-----|----------|-----------|---------|-------------|------------|------------|---------------------|------------------|
| Baseline | — | — | 0.8222 | 0.5466 | 0.84 | 0.09 | 0.883 | 0.967 |
| `ta_loss_only` | 0.0 (loss only) | — | 0.8222 | 0.5466 | 0.84 | 0.09 | 0.883 | 0.967 |
| `ta_penalty_weak` | 0.25 | — | 0.8315 | 0.5522 | 0.77 | 0.12 | 0.733 | 0.967 |
| `ta_penalty_pe` | 0.25 | 0.05 | 0.8611 | 0.5712 | 0.58 | 0.14 | 0.650 | 0.900 |
| `ta_p025_pe003` | 0.25 | 0.03 | 0.8333 | 0.5534 | 0.76 | 0.12 | 0.833 | 0.967 |
| `ta_p030_pe002` | 0.30 | 0.02 | 0.8722 | **0.5776** | 0.46 | 0.20 | 0.617 | 0.867 |
| `ta_p035_pe002` | 0.35 | 0.02 | **0.9352** | **0.6195** | **0.07** | 0.23 | 0.600 | 0.817 |

### Run-level notes

**`ta_loss_only`:** Detached adapter BCE training with penalty disabled. Results are identical to baseline. This confirms the adapter branch is architecturally safe when the final penalty is not applied — the detach correctly blocks any gradient path to the main model.

**`ta_penalty_weak` (scale=0.25, no PE):** Small but real temporal/OOD improvement (+0.006 macro-F1, temporal FE 0.84→0.77) with predicate disentanglement fully preserved (0.967). Paraphrase preservation drops moderately (0.883→0.733). Best preservation-safe direction at low penalty.

**`ta_penalty_pe` (scale=0.25, PE=0.05):** A useful middle candidate. Temporal FE falls to 0.58 and OOD macro rises to 0.571. Preservation costs are real but smaller than Stage23 direct TD at comparable temporal gain. Not a clean final claim.

**`ta_p025_pe003` (scale=0.25, PE=0.03):** Preservation-safe at (0.833, 0.967) with modest temporal improvement (FE 0.84→0.76). Best preservation-to-improvement tradeoff at weak PE.

**`ta_p030_pe002` (scale=0.30, PE=0.02):** Best temporal/macro frontier candidate under macro selection. Temporal FE drops to 0.46; OOD macro rises to 0.578. Paraphrase preservation at 0.617 remains too low for a clean claim, but the predicate-disentanglement cost (0.867) is moderate.

**`ta_p035_pe002` (scale=0.35, PE=0.02):** Strongest temporal OOD result in Stage24. Temporal FE reduced to 0.07; OOD acc = 0.935; OOD macro-F1 = 0.620. Surface FNE = 0.23 (high but not catastrophic). Paraphrase preservation = 0.600 — too low for a final claim, but critically, surface FNE did not collapse to 0.68 as in Stage23 `td_005`. The detached penalty achieves comparable temporal rejection with substantially less surface-FNE damage.

---

## 7. Direct TD vs Detached TA: Comparison at Similar Temporal FE

| Mechanism | Temporal FE | Surface FNE | Paraphrase preserved | Pred disentangled | OOD macro-F1 |
|-----------|------------|------------|---------------------|------------------|-------------|
| Stage23 `td_005` (direct, FE≈0.01) | 0.01 | **0.68** | 0.650 | 0.383 | 0.556 |
| Stage24 `ta_p035_pe002` (detached, FE≈0.07) | 0.07 | **0.23** | 0.600 | 0.817 | 0.620 |
| Stage23 `td_pe_005` (direct+PE, FE≈0.31) | 0.31 | 0.28 | 0.583 | 0.717 | 0.586 |
| Stage24 `ta_penalty_pe` (detached+PE, FE≈0.58) | 0.58 | 0.14 | 0.650 | 0.900 | 0.571 |

The detached adapter substantially reduces surface FNE damage at comparable temporal rejection levels. At FE≈0.07 (`ta_p035_pe002`), surface FNE is 0.23 vs 0.68 for direct TD at FE≈0.01. Predicate disentanglement improves dramatically (0.817 vs 0.383). This is the main qualitative advance of Stage24 over Stage23: the temporal/preservation trade-off frontier has moved, not just the operating point on the same frontier.

---

## 8. Preservation-Constrained Selection Results

A clean-dev-only preservation-constrained checkpoint selector was implemented and applied to `ta_p030_pe002` and `ta_p035_pe002`.

**Constraints:** paraphrase_preserved ≥ 0.70 AND predicate_disentangled ≥ 0.85 (clean-dev pairwise checks only; Stage15 never consulted).

| Run | Selector | Selected epoch | Eligible epochs | pp selected | pd selected | OOD macro-F1 | Temporal FE | Surface FNE |
|-----|----------|---------------|----------------|------------|------------|-------------|------------|------------|
| `ta_p030_pe002` | unconstrained | best macro | — | 0.617 | 0.867 | 0.5776 | 0.46 | 0.20 |
| `ta_p030_pe002_presel` | pres-constrained | 56 | 16 | 0.750 | 0.950 | 0.5432 | 0.78 | 0.17 |
| `ta_p035_pe002` | unconstrained | best macro | — | 0.600 | 0.817 | 0.6195 | 0.07 | 0.23 |
| `ta_p035_pe002_presel` | pres-constrained | 59 | 18 | 0.750 | 0.917 | 0.5529 | 0.70 | 0.16 |

**Interpretation:** The selector worked correctly — eligible epochs were found (16 and 18 respectively), fallback was not triggered, and Stage15 was not used. However, preservation-constrained selection recovers preservation at the cost of losing most of the temporal gain. The selected epochs are earlier in training, where the penalty has not yet shifted the model strongly enough to reduce temporal FE. This reveals that the temporal improvement and the preservation cost are coupled through the same penalty scale and PE weight — there is no epoch within the current training dynamics that simultaneously satisfies the preservation constraints and achieves strong temporal rejection.

---

## 9. Failure Mode and Frontier Analysis

**The trade-off is real but narrower than Stage23.**

The detached adapter penalty mechanism creates a temporal/preservation frontier:
- Higher penalty scale → stronger temporal rejection → lower paraphrase preservation
- Lower penalty scale → preservation intact → weak temporal rejection
- PE loss partially counteracts preservation degradation but also reduces temporal gain

The selector confirms this is not a checkpoint-selection artifact. No epoch in any Stage24 run simultaneously achieves temporal FE ≤ 0.50 and paraphrase_preserved ≥ 0.75 under the current configurations.

**Why the frontier still exists despite detached adapter:**

The detached adapter input isolates the temporal BCE gradient from FrameGate. However, the final-logit penalty (`sigmoid(adapter_logit) * scale`) is applied unconditionally to all examples at inference. Examples that are temporal mismatches receive correct NOT_ENTITLED boost, but non-temporal NE examples (paraphrase, surface control) also receive some penalty proportional to whatever temporal signal the adapter has learned for them. The penalty is per-example but not class-gated, so it applies a continuous soft push across all examples — not only time_swap.

**Key structural difference from Stage23:** In Stage23, the gradient conflict corrupted `frame_pair_repr` globally, damaging both temporal and non-temporal representations. In Stage24, the `frame_pair_repr` weights are unchanged; only the final logit is adjusted. This is why surface FNE = 0.23 at `ta_p035_pe002` vs 0.68 in Stage23 `td_005`. The mechanism is less destructive, but the trade-off remains.

---

## 10. Current Conclusion

Stage24 established the following facts:

1. **Detached adapter loss alone is a no-op on final behavior.** The `ta_loss_only` result exactly matches baseline. Gradient isolation via `detach()` works as designed.

2. **The detached adapter penalty can substantially reduce temporal FE without catastrophic surface FNE.** At `ta_p035_pe002`, temporal FE = 0.07 and surface FNE = 0.23, compared with Stage23 `td_005` at FE = 0.01 and surface FNE = 0.68. The model-design frontier has genuinely moved.

3. **No Stage24 run is preservation-safe for a final claim.** Paraphrase preservation drops below acceptable levels (< 0.70) in all runs with strong temporal improvement. The best preservation-safe candidate (`ta_p025_pe003`) achieves only modest temporal gain (FE 0.84→0.76).

4. **Preservation-constrained selection resolves the preservation problem but not the temporal gain.** The selector works correctly but confirms the temporal/preservation trade-off is not a checkpoint-selection artifact.

5. **Stage24 is a positive diagnostic/model-design result.** It identifies the correct gradient isolation principle, shows the frontier has moved, and rules out several simplistic approaches (adapter loss alone, unconstrained penalty scaling). It is not a final improvement claim and should not be reported as one.

---

## 11. Recommended Next Model Direction

**Do not proceed to 3-seed evaluation of any Stage24 run as the final model.** The preservation failures would replicate across seeds.

**Do not add further selectors or loss components to the current architecture.** The current model has accumulated CE loss, pairwise losses, boundary head, frame-violation head, predicate isolation head, preservation entitlement head, temporal diagnostic head, temporal residual adapter, adapter penalty, and pairwise selection logic. Additional components increase audit burden without addressing the structural frontier issue.

**Recommended next step: loss/logit/selector audit ledger.**

Before any further Stage25 experiment, produce a structured ledger per run that includes:
- `active_training_losses` — list of losses that contributed nonzero gradients
- `active_final_logit_modifiers` — comparator alphas, temporal penalty, other adjustments
- `active_selection_rules` — which selector was active; fallback status
- `loss_component_epoch_avg` — mean per-loss value over training
- `aux_to_ce_ratio` — ratio of auxiliary loss total to CE loss, as a complexity signal

**Recommended lean final-model design target:**

1. Remove the direct TD head from the final path (Stage23 confirmed it is unsafe).
2. Keep the gradient-isolation principle from Stage24, but move from `frame_pair_repr` to an independent temporal channel reading from `base_pair_repr` (or directly from encoder slot outputs), fully decoupled from FrameGate.
3. Gate the temporal logit adjustment through PE: apply the NOT_ENTITLED penalty only when the preservation entitlement head signals entitlement is NOT preserved (preventing the penalty from firing on paraphrase-preserved examples).
4. Allow only one local final-logit arbitration mechanism per run — not a combination of comparator alphas, temporal penalty, and PE gating simultaneously without an explicit priority rule.
5. Keep total training loss minimal: CE + temporal BCE (on independent channel) + optional preservation BCE. Remove or disable heads that are not load-bearing for the current test.

**Options deferred until lean temporal channel is stable:**
- Prototype memory features
- Alternative Mamba-layer representations
- Multi-seed evaluation of any temporal-correction design
