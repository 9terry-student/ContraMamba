# Stage23: Temporal Diagnostic Head — Summary and Negative Result

## 1. Purpose

Stage23 investigated whether adding a temporal diagnostic head trained on a controlled `time_swap`-based diagnostic dataset could reduce Stage15 OOD temporal mismatch false-entitled errors (FE rate = 0.84 at baseline) without destroying clean-dev preservation or predicate disentanglement.

This document is a **diagnostic/negative-result summary**. Stage23 identified a real temporal signal and a concrete failure mechanism. It is not a final performance claim and should not be treated as one.

---

## 2. Non-Leakage / Data Provenance

All constraints were enforced:

- **Stage15 is eval-only.** It was not used for training, calibration, threshold selection, shift selection, checkpoint selection, or diagnostic dataset construction at any point in Stage23.
- **`time_swap` was not reintroduced into the main clean train/eval classification data.**
- Main clean training data: `data/controlled_v5_v3_without_time_swap.jsonl` (3600 records, 12 intervention types).
- Temporal diagnostic data: `data/temporal_diagnostic_v1_from_controlled_v5_v3.jsonl`, built from the controlled dataset only:
  - `time_swap` → temporal mismatch, diagnostic label 1
  - `none`, `paraphrase` → temporal control, diagnostic label 0
  - all other intervention types excluded
- TD data was loaded as a **separate file** with a separate forward pass during training; TD records were never mixed into the main classification CE or pairwise losses.
- Checkpoint selection used clean dev `final_macro_f1` by default. TD-aware constrained selection (when enabled) used only clean dev pairwise checks and TD dev metrics — never Stage15.
- `stage15_used_for_temporal_diagnostic_training: false` and `stage15_used_for_td_constrained_selection: false` are enforced in all run provenance.

---

## 3. Architecture Note

**The TD head uses `frame_pair_repr` as its input representation.**

`frame_pair_repr` is the output of `FrameGate`, the shared frame compatibility sub-module. Temporal mismatch records (`time_swap`) have `primary_failure_type = 'frame'` and `frame_compatible_label = 0`, which justifies using the frame representation as a temporal probe. However, this choice means:

- TD supervision gradients flow through the shared `FrameGate` path.
- TD gradients are **not** isolated from the frame representation used for preservation and predicate disentanglement checks.
- The temporal diagnostic head is **not** an independent temporal channel. It is a frame-level temporal mismatch probe.

This architectural coupling is the central failure mode of Stage23 (see Section 8).

---

## 4. Experimental Conditions

| Run | TD weight | PE weight | Constrained selector |
|-----|-----------|-----------|----------------------|
| Baseline (clean seed1) | — | — | — |
| `td_005` | 0.05 | — | off |
| `td_010` | 0.10 | — | off |
| `td_pe_005` | 0.05 | 0.05 | off |
| `td_005_finalsel` | 0.05 | — | final-decision constrained |
| `td_pe_005_finalsel` | 0.05 | 0.05 | final-decision constrained |
| `td_003_sel` (prior) | 0.03 | — | head-prob constrained |
| `td_005_sel` (prior) | 0.05 | — | head-prob constrained |
| `td_pe_005_sel` (prior) | 0.05 | 0.05 | head-prob constrained |

All runs: single seed (seed1), Mamba backbone, frozen encoder, clean dev `final_macro_f1` as primary checkpoint metric.

---

## 5. Main Results

| Run | Best epoch | OOD acc | OOD macro-F1 | Temporal FE | Surface FNE | Temporal-erased FNE | Frame-loc FE | Frame-role FE | Pred FE | Dev macro-F1 | Paraphrase preserved | Pred disentangled |
|-----|-----------|---------|-------------|------------|------------|---------------------|-------------|--------------|--------|-------------|---------------------|------------------|
| Baseline | 57 | 0.822 | 0.547 | **0.84** | 0.09 | 0.00 | 0.05 | 0.05 | 0.01 | 0.962 | 0.883 | 0.967 |
| `td_005` | 48 | 0.859 | 0.556 | **0.01** | 0.68 | 0.06 | 0.05 | 0.00 | 0.00 | 0.976 | 0.650 | 0.383 |
| `td_010` | 60 | 0.826 | 0.549 | 0.87 | 0.04 | 0.00 | 0.05 | 0.05 | 0.01 | 0.966 | 0.917 | 0.950 |
| `td_pe_005` | 58 | **0.887** | **0.586** | 0.31 | 0.28 | 0.00 | 0.05 | 0.00 | 0.01 | 0.979 | 0.583 | 0.717 |

**Key:** FE = false-entitled rate; FNE = false-not-entitled rate. Bold indicates notable values.

---

## 6. Interpretation of Main Results

**`td_005` (TD weight 0.05):** The temporal diagnostic signal transfers strongly to Stage15 temporal mismatch rejection. Temporal FE dropped from 0.84 to 0.01 — nearly complete suppression. However, surface control FNE rose from 0.09 to 0.68, paraphrase preservation collapsed from 0.883 to 0.650, and predicate disentanglement collapsed from 0.967 to 0.383. This is a severe preservation failure caused by TD gradients interfering with the shared frame representation.

**`td_010` (TD weight 0.10):** Higher TD weight did not monotonically improve temporal rejection. Temporal FE remained at 0.87 (essentially matching baseline), while preservation metrics largely recovered. This indicates the relationship between TD weight and temporal rejection is non-monotone within a single seed — the optimization landscape is unstable, and checkpoint selection on `final_macro_f1` alone may select epochs where the TD signal has not taken hold.

**`td_pe_005` (TD weight 0.05 + PE weight 0.05):** Best OOD accuracy (0.887) and OOD macro-F1 (0.586) of Stage23. Temporal FE reduced to 0.31 — a meaningful improvement over baseline, though not the near-complete suppression seen in `td_005`. Preservation and predicate-disentanglement costs are smaller than in `td_005` but still large: paraphrase preservation = 0.583, predicate disentanglement = 0.717. Not a clean claim.

**Overall:** There is a sharp temporal-rejection / preservation trade-off along the TD supervision axis. The trade-off cannot be resolved by tuning TD weight alone.

---

## 7. Constrained Checkpoint Selection Results

TD-aware constrained checkpoint selection was implemented to explore whether a better epoch existed within each run that satisfied both temporal rejection and preservation constraints, using only clean-dev and TD-dev metrics (never Stage15).

**Final-decision constrained selector runs (`td_005_finalsel`, `td_pe_005_finalsel`):**

- Both fell back to unconstrained `final_macro_f1` selection (`eligible_epoch_count = 0`).
- Results matched their unconstrained counterparts (`td_005` and `td_pe_005`) exactly.
- No epoch within either run simultaneously satisfied temporal rejection constraints on the TD dev set and preservation constraints on the clean dev set.
- Conclusion: the failure is not a checkpoint selection artifact. No preservation-safe temporal improvement exists within the current training dynamics.

**Prior head-prob constrained selector runs (`td_003_sel`, `td_005_sel`, `td_pe_005_sel`):**

| Run | Temporal FE | Frame-loc FE | Frame-role FE | Outcome |
|-----|------------|--------------|---------------|---------|
| `td_003_sel` | 0.71 | 0.85 | 0.70 | Invalid — frame rejection collapse |
| `td_005_sel` | 0.84 | 0.25 | — | No improvement over baseline |
| `td_pe_005_sel` | 0.91 | 0.30 | — | Worse than baseline |

All prior constrained-selector runs either produced invalid results (frame collapse in `td_003_sel`) or selected epochs that provided no useful temporal improvement.

**Conclusion:** Constrained selection does not resolve the fundamental representation conflict.

---

## 8. Failure Mode Analysis

The primary failure mechanism is **gradient coupling through the shared frame representation**.

The TD head operates on `frame_pair_repr`, the output of `FrameGate`. When TD BCE loss flows back through `frame_pair_repr`, it also modifies the same representation that downstream preservation checks (`paraphrase_preserved`), predicate-disentanglement, and sufficiency judgments depend on.

Under sufficient TD weight (`td_005`):
1. The gradient pressure on `frame_pair_repr` from TD BCE loss pushes the representation to reject `time_swap` patterns.
2. This same representational shift causes the model to incorrectly assign higher NOT_ENTITLED probability to surface controls (`none`, `paraphrase`) and to paraphrase variants of preserved claims, causing FNE collapse.
3. Predicate disentanglement collapses because `frame_pair_repr` is also read by the predicate-coverage downstream path.

Under insufficient TD weight (`td_010`):
1. The gradient from TD loss is too small relative to the main classification and pairwise losses to persistently reshape `frame_pair_repr`.
2. The checkpoint selected by `final_macro_f1` tends to be one where main-task performance dominates and TD signal has not taken hold.

**The trade-off is structural, not a tuning problem.** Checkpoint selection, weight sweeping, and combination with PE loss cannot fully decouple temporal rejection from preservation when both are routed through the same `frame_pair_repr`.

---

## 9. Current Conclusion

Stage23 established the following facts:

1. **The temporal diagnostic signal is real.** TD supervision from a controlled `time_swap` dataset can transfer to Stage15 OOD temporal mismatch rejection, reducing false-entitled rate from 0.84 to as low as 0.01 (at `td_005`).
2. **The current implementation creates an unacceptable trade-off.** Under the shared `frame_pair_repr` routing, achieving temporal rejection requires degrading surface preservation, paraphrase preservation, and predicate disentanglement.
3. **Constrained checkpoint selection cannot recover a safe epoch.** No epoch within any Stage23 run simultaneously satisfied temporal rejection and preservation thresholds. The conflict is not a selection artifact.
4. **Stage23 is a diagnostic/negative result.** It should not be reported as a final improvement. The best OOD macro-F1 result (`td_pe_005` at 0.586 vs baseline 0.547) comes with preservation costs that are too large to be a clean claim, especially for a single seed.

---

## 10. Next Design Recommendation

**Do not proceed to 3-seed evaluation with the current Stage23 setup.** The preservation failure would replicate across seeds.

**Recommended next step: temporal residual adapter.**

Instead of routing TD supervision through the shared `frame_pair_repr`, introduce a small TD-specific adapter module:

- Keep the main `frame_pair_repr` / FrameGate path intact and unchanged.
- Branch a separate, small adapter from `frame_pair_repr` (or from an earlier representation) that is dedicated to temporal diagnostic supervision.
- Route TD BCE loss gradients into the adapter only, using a gradient stop (`detach()`) on the main `frame_pair_repr` input to the adapter, so TD gradients cannot propagate back into the shared frame representation.
- The adapter output feeds only `temporal_diagnostic_logit`; it never connects to `output["logits"]` or any main task path.

This preserves the output contract (`output["logits"] = final_logits`, `output["base_logits"] = base_logits`) and keeps all other losses unchanged. The architectural risk is much smaller than redesigning a full separate temporal channel.

**Alternative options for future investigation (in order of increasing cost):**

1. TD gradient guard via `frame_pair_repr.detach()` as adapter input — minimal code change, tests whether full gradient isolation is enough.
2. PCGrad / gradient conflict mitigation — more complex, modifies optimizer behavior, not recommended until the simpler adapter is tested.
3. Separate temporal representation — separate temporal slot-level branch trained end-to-end; largest change, highest potential, highest risk of unintended interactions.

**Do not claim:**
- Independent temporal channel (the current head is not independent).
- Final improvement over baseline (no preservation-safe result was achieved in Stage23).
- That 3-seed evaluation would confirm Stage23 results (it would replicate the trade-off, not resolve it).
