# ContraMamba-v6C Lean: Design Document

## 1. Purpose

This is a design proposal for the next controlled stage of ContraMamba development. It is not an implementation patch and not a final architecture claim.

v6C Lean defines the intended direction, permitted components, and evaluation criteria for the next architecture iteration after Stage21–24. It should be read alongside `experiments/active_component_policy.md` and `experiments/stage21_24_midpoint_review.md`. Nothing in this document authorizes adding new code until the design is reviewed and the active component policy is committed.

---

## 2. Why v6C Lean Is Needed

Stage21–24 produced genuine diagnostic evidence but left the project in a state of accumulated complexity with no preservation-safe final result:

**Stage23** demonstrated that a real temporal signal exists in `frame_pair_repr` and is detectable by a supervised temporal diagnostic head. However, routing temporal BCE supervision through `frame_pair_repr` (a shared representation used by FrameGate, PredicateCoverageHead, and SufficiencyGate) caused gradient coupling that collapsed paraphrase preservation (0.883 → 0.650) and predicate disentanglement (0.967 → 0.383) at weight ≥ 0.05. No epoch within any Stage23 run simultaneously satisfied temporal rejection and preservation thresholds. The trade-off was structural, not a tuning or checkpoint-selection artifact.

**Stage24** introduced a detached temporal residual adapter that gradient-isolates the temporal BCE from `frame_pair_repr`. The adapter alone (no final penalty) is a no-op on final behavior — confirming the `detach()` works as designed. Adding a per-example NOT_ENTITLED boost driven by the adapter improved the temporal/preservation frontier over Stage23 (surface FNE 0.23 vs 0.68 at comparable temporal FE), but preservation still degraded below acceptable levels at any penalty scale strong enough to produce meaningful temporal improvement. The preservation-constrained checkpoint selector recovered preservation by selecting earlier epochs but lost most of the temporal gain in the process. No Stage24 run is preservation-safe at a temporally meaningful operating point.

**Audit findings** after Stage24 showed that:
- The "baseline" includes an active ranking/intervention objective with a large weighted contribution. At default `ranking_weight`, `aux_to_ce_loss_ratio_weighted` is substantially above zero even before any optional auxiliary losses are added. Comparisons that call this CE-only are incorrect.
- Active components had never been formally tracked per-run. The audit ledger patch added this tracking, but it also revealed how many independently toggleable components exist.
- Comparator alphas (temporal, predicate) are architectural components trained through backpropagation, not post-hoc modifiers — but they had been documented as modifiers.

The conclusion is that the project should not continue stacking losses, selectors, and post-hoc penalties around the existing shared-representation architecture. The next step must separate channels: the temporal evidence pathway should not share its gradient path or its representation with the frame, predicate, and sufficiency judgment channels.

---

## 3. Guideline Alignment

ContraMamba is not intended to be a sequence classifier that maximizes OOD metrics through score calibration. Its purpose is an evidence-grounded epistemic judgment model that forms warranted entitlement claims by decomposing evidence into structured sub-judgments.

The intended judgment order is:

1. **Frame compatibility first.** Is the evidence temporally and structurally compatible with the claim frame? Temporal mismatch is a frame-compatibility failure — the evidence is from the wrong time, making it structurally non-entitling regardless of content.
2. **Predicate coverage second.** Does the evidence address the predicates asserted by the claim? Predicate noncoverage is a different failure from temporal mismatch — it is about what is asserted, not when it was asserted.
3. **Sufficiency and entitlement third.** Given compatible frame and covered predicates, is the evidence sufficient to license an entitlement decision?
4. **Polarity only after entitlement.** Given sufficient entitlement, does the evidence support or refute the specific polarity of the claim?
5. **Uncertainty and abstention as controlled judgment.** NOT_ENTITLED should reflect a model determination that entitlement conditions are not met, not a low-confidence fallback from a poorly calibrated classifier.

v6C Lean aligns with this order in two ways:

- **Temporal mismatch belongs before frame-level judgment, not alongside it.** The TemporalChannel should detect temporal incompatibility between evidence and claim at the evidence-level, before FrameGate processes paired representation. It should not share a representation with FrameGate because temporal incompatibility is a different kind of failure than frame-level inconsistency.
- **Preservation entitlement belongs at the sufficiency layer.** Paraphrase-preserved examples should be entitlement-preserving by construction. The PE head at the sufficiency layer is the right location for this signal, not a post-hoc gating term applied after final logits are produced.

v6C Lean respects this order by keeping each judgment channel in its appropriate position and not routing temporal signals through the FrameGate representation.

---

## 4. Proposed Architecture

The conceptual structure for v6C Lean:

```
Mamba encoder
  ↓
base_pair_repr
  ├── FrameGate branch         → frame_pair_repr → PredicateCoverage → Sufficiency → Polarity → final_logits
  ├── PredicateCoverage branch (reads frame_pair_repr)
  ├── Sufficiency / Entitlement branch (reads sufficiency_repr, includes PE head)
  ├── Polarity branch (reads polarity_repr)
  └── TemporalChannel branch   → temporal_channel_logit, temporal_channel_prob
```

**Critical design constraints:**

- **TemporalChannel does not use `frame_pair_repr` as its primary input.** `frame_pair_repr` is the output of FrameGate and is shared by PredicateCoverageHead and SufficiencyGate. Routing temporal supervision through it caused Stage23 collapse. TemporalChannel should read from `base_pair_repr` (the encoder output before FrameGate) or from a dedicated lightweight projection of `base_pair_repr` that is separate from the FrameGate path.

- **`detach=True` is the default for TemporalChannel input.** Unless gradient sharing is explicitly being tested, TemporalChannel's input should be `base_pair_repr.detach()`. This prevents TemporalChannel's BCE loss from propagating into the Mamba encoder weights and creating coupling with FrameGate's gradient path. Explicit gradient-sharing experiments are a separate test condition, not the default.

- **TemporalChannel is not a replacement for FrameGate.** FrameGate detects whether the evidence is structurally compatible with the claim frame. TemporalChannel detects whether the evidence is temporally compatible with the claim time reference. These are related but not equivalent failure modes. TemporalChannel supplements FrameGate; it does not replace it.

- **Temporal mismatch ≠ frame mismatch.** A temporally mismatched piece of evidence can be perfectly frame-compatible (same entities, same topic, correct polarity — but wrong time period). FrameGate should not be expected to detect this signal; TemporalChannel is the correct location.

- **Predicate swap ≠ frame failure.** Predicate noncoverage (the evidence addresses different predicates than the claim asserts) is a coverage failure, not a frame-compatibility failure. PredicateCoverageHead handles this. TemporalChannel should not be trained or evaluated against predicate-swap examples.

---

## 5. Final Arbitration Policy

**Bad direction (what Stage24 revealed as insufficient):**

```
final_logits += many independent shifts and penalties
             += temporal_adapter_penalty (unconditional, proportional to temporal prob)
             += dev-calibrated NE shift
             += comparator alpha adjustments
```

This fires on all examples in proportion to the relevant scores, regardless of whether the example is a temporal mismatch or a paraphrase-preserved surface control. The accumulation of independent adjustments makes attribution impossible and causes preservation damage at the scales needed for meaningful temporal improvement.

**Preferred direction:**

```
base/final logits from the normal judgment path
  + at most one local arbitration term, conditionally gated
```

**Candidate local arbitration term for v6C Lean:**

```
temporal_NE_boost = scale × temporal_channel_prob × (1 - preservation_entitlement_prob)
```

Applied per-example: `final_logits[:, NOT_ENTITLED] += temporal_NE_boost`

Interpretation:
- `temporal_channel_prob`: adapter/channel confidence that the example is a temporal mismatch (label 1 in temporal diagnostic data)
- `1 - preservation_entitlement_prob`: complement of the PE head's probability that evidence-claim entitlement is preserved; high when PE head predicts non-entitlement
- The product is near zero when either temporal confidence is low OR preservation entitlement is high
- The product is nonzero only when the model simultaneously believes the example is temporally mismatched AND that entitlement is not preserved

This prevents the penalty from firing on paraphrase-preserved examples (where `preservation_entitlement_prob` should be high) and on surface control examples where neither condition holds.

**What this arbitration term must not be:**

- It is not an OOD-calibrated shift. The `scale` parameter must be chosen on clean controlled dev, not on Stage15.
- It is not a global NE shift. It applies per-example, not uniformly.
- It is not a post-hoc model fix. It is one explicitly designed local arbitration mechanism. If it is active, it is the only active post-hoc modifier.
- It is not a substitute for structural model improvement. If this gating approach fails (preservation still collapses, or temporal signal is still insufficient), the correct response is to revisit the temporal channel architecture, not to add another modifier.

---

## 6. Loss Policy for v6C Lean

**Permitted loss families for initial experiments:**

| Loss | Status | Notes |
|------|--------|-------|
| CE on `output["logits"]` | Always active | Core task signal; weight = 1.0 |
| Ranking/intervention objective | Active (controlled/ranking baseline) | Explicit in audit ledger; report weighted contribution |
| TemporalChannel BCE | Optional; off by default | Supervises `temporal_channel_logit`; input must be from `base_pair_repr.detach()` path |
| Preservation entitlement BCE | Optional; off by default | Supervises `preservation_entitlement_logit` on sufficiency branch |

**Preferred initial experiment sequence:**

1. `v6c_tc_loss_only`: CE + ranking + TemporalChannel BCE. No final penalty. Verifies TemporalChannel can learn a temporal signal without corrupting the main model. Expected result: identical to baseline on final metrics (if `detach=True` works), with TemporalChannel accuracy > chance.
2. `v6c_tc_gated_penalty`: CE + ranking + TemporalChannel BCE + PE-gated arbitration. Tests whether the gated penalty produces temporal improvement with preserved surface controls.
3. `v6c_tc_gated_penalty_preservation`: adds PE BCE loss to the gated penalty run. Asks whether explicitly supervising the PE head improves gating quality.

**Do not combine in initial v6C experiments:**

- Direct temporal diagnostic loss through `frame_pair_repr` (Stage23 failure confirmed)
- Pair contrastive frame loss (not relevant to temporal channel design)
- Boundary head BCE (orthogonal to temporal channel question)
- Frame violation BCE (orthogonal)
- Multiple active selectors simultaneously
- Prototype memory
- Raw unconditional temporal adapter penalty (Stage24 failure mode confirmed)
- OOD-tuned shifts
- Dev-calibrated NE shift simultaneously with gated arbitration (would create two active post-hoc modifiers)

**Required audit reporting for every v6C run:**

- `loss_component_epoch_avg_raw` and `loss_component_epoch_avg_weighted`
- `aux_to_ce_loss_ratio_weighted`
- `active_training_losses` with `enabled` field for each component
- `active_final_logit_modifiers` with `enabled` field for each
- `active_architectural_logit_components`
- `active_selection_rules`
- `audit_warnings`

**Policy:** If `aux_to_ce_loss_ratio_weighted > 0.5`, the run is auxiliary-dominated and must be documented as such. This is not automatically disqualifying, but the claim interpretation must account for it.

---

## 7. Component Status Table

| Component | Status for v6C Lean | Rationale |
|-----------|--------------------|-|
| Mamba backbone (real) | **Keep** | Stage21 confirmed correct substrate; dummy is plumbing only |
| Clean no-time-swap main data | **Keep** | Permanent constraint; `time_swap` in temporal diagnostic file only |
| Stage15 eval-only discipline | **Keep** | Non-negotiable; any Stage15-derived training/selection decision invalidates a run |
| Frame/predicate/sufficiency/polarity decomposition | **Keep** | Architectural spine; intact in v6B and must remain intact in v6C |
| Audit ledger | **Keep** | Prerequisite for all v6C runs; no result interpretable without it |
| Comparator alphas (temporal, predicate) | **Keep as architectural** | Learned parameters in forward pass; not post-hoc modifiers; documented in `active_architectural_logit_components` |
| Direct temporal diagnostic loss through `frame_pair_repr` | **Diagnostic-only** | Stage23 confirmed gradient coupling causes preservation/predicate collapse; not final-model safe |
| OOD-tuned NE shift | **Diagnostic-only** | Stage15 derived; cannot be a final method; performance upper bound only |
| TD constrained checkpoint selector | **Diagnostic-only** | Post-hoc correction for structural gradient problem; does not fix the underlying model |
| Preservation constrained checkpoint selector | **Diagnostic-only** | Useful analysis instrument; confirmed that temporal/preservation trade-off is structural; not a final solution |
| Raw unconditional temporal adapter final penalty | **Diagnostic-only** | Fires on all examples including paraphrase-preserved; confirmed source of Stage24 preservation cost |
| Detached temporal residual adapter | **Promising; not yet final** | `detach()` gradient isolation is correct; penalty application method needs preservation gating |
| TemporalChannel (v6C) | **Promising; not yet final** | New proposal reading from `base_pair_repr`; gradient-isolated; yet to be tested |
| PE-gated local arbitration | **Promising; not yet final** | Single conditional modifier; directly addresses Stage24 failure mode; not yet tested |
| Prototype memory | **Defer** | Not relevant to single-example entitlement problem; introduces new information pathways not yet understood |
| Alternative Mamba hidden-layer extraction | **Defer** | Potentially useful but not the current bottleneck |
| Multi-dataset benchmark expansion | **Defer** | Not appropriate before controlled diagnostic setup is resolved |
| SOTA claim or paper framing | **Defer** | No Stage21–24 result meets multi-seed preservation-safe bar |
| Large architecture expansion before simplification | **Defer** | Increases audit burden without addressing the Stage24 frontier issue |

---

## 8. Minimal Implementation Plan

This is high-level only. No code changes are authorized until the design is reviewed and the active component policy is committed.

**Step 1: Add default-off `TemporalChannelV1` module to the model.**

A small MLP reading from `base_pair_repr`. Default: disabled. When enabled, `base_pair_repr.detach()` is the default input (same gradient-isolation principle as the Stage24 adapter). The module should be architecturally separate from the `temporal_residual_adapter` — different input source (`base_pair_repr` not `frame_pair_repr`), documented independently.

**Step 2: Add `temporal_channel_logit` and `temporal_channel_prob` to model output.**

These appear in `output` when TemporalChannel is enabled. They must not be used as `output["logits"]` and must not be included in any CE computation. They are auxiliary outputs only.

**Step 3: Add optional `use_temporal_channel_loss` CLI flag and BCE training.**

Supervised on the same temporal diagnostic data as the Stage23/24 heads (`time_swap` → label 1; `none`, `paraphrase` → label 0). The training uses `temporal_channel_logit` from a separate forward pass on the temporal diagnostic batch, not the main classification batch. Separate from main CE.

**Step 4: Add optional `use_temporal_channel_gated_penalty` CLI flag.**

If enabled, computes `temporal_NE_boost = scale × sigmoid(temporal_channel_logit).detach() × (1 - sigmoid(preservation_entitlement_logit).detach())` per example and applies it as a NOT_ENTITLED boost to `final_logits`. Both logits are detached — no gradient from the penalty back to either the TemporalChannel or the PE head. Scale is a fixed hyperparameter chosen on clean controlled dev.

**Step 5: Report all new active components through the audit ledger.**

`active_training_losses` must include `temporal_channel_loss` with `enabled`, `weight`, `target`, `gradient_isolated: True`, `input_source: "base_pair_repr.detach()"`. `active_final_logit_modifiers` must include `temporal_channel_gated_penalty` with `enabled`, `scale`, `type: "local_gated_by_preservation_entitlement"`, `stage15_used_for_selection_or_calibration: False`.

**Step 6: Keep all new flags default-off.**

The standard training command must produce identical behavior to the current v6B minimal baseline when no v6C flags are specified. No behavior change to existing code paths.

---

## 9. Minimal Experiment Plan

No Kaggle or Lightning commands are included. These are conceptual run names only. Actual commands should be derived from the v6C implementation and confirmed before any training is run.

| Run | Active losses | Gated penalty | Selection | Purpose |
|-----|--------------|---------------|-----------|---------|
| `v6c_tc_loss_only` | CE + ranking + TC BCE | Off | Standard clean-dev | Verify TC learns temporal signal without model corruption; expected: baseline identical |
| `v6c_tc_gated_penalty` | CE + ranking + TC BCE | On (TC × PE gate) | Standard clean-dev | First test of gated arbitration; should show temporal FE reduction with preserved surface FNE |
| `v6c_tc_gated_penalty_preservation` | CE + ranking + TC BCE + PE BCE | On (TC × PE gate) | Standard clean-dev | Test whether explicit PE supervision improves gate calibration |
| CE-only diagnostic baseline (optional) | CE only (`ranking_weight=0`) | Off | Standard clean-dev | Establish true CE-only reference point; isolates ranking objective contribution |

**Evaluation criteria per run (all must be reported):**

- Clean controlled dev macro-F1 (primary selection metric)
- Stage15 OOD macro-F1 (eval-only; no selection role)
- `temporal_mismatch` false-entitled rate (target: < 0.84 baseline)
- `surface_control` false-not-entitled rate (must not rise sharply above 0.09 baseline)
- `temporal_erased` false-not-entitled rate (should remain near baseline)
- `paraphrase_preserved` pass rate (must remain acceptable; < 0.70 is a preservation failure)
- `predicate_disentangled` pass rate (must not collapse; < 0.80 warrants investigation)
- `loss_component_epoch_avg_weighted` for all active components
- `aux_to_ce_loss_ratio_weighted`
- `audit_warnings` list (must be empty or explicitly explained)
- `active_training_losses` summary

**Multi-seed policy:** Do not scale to 3 seeds until seed 1 shows a clean, interpretable result — specifically, temporal FE improvement without surface FNE collapse and without paraphrase preservation falling below 0.75. A result that only holds at seed 1 before preservation is confirmed is a pilot, not a finding.

---

## 10. Risks

**Risk 1: TemporalChannel may still learn surface shortcuts.**
Reading from `base_pair_repr` does not guarantee the TemporalChannel learns temporal mismatch specifically. The encoder may encode surface features in `base_pair_repr` that correlate with temporal mismatch in the training data. The temporal diagnostic dataset design (time_swap vs. none/paraphrase) controls for this, but it is not guaranteed. Evaluation on `temporal_erased` (time-swap examples where the time reference is removed) is the primary check.

**Risk 2: The preservation gate may suppress true temporal mismatches.**
If the PE head's `preservation_entitlement_prob` is poorly calibrated — in particular, if it does not distinguish `time_swap` examples (low entitlement, high temporal prob) from paraphrase examples (high entitlement, low temporal prob) — then `(1 - preservation_entitlement_prob)` will be near zero for temporal mismatches and near one for paraphrases, reversing the intended gating. This would make the penalty fire on paraphrase examples and be suppressed on temporal mismatches — the exact wrong behavior.

**Risk 3: Preservation probability calibration may require explicit supervision.**
The PE head was introduced in Stage22 and trained with PE BCE loss. Whether it is adequately calibrated to distinguish `time_swap` (non-entitlement-preserving) from `paraphrase` (entitlement-preserving) without additional supervision is not known. The `v6c_tc_gated_penalty_preservation` run (with active PE BCE) tests whether this matters.

**Risk 4: If TemporalChannel BCE shares gradients through `base_pair_repr`, Stage23 collapse may recur.**
The `detach()` on `base_pair_repr` before TemporalChannel is the protection against this. If `detach=False` is ever tested, the Stage23 failure mode (temporal supervision corrupting shared representations) may reproduce. Stage23 showed that even a 0.05-weight temporal BCE through a shared representation is sufficient to collapse predicate disentanglement. Any gradient-sharing experiment must be clearly labeled and treated as diagnostic.

**Risk 5: Auxiliary losses may dominate CE if scaled without checking the ratio.**
The controlled/ranking baseline already has `aux_to_ce_loss_ratio_weighted` substantially above zero. Adding TemporalChannel BCE and PE BCE without checking the ratio could push auxiliary dominance to a level where CE is no longer the primary training signal. The audit warning threshold of `> 0.5` is a concrete check; if it fires, the weighting must be revisited before interpreting results.

**Risk 6: Too many knobs reintroduce the Stage24 interpretability problem.**
The v6C Lean plan has TC loss weight, TC penalty scale, PE loss weight, PE gate threshold, and the comparator alphas as simultaneously active tunable parameters in the gated penalty runs. This is already four or five knobs. The commitment to test one configuration at a time and not stack them without ablation is the safeguard. If experiments start combining two or more new parameters simultaneously, the Stage24 problem recurs.

---

## 11. Stop Conditions

Abandon or pause the v6C Lean direction and return to design review if any of the following are observed:

- `aux_to_ce_loss_ratio_weighted` exceeds 1.0 in any v6C run without explicit justification and ablation.
- `surface_control` FNE rises sharply above 0.20 in any run with active gated penalty (indicating the gate is not suppressing the penalty on surface controls).
- `paraphrase_preserved` pass rate falls below 0.70 in any run that claims temporal improvement (preservation failure threshold from the active component policy).
- `predicate_disentangled` pass rate falls below 0.80, indicating gradient coupling is occurring.
- The audit ledger shows more than one true post-hoc final-logit modifier active in any final-candidate run.
- Stage15 or OOD labels are found to have been consulted at any point other than final evaluation.
- The implementation requires adding more than one new model module and more than three new CLI flags in a single patch (sign that the plan has expanded beyond "lean").
- `v6c_tc_loss_only` does not produce baseline-equivalent final metrics (indicating `detach=True` is not correctly isolating gradients).

The last condition is a critical early check: if the loss-only run is not baseline-equivalent, something is wrong with the gradient isolation, and the gated penalty experiments should not proceed.

---

## 12. Current Conclusion

v6C Lean should be treated as the next controlled design candidate, not the final model.

The goal is specific: test whether temporal evidence can be separated from frame/predicate/sufficiency judgments — by moving the temporal signal to a dedicated channel reading from `base_pair_repr` — without recreating the Stage23 preservation collapse or the Stage24 unconditional-penalty trade-off.

If the gated arbitration (`TC prob × (1 - PE prob)`) works, it would resolve the key Stage24 failure: the penalty currently fires on paraphrase-preserved examples because it is unconditional. A PE-gated term suppresses it on those examples by construction.

If it does not work, the likely failure modes are PE miscalibration (Risk 2/3) or TemporalChannel learning surface features instead of temporal ones (Risk 1). Both are diagnosable from the per-run metrics: preservation check shows PE gate failure; temporal_erased FNE shows surface-feature learning.

This design is conservative by intent. It introduces one new module (TemporalChannel reading `base_pair_repr.detach()`), one new gating formula, and no new selectors. It does not change the model's judgment structure, the output contract, the data pipeline, or the checkpoint selection default. Every run is auditable through the existing ledger infrastructure.

The project should not proceed beyond this design until the active component policy is committed and the v6C Lean design is reviewed.
