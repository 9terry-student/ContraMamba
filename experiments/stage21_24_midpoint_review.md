# Stage21–Stage24 Midpoint Review

## 1. Purpose of This Document

This is a deliberate pause before adding more components to the ContraMamba training pipeline. It is not a final model proposal and not a performance claim. It is a checkpoint document intended to prevent project direction from drifting through accumulated complexity after a long sequence of experiments.

The review asks four questions:

1. What has actually been learned from Stage21–24?
2. Which results and mechanisms are robust enough to preserve?
3. Which mechanisms are diagnostic only and should not be promoted to final-model status?
4. What conditions should be true before the next architecture change?

---

## 2. Current Project Status

ContraMamba is in a **controlled diagnostic phase**, not a final-model phase.

- No Stage21–24 result has been confirmed across multiple seeds with clean preservation metrics.
- No result is ready to be reported as a final improvement over baseline.
- The best temporal OOD result (`ta_p035_pe002`, macro-F1 = 0.620, temporal FE = 0.07) carries a paraphrase preservation cost (0.600) that disqualifies it as a final claim.
- The best preservation-safe temporal result achieves only modest improvement (FE 0.84 → 0.76).

The appropriate frame is: Stage21–24 is controlled diagnostic progress within a model-design phase. Results are real and informative. They do not constitute a final model.

---

## 3. Guideline Alignment Check

ContraMamba is not intended to be merely a Mamba sequence classifier. Its long-term goal is an **evidence-grounded epistemic judgment/controller model** — a system that forms warranted entitlement claims by decomposing evidence into structured sub-judgments before reaching a final label.

The core judgment order should remain:

1. **Frame compatibility first** — Is the evidence temporally and structurally compatible with the claim frame?
2. **Predicate coverage second** — Does the evidence address the predicates asserted by the claim?
3. **Sufficiency and entitlement** — Is the evidence sufficient to license an entitlement decision?
4. **Polarity** — Does the evidence support or refute the specific polarity of the claim?
5. **Uncertainty / abstention** — Under what conditions should the model withhold judgment?

This order reflects an epistemically principled architecture. Shortcuts that bypass it — score-based NE shifts, raw logit penalties without structured gating, loss combinations that blur the judgment channels — may improve OOD metrics temporarily but undermine the long-term design goal.

**Why confidence or OOD accuracy alone is not enough:**
A model that achieves high OOD accuracy by collapsing the NOT_ENTITLED threshold globally is not making a principled judgment. It is tuning a decision surface without grounding. ContraMamba's value is in making the sub-judgments interpretable and individually testable — frame compatibility, predicate coverage, sufficiency — not in maximizing a single number at one evaluation point.

---

## 4. Stage21 Baseline Recap

Stage21 established the real Mamba baseline for controlled evaluation. Key facts:

- The baseline Mamba model trained on `data/controlled_v5_v3_without_time_swap.jsonl` achieves clean controlled dev macro-F1 ≈ 0.962, paraphrase_preserved ≈ 0.883, predicate_disentangled ≈ 0.967.
- Stage15 OOD evaluation was established as **eval-only**: Stage15 is never used for training, calibration, threshold selection, shift selection, candidate selection, checkpoint selection, adapter construction, or penalty selection.
- `time_swap` was removed from the main training data and placed in a separate temporal diagnostic file. This separation is a permanent constraint.
- Temporal mismatch false-entitled rate at baseline: 0.84. This was identified as the primary OOD failure mode.

The baseline is a legitimate reference point. Any future result should be compared against it on both OOD metrics and clean preservation metrics simultaneously.

---

## 5. Stage22 Recap

Stage22 explored shift-based and calibration-based approaches to the NE false-entitled problem.

Key findings:

- OOD-tuned NE shift achieved meaningful OOD improvement but was derived from Stage15 OOD results directly. This violates the Stage15 eval-only discipline and is not valid as a final method.
- Dev-only calibration (without OOD access) did not recover the OOD-tuned improvement. The OOD-calibrated gain did not generalize to clean calibration.
- Pair contrastive frame loss and other auxiliary losses were introduced but did not independently solve the OOD failure.

**Conclusion from Stage22:** Shift-based fixes are diagnostic only. They demonstrate that the model's NE decision boundary is adjustable and that temporal mismatch sensitivity can in principle be increased. But shift tuning is not a principled model design, and OOD-derived shifts are methodologically invalid as final model components. Stage22 results should not be carried forward as final-model components.

---

## 6. Stage23 Recap

Stage23 introduced a temporal diagnostic head supervised on a separate `time_swap`-based dataset.

Key findings:

- Direct TD supervision (weight=0.05) reduced temporal mismatch FE from 0.84 to 0.01 — the strongest temporal improvement seen in any stage.
- However, paraphrase preservation collapsed from 0.883 to 0.650 and predicate disentanglement from 0.967 to 0.383. Surface control FNE rose from 0.09 to 0.68.
- The failure mechanism is structural: TD gradients flowing through the shared `frame_pair_repr` shifted FrameGate toward temporal discrimination, corrupting the representation used by preservation and predicate checks.
- Constrained checkpoint selection (TD-aware and final-decision) did not recover a preservation-safe epoch. No epoch within any Stage23 run simultaneously satisfied temporal rejection and preservation thresholds.
- The trade-off is not a tuning or checkpoint-selection artifact. It is a gradient routing problem.

**Conclusion from Stage23:** Direct temporal diagnostic supervision through `frame_pair_repr` is not final-model safe. It confirms a real temporal signal exists in the data and is detectable from the frame representation, but the shared-representation architecture creates an unacceptable conflict. The conflict must be resolved at the architecture level, not the weight or selector level.

---

## 7. Stage24 Recap

Stage24 introduced a detached temporal residual adapter with an optional per-example final-logit penalty.

Key findings:

- **Adapter loss only:** Identical to baseline when final penalty is disabled. The `detach()` on the adapter input correctly blocks gradient propagation into FrameGate. This is the cleanest result of Stage24.
- **Weak temporal penalty (scale=0.25):** Small temporal improvement (FE 0.84→0.77) with predicate disentanglement intact (0.967). Paraphrase preservation dropped (0.883→0.733). Modest but structurally safer than Stage23.
- **Strong temporal penalty + PE loss (`ta_p035_pe002`):** Temporal FE = 0.07, OOD macro-F1 = 0.620, surface FNE = 0.23. This is substantially better than Stage23's direct TD at similar temporal FE (surface FNE was 0.68 there). However, paraphrase preservation = 0.600 — not acceptable for a final claim.
- **Preservation-constrained selection:** The clean-dev-only selector worked as designed (Stage15 not used, eligible epochs found, no fallback triggered). But it revealed that preservation and temporal improvement are coupled through the penalty scale — no epoch in any run simultaneously satisfies the preservation thresholds and achieves strong temporal rejection.

**Conclusion from Stage24:** The detached adapter moves the temporal/preservation trade-off frontier in the right direction but does not eliminate the trade-off. The frontier has genuinely improved over Stage23 (lower surface FNE at comparable temporal FE, better predicate disentanglement), but no Stage24 configuration is preservation-safe enough for a final claim. The trade-off persists because the raw temporal penalty fires on all examples proportionally to adapter confidence — it is not gated by preservation entitlement or any evidence-quality check.

---

## 8. Complexity Warning

The current `scripts/train_controlled_v6b_minimal.py` contains the following optional active components, most of which are independently controllable at the CLI:

**Training losses:**
- CE loss (always active)
- Intervention ranking losses (frame preserve, frame anchor, predicate contrast, predicate anchor, sufficiency contrast, polarity flip, paraphrase preserve, entitlement preserve, logit preserve)
- Boundary head BCE
- Frame violation head BCE
- Predicate isolation head BCE
- Preservation entitlement head BCE
- Temporal diagnostic head BCE
- Temporal residual adapter BCE

**Final-logit modifiers:**
- Temporal comparator alpha (learned)
- Predicate comparator alpha (learned)
- Temporal adapter final penalty (per-example)

**Checkpoint selection rules:**
- Default: unconstrained final_macro_f1
- TD-constrained selection (head-prob and final-decision variants)
- Generic preservation-constrained selection

This is a large number of independent components. Each additional component:

1. Increases the difficulty of attributing any observed behavior to a specific mechanism.
2. Introduces potential gradient conflicts that are hard to detect without explicit component tracking.
3. Makes ablation more expensive (more runs needed per experiment).
4. Creates audit risk: it becomes possible to tune toward OOD metrics by adjusting component combinations without a principled model design argument.

The aux-to-CE loss ratio is currently untracked. There is no formal check on whether auxiliary losses are dominating the CE signal in late-stage training. This is a known risk given the number of optional BCE objectives.

The core judgment sub-modules (FrameGate, PredicateCoverageHead, SufficiencyGate, PolarityEnergyHead, FinalEntitlementDecisionHead) are still architecturally intact. The risk is not that they have been damaged, but that layers of auxiliary supervision, modifiers, and selectors are being added around them without a clear principle for when to stop.

---

## 9. What Should Be Preserved

The following results, constraints, and design principles are robust and should be preserved unconditionally:

- **Real Mamba backbone.** Stage21 confirmed that the Mamba backbone is the correct substrate for this work. The dummy backbone is for testing only.
- **Clean no-time-swap main data.** `data/controlled_v5_v3_without_time_swap.jsonl` is the permanent main training/eval dataset. `time_swap` records belong only in the temporal diagnostic file.
- **Stage15 eval-only discipline.** No result derived from Stage15 access may be used for training, calibration, or selection. This is a non-negotiable methodological constraint.
- **Frame/predicate/sufficiency decomposition.** The V5/V6B sub-module decomposition (FrameGate → PredicateCoverage → SufficiencyGate → PolarityEnergy → FinalDecision) should be preserved as the architectural spine.
- **Gradient isolation principle.** The `detach()` insight from Stage24 is correct and should be the default for any future auxiliary head that reads a shared representation. No auxiliary head should be allowed to propagate gradients into a shared representation without explicit justification.
- **Preservation-constrained selection as an analysis tool.** The clean-dev-only preservation-constrained selector is a useful diagnostic for understanding the epoch-level temporal/preservation trade-off. It is not a final solution, but it is a principled analysis instrument.

---

## 10. What Should Be Downgraded to Diagnostic-Only

The following mechanisms have produced informative experimental results but should not be promoted to final-model components:

- **OOD-tuned NE shifts (Stage22).** Any shift derived from Stage15 access is methodologically invalid. It can be reported as a performance upper bound, but it cannot be a final design component.
- **Direct temporal diagnostic supervision through `frame_pair_repr` (Stage23).** Stage23 confirmed this causes unacceptable preservation collapse. It is useful as a research negative result, not as a model design.
- **Raw temporal final-logit penalty without preservation gating (Stage24).** The unconditional penalty fires on preservation-safe examples (paraphrases, surface controls) in proportion to adapter confidence. This is the proximate cause of the remaining Stage24 trade-off. The penalty may be a useful diagnostic but is not a principled final-model component without entitlement gating.
- **Selector-only fixes for structural failures.** Constrained checkpoint selection is a post-hoc correction, not a solution. When a selector consistently falls back or selects epochs with weaker task performance to recover preservation, it is signaling that the training dynamics are wrong, not that the selector should be tuned further.

---

## 11. What Should Be Deferred

The following directions are potentially valuable but should not be started until the current architecture is simplified and audited:

- **Prototype memory.** An interesting long-term direction for evidence accumulation over multi-hop chains, but not relevant to the current single-example entitlement problem. Deferred until the base temporal/preservation frontier is resolved.
- **Alternative Mamba hidden-layer extraction.** Reading from intermediate Mamba layers rather than the final hidden state could improve representation quality. Deferred until the loss/head audit is complete.
- **Large independent temporal architecture (full TemporalChannel).** The Stage24 next-direction recommendation includes an independent temporal channel reading from `base_pair_repr`. This is the right direction but requires a clean codebase with minimal competing auxiliary objectives before it can be properly evaluated.
- **Multi-dataset final benchmark expansion.** ContraMamba is not ready to be evaluated on additional benchmarks. The controlled diagnostic setup should be resolved first.
- **SOTA claim or paper-ready performance framing.** No Stage21–24 result meets the bar for a clean multi-seed final claim. Any SOTA framing is premature.

---

## 12. What Should Be Considered Next, After This Review

The following steps are appropriate after this review is committed, in priority order:

**First: loss/logit/selector audit ledger.** Before any new mechanism is added, produce a structured per-run ledger that records:
- `active_training_losses`: which loss components contributed nonzero gradients during training
- `active_final_logit_modifiers`: which adjustments to final logits were active at eval time
- `active_selection_rules`: which checkpoint selector was active; whether fallback was triggered
- `loss_component_epoch_avg`: mean per-loss value averaged over training epochs
- `aux_to_ce_ratio`: ratio of total auxiliary loss to CE loss, as a complexity signal

This ledger should be automatically generated and attached to every run report. It is a prerequisite for meaningful ablation.

**Second: active component simplification policy.** At any given time, the model should have at most:
- One auxiliary supervised head active per representation channel (frame, predicate, sufficiency)
- One final-logit modifier at a time (comparator alpha OR temporal penalty, not both without a gating rule)
- One checkpoint selection rule (default or pres-constrained, not both; not combined with TD-constrained)

**Third: entitlement-gated temporal penalty.** If the temporal adapter penalty is retained in any future run, it should be gated by preservation entitlement: apply the NOT_ENTITLED boost only when the preservation entitlement head predicts non-entitlement (PE score below threshold). This prevents the penalty from firing on paraphrase-preserved examples. This is a single, principled change that directly addresses the Stage24 failure mode without adding new components.

**Fourth: independent temporal channel prototype (deferred to after audit).** If the gated penalty approach does not resolve the trade-off, the appropriate next step is a fully independent temporal channel that reads from `base_pair_repr` (encoder output before FrameGate), not `frame_pair_repr`. This keeps the temporal signal separate from the FrameGate path entirely. This should be prototyped as a dedicated, minimal experiment — not layered on top of the current Stage24 configuration.

---

## 13. Recommended Pause Policy

The following are concrete constraints for the period immediately following this review:

1. **No new architecture changes until this review is committed.** This document must be written, reviewed, and committed before any new model component (head, loss, penalty, channel) is added.

2. **No OOD-informed hyperparameter search.** Do not run sweeps where Stage15 results influence which hyperparameter is tried next. All hyperparameter decisions must be justifiable from clean-dev metrics and architectural reasoning alone.

3. **No 3-seed scaling until the model path is simplified.** Multi-seed evaluation is expensive and should be reserved for a configuration that is (a) architecturally principled, (b) preservation-safe at the chosen operating point, and (c) not dependent on a combination of auxiliary objectives whose individual contributions are unaudited.

4. **No prototype memory until temporal/preservation frontier is understood.** Prototype memory adds a new information pathway that would interact with all existing heads and losses. Do not introduce it while the temporal/preservation trade-off is unresolved.

5. **Document before adding.** For every new mechanism (loss, head, penalty, selector), write a one-paragraph design rationale that explains (a) which judgment channel it is intended to improve, (b) which representation it reads, (c) whether its gradient path is isolated, and (d) what the expected failure mode is if it does not work. This rationale should appear in the experiment document, not just the commit message.

---

## 14. Current Conclusion

Stage21–24 produced real, reproducible controlled diagnostic evidence. The key findings are:

- A real temporal signal exists in `frame_pair_repr` and is detectable by a probe trained on the controlled temporal diagnostic dataset.
- Gradient isolation via `detach()` is the correct principle for auxiliary heads that read shared representations.
- The temporal/preservation trade-off is structural in the current configuration. It can be moved (Stage24 improved the frontier over Stage23) but not eliminated by weight tuning, loss combinations, or checkpoint selection alone.
- OOD metric improvements without clean preservation maintenance are not final results.

ContraMamba is still in a model-design phase. The appropriate response to Stage24 is not to continue adding mechanisms in the same direction, but to pause, audit what is active, and design the next single change carefully.

The next step is simplification and audit, not acceleration toward a final model.
