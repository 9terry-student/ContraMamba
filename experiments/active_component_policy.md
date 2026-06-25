# ContraMamba Active Component Policy

## 1. Purpose

This is a policy and checkpoint document, not a final model proposal. It is written after the Stage21–24 midpoint review and after the audit ledger patch was committed to `scripts/train_controlled_v6b_minimal.py`.

The audit ledger can now report which losses, final-logit modifiers, selection rules, and architectural components were active in any given run, along with raw and weighted loss averages and aux-to-CE ratios. That machinery is a prerequisite. This document defines what those reports should contain in a valid run, what combinations are allowed or forbidden, and what must be true before the next architecture change is introduced.

The audience is the project team and future sessions that continue this work. The goal is to prevent the project from drifting through accumulation of uncoordinated optional components.

---

## 2. Core Principle

A single training run should have a small number of intentionally active causal mechanisms. The audit ledger should be readable and interpretable. The following rules follow from this:

1. **One active causal hypothesis per experiment.** A run that simultaneously changes the training loss family, the final-logit modifier, and the checkpoint selector is testing three things at once. If it improves, the cause is unattributable. If it degrades, the cause is equally unattributable.

2. **Audit-readability is a prerequisite for claimability.** A result is not claimable if the audit ledger shows more than one non-default mechanism active simultaneously without explicit ablation evidence for each.

3. **OOD / Stage15 is eval-only, with no exceptions.** Stage15 cannot be used for training, calibration, shift selection, hyperparameter selection, candidate selection, checkpoint selection, adapter construction, penalty selection, or dataset construction. Any field that reads `stage15_used_for_* = true` in the audit ledger invalidates that mechanism as a final-model component.

4. **The baseline is not necessarily CE-only.** The v6B training script includes a ranking/intervention objective that is active when `ranking_weight > 0`. At default settings, the `aux_to_ce_loss_ratio_weighted` is substantially above zero even without any explicitly added auxiliary losses. Any comparison that calls itself a "CE-only baseline" must verify this from the audit ledger. Future claims must distinguish between:
   - CE-only baseline (`ranking_weight=0`, no auxiliaries)
   - Controlled/ranking baseline (CE + ranking/intervention objective)
   - Auxiliary-loss variants (controlled baseline + one auxiliary family)
   - Final-logit modifier variants (any of the above + one post-hoc modifier)
   - Selector variants (any of the above + a non-default selector)

---

## 3. Component Taxonomy

### A. Core Model Architecture

These are fixed structural components of the current model. They are not toggleable per-experiment and are not optional unless explicitly removing or replacing them.

- **Mamba backbone**: the primary encoder. Real Mamba only; dummy backbone is for plumbing tests only.
- **FrameGate → PredicateCoverageHead → SufficiencyGate → PolarityEnergyHead → FinalEntitlementDecisionHead**: the V5/V6B sub-module decomposition. This is the architectural spine. Any change to this ordering or composition is a major architecture change, not an auxiliary experiment.
- **`output["logits"] = final_logits` and `output["base_logits"] = base_logits`**: these output keys are inviolable. No experiment should change what `output["logits"]` represents. CE loss always uses `output["logits"]`.
- **Architectural logit components** (comparator alphas): `temporal_comparator_alpha` and `predicate_comparator_alpha` are learned `nn.Parameter` scalars applied inside `model.forward()` via flag gating. They are part of the model graph and trained through backpropagation. They are not post-hoc modifiers. They appear in `active_architectural_logit_components` in the audit ledger.

### B. Training Losses

These are optional and individually weighted. Each adds to `total_loss` in the training step.

| Loss | Default state | Targets | Notes |
|------|--------------|---------|-------|
| CE (`label` component of `losses["total"]`) | Always active | `output["logits"]` | Core task signal |
| Ranking/intervention objective | Active by default via `ranking_weight > 0` | `output["logits"]` | Not CE-only when enabled; weight matters |
| Boundary head BCE | Off by default | `boundary_logit` | Diagnostic head; gradient isolated from final logits |
| Frame violation BCE | Off by default | `frame_violation_logit` | Diagnostic head |
| Pair contrastive frame loss | Off by default | `frame_violation_logit` (margin) | Diagnostic head |
| Predicate isolation BCE | Off by default | `predicate_noncoverage_logit` | Diagnostic head |
| Preservation entitlement BCE | Off by default | `preservation_entitlement_logit` | Diagnostic head |
| Temporal diagnostic BCE | Off by default | `temporal_diagnostic_logit` | Stage23: causes preservation collapse via shared `frame_pair_repr` at weight >= 0.05 |
| Temporal adapter BCE | Off by default | `temporal_adapter_logit` | Input is `frame_pair_repr.detach()`; gradient isolated |

Auxiliary losses labeled "diagnostic head" supervise auxiliary heads only and do not directly supervise `output["logits"]` or `output["base_logits"]`. This must remain true.

### C. Final-Logit Modifiers

These are post-hoc adjustments applied after the model's forward pass produces `output["logits"]`. They change the final prediction directly without passing through the training graph (or, in the case of the temporal adapter penalty, using a detached signal inside forward).

| Modifier | Type | Stage15 used | Final-model status |
|----------|------|-------------|-------------------|
| Temporal adapter final penalty | Local example-dependent | No | Diagnostic; causes preservation cost |
| Dev-calibrated NE shift | Global post-hoc | No (controlled dev only) | Diagnostic; not validated final method |
| OOD-tuned unflagged NE shift | Global OOD eval sweep | Yes | Diagnostic upper bound only |
| OOD-tuned selective NE shift | Selective OOD eval sweep | Yes | Diagnostic upper bound only |

### D. Selection Rules

These control which epoch's checkpoint is used as the final model.

| Rule | Source | Stage15 used | Notes |
|------|--------|-------------|-------|
| Standard clean-dev metric | Controlled dev | No | Default; always active |
| TD constrained selection | Controlled dev + TD metrics | No | Post-loop override; fallback to standard if no epoch qualifies |
| Preservation constrained selection | Controlled dev pairwise | No | Post-loop override; mutually exclusive with TD constrained |
| Dev-calibrated NE selector | Controlled dev (G2/G3) | No | Selects NE shift, not epoch |

### E. Evaluation-Only Diagnostics

These produce results that may be reported but cannot be used to select training decisions.

- Stage15 OOD group metrics and class-level breakdown
- OOD ablation modes
- OOD unflagged and selective NE shift sweeps (eval sweep results only)
- Temporal/preservation frontier analysis (comparing FE vs surface FNE across runs)
- Per-intervention dev breakdowns when used only for analysis

---

## 4. Allowed Default Policy

A safe default run is defined as:

1. **Data**: `data/controlled_v5_v3_without_time_swap.jsonl` only. No `time_swap` in main train/eval data. No Stage15 records in training.
2. **Evaluation**: Stage15 OOD evaluation runs after checkpoint selection. Stage15 results are not used to select checkpoints.
3. **Training loss**: CE plus at most one established core auxiliary objective. The ranking/intervention objective counts as an auxiliary objective for policy purposes even if it is active by default.
4. **Final-logit modifier**: none active in a default run. If a modifier is tested, it is the only non-default modifier in that run.
5. **Checkpoint selection**: standard clean-dev metric selection. No selector stacking. TD constrained and preservation constrained are not used simultaneously.
6. **Shifts**: no OOD-tuned shift. No global NE shift except as explicitly diagnostic.
7. **Prototype memory**: not active. Defer until temporal/preservation frontier is resolved.

---

## 5. Loss Policy

1. **Always report raw and weighted loss components.** The audit ledger `loss_component_epoch_avg_raw` and `loss_component_epoch_avg_weighted` must both be present in the run report.

2. **Always report `aux_to_ce_loss_ratio_weighted`.** This is the primary complexity signal. It must be read before any training result is interpreted.

3. **If `aux_to_ce_loss_ratio_weighted > 0.5`, the run is auxiliary-dominated.** This is not necessarily wrong, but it must be explicitly acknowledged. A claim based on an auxiliary-dominated run without ablation of which loss component drives the improvement is not claimable.

4. **Distinguish run types clearly.** Do not describe a run with `ranking_weight > 0` as a CE-only baseline. The audit ledger `active_training_losses["ranking_loss"]["enabled"]` must be checked. Baseline comparisons must use the same run type.

5. **Do not add a new auxiliary loss until the previous auxiliary mechanism is understood.** If two auxiliary losses are active simultaneously and one or both is new, the result is not attributable. The Stage21–24 setup accumulated up to seven optional loss components. This should not increase without completing ablations on existing ones.

6. **Prefer CE + one auxiliary family for clean ablations.** An auxiliary family is a coherent group targeting the same judgment channel (e.g., frame-quality losses = frame violation + boundary head + pair contrastive; temporal losses = temporal diagnostic OR temporal adapter, not both). Mixing families in one run without ablation evidence makes the contribution attribution unclear.

7. **The V5 controlled losses sub-components (frame, predicate, sufficiency, polarity) are always active inside `losses["total"]`.** These are not separately audited per-epoch but contribute to `total_loss` above the CE component. The difference between `total_loss` and `ce_loss` in `loss_component_epoch_avg_weighted` reflects both the ranking/intervention objective and these V5 sub-losses.

---

## 6. Final-Logit Modifier Policy

1. **At most one true post-hoc final-logit modifier should be active in a final-candidate run.** The audit ledger `active_final_logit_modifiers` counts enabled post-hoc modifiers. If the count exceeds one, the result is not claimable without individual ablations.

2. **OOD-tuned shifts are diagnostic-only.** Any shift where Stage15 results were used to select or tune the shift value cannot be a final-model component. It can be reported as a performance upper bound, but it cannot be presented as a model design. The audit flag `stage15_used_for_selection_or_calibration: true` identifies these.

3. **Global NE shifts are diagnostic-only unless separately justified.** A global NE shift changes the decision threshold for all examples. This is not a model design choice; it is a post-hoc calibration. It can reveal what an ideal calibration would look like, but it does not constitute a model that achieves that calibration principled.

4. **The temporal adapter final penalty is a high-risk mechanism.** Unlike training losses that affect only auxiliary heads, the temporal adapter final penalty directly modifies `output["logits"]` (the final classification decision) on every forward pass. It fires proportionally to the adapter's temporal mismatch confidence for every example — including paraphrase-preserved and surface-control examples that should not receive NOT_ENTITLED boosts. Stage24 confirmed this is the proximate cause of the preservation cost at scales above 0.25. Future use of this mechanism must document: the scale value, how it was chosen, whether preservation entitlement gating was applied, and what the surface FNE cost was.

5. **Local example-dependent arbitration (comparator alphas) is architectural, not a post-hoc modifier.** The comparator alphas are trained parameters that appear in the model graph. They are documented in `active_architectural_logit_components`, not `active_final_logit_modifiers`. Enabling or disabling them is an architectural decision, not a post-hoc tuning decision.

---

## 7. Selection Policy

1. **Standard selection uses clean controlled dev only.** The primary selection metric (`final_macro_f1` or equivalent) is computed on `data/controlled_v5_v3_without_time_swap.jsonl` dev split. This is always the first-line selection signal.

2. **Stage15/OOD cannot select checkpoints.** No selector may consult Stage15 labels, Stage15 metrics, or any OOD-derived signal to choose which epoch or model state to use. The audit fields `stage15_used_for_checkpoint_selection: false` must hold.

3. **TD constrained and preservation constrained selectors are mutually exclusive.** Enabling both simultaneously is a `ValueError` at startup. Each is a post-loop override of `best_epoch`. Combining them would create ambiguous override semantics with no principled priority rule.

4. **Selector-only fixes are diagnostic, not solutions.** When a constrained selector consistently falls back to the unconstrained best, it is signaling that no epoch satisfies the constraints — meaning the training dynamics are producing a trade-off that no checkpoint can resolve. The selector cannot fix this; the model design must.

5. **The Stage24 preservation-constrained selection result is analysis evidence, not a final solution.** The selector worked correctly — it found eligible epochs, fallback was not triggered, and Stage15 was not used. But it revealed that the temporal improvement and the preservation cost are coupled through the penalty scale. Selecting an earlier epoch recovers preservation but loses most of the temporal gain. This confirms the trade-off is real and structural, not a checkpoint-selection artifact.

6. **Fallback-triggered selectors generate an audit warning.** If `preservation_constrained_selection_fallback_used` or `td_constrained_selection_fallback_used` is true in the ledger, the result is based on the unconstrained best epoch, not the constrained one. The constrained selector label in the run name is then misleading.

---

## 8. Stage21–24 Component Classification

### Keep as Core / Safe

These results and constraints are robust and unconditional.

- **Real Mamba backbone.** Stage21 confirmed this is the correct substrate. Dummy backbone is plumbing only.
- **Clean no-time-swap main data.** `data/controlled_v5_v3_without_time_swap.jsonl` is permanent. `time_swap` records belong only in the separate temporal diagnostic file.
- **Stage15 eval-only discipline.** Non-negotiable methodological constraint. No result derived from Stage15 access may be used for any training-time decision.
- **Frame/predicate/sufficiency decomposition.** The V5/V6B architectural spine is intact. Any future model must preserve this structure or explicitly replace it with a documented substitute.
- **Gradient isolation via `detach()` for auxiliary heads reading shared representations.** Stage24 demonstrated this is the correct principle. Any future auxiliary head that reads a shared representation must detach its input by default.
- **Audit ledger.** The reporting infrastructure added after Stage24 is now a prerequisite for any new experiment. A run without a readable audit ledger is not interpretable.

### Diagnostic-Only (Not Final-Model Components)

These mechanisms produced informative results but cannot be final model design components.

- **OOD-tuned NE shifts (Stage22).** Derived from Stage15 access. Diagnostic upper bound only.
- **Direct temporal diagnostic loss through `frame_pair_repr` (Stage23).** Causes preservation and predicate collapse at weight ≥ 0.05. Not recoverable by selector. Negative result confirmed.
- **TD constrained checkpoint selector.** A post-hoc correction for a gradient routing problem. When it falls back, the model is wrong; when it does not fall back, the improvement is real but small. It is an analysis instrument, not a model design.
- **Preservation constrained checkpoint selector (in the Stage24 context).** Confirmed the trade-off is structural. Useful for analysis. Not a solution on its own.
- **Raw temporal final-logit penalty without preservation entitlement gating (Stage24).** Fires on all examples proportional to adapter confidence. Produces preservation cost at scales above 0.25. Not valid as a final component at the scales needed for meaningful temporal improvement.

### Promising but Not Final

These directions are worth pursuing in a future stage, but have not yet been confirmed as final-model safe.

- **Detached temporal residual adapter** as a temporal signal probe. The `detach()` architecture is correct; the penalty application method is not yet clean.
- **Preservation-entitlement-gated temporal penalty.** The hypothesis: apply the NOT_ENTITLED boost only when the PE head signals entitlement failure, preventing the penalty from firing on preserved examples. Stage24 identified this as the most promising minimal next step. Not yet implemented or tested.
- **Independent temporal channel reading from `base_pair_repr`.** Reading the temporal signal from encoder output before FrameGate, not from `frame_pair_repr`, would fully decouple temporal diagnosis from the FrameGate representation. Not yet implemented.

### Defer

These directions should not be started until the current model is simplified and the preservation/temporal trade-off is resolved.

- **Prototype memory.** Not relevant to the current single-example entitlement problem. Introduces new information pathways that would interact with all existing heads and losses.
- **Alternative Mamba hidden-layer extraction.** Reading from intermediate Mamba layers rather than the final hidden state. Potentially valuable but not the current bottleneck.
- **Multi-dataset final benchmark expansion.** Not appropriate until the controlled diagnostic setup is resolved.
- **SOTA claim or paper-ready performance framing.** No Stage21–24 result meets the bar for a clean multi-seed final claim.
- **Large architecture expansion before simplification.** Adding more components in the current direction increases audit burden without addressing the structural frontier issue confirmed in Stage24.

---

## 9. Before the Next Architecture

Before introducing v6C Lean or a TemporalChannel prototype, all of the following must be true:

1. **The audit ledger must pass.** The run report for any Stage25+ experiment must include `active_training_losses`, `active_final_logit_modifiers`, `active_architectural_logit_components`, `active_selection_rules`, `loss_component_epoch_avg_raw`, `loss_component_epoch_avg_weighted`, `selected_epoch_loss_component_avg`, `final_epoch_loss_component_avg`, and `aux_to_ce_loss_ratio_weighted`.

2. **This active component policy must be committed to the repository.**

3. **Every next experiment must specify, in its design document before training:**
   - Which losses are active (`enabled: true` in the audit ledger)
   - Which final-logit modifiers are active (expected to be none in default runs)
   - Which selection rule is active
   - Expected `aux_to_ce_loss_ratio_weighted` range
   - Run classification: CE-only / controlled-ranking / loss-ablation / modifier-ablation / diagnostic

4. **The description of each new mechanism must include:**
   - Which judgment channel it is intended to improve (frame, predicate, sufficiency, temporal, polarity)
   - Which representation it reads
   - Whether its gradient path is isolated from shared representations
   - What the expected failure mode is if it does not work

---

## 10. Recommended Next Step After This Document

After this policy is committed:

1. **Design v6C Lean** — a simplified configuration that keeps CE + ranking/intervention objective, removes all inactive auxiliary heads, and tests the PE-gated temporal penalty as the single active modifier. This is one new mechanism. It should be documented in an experiment design note before any training runs.

2. **Keep the first v6C patch minimal.** The single change should be: apply the temporal adapter final penalty only when `preservation_entitlement_logit > threshold` (PE head signals entitlement failure). This prevents the penalty from firing on paraphrase-preserved examples. Everything else stays the same.

3. **Do not introduce prototype memory or alternative Mamba layer extraction** before the temporal/preservation trade-off is resolved and confirmed across multiple seeds.

4. **If a TemporalChannel is introduced**, it must read from `base_pair_repr` (or directly from encoder slot outputs) — not from `frame_pair_repr`. It must be audited as a single active auxiliary family. The audit ledger's `active_training_losses` must show only the TemporalChannel BCE as the active auxiliary, with all other optional auxiliaries disabled. A TemporalChannel experiment that also has active boundary, frame-violation, predicate-isolation, and PE losses simultaneously is not a clean test of the TemporalChannel hypothesis.

5. **Multi-seed evaluation should be reserved** for a configuration that is preservation-safe at the chosen operating point and does not depend on a combination of unablated auxiliary objectives. No current Stage21–24 configuration satisfies this bar.

---

## 11. Current Conclusion

Stage21–24 produced real, reproducible, controlled diagnostic evidence. The core findings are confirmed:

- A real temporal signal exists in `frame_pair_repr` and is detectable via a gradient-isolated adapter trained on the controlled temporal diagnostic dataset.
- Gradient isolation via `detach()` is the correct principle for auxiliary heads that read shared representations.
- The temporal/preservation trade-off is structural in the current architecture. It can be moved (Stage24 improved the frontier over Stage23) but not eliminated by weight tuning, loss combinations, or checkpoint selection alone.
- OOD metric improvements without clean preservation maintenance are not final results.
- The baseline is not CE-only and the audit ledger now makes this explicit.

The project should move from exploratory stacking to auditable, minimal, channel-separated experiments. Active component complexity should decrease before the next architecture is introduced, not increase. The next model design should be justified by Stage21–24 evidence, constrained by this policy, and limited to one new hypothesis per experiment.

The audit ledger and this policy document together form the basis for that transition.
