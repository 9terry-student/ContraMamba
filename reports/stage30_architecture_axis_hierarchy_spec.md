# Stage30-ARCH: Entitlement-First Architecture Contract

---

## Executive Summary

**Stage30-ARCH freezes the target architecture contract for ContraMamba.**

The current v7 implementation is a **proxy/bridge stack**, not the target architecture. It is a pragmatic approximation built from available heads and caps, not a full implementation of the structured axis hierarchy that defines ContraMamba's conceptual design.

**Stage30-E diagnostic conclusion**: The temporal mismatch risk signal and the temporal preservation-aware cap both worked as designed. With gamma=2, time_swap entitlement was reduced from approximately 0.928 to 0.283. However, time_swap predictions remained SUPPORT 60/60, NE 0/60, with none/paraphrase and clean macro staying stable (macro ~0.985). The bottleneck is the **H1 final composer**, not the temporal heads or caps. Lowered entitlement did not translate into NOT_ENTITLED predictions because the H1 composer's ne_score formula is not strong enough to surface NE at that entitlement level.

This result motivates a **structured architecture reset** — not another round of cap tuning, gamma adjustment, or NE-boost patches. Adding a composer-side NE boost is available as an optional stress-test ablation (Stage30-E2), but it is not the main architecture path. The main path is Stage31 onward: a clean implementation of the target axis hierarchy.

---

## 1. Current Implemented Proxy Stack

The following describes what is **actually implemented** in the v7_hierarchical system as of Stage30-E. This is a bridge stack, not the target architecture.

```
Mamba Encoder
  -> v7_hierarchical channel heads
       - FrameGate           -> frame_prob
       - PredicateCoverage   -> predicate_coverage_prob
       - SufficiencyGate     -> sufficiency_prob
       - PolarityChannelV7   -> support_energy, refute_energy
  -> H1 product entitlement
       base_entitlement = (frame_prob * predicate_coverage_prob * sufficiency_prob) ** 0.90
  -> Stage28-I location boundary cap
       entitlement_after_location_cap =
           base_entitlement * location_boundary_prob ^ gamma
  -> Stage30-D temporal mismatch multihead risk signal
       temporal_mismatch_fused_prob  (frame/predicate/sufficiency heads fused)
  -> Stage30-E temporal preservation-aware cap
       effective_temporal_penalty =
           temporal_mismatch_fused_prob * (1 - temporal_preservation_prob)
       entitlement_after_temporal =
           entitlement_after_location_cap * (1 - effective_temporal_penalty) ^ gamma
  -> H1 final composer
       support_score = entitlement * support_energy
       refute_score  = entitlement * refute_energy
       ne_score      = ne_bias + alpha * (1 - entitlement)
  -> argmax([refute_score, ne_score, support_score])
  -> REFUTE / NOT_ENTITLED / SUPPORT
```

### What the proxy stack approximates

| Proxy element | Target architecture element it approximates |
|---|---|
| FrameGate | Partial coverage of Hard Core Validity (temporal slot matching) |
| PredicateCoverage | Partial coverage of Coverage/Entailment (predicate scope) |
| SufficiencyGate | Partial coverage of Residual Adjudication (sufficiency signal) |
| Stage28-I location cap | Partial Hard Core (location/circumstance validity) |
| Stage30-D mismatch risk | Hard Core temporal validity signal |
| Stage30-E preservation cap | Residual Adjudication (temporal residual preservation) |
| H1 composer (ne_score formula) | Final Composer — **current bottleneck** |

### What the proxy stack does NOT implement

- A clean structural separation between Hard Core and Coverage/Entailment
- All/some, specific/general, part/whole, only/also coverage failure detection
- Full Residual Adjudication with a structured residual vector
- ANI-style epistemic diagnostic readouts
- A robust Final Composer that can reliably surface NOT_ENTITLED from lowered entitlement

---

## 2. Stage30-E Diagnostic Result

| Metric | Value |
|---|---|
| gamma tested | 2 |
| time_swap entitlement (before cap) | ~0.928 |
| time_swap entitlement (after preservation cap) | ~0.283 |
| time_swap predictions: SUPPORT | 60/60 |
| time_swap predictions: NOT_ENTITLED | 0/60 |
| none/paraphrase predictions | Stable (no regression) |
| Clean macro F1 | ~0.985 (stable) |

### Interpretation

- **The temporal risk head is not the bottleneck.** Fused mismatch probability correctly fires.
- **The temporal preservation-aware cap is not the bottleneck.** Entitlement was successfully reduced from 0.928 to 0.283.
- **The H1 final composer is the bottleneck.** At entitlement=0.283 with high support_energy, the product `entitlement * support_energy` still exceeds `ne_score = ne_bias + alpha * (1 - entitlement)`.
- **Adding NE boost is an optional stress-test ablation**, not the main architecture path. It would confirm the bottleneck hypothesis but does not address the root cause: the H1 composer formula is not structurally suited to surface NE from entitlement reduction alone.
- **The correct response** is not more cap/gamma/boost iteration. It is Stage31: implementing Coverage/Entailment as a proper structural axis.

---

## 3. Target Architecture

The following is the **target architecture** that ContraMamba is designed toward. It is not yet fully implemented.

```
Mamba Encoder
  -> Structural Entitlement Layer
       [1] Hard Core Validity
             Is this pair even about the same judgeable target/event/relation?
       [2] Coverage / Entailment
             Does the evidence cover or entail the claim's required scope,
             strength, or specificity?
       [3] Residual Adjudication
             Is this remaining difference a real mismatch, or a
             harmless/preserved variation?
  -> ANI-style Epistemic Diagnosis
       - novelty-like failure
       - ambiguity-like failure
       - ignorance-like failure
       (diagnostic readout over structural entitlement failures — NOT primary heads)
  -> Polarity Layer
       If the claim is judgeable, does the evidence support or refute it?
  -> Final Composer
       Assembles REFUTE / NOT_ENTITLED / SUPPORT using the Single Owner Rule
  -> REFUTE / NOT_ENTITLED / SUPPORT
```

---

## 4. Module Definitions

### [1] Hard Core Validity

**Question:** Is this pair even about the same judgeable target, event, or relation?

Hard Core Validity determines whether the claim and evidence are referring to the same evaluable object. It is not about evidence quality or scope — it is about whether the pair is even in the same semantic space for evaluation.

**Covers:**
- Different entity or referent (wrong subject)
- Different event (temporal mismatch: claim about event A, evidence about event B)
- Clear role reversal (subject/object swap that changes truth value)
- Clear relation mismatch (developed vs. published, caused vs. correlated)
- Hard circumstance mismatch (clear location, modality, or condition invalidity)
- Minimum evidence absence (evidence is structurally empty or disconnected)

**Rule:** Hard Core failure cannot be rescued by preservation logic, coverage adjustment, or residual adjudication. Hard Core fires, Final Composer emits NOT_ENTITLED; polarity is not consulted.

---

### [2] Coverage / Entailment

**Question:** Does the evidence cover or entail the claim's required scope, strength, or specificity?

Coverage/Entailment determines whether the evidence is semantically sufficient to license a verdict on the claim, given the claim's logical form.

**Examples:**
- `All -> Some`: evidence about all licenses support of some. Valid support.
- `Some -> All`: evidence about some does not license a universal claim. Coverage failure.
- `Specific -> General`: evidence about a specific case can support a general claim. Valid.
- `General -> Specific`: general evidence does not license a specific claim. May fail.
- `Only -> Winner`: "only participant" licenses "winner." Valid.
- `Also -> Only`: "also" does not license "only." Failure.

**Important:** Coverage/Entailment is distinct from frame mismatch. Temporal or source-frame invalidity is a Hard Core concern. Coverage addresses semantic scope and logical entailment. Do not collapse all/some, part/whole, or specificity cases into frame mismatch.

---

### [3] Residual Adjudication

**Question:** Is this remaining difference a real mismatch, or a harmless/preserved variation?

Residual Adjudication operates over evidence differences that were not resolved by Hard Core Validity or Coverage/Entailment. It determines whether residual surface variation should actually reduce entitlement.

**Examples of preserved variation (should NOT reduce entitlement):**
- Temporal expression variation (paraphrase of a time reference)
- Surface paraphrase (different wording, same proposition)
- Location alias or granularity (city vs. metro area when not semantically critical)
- Format shift (numeric vs. written)

**Examples of real mismatch (should reduce entitlement):**
- Relation nuance change (planned vs. actual)
- Scope ambiguity that changes truth value
- Temporal offset that invalidates the claim window

**Important:** Residual Adjudication should not be implemented as a pile of direct caps stacked on top of each other. It should function as a structured decision about whether residual differences matter. Stage30-E's temporal preservation-aware cap is a primitive approximation of this — a direct cap is not the same as structured adjudication.

---

### [4] ANI-style Epistemic Diagnosis

Novelty, Ambiguity, and Ignorance are **not primary decision heads** in the target architecture. They are diagnostic readouts over structural entitlement failures.

**Mapping:**
- **Ignorance:** Insufficient evidence, coverage failure, missing scope support. Maps to Coverage/Entailment failure.
- **Ambiguity:** Unresolved residual, multiple plausible readings, borderline relation or scope. Maps to Residual Adjudication uncertainty.
- **Novelty:** Claim introduces information not licensed by evidence — claim is beyond the entailed scope. Maps to Coverage/Entailment or Hard Core overflow.

ANI axes are available to the Final Composer as diagnostic signals but cannot unilaterally change a verdict. They are not trained as primary classifiers in the target architecture.

---

### [5] Polarity Layer

**Question:** If the claim is judgeable, does the evidence support or refute it?

The Polarity Layer operates only after the Structural Entitlement Layer has passed the claim. It outputs support or refute energy. High support/refute energy must not override low entitlement — entitlement gates polarity, not the reverse.

---

### [6] Final Composer

Assembles the final label from all upstream signals using the **Single Owner Rule**: each final label (SUPPORT, REFUTE, NOT_ENTITLED) has exactly one primary upstream axis responsible for producing it. No two axes may claim the same label for conflicting reasons.

---

## 5. Architecture Rules

The following rules are explicit constraints on the target architecture and on any bridge/proxy implementation.

1. **Entitlement before polarity.** The model must resolve whether the claim is judgeable before computing polarity. High polarity energy does not override low entitlement.

2. **Single Owner Rule.** Each final label has exactly one primary owning axis. Overlapping ownership is a spec violation.

3. **No double penalty.** A claim already penalized by Hard Core must not receive an independent penalty from Residual Adjudication for the same root cause.

4. **No global preservation rescue.** A temporal preservation signal cannot rescue a claim blocked by Hard Core Validity. Preservation logic is local to Residual Adjudication.

5. **Hard Core failures cannot be rescued.** Once Hard Core fires, the final label is NOT_ENTITLED. No downstream signal reverses this.

6. **Coverage/Entailment is not the same as Frame.** Temporal or source-frame invalidity is Hard Core. Coverage addresses semantic scope and logical entailment. Do not conflate.

7. **Residual Adjudication is not direct cap stacking.** Stacking caps (location cap, temporal cap, preservation cap) is a proxy approximation. The target is a structured adjudication decision.

8. **ANI is diagnostic, not the primary decision mechanism.** ANI signals cannot unilaterally change a verdict. They inform calibration and confidence.

9. **Current v7 proxy stack is a bridge, not the target architecture.** Stage30-E results confirm the H1 composer is the bottleneck. The correct path is architecture reset, not composer patching.

---

## 6. What ContraMamba Is and Is Not

**ContraMamba is:**
- An entitlement-first verification architecture that separates judgeability from polarity.
- A system that targets confident errors caused by entitlement failure: cases where the model predicts SUPPORT/REFUTE despite structurally insufficient or mismatched evidence.
- An architecture that treats NOT_ENTITLED as a first-class output, not a fallback.

**ContraMamba is not:**
- A universal hallucination solver.
- A system that verifies world truth without reliable evidence.
- A claim that Mamba's architecture is the novelty. The novelty is separating judgeability/entitlement from polarity.
- A system where temporal cap gamma tuning is the final solution.

---

## 7. Roadmap

### Stage30-ARCH (this document)
- Freeze the target architecture contract.
- Document the current proxy stack vs. the target architecture.
- Record Stage30-E diagnostic results and their correct interpretation.

### Stage30-E (bridge experiment — complete)
- Temporal residual preservation-aware cap, reusing Stage30-D mismatch risk.
- Treat as bridge experiment. Not the final architecture.
- Result: temporal heads and cap working; H1 composer is the bottleneck.

### Stage30-E2 (optional ablation only)
- Residual-aware NE boost in the H1 composer.
- Stress-test only: confirm that lowering entitlement to ~0.283 is sufficient to surface NE when the composer is given a stronger NE signal.
- **Not the main architecture path.** Does not address the structural separation problem.

### Stage31: Coverage / Entailment Diagnostic
- Implement a diagnostic/probe for Coverage/Entailment failures.
- Focus: all/some, specific/general, part/whole, only/also failure modes.
- Do not conflate with frame mismatch.

### Stage32: Relation / Role / Nuance Diagnostic
- Implement diagnostics for relation and role mismatch.
- Focus: published/distributed, developed/published, caused/correlated, set-to-replace/replaced.

### Stage33: Order / Modality / Condition Diagnostic
- Implement diagnostics for ordering and modal/conditional mismatch.
- Focus: before/after, planned/actual, may/cause, necessary/sufficient, if/only-if.

### Stage34: Axis Pruning and Owner Conflict Audit
- Audit which axes own which failure modes.
- Prune overlapping ownership and resolve Single Owner Rule violations.
- Prepare for Final Composer multi-axis fusion.
