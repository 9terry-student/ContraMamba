# Stage30 Architecture Axis Hierarchy Spec

## Overview: Entitlement-First Verification

ContraMamba adopts an **entitlement-first verification** design: before deciding *how* a claim should be judged, the architecture first determines *whether* it is judgeable at all.

This separation prevents the model from confidently labeling claims that fall outside the scope of the provided evidence — penalizing assertoric confidence in low-coverage or unresolvable situations before the polarity decision is made.

---

## Core Conceptual Separation

### Judgeability vs. Polarity

| Dimension | Question | Handled by |
|-----------|----------|------------|
| **Judgeability** | Can the evidence license any verdict? | Hard Core Validity, Coverage/Entailment, Residual Adjudication, ANI Diagnosis |
| **Polarity** | If judgeable, does the evidence support or refute the claim? | Polarity Layer |

These two dimensions are **strictly separated** in the axis hierarchy. No polarity signal may rescue a claim that has been flagged as unjudgeable by an upstream axis.

---

## Final Architecture Order

```
Mamba Encoder
    │
    ▼
[1] Hard Core Validity
    │  (blocks if temporal/source frame is invalid)
    ▼
[2] Coverage / Entailment
    │  (blocks if evidence does not cover the claim domain)
    ▼
[3] Residual Adjudication
    │  (handles partially covered, ambiguous, or conflicting evidence)
    ▼
[4] ANI-style Epistemic Diagnosis
    │  (diagnostic readout: not a decision axis)
    ▼
[5] Polarity Layer
    │  (support / refute only if judgeable)
    ▼
[6] Final Composer
         (single label: Supported / Refuted / NEI)
```

---

## Module Definitions

### [1] Hard Core Validity

Validates whether the claim can be evaluated given the available temporal or source context.

- Checks for temporal mismatch between claim and evidence frames.
- Checks for source domain validity.
- **Hard Core cannot be overridden** by any downstream axis.
- If Hard Core fires, the Final Composer must emit NEI or equivalent; polarity is not consulted.

### [2] Coverage / Entailment

Determines whether the evidence semantically covers the claim.

- Measures overlap between claim predicates and retrieved evidence.
- **Coverage/Entailment is distinct from frame mismatch**: frame mismatch is a Hard Core concern; coverage is a semantic scope concern.
- Low coverage → NEI before polarity is computed.
- Must not be conflated with temporal residual preservation.

### [3] Residual Adjudication

Handles claims that passed Hard Core and Coverage checks but remain ambiguous.

- Operates over residual evidence not resolved by Coverage/Entailment.
- Surfaces conflict, partial support, or out-of-scope fragments.
- **No global preservation rescue**: residual adjudication cannot rescue a claim blocked by Hard Core.
- **No double penalty**: a claim already penalized by Hard Core must not receive an additional residual penalty.

### [4] ANI-style Epistemic Diagnosis

Measures model uncertainty and epistemic state across the claim-evidence pair.

- **ANI is a diagnostic readout, not a primary decision axis.**
- ANI output is available to the Final Composer but cannot unilaterally change a verdict.
- Used for calibration, confidence scoring, and post-hoc analysis.

### [5] Polarity Layer

Determines the direction of evidence relative to the claim, assuming judgeability has been confirmed.

- Outputs: `supports`, `refutes`, or abstains if upstream blocked.
- Never receives a claim that was blocked upstream.

### [6] Final Composer

Assembles the final label from all upstream signals.

- Applies the **Single Owner Rule**: each label (Supported / Refuted / NEI) has exactly one primary axis responsible for it; no two axes may claim the same label for conflicting reasons.
- Emits the final prediction with attached confidence.

---

## Architecture Invariants

### Single Owner Rule
Each verdict outcome is owned by exactly one axis. Overlapping ownership is a spec violation.

### No Global Preservation Rescue
A temporal preservation signal cannot be used to rescue a claim that Hard Core has invalidated. Preservation signals are local to axis [3] (Residual Adjudication) and must not propagate upward.

### No Double Penalty
If Hard Core fires a penalty for a given claim, downstream axes must not apply an independent penalty for the same root cause.

### Hard Core Cannot Be Overridden
No signal from Coverage/Entailment, ANI, or Polarity may reverse a Hard Core block.

### Coverage/Entailment Is Distinct from Frame Mismatch
Temporal or source-frame invalidity is a Hard Core concern. Coverage/Entailment addresses semantic scope only.

### ANI Is Diagnostic, Not Decisional
ANI readout may inform confidence but must not serve as a primary decision input.

---

## Stage30-E Implementation Scope

Stage30-E is a **minimal, targeted** implementation that touches only the following:

| Item | Description |
|------|-------------|
| **Temporal residual preservation-aware cap** | Caps the temporal mismatch penalty using a preservation signal |
| **Reuse of Stage30-D temporal mismatch risk** | Reads the fused temporal mismatch probability from Stage30-D output |
| **New temporal preservation signal** | Introduces `temporal_preservation_prob` as a new signal |
| **Effective penalty formula** | `temporal_mismatch_fused_prob × (1 − temporal_preservation_prob)` |

### Stage30-E Explicitly Does NOT Implement

- Coverage/Entailment axis (reserved for a future stage)
- Full ANI readout (reserved for a future stage)
- Full residual vector (reserved for a future stage)
- Hard Core refactor (Hard Core structure is fixed in this stage)
- Global preservation rescue (invariant: forbidden by architecture contract)
- Direct use of Stage30-D cap output as a final candidate score

Stage30-E may only modify components directly related to the temporal residual preservation-aware cap. Any changes to Hard Core, Coverage/Entailment, or ANI modules are out of scope.

---

## Allowed Future Stages

| Stage | Scope |
|-------|-------|
| Stage31 | Coverage/Entailment axis implementation |
| Stage32 | Full residual vector and Residual Adjudication |
| Stage33 | ANI-style epistemic readout integration |
| Stage34+ | Final Composer multi-axis fusion and calibration |

---

## Summary

ContraMamba Stage30 defines an entitlement-first hierarchy where judgeability is settled before polarity. The axis order is fixed: Hard Core → Coverage/Entailment → Residual Adjudication → ANI Diagnosis → Polarity → Final Composer. Stage30-E implements only the temporal residual preservation-aware cap as a minimal, scoped addition to this hierarchy.
