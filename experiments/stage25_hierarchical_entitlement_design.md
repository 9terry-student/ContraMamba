# Stage25-A: Hierarchical Entitlement Architecture — Design

**Status:** Design-only. No new experiment results. No code behavior changes. No training has
been run for this stage.

**Supersedes:** Stage24 TemporalChannel penalty approach as primary forward direction.
**Builds on:** Stage23–24 diagnostic findings, ContraMamba six-axis decomposition, EpistemicBERT
codebook judgment order.

---

## 1. Historical Evolution of the Six-Axis Framework

### 1.1 Origin: ContraMamba, not EpistemicBERT

The six-axis epistemic decomposition originated within ContraMamba, not from EpistemicBERT.

ContraMamba began as a simple three-way epistemic judgment system:

```
-1 = false
 0 = unknown / not sure
+1 = true
```

This geometry was a natural starting point: it mirrors classical truth-value semantics and is
sufficient for simple fact-verification. Through early ContraMamba experiments, the three-way
space proved unable to correctly represent all knowledge states encountered in entailment and
evidence evaluation tasks. Cases arose that did not fit cleanly into true/unknown/false:

- Cases where interpretation was underdetermined — it was not clear which reading of a predicate
  or entity the evidence was addressing. The label was not "unknown" but "the question itself is
  ambiguous." This led to the **ambiguity** axis.
- Cases where evidence was entirely absent or insufficient — not a negative, not an unknown, but
  a positive absence of grounding. This led to the **ignorance** axis.

The resulting five-axis formulation (truth, false/error, contradiction, ambiguity, ignorance)
was still not complete. Cases appeared that were outside the known epistemic space entirely —
claims that presupposed a frame of reference not covered by the evidence, or temporally displaced
claims that could not be evaluated on the same epistemic surface. This led to the **novelty** axis.

The six-axis decomposition was thus produced by ContraMamba experimental necessity, not
architectural design from the start.

### 1.2 EpistemicBERT as a Pragmatic Detour

At the time the six-axis framework matured, building a full ContraMamba architecture that could
cleanly operationalize all six axes was technically difficult. As a pragmatic solution, the
ContraMamba-originated six-axis philosophy was transplanted into a strong baseline backbone —
this became EpistemicBERT.

EpistemicBERT was a testbed, not a source. It operationalized the ContraMamba epistemic
decomposition in a simpler, more tractable strong-backbone setting. The EpistemicBERT codebook
formalized an annotation judgment order:

1. FRAME anchor / same referent-space?
2. Sufficiency / enough evidence?
3. Difference or mismatch?
4. I / A / M decision (Ignorance / Ambiguity / Mismatch)

This codebook order is important not because EpistemicBERT created it, but because annotators
following it produced consistent, decomposed judgments — which revealed that the judgment order
itself is an architectural ordering principle, not just annotation convenience.

### 1.3 Stage25 as a Return

Stage25 brings the clarified judgment order back into the original ContraMamba architecture.
The six-axis decomposition that ContraMamba generated, tested through EpistemicBERT's codebook,
now becomes the structural blueprint for a hierarchical judgment pipeline within Mamba.

---

## 2. Why Stage25 Is Needed

### 2.1 Stage24 TemporalChannel Findings

Stage24 established the following facts:

**TemporalChannelV1 loss-only (functional sanity):**
- TC BCE loss computed correctly; `temporal_channel_loss > 0` in raw and weighted trackers.
- TC head reads `cat([claim_frame_state, evidence_frame_state])` — pre-pair-projector slot
  states from FrameGate. Gradient isolation via `detach()` confirmed working.
- Loss-only has zero effect on final logits. Clean dev metrics unchanged.
- Stage15 provenance: all `false`. `time_swap` absent from main clean train/eval.

This confirms: **the temporal signal exists and is learnable from pre-projector frame states.**

**PE-gated final-logit penalty:**
- p0.25: weak effect on temporal mismatch, surface control mostly stable but insufficient signal.
- p0.75: temporal mismatch improves, but surface_control FNE increases. Trade-off reappears.
- p1.5: non-monotonic, unstable interaction between TC signal and PE signal at high scale.

### 2.2 The Bottleneck Is Arbitration, Not Signal

The Stage24 TC findings reveal the same structural problem as Stage23–24 adapter penalty
experiments: the temporal signal is real and extractable, but applying it as a post-hoc
final-logit boost cannot resolve the temporal/preservation trade-off.

The issue is not that the TC head fails to detect temporal mismatch. It does. The issue is that
converting a continuous temporal-probability signal into a direct NOT_ENTITLED logit boost —
even when gated by PE probability — applies a soft penalty across all examples proportionally.
At scales sufficient to move temporal mismatch decisions, the boost also fires on
preservation-safe examples (surface controls, paraphrases) where PE probability is not exactly 1.

**The bottleneck is crude final arbitration, not temporal signal availability.**

### 2.3 Why Final-Logit Penalty Is Not the Preferred Path Forward

Direct final-logit interventions (adapter penalty, TC-gated penalty) have now failed to resolve
the trade-off across Stage23, Stage24, and Stage25-precursor configurations. The pattern is:

- Weak penalty: insufficient effect on temporal decisions.
- Strong penalty: preservation degrades.
- PE gating: reduces but does not eliminate the trade-off at useful scales.

These are not parameter-tuning failures. They are structural: a scalar boost on a 3-class logit
vector cannot simultaneously satisfy temporal rejection and preservation safety unless the
underlying representation already encodes the distinction — which it does not, because frame
representation is shared between temporal and preservation judgments.

The preferred path forward is not a better penalty formula. It is a different arbitration
structure.

---

## 3. Six-Axis Reinterpretation as Hierarchical Failure Modes

The six axes of the ContraMamba decomposition are not symmetric. They do not operate at the
same level of the judgment pipeline. Treating them as six parallel classification targets is
a category error:

| Axis | Level | Meaning |
|------|-------|---------|
| Novelty | Frame | Referent or frame of reference is outside known space |
| Ambiguity | Predicate | Semantic coverage is underdetermined — which predicate? |
| Ignorance | Sufficiency | Evidence is absent or insufficient |
| Temporal invalidity | Entitlement | Temporal conditions break entitlement to judge |
| Truth | Polarity | Post-entitlement: evidence supports the claim |
| Contradiction / Error | Polarity | Post-entitlement: evidence refutes the claim |

**Three axes are pre-entitlement failure modes.** If novelty, ambiguity, or ignorance are
detected, the judgment does not proceed to polarity. The correct label is NOT_ENTITLED, not
a polarity-qualified SUPPORT or REFUTE.

**One axis is an entitlement-breaking precondition.** Temporal invalidity — detected by the
TemporalChannel — signals that the claim and evidence are in incompatible temporal frames.
This should stop the judgment at the entitlement gate, not add a penalty after polarity has
been computed.

**Two axes are post-entitlement.** Truth and contradiction/error are meaningful only when
entitlement is established. Polarity should operate under the condition that frame, predicate,
sufficiency, and temporal preconditions are satisfied.

**Warning against flat six-head parallelism:** Implementing six parallel BCE heads, each
targeting one axis independently, violates the dependency structure above. A model with six
flat heads can be trained to detect all six conditions simultaneously, but it will not learn
their hierarchical relationship. Polarity will not be conditioned on entitlement. Temporal
signals will compete with polarity signals at the same output level. The result is the same
arbitration problem Stage24 already encountered.

---

## 4. Proposed Hierarchical Pipeline

```
Mamba encoder
  ↓
FrameState / FrameChannel
  (novelty / frame mismatch detected here)
  ↓
PredicateState / PredicateCoverage
  (ambiguity / predicate noncoverage detected here)
  ↓
SufficiencyState + TemporalState
  (ignorance / insufficient evidence detected here)
  (temporal invalidity detected here)
  ↓
EntitlementGate
  (aggregates frame, predicate, sufficiency, temporal signals)
  (outputs: entitlement probability → gates final decision)
  ↓
PolarityState
  (active only under entitlement)
  (truth / contradiction / error evaluated here)
  ↓
Final decision: SUPPORT / NOT_ENTITLED / REFUTE
```

The pipeline is sequential in judgment but not necessarily sequential in computation. The
channels can run in parallel; the EntitlementGate aggregates their outputs before polarity
is consulted for the final label.

**Key structural property:** Final logits are determined by EntitlementGate × PolarityState,
not by any single channel's direct logit modification.

---

## 5. Architectural Principles

These principles govern Stage25 design choices. Any proposed addition must be consistent with
all of them:

**Role separation, not full independence.** Each channel serves a specific epistemic role
(frame, predicate, sufficiency, temporal, polarity). Channels should not duplicate each other's
function. Adding a new channel requires identifying which of the six axes it represents.

**Hierarchical dependency, not flat parallel heads.** Later pipeline stages depend on earlier
ones. The EntitlementGate is downstream of all four precondition channels. Polarity is
downstream of the EntitlementGate. This dependency must be preserved in the architecture.

**Diagnostic heads must not directly modify final logits.** TemporalChannel, FrameChannel,
PredicateChannel, and SufficiencyChannel are diagnostic heads. Their direct outputs are
intermediate representations and auxiliary training signals. They feed into the EntitlementGate;
they do not add boosts to the final logit vector independently.

**TemporalChannel is an entitlement input, not a direct NE-logit boost.** This is the primary
architectural correction from Stage24. The temporal mismatch signal should enter the
EntitlementGate, which then influences the final label. Bypassing the gate with a direct logit
modifier loses the hierarchical structure.

**Polarity should only operate under entitlement.** The PolarityState's contribution to final
logits should be gated or weighted by EntitlementGate output. When entitlement is low, polarity
signals should be suppressed, not allowed to compete with NOT_ENTITLED at full strength.

**Predicate mismatch is not frame failure.** A claim that uses a predicate not covered by the
evidence is an ambiguity/NOT_ENTITLED failure at the predicate level, not a frame-level novelty.
Frame and predicate channels must not conflate these.

**Temporal mismatch is not polarity.** A temporally displaced claim is NOT_ENTITLED, not
incorrectly REFUTED. The TemporalChannel should route through the EntitlementGate, not through
a polarity-level signal.

**`time_swap` must stay out of main clean training and evaluation data.** Temporal diagnostic
records (including `time_swap`) are used only in the separate temporal diagnostic dataset for
supervising the TemporalChannel. They are never mixed into the main CE classification data.

**Stage15 remains eval-only.** No entitlement gate threshold, no penalty scale, no channel
weight, and no checkpoint is selected using Stage15 OOD results. All selection uses clean dev
metrics only.

---

## 6. Minimal Next Implementation: v7 Hierarchical Entitlement Minimal

The next implementation should be conservative. The goal is to demonstrate that the hierarchical
structure produces better temporal/preservation behavior than flat penalty approaches, not to
build a complete six-channel system at once.

**Proposed name:** `v7 Hierarchical Entitlement Minimal`

**Allowed conceptual components:**

| Component | Role | Axis covered |
|-----------|------|--------------|
| FrameChannel | Pre-entitlement | Novelty / frame mismatch |
| PredicateChannel | Pre-entitlement | Ambiguity / predicate noncoverage |
| SufficiencyChannel | Pre-entitlement | Ignorance / evidence insufficiency |
| TemporalChannel | Pre-entitlement | Temporal invalidity |
| EntitlementGate | Aggregation | Combines pre-entitlement signals |
| PolarityChannel | Post-entitlement | Truth / contradiction / error |

**This is not "more heads for every case."** It is a structured judgment pipeline where each
component has a well-defined epistemic role and the components are connected in the correct
hierarchical order.

**What minimal means here:**
- Each channel should be a small MLP reading from appropriate intermediate representations.
- The EntitlementGate should be a gating mechanism, not a large black-box network.
- Auxiliary BCE losses supervise diagnostic channels; CE loss remains the primary task signal.
- At most one final-logit modifier at a time (the EntitlementGate's output, not stacked penalties).
- Audit ledger tracking of all active losses and their contributions is mandatory.

**What to defer:**
- Case-specific heads (number_swap head, location_swap head, role_swap head, etc.) are explicitly
  out of scope. Every intervention type does not need its own head.
- Multi-seed evaluation of any v7 configuration until at least one seed shows
  preservation-safe temporal improvement.
- Large EntitlementGate architectures. Start with a simple gating formula.

---

## 7. What NOT to Do Next

These are explicit exclusions, stated to prevent backsliding into the patterns Stage23–24
identified as structurally inadequate:

- **Do not continue arbitrary final-logit scale sweeps.** The TC-gated penalty has been
  evaluated at p0.25, p0.75, and p1.5. The trade-off pattern is clear. Sweeping more scales
  will not resolve the structural problem.

- **Do not add intervention-type-specific heads.** A `number_swap_head`, `time_swap_head`,
  `surface_head`, `role_head`, or `location_head` is not the six-axis decomposition. It is
  case enumeration. The architecture should generalize across intervention types through
  principled epistemic channels, not by memorizing each intervention type.

- **Do not use OOD/Stage15 to tune gates.** No EntitlementGate threshold, penalty scale,
  or channel weight should be set by reference to Stage15 OOD results. Stage15 is eval-only.

- **Do not stack more final-logit penalties.** Adding a TC penalty on top of an existing
  adapter penalty on top of a PE modifier is not a solution. It is accumulated arbitration
  complexity. Active component policy: at most one final-logit modifier per run.

- **Do not collapse TC/PE/Frame signals into one uncontrolled final boost.** The EntitlementGate
  should aggregate signals in a structured, transparent way. A linear combination of probabilities
  fed into an ad-hoc boost is not an EntitlementGate.

- **Do not present the Stage24 direct penalty as the final architecture.** It is a diagnostic
  finding — proof that the signal exists and can move decisions — not a design recommendation.

---

## 8. Testable Predictions for the Next Implementation

If the hierarchical structure is correct, the following should hold. These are predictions, not
guarantees — they are the conditions under which v7 would be considered a positive result:

1. **Temporal mismatch rejection should improve without large surface_control FNE increase.**
   The EntitlementGate should be able to distinguish temporal mismatch from surface control
   because the TemporalChannel reads pre-projector frame states that should encode temporal
   frame conditions independently of surface-level predicate matching.

2. **`temporal_erased` should remain preservation-safe.** Cases where temporal information is
   erased (no temporal mismatch, no temporal advantage) should not be pushed to NOT_ENTITLED
   by the temporal signal. The TemporalChannel trained on `time_swap` should not fire on
   `temporal_erased`.

3. **`predicate_mismatch` should remain NOT_ENTITLED without being treated as frame failure.**
   PredicateChannel signals should route through NOT_ENTITLED at the EntitlementGate without
   activating FrameChannel's novelty signal.

4. **`frame_location` and `frame_role` mismatches should remain mostly NOT_ENTITLED.**
   FrameChannel should detect these; they should not slip through to polarity.

5. **Clean dev selection should not use Stage15.** All checkpoint and threshold selection
   should produce valid results from clean dev metrics alone, with Stage15 confirming or
   not confirming — never informing — the selection.

6. **Aux-to-CE loss ratio must remain below 0.5** (or the audit warning threshold). Adding
   more channels must not push auxiliary losses to dominate the CE signal.

---

## 9. Open Risks

**Over-engineering.** More pipeline stages mean more parameters, more interactions, and more
potential for auxiliary losses to conflict. The minimal implementation policy is a guard against
this, but it must be actively enforced.

**EntitlementGate as a black box.** If the EntitlementGate is too large or too unconstrained,
it may learn a decision boundary that is not interpretable as entitlement gating. The gate
should be small and its inputs should be inspectable.

**Weak channel supervision.** If individual channel losses are too small (weight too low) or
the auxiliary data is insufficient, channels may not learn their intended roles before the
CE loss dominates. This requires per-channel loss tracking and audit.

**Too many auxiliary losses distorting training.** Each channel that has its own BCE loss
adds to the total auxiliary weight. The existing audit warning on aux-to-CE ratio must remain
active. No new channel should be added without tracking its contribution.

**Hierarchical channels becoming case-specific patches.** The risk is that TemporalChannel
becomes a `time_swap` detector, FrameChannel becomes a `frame_mismatch` detector, and so on —
each patching one observed OOD failure type rather than generalizing across intervention types
from principled epistemic representations.

**Entitlement/polarity coupling.** If the EntitlementGate output is used as a multiplicative
gate on PolarityState, gradient flow between entitlement and polarity becomes non-trivial.
This needs careful implementation so that entitlement learning and polarity learning do not
interfere destructively.

---

## 10. Decision Summary

**What Stage24 established:**
- TemporalChannelV1 is a functional signal probe. TC BCE trains a temporal mismatch detector
  from pre-pair-projector frame states without corrupting FrameGate.
- The temporal mismatch signal is real and learnable.
- Direct final-logit arbitration (both Stage24 adapter penalty and Stage25-precursor TC-gated
  penalty) cannot resolve the temporal/preservation trade-off at useful scales.

**What Stage25 proposes:**
- The six-axis epistemic decomposition that ContraMamba originated — and that EpistemicBERT
  operationalized in codebook annotation order — should become the structural blueprint for
  ContraMamba's next architecture.
- Novelty, ambiguity, ignorance, and temporal invalidity are pre-entitlement failure modes.
  They belong upstream of final-label computation.
- Truth, contradiction, and error are post-entitlement judgments. Polarity belongs downstream
  of an EntitlementGate that aggregates pre-entitlement channel signals.
- TemporalChannel should be an input to the EntitlementGate, not a direct NE-logit modifier.
- The next implementation (v7 Hierarchical Entitlement Minimal) should test this structure
  conservatively before any multi-seed evaluation or OOD reporting.

**What Stage25 does not claim:**
- v7 is not proven to work. These are architectural predictions, not empirical results.
- Stage25 does not claim TemporalChannel solved the OOD temporal problem.
- Stage25 does not claim EpistemicBERT originated the six-axis framework.
- Stage25 does not use Stage15 for any selection or calibration.
- The direct penalty approach (Stage24) is recorded as a diagnostic finding with real signal,
  not dismissed as a failure. It confirmed signal availability while revealing an arbitration
  structural limitation.
