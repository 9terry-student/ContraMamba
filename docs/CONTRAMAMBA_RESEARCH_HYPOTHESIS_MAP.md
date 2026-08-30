# ContraMamba Research Hypothesis Map

Status: LONG-TERM RESEARCH HYPOTHESIS MAP / NON-AUTHORITY

Purpose: preserve the current long-term ContraMamba hypothesis map, candidate
mechanisms, measurement ideas, falsification structure, and scope locks. This
document complements `docs/CONTRAMAMBA_RESEARCH_VISION.md`; it does not replace
or rewrite that vision.

Authority boundary: this document does not authorize implementation, training,
evaluation, scientific execution, Kaggle execution, promotion, or scientific
claims. It records hypotheses and future experiment concepts only. Every
implementation, run, artifact, promotion decision, or scientific claim requires
separate stage or experiment authority and provenance.

Repository-memory hierarchy:

```text
Research Vision              = enduring why
Hypothesis Map               = current long-term hypotheses, candidate
                               mechanisms, measurements, falsification, and
                               scope lock
Stage/Experiment Authority   = what may actually be executed
Executed Artifacts/Reports   = evidence actually obtained
```

## 1. Linked Research Questions

The original backbone-level objective remains central:

Before a model emits an unsupported or hallucinated factual commitment, are
there measurable precursor signals already present in Mamba's internal
state/dynamics?

This map does not assume that one predefined scalar already exists. The research
program instead keeps two linked questions separate.

### Q1 - Native Precursor Discovery

Can frozen/native Mamba state trajectories contain signals predictive of future
unsupported commitment before final output?

More concretely: before an unsupported or hallucinated factual commitment is
emitted, are there measurable signals already present in native Mamba
hidden/state dynamics?

### Q2 - Constructed Precursor

If native signals are weak, late, entangled, or unstable, can semantic-state
ownership create cleaner, earlier, or more causally interpretable
pre-commitment epistemic risk signals?

The research order is:

```text
Observe existing dynamics
-> test precursor predictability
-> introduce architectural structure only if justified
```

Semantic ownership is a candidate construction mechanism. It must not replace
the original native precursor-discovery objective.

## 2. Why Mamba

Mamba remains scientifically relevant because of:

- state-space dynamics;
- selective update;
- meaningful use of discourse/sequence time;
- efficiency as useful experimental headroom, not the primary scientific
  objective.

Reasoning is treated as state evolution:

```text
s0 -> s1 -> ... -> sT
```

The primary default hypothesis is Primary Discourse Dynamics: reasoning,
support, contradiction, sufficiency, and risk may appear as changes through the
sequence/discourse state.

An optional special subsystem is Semantic-Order Dynamics for explicit
chronology, causal, procedural, or dependency ordering when sequence order and
semantic order differ. World-time is not a permanent basic ontology; it is a
possible special subsystem only if evidence requires it.

Priority order:

```text
semantic integrity
> epistemic reliability
> causal interpretability
> OOD robustness
> compute efficiency
```

Higher constant-factor compute is acceptable for clean causal prototypes.
Ownership is not compression.

## 3. Core Hypothesis Families

### H1 - Structured Decision

Explicit semantic primitives should compose the final decision rather than an
unrestricted final classifier deciding everything.

### H2 - Gradient Ownership

Forward read permission is not the same as backward modification permission.
This requires preserving a distinction between the information graph `G_I` and
the gradient graph `G_G`.

### H3 - Semantic State Ownership

Reasoning-relevant memories may benefit from separately owned recurrent
trajectories rather than one globally shared latent state.

### H4 - Semantic Dynamics / Epistemic Reliability

Reasoning and hallucination risk may appear as state evolution: support,
contradiction, insufficient evidence, instability, or failure to incorporate
relevant evidence. H4 is an expansion axis, not a mandatory fourth
architectural doctrine.

## 4. Three Graphs

ContraMamba should distinguish at least three graphs.

`G_I` = representation / information communication graph.

`G_G` = gradient modification-authority graph.

`G_D` = structured decision / authorization graph.

Important correction: `Frame -> Predicate -> Sufficiency -> Authorization` does
not have to mean neural owner-to-owner communication. It may exist only in
`G_D` as logical authorization order.

Default first backbone hypothesis:

```text
NO cross-owner communication during representation formation.
```

Owners may compute independent semantic judgments from the same competent base
representation and meet only at structured decision time.

## 5. Competence Versus Epistemic Control

ContraMamba semantic owners should not be required to relearn everything a
pretrained language model already handles well.

The base/pretrained Mamba remains responsible for ordinary competence:

- syntax;
- lexical semantics;
- normal contextual understanding;
- ordinary non-epistemic processing.

ContraMamba focuses on epistemically relevant factual commitment and internal
risk/authorization structure. Preferred language is "risk of unsupported
commitment" or "pre-commitment epistemic risk." This map does not claim that a
perfect pre-routing hallucination detector exists.

## 6. Minimal ContraMamba Backbone Hypothesis

Minimal candidate:

```text
Ownership Router
+ Independent Semantic Mamba State Trajectories
+ Structured Decision
```

Initial experimental semantic owners:

```text
F = Frame
P = Predicate
S = Sufficiency
Q = Polarity
```

This ontology is experimental, not permanent.

Shared pretrained parameters are allowed while runtime recurrent histories are
independent. A candidate whole-transition ownership update is:

```text
H_t^r =
H_{t-1}^r
+ g_t^r (H_tilde_t^r - H_{t-1}^r)
```

Interpret `g_t^r` as write/commit authority: how much current information is
allowed to modify owner `r`'s recurrent memory.

Use independent sigmoid-style gates conceptually, not mandatory softmax
competition. Ownership is not exclusivity.

## 7. Communication Minimalism

Default:

```text
NO CROSS-OWNER COMMUNICATION.
```

Do not assume Predicate needs Frame judgment merely because the final semantic
decision is hierarchical.

Motivating counterfactual:

```text
Claim:    John acquired X.
Evidence: Mary acquired X.

Frame     = mismatch
Predicate = match
Final decision blocked by Frame.
```

Predicate need not lower itself because Frame failed.

Likewise:

```text
Claim:    John acquired X.
Evidence: John invested in X.

Frame     = match
Predicate = mismatch
```

This separation is useful causal evidence. Communication must earn its
existence through an observed identifiability or information deficit.

If later needed, the progression is:

```text
NO EDGE
-> READ-ONLY NARROW MESSAGE
-> SMALL ADAPTER
-> PARTIAL GRADIENT THAW
```

Never assume full hidden-state concatenation. Communication is not merge.
Progressive Semantic Thaw is a PARKED HYPOTHESIS, not current core.

## 8. Multi-Stream Candidate

Potential long-term form:

```text
Shared Perception
-> Owned Reasoning Dynamics
```

The same pretrained/shared Mamba parameters may process multiple independent
semantic runtime streams. Independent streams need not merge between layers.

Possible provisional names:

- ContraMamba-S = shared backbone + owned semantic sidecar;
- ContraMamba-M = multi-stream owned Mamba backbone.

These names are provisional and non-authoritative.

Split depth remains open:

- full split from early layers;
- mid/late split after shared perception.

No choice is frozen yet.

## 9. Polarity And Conflict

A separate Conflict owner is not required initially. Support and refute evidence
may coexist conceptually:

```text
high Q+, low Q- = support
low Q+, high Q- = refute
high both       = conflict
low both        = weak/insufficient polarity information
```

This preserves the historical S+/S- intuition only at the conceptual level.
DCT, Parseval, fixed orthogonality, and physical energy interpretations are
historical hypothesis-pool items, not current doctrine.

## 10. O0 - Native Mamba Precursor Observation

O0 - Native Mamba Precursor Observation is a FUTURE EXPERIMENT CONCEPT / NOT
AUTHORIZED.

Question:

Can native Mamba hidden/state dynamics predict future unsupported decisive
output before final commitment?

Potential observations:

- hidden/state transition magnitude;
- consecutive-state cosine or transition direction;
- acceleration/curvature;
- evidence-responsive state incorporation;
- perturbation stability;
- layer-wise disagreement;
- later, only if technically accessible and separately authorized: internal
  selective-SSM state and A/B/C/Delta-related diagnostics.

None of these are known hallucination indicators from this document. They are
candidate observations for a future authorized study.

## 11. Exploratory Downstream Precursor Diagnostics

These are exploratory diagnostics, not validated detectors.

Authorization:

```text
A = F * P * S
```

First-blocker masses:

```text
qF = 1 - F
qP = F * (1 - P)
qS = F * P * (1 - S)
qA = F * P * S
```

Candidate unsupported-conviction diagnostic, also described in prose as
unsupported conviction:

```text
R_UC = (1 - A) * decisive_conviction
```

Interpretation: strong answer-direction conviction despite weak semantic
authorization.

The existing chat-level numerical thought experiment is NON-SCIENTIFIC /
HYPOTHESIS SCREENING ONLY. It is not scientific evidence.

Candidate conflict diagnostic:

```text
support_refute_conflict = min(p_support, p_refute)
```

Reason entropy:

```text
entropy(qF, qP, qS, qA)
```

Interpret reason entropy as ambiguity/indecision, not a generic hallucination
detector.

Decision fragility is retained as an instability diagnostic only. It should not
be promoted to a hallucination signal without separate evidence.

Preferred conceptual risk representation is a vector, not one forced scalar:

```text
R_t = [
  unsupported_conviction,
  conflict,
  ambiguity,
  fragility,
  native_state_dynamics_features
]
```

## 12. Detect -> Disentangle -> Control

The possible long-term scientific story is:

```text
discover
-> disentangle
-> control
```

Example hypothesis:

A weak precursor may already exist in shared Mamba dynamics; structured
semantic primitives may make it more interpretable; owned semantic states may
make it earlier, more local, or more controllable.

This is a hypothesis only, not an established result.

## 13. Minimal Future Ablation Family

This matrix records a future causal ablation family only. It does not authorize
implementation.

```text
O0 - Native backbone precursor observation

M0 - Shared State Control

M1 - Replicated-State Null Control
     Same inputs/updates; expected functional equivalence to shared state.

M2 - Arbitrarily Diversified State Control
     Different trajectories without semantic routing.

M3 - Semantic Ownership
     Independent state trajectories + learned semantic write authority +
     owner-local supervision.

M4 - Semantic Ownership + Joint Gradient
     Same forward architecture as M3, but downstream/final gradient may reshape
     owners.
```

Core causal comparisons:

```text
M0 vs M1 = mere state replication
M1 vs M2 = trajectory diversity
M2 vs M3 = semantic ownership
M3 vs M4 = gradient ownership
```

No implementation authority is created by this matrix.

## 14. Semantic Intervention Suite

Intended causal evaluation principle:

Change one semantic factor while preserving others as much as possible.

Intervention families:

- Frame: entity/object/event/context swap;
- Predicate: relation/action swap;
- Sufficiency: remove conclusion-critical evidence while preserving
  frame/predicate where possible;
- Polarity: negation/refutation while preserving frame/predicate.

Measure:

- semantic readout/output locality;
- recurrent-state locality;
- trajectory locality;
- gate/state consistency;
- invariance;
- double interventions;
- OOD factor composition.

Conceptual Intervention Locality Matrix:

```text
I_{r,k} = E[d(H_r(x), H_r(x^(k)))]
```

Perfect diagonal structure is not required, but the intended owner should be
selectively more responsive.

## 15. Anti-Cheating Rules

Explicit constraints:

- no unrestricted raw hidden-state bypass to final decision;
- no unrestricted concat of all semantic owner states into a generic MLP;
- no duplicated identical state streams called semantic ownership;
- control parameter count, FLOPs, and state capacity;
- probes alone are insufficient;
- auxiliary-loss improvement alone is insufficient architecture evidence;
- reject algebraic equivalence;
- accuracy cannot hide semantic unreliability;
- semantic cleanliness cannot justify capability collapse.

## 16. Failure Taxonomy

Latent Soup: all owners respond similarly.

Dead Owner: one owner state barely changes.

Always-Open Gate: `g` approximately 1 everywhere.

Readout Illusion: only semantic heads specialize; recurrent states remain
entangled.

Over-Isolation: semantic locality improves but capability collapses.

Arbitrary Diversification: M2 ~= M3, weakening the semantic-ownership
interpretation.

Shortcut Owner: an owner relies on a lexical shortcut rather than the intended
semantic factor.

## 17. Continue / Reformulate / Kill

CONTINUE: reproducible improvements across multiple relevant axes that are not
explained by parameter, compute, or state-capacity confounds.

REFORMULATE: mixed, weak, unstable, or model-family-specific effects.

KILL CURRENT OWNERSHIP FORMULATION: after adequate controls, no meaningful
semantic specialization, precursor improvement, reliability effect, OOD effect,
or generalization beyond arbitrary state diversification.

Failure of semantic ownership does not falsify the broader native-precursor
discovery question.

Do not add complexity merely to rescue a favored architecture.

## 18. PARKED HYPOTHESES

Until observed evidence requires them, keep these parked:

- owner-to-owner communication;
- progressive semantic thaw;
- owner-specific Mamba A/B/C/Delta;
- explicit uncertainty owner;
- auxiliary semantic/world-time Mamba;
- DCT;
- orthogonality;
- energy regularization;
- sparse/top-k ownership routing;
- unrestricted owner-state merging;
- generative extension.

Guiding rule:

```text
Measure first.
Explain second.
Modify third.
```

## 19. Relationship To Current URP Work

The current URP/reason-router work is a separate active research track.

This Hypothesis Map does not change, authorize, reinterpret, or consume:

- URP authority;
- URP attempts;
- URP checkpoints;
- URP Kaggle runs;
- URP scientific conclusions.

URP results may later become evidence relevant to long-term hypotheses, but
authority and provenance remain separate.

In particular, this document does not change:

- `reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`;
- `reports/stage180a_pass2_annotations_completed.csv`;
- any root `.patch` file;
- `docs/CONTRAMAMBA_RESEARCH_VISION.md`;
- any `reports/`;
- any `src/`;
- any `scripts/`;
- any `tests/`;
- any URP authority or artifact;
- any existing file.

## 20. Repository Memory Rule

This file serves as the durable current hypothesis map, complementing:

```text
docs/CONTRAMAMBA_RESEARCH_VISION.md
```

The Research Vision remains the durable "why." This Hypothesis Map records the
current long-term hypothesis structure and falsification map. Stage and
experiment authorities remain the only sources for execution permissions.
Executed artifacts and reports remain the only sources for evidence actually
obtained.

Do not turn chat discussion into established evidence.
