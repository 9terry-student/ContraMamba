# ContraMamba Native State Kinematics and Temporal Epistemic Dynamics

- **Status:** LONG-TERM RESEARCH HYPOTHESIS / NON-AUTHORITY
- **Origin date:** 2026-09-11 (Asia/Seoul)
- **Scientific evidence created by this document:** NONE
- **Implementation authority:** NO
- **Training / evaluation authority:** NO
- **Kaggle execution authority:** NO
- **Promotion authority:** NO

## 1. Purpose

This document preserves a long-term ContraMamba hypothesis that emerged after the completed A-series factorial work and the O0b/O0c native-state observation line.

The hypothesis is deliberately narrower than a new architecture proposal.

The immediate scientific question is:

> **Does a confident error have a different native Mamba recurrent-state trajectory before it has a different answer?**

Equivalently:

> **Can a wrong factual commitment be preceded by a detectably wrong trajectory in Mamba's native selective-SSM recurrent state?**

The central object is not a final hidden representation, a semantic head, or a learned hallucination detector. The central object is the **time-indexed native recurrent state trajectory itself**.

The first research move should therefore be:

```text
native recurrent state
-> trajectory measurement
-> precursor falsification
```

not:

```text
invent semantic states
-> add architecture
-> train a detector
```

This document records hypotheses, candidate mathematics, constraints, controls, and falsification logic only. It does not authorize an experiment.

---

## 2. Relationship to the existing ContraMamba program

ContraMamba already separates three conceptual graphs:

```text
G_I = information / representation communication graph
G_G = gradient modification-authority graph
G_D = structured decision / authorization graph
```

The current reason-router URP line primarily studies explicit decision structure and gradient ownership while holding the encoder fixed.

The long-term native-Mamba line asks a different question:

```text
URP / A-D line:
How should structured downstream computation and learning-signal authority be controlled?

Native-state line:
How does Mamba's own recurrent state evolve while evidence is being incorporated,
and does that evolution reveal epistemic failure before the final decision?
```

These questions may later inform one another, but they are not the same experiment and should not be merged prematurely.

Current operating rule:

```text
URP experimental line:
D1 remains the bounded active closing question for the A-series gradient-ownership result.

Long-term native-state line:
Native State Kinematics remains theory/design only until separately authorized.
```

A positive or negative D1 result does not automatically authorize the native-state study. Likewise, this document does not alter D1 scope.

---

## 3. Established evidence versus new hypothesis

### 3.1 Established A-series context

The A0-A3 factorial work established, under the frozen-encoder controlled design, that the tested hard `explicit_local` ownership intervention was reproducibly harmful under both router settings.

The strongest bounded interpretation is an **over-isolation-like failure pattern**, not a falsification of the broader Gradient Ownership research axis.

This matters to the present hypothesis because it warns against assuming that epistemically meaningful computation should be forced into completely isolated semantic streams.

The current native-state hypothesis therefore begins **before** any semantic-state construction.

### 3.2 Established O-series context

O0b provided a narrow sufficiency-sensitive clue in hidden-state proxies, but it did not establish a detector, causal mechanism, significance claim, or general precursor.

O0c then directly instrumented the native selective-SSM recurrent state.

Validated O0c result identity:

```text
validated native-state result commit:
ff2fb076f6e66a34a632515bb8502d8b1c90ad7f
```

O0c captured the native recurrent state at post-consumption time `s_t` and preserved complete native-state artifacts with validated provenance.

The O0c measurement family included:

```text
normalized_l2_state_distance
paired_transition_delta
transition_direction_cosine
```

at preregistered anchors including:

```text
anchor_pre_minus_1
anchor_divergence
anchor_post_plus_1
anchor_post_plus_2
anchor_post_plus_4
anchor_terminal
```

Its bounded scientific conclusion was:

```text
BROAD_NATIVE_PRECURSOR_NOT_SUPPORTED
TERMINAL_LOCALIZED_RECURRENT_STATE_SEPARATION_OBSERVED
```

Therefore, O0c does **not** support a simple broad native precursor across the preregistered anchor/layer pattern.

### 3.3 What O0c did not test

O0c did not test the present hypothesis in its full form.

In particular, it did not establish or falsify:

```text
full token-time trajectory kinematics
velocity fields over the entire sequence
acceleration trajectories
turning / directional persistence trajectories
path-length versus displacement structure
dynamically defined onset times
trajectory stabilization or lock-in
evidence-event response latency
state response to evidence-order perturbation
confident-correct versus confident-wrong matched trajectories
decision-space projected velocity
commitment-before-entitlement timing
wrong-attractor dynamics
```

The present hypothesis must therefore be treated as a **new long-term hypothesis**, not as a reinterpretation of O0c.

---

## 4. Mamba-specific research lock

The long-term ContraMamba identity should use Mamba's native recurrent/selective-SSM state as a primary scientific object.

The target is **not**:

```text
Mamba final hidden representation
-> generic semantic heads
-> generic classifier
```

where the backbone could be replaced by a Transformer without changing the essential mechanism.

The target research question is instead:

> **Is factual/epistemic reasoning partly a property of evidence-conditioned recurrent-state evolution rather than only a property of the final representation?**

Mamba is relevant because it exposes a persistent recurrent state whose update is explicitly tied to sequence progression and selective state-space computation.

This does not imply that Transformers lack sequence processing. The intended distinction is narrower:

```text
Mamba-specific object:
native selective-SSM recurrent-state trajectory

Research emphasis:
state evolution through sequence/model time
```

Transformer models may later be useful as controls, but the defining mechanism of this research line should remain native Mamba state dynamics.

---

## 5. Two spaces must be distinguished

The phrase "state space" is ambiguous in this project.

Two different spaces are required.

### 5.1 Native recurrent-state space

For layer `l`, let the native selective-SSM recurrent state after token `t` be:

```text
S_t^(l)
```

The raw state may be tensor-valued. For a minimal geometric treatment, define:

```text
s_t^(l) = vec(S_t^(l))
```

where vectorization is only a coordinate representation, not a claim that tensor structure is scientifically irrelevant.

This is the **native recurrent-state space**.

Questions in this space include:

- how far the state moves per token;
- in what direction it moves;
- when its motion changes;
- whether it reverses after contradictory evidence;
- whether it stabilizes early;
- whether it wanders;
- whether its path differs between confident-correct and confident-wrong cases.

### 5.2 Epistemic decision space

The model's factual decision semantics may live in a different space.

The current external labels are:

```text
REFUTE
NOT_ENTITLED
SUPPORT
```

However, treating them as three symmetric categorical vertices may be epistemically misleading.

`NOT_ENTITLED` is not obviously a third polarity between SUPPORT and REFUTE. It represents a lack of sufficient authorization to make either decisive factual commitment.

A key open problem is therefore the geometry of the **epistemic decision space**.

At minimum, the research should compare:

```text
Candidate A:
flat 3-class simplex

Candidate B:
Authorization x Signed Polarity factorization
```

and should not assume in advance that either is correct.

---

## 6. Native-state kinematics

Use discrete model time initially.

Let:

```text
t = token index
Delta t = 1 token
```

for the first formulation.

For one fixed model and one fixed layer, define state velocity:

\[
\mathbf{v}^{(\ell)}_t
=
\mathbf{s}^{(\ell)}_t
-
\mathbf{s}^{(\ell)}_{t-1}.
\]

Velocity is a vector. Its magnitude is speed:

\[
\nu^{(\ell)}_t
=
\left\|
\mathbf{v}^{(\ell)}_t
\right\|.
\]

Its unit direction, when the norm is nonzero, is:

\[
\hat{\mathbf{v}}^{(\ell)}_t
=
\frac{
\mathbf{v}^{(\ell)}_t
}{
\left\|
\mathbf{v}^{(\ell)}_t
\right\|
}.
\]

Define discrete acceleration:

\[
\mathbf{a}^{(\ell)}_t
=
\mathbf{v}^{(\ell)}_t
-
\mathbf{v}^{(\ell)}_{t-1}.
\]

Acceleration magnitude is:

\[
\alpha^{(\ell)}_t
=
\left\|
\mathbf{a}^{(\ell)}_t
\right\|.
\]

A minimal directional-turning statistic is:

\[
\kappa^{(\ell)}_t
=
1 -
\cos\left(
\mathbf{v}^{(\ell)}_t,
\mathbf{v}^{(\ell)}_{t-1}
\right).
\]

This is a turning statistic, not a claim of differential-geometric curvature.

For a prefix `[0,T]`, define path length:

\[
L^{(\ell)}_{0:T}
=
\sum_{t=1}^{T}
\left\|
\mathbf{v}^{(\ell)}_t
\right\|.
\]

Define displacement:

\[
D^{(\ell)}_{0:T}
=
\left\|
\mathbf{s}^{(\ell)}_T
-
\mathbf{s}^{(\ell)}_0
\right\|.
\]

A simple path-efficiency statistic is:

\[
E^{(\ell)}_{0:T}
=
\frac{
D^{(\ell)}_{0:T}
}{
L^{(\ell)}_{0:T}
}
\]

when path length is nonzero.

Interpretation:

```text
high path length + modest displacement
-> potentially wandering / circuitous dynamics

high displacement/path efficiency
-> potentially directed / lock-in dynamics
```

Neither pattern is assumed to be correct or erroneous in advance.

---

## 7. Time is model time, not physical time

Token index is not physical time.

The first coordinate is:

```text
token time:
t = 0, 1, 2, ..., T
```

This should be called **discrete model time** or **token time**.

Different tokens do not carry equal semantic information, so token time should later be complemented by **semantic-event-relative time**.

Candidate semantic events include:

```text
entity introduction
predicate-bearing phrase
critical supporting evidence
critical refuting evidence
negation
contradiction
conclusion-critical evidence
insufficiency-inducing omission boundary
```

Then the research can define response quantities such as:

```text
critical evidence at tau_e
-> velocity change at tau_e + delta_1
-> direction reversal at tau_e + delta_2
-> stabilization at tau_e + delta_3
```

This creates a measurable concept of **evidence-response latency**.

For variable-length sequences, at least three time parameterizations may eventually be compared:

```text
absolute token time
normalized sequence progress t/T
event-relative time t - tau_event
```

They should not be conflated.

---

## 8. Native-state geometry is not automatically Euclidean truth

Using Euclidean norms is a minimal operational choice, not a claim that Mamba's latent coordinates form a privileged physical Euclidean space.

Potential problems include:

- channel scale differences;
- state-dimension anisotropy;
- layer-specific norm scales;
- arbitrary coordinate rescaling;
- dominance by high-variance dimensions;
- flattening artifacts;
- comparing non-comparable layers.

Initial restriction:

> **Compare trajectories within the same frozen model, same layer, same native coordinate system before making cross-layer geometric claims.**

In particular, do not infer:

```text
layer 18 is "faster" than layer 3
```

from raw norm magnitude alone.

If a signal is found, stronger robustness tests may include:

```text
per-layer normalization
reference-distribution standardization
covariance whitening
Mahalanobis-style geometry
orthogonal-basis invariance checks
channel/state-structured measurements
```

These are validation tools, not default complexity.

The first experiment should use the simplest geometry that can be falsified.

---

## 9. Candidate epistemic decision geometry

### 9.1 Flat 3-class geometry

The simplest existing view is a three-class decision:

```text
REFUTE
NOT_ENTITLED
SUPPORT
```

with a softmax-style simplex.

This is operationally convenient but may conflate:

```text
direction of factual belief
```

with:

```text
authorization to make a factual commitment
```

### 9.2 Authorization x Signed Polarity geometry

A stronger candidate is a two-factor representation:

```text
A = authorization / entitlement strength
Q = signed polarity / commitment direction
```

with conceptual interpretation:

```text
A low, Q any
-> NOT_ENTITLED region

A high, Q negative
-> REFUTE region

A high, Q positive
-> SUPPORT region
```

This geometry allows a state that is directionally decisive but epistemically unauthorized.

For example:

```text
Q strongly positive
A still low
```

can represent:

```text
strong SUPPORT tendency
without sufficient authorization
```

This is a natural location for the existing concept of **unsupported conviction**.

The factorization remains a hypothesis. It must be compared against simpler alternatives and must not be treated as established ontology.

### 9.3 Commitment-before-entitlement

If time-dependent decision coordinates can be defined:

```text
A_t
Q_t
```

then one candidate temporal failure is:

```text
directional commitment becomes strong
before authorization becomes sufficient
```

Operationally, if:

```text
tau_Q = onset of stable decisive polarity
tau_A = onset of sufficient authorization
```

then a candidate premature-commitment pattern is:

\[
\tau_Q < \tau_A.
\]

A larger lead:

\[
\Delta \tau_{QA}
=
\tau_A - \tau_Q
\]

could represent stronger commitment-before-entitlement.

This must **not** be assumed to characterize errors. Correct examples may also show early polarity formation. The scientific question is whether its distribution differs under carefully matched confident-correct and confident-wrong conditions.

---

## 10. Connecting native-state motion to decision-space motion

A major open question is:

> **When a native state moves, what epistemically meaningful direction is it moving toward?**

Raw velocity:

\[
\mathbf{v}_t
=
\mathbf{s}_t
-
\mathbf{s}_{t-1}
\]

is defined in native coordinates.

That vector may reflect syntax, lexical processing, position, world knowledge, evidence incorporation, or many factors unrelated to factual authorization.

Therefore the project must distinguish:

```text
raw-state trajectory
```

from:

```text
decision-relevant projected trajectory
```

Candidate mappings include:

1. **Existing frozen task-head geometry**
   Use the already-trained frozen decision head only as a readout coordinate system, without retraining the backbone.

2. **Class-margin geometry**
   Use fixed decision margins such as SUPPORT-vs-REFUTE and entitled-vs-not-entitled coordinates.

3. **Jacobian-local projection**
   For a fixed readout `f(s)`, study the local image of native velocity:
   \[
   \dot{y}_t \approx J_f(\mathbf{s}_t)\mathbf{v}_t.
   \]
   In discrete time, this is a local sensitivity interpretation, not a continuous-time identity.

4. **Simple supervised linear readout**
   A fixed linear probe may be used diagnostically if fitted under a clean protocol. Probe accuracy alone is not architecture evidence.

5. **Intervention-derived axes**
   Define directions through controlled semantic interventions rather than only through label supervision.

No large learned nonlinear trajectory detector should be the first mapping.

A positive precursor should survive at least one interpretation that is simple enough to audit.

---

## 11. Candidate trajectory signatures

The following are **hypothesis names only**, not established mechanisms.

### Premature Commitment

The trajectory moves strongly toward a decisive region before decision-relevant evidence is sufficient.

### Wrong Attractor / Premature Lock-In

The state rapidly enters a decision-consistent direction and subsequently exhibits low turning despite later contradictory evidence.

### Evidence Non-Incorporation

A conclusion-critical evidence event produces unexpectedly weak state response.

### Commitment Inertia

Contradictory evidence arrives, but the decision-relevant velocity direction remains persistent.

### Failed Correction

The state responds to corrective evidence, but the correction is too small or too late to reverse the eventual wrong commitment.

### Wandering / Instability

The trajectory accumulates large path length and/or frequent turning before terminating.

### Prematurely Stable Wrong Dynamics

The trajectory becomes unusually stable early, but toward the wrong decision.

The project must not assume:

```text
error = instability
```

or:

```text
error = early stability
```

Both remain live alternatives until measured.

---

## 12. First scientific target: confident error, not hallucination

The first controlled target should be:

```text
CONFIDENT-CORRECT
vs
CONFIDENT-WRONG
```

not generic correct-vs-wrong comparison.

The purpose is to test whether state dynamics contain information beyond final confidence.

Where feasible, matching or stratification should account for:

```text
final confidence
gold class
predicted class
input/sequence length
intervention family
lexical similarity
claim/evidence structure
position of decisive evidence
seed
layer
```

The initial claim should be **confident-error precursor**.

Only later, under an explicitly generative setup, may the project test:

```text
unsupported factual commitment precursor
```

and then:

```text
generative hallucination precursor
```

Classification error and hallucination must not be treated as synonyms.

---

## 13. What qualifies as a precursor

Terminal separation is not enough.

A candidate signal should not be called a precursor unless it satisfies strict temporal criteria.

At minimum:

1. It can be computed from a prefix before the final decision.
2. It distinguishes confident-correct from confident-wrong under meaningful confidence control.
3. The separation is not explained only by sequence length, class identity, or one lexical artifact.
4. It appears across multiple samples and seeds.
5. Its time of emergence is explicitly localized.
6. It is distinguished from terminal-only state separation.
7. A prefix-only evaluation is possible.
8. Positive findings survive at least one held-out or intervention-based test.
9. The selected layer/time/metric is not chosen by post-hoc cherry-picking.
10. The effect remains interpretable under the declared state-space geometry.

A stronger precursor would additionally:

- appear before conclusion-critical evidence has fully accumulated;
- predict failure under evidence-order perturbation;
- respond systematically to targeted semantic interventions;
- show partial robustness to coordinate normalization or basis changes;
- causally influence final commitment when the relevant state dynamics are perturbed.

---

## 14. Falsification structure

The hypothesis must be killable.

### Native-kinematics precursor: SUPPORT

A strong positive result would require a preregistered small set of trajectory observables to distinguish confident-wrong from matched confident-correct cases **before** final decision, with reproducibility and nontrivial controls.

### Native-kinematics precursor: REFORMULATE

Reformulate if:

```text
signal is weak
signal appears only in a narrow layer/time region
effect is pair-specific
effect is dataset-specific
effect disappears after confidence matching
effect depends strongly on coordinate scaling
effect is terminal rather than early
```

### Native-kinematics precursor: NOT SUPPORTED

The simple native-kinematics formulation should be considered unsupported if, under adequate powered controls:

```text
no interpretable preregistered trajectory observable
provides reproducible early separation
beyond final confidence and obvious confounds
```

A negative result must not be rescued by adding dozens of trajectory metrics, unrestricted nonlinear detectors, or post-hoc layer searches.

### Consequence of a negative result

A negative result would **not** falsify all Mamba-state research.

It could instead support the narrower interpretation:

```text
native recurrent dynamics are too entangled for the simple geometry
```

which may justify a later structured semantic-state construction experiment.

That architectural step must still receive separate authority.

---

## 15. Evidence-order perturbation as a particularly strong test

If trajectory dynamics are truly tied to evidence incorporation, changing evidence order while preserving content should alter trajectory timing in a predictable way.

Candidate comparison:

```text
Condition 1:
decisive evidence appears before commitment-triggering cue

Condition 2:
commitment-triggering cue appears before decisive evidence
```

A meaningful temporal mechanism should respond to this order manipulation.

Potential observables:

```text
onset time
velocity peak
acceleration peak
direction reversal
response latency
stabilization time
decision-space lead/lag
```

This may be especially valuable because it uses Mamba's sequence-time recurrence directly.

However, order perturbations must preserve semantics carefully enough to avoid creating a different task.

---

## 16. Minimal observable family

To avoid metric fishing, the first formal experiment should use a small preregistered set.

A reasonable minimal family is:

```text
1. speed
2. acceleration magnitude
3. directional turning / persistence
4. path length
5. displacement / path efficiency
6. event-relative response latency
7. one explicitly defined decision-space directional alignment measure,
   only if the decision geometry is frozen in advance
```

Possible later quantities such as jerk, spectral summaries, nonlinear manifold statistics, and large learned trajectory embeddings should remain parked until a documented failure requires them.

---

## 17. Layer selection must be controlled

Mamba provides many layers, creating a major multiple-comparison risk.

Do not:

```text
scan every layer
scan every token
scan every metric
report the best-looking cell
```

Possible clean strategies include:

```text
preselect a small layer set from prior non-outcome-based rationale
use held-out discovery/confirmation splits
aggregate using a preregistered layer rule
use hierarchical statistics with multiplicity control
```

The choice must be frozen before confirmatory evaluation.

O0c's previous anchor/layer results may inform feasibility, but should not be used to cherry-pick a new positive claim from the same evidence.

---

## 18. Existing O0c artifacts as a feasibility sandbox

The full O0c native recurrent-state artifact can be useful for:

```text
checking tensor shapes
verifying discrete velocity calculation
verifying acceleration calculation
testing numerical stability
testing trajectory serialization
testing visualization
testing geometry normalization code
```

This use is **methodological only**.

The O0c dataset was not designed as a powered confident-correct versus confident-wrong population.

Therefore, reanalyzing O0c may support:

```text
measurement feasibility
```

but cannot by itself establish:

```text
confident-error precursor
```

unless a new authority explicitly defines and validates an appropriate scientific reanalysis with adequate evidence.

---

## 19. Semantic states are downstream of evidence, not a starting assumption

Do not begin by assuming separate recurrent states for:

```text
Frame
Predicate
Sufficiency
Polarity
```

The preferred causal research order is:

```text
RAW NATIVE MAMBA STATE
        |
        v
KINEMATIC / TRAJECTORY TEST
        |
        +---- positive ----> characterize native temporal precursor
        |
        +---- weak/entangled ----> consider semantic-state construction
```

Only if native trajectories are insufficiently identifiable should the project consider structured recurrent semantic states.

If semantic states are later introduced, their ontology must be:

```text
minimal
necessary
operationally separable
intervention-testable
```

and not merely named heads over shared latent computation.

Semantic definitions may be non-overlapping while their learned representations remain statistically correlated. Semantic ownership does not require artificial independence.

---

## 20. If semantic decision factors are later used

The current candidate semantic factors remain:

```text
Frame
Predicate
Sufficiency
Polarity
Authorization
```

but their final ontology is not frozen.

A particularly important separation is:

```text
Directional belief / polarity
!=
Authorization to commit
```

This is necessary if the model is to expose a state such as:

```text
strong SUPPORT tendency
+
insufficient entitlement
```

rather than forcing the polarity stream to remain neutral whenever authorization is weak.

If polarity is architecturally prevented from moving until sufficiency is satisfied, then premature commitment becomes impossible by construction and cannot be studied as an endogenous failure mode.

Therefore any future structured design must preserve the possibility of measuring:

```text
what the model tends to believe
```

separately from:

```text
whether the model is entitled to state it
```

unless evidence later falsifies this factorization.

---

## 21. Anti-cheating and anti-complexity rules

This research line should fail closed against the following:

- no giant nonlinear detector as the first experiment;
- no dozens of post-hoc trajectory metrics;
- no best-layer cherry-picking;
- no best-token cherry-picking;
- no redefining the precursor threshold after seeing test outcomes;
- no calling terminal separation a precursor;
- no using final confidence leakage in a prefix detector;
- no treating classification error as hallucination;
- no assuming `NOT_ENTITLED` is merely a middle polarity class;
- no cross-layer raw-norm comparisons without normalization justification;
- no semantic-state architecture before native-state evidence requires it;
- no parameter-count or capacity confound disguised as a state-dynamics effect;
- no hyperparameter sweep to rescue a failed mechanism;
- no rewriting O0c's negative broad conclusion;
- no complexity added merely to protect a favored theory.

The preferred order remains:

```text
Measure first.
Explain second.
Modify third.
```

---

## 22. Relationship to D1 and current URP execution

D1 remains conceptually separate.

Its bounded question is whether the harmful A2/A3 hard-isolation result reflects:

```text
failure of the entire gradient-ownership axis
```

or:

```text
failure of the binary 0/1 hard boundary
```

D1 should remain a minimal causal closing experiment, not become a broad tuning study.

The Native State Kinematics line should proceed only at the level of theory and measurement design while D1 is active.

After D1 closes, the next long-term execution should be chosen from the evidence then available.

This document does **not** imply that edge-specific D2 must precede a native-state kinematics experiment.

A plausible future priority, if no stronger authority intervenes, is:

```text
close D1 cleanly
-> freeze native-state geometry / decision-space hypothesis
-> run the first proper confident-error native-kinematics experiment
-> only then decide whether semantic-state architecture is justified
```

This is a research-order hypothesis, not execution authority.

---

## 23. Possible long-term architectural consequences

No architecture is selected by this document.

If raw native-state kinematics produce a reproducible, early, interpretable precursor, the simplest future ContraMamba may require only:

```text
native Mamba recurrent state
+
small auditable temporal readout
+
structured authorization / abstention
```

If native state is predictive but semantically entangled, a later model may introduce a small number of structured recurrent epistemic states.

If native state contains no useful early signal even under appropriate geometry and controls, then any future semantic-state architecture must justify itself as a constructed mechanism rather than pretending to uncover an already-existing native signal.

The final model should remain as simple as the evidence permits.

---

## 24. Potential general scientific contribution

The strongest possible result would not be:

```text
one trajectory metric correlates with error
```

but something closer to:

> **Confident factual errors can begin as altered recurrent-state dynamics before the final prediction becomes distinguishable, and these dynamics reflect evidence incorporation rather than only terminal confidence.**

A stronger Mamba-specific result would show that:

```text
native selective-SSM recurrence
+
sequence-time evidence processing
```

supports a reproducible early-warning geometry that is not reducible to the final classifier state.

A still stronger causal result would show that changing evidence order or perturbing the identified dynamic changes eventual commitment in the predicted direction.

None of these claims currently exists.

---

## 25. Open theoretical questions

The following must be resolved before a strong confirmatory experiment:

1. What is the minimal valid metric on the native recurrent-state tensor?
2. Should the state be vectorized, channel-normalized, whitened, or treated blockwise?
3. How should layer-dependent scale be handled?
4. Which trajectory quantities are preregistered and which remain exploratory?
5. How is confident error operationally defined without circular threshold tuning?
6. How is final confidence matched?
7. How is decisive evidence time annotated or derived?
8. What constitutes an early-enough precursor?
9. How is multiplicity across layers and time controlled?
10. What is the simplest defensible epistemic decision geometry?
11. Is `Authorization x Signed Polarity` identifiable from existing outputs?
12. Can decision-space projection be defined without training a powerful new black box?
13. Which order perturbations preserve task semantics?
14. Which negative result would justify semantic-state construction?
15. Which positive result would make additional architecture unnecessary?

These questions belong to the design phase, not implementation.

---

## 26. Current verdict

**VERDICT: PROMOTE TO LONG-TERM CORE HYPOTHESIS CANDIDATE / DO NOT EXECUTE YET**

Rationale:

- it directly uses Mamba's native recurrent/selective-SSM state;
- it makes sequence-time dynamics scientifically central rather than incidental;
- it does not require semantic states to be assumed in advance;
- it is compatible with the negative broad O0c result without rewriting it;
- it can be falsified with a small set of interpretable observables;
- it may explain why terminal-only state separation is insufficient;
- it offers a path to confident-error precursor research before claiming hallucination;
- it may simplify, rather than complicate, the eventual ContraMamba architecture.

The hypothesis should be preserved in repository memory, but implementation/training/evaluation must wait for a separate scientific design authority.

---

## 27. One-sentence memory

> **A wrong answer may be preceded by a wrong trajectory: test whether Mamba's native recurrent state moves differently in speed, direction, acceleration, turning, and evidence-relative timing before a confident error becomes visible at the output.**

More formally:

> **ContraMamba should test whether factual reliability is encoded not only in where the native Mamba state ends, but in how it moves through recurrent state space while evidence is being incorporated.**
