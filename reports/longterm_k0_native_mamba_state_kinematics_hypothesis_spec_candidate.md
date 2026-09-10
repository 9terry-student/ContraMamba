# K0 Native Mamba State Kinematics Hypothesis Specification Candidate

**Status:** K0 HYPOTHESIS / MATHEMATICAL SCIENTIFIC DESIGN CANDIDATE

**Authority boundary:** This is a mathematical scientific-design candidate only: not implementation, training, evaluation, Kaggle, or K1 execution authority; not scientific evidence. O0c remains CLOSED/PARKED. K-series is independent from the D-series, and D1 remains a separate pre-D/URP research direction.

## 1. Scientific motivation and core hypothesis

The primary question is: **Does a confident error have a different native Mamba recurrent-state trajectory before it has a different answer?**

Three levels are distinct: **state position** is where a state lies at an anchor; **local state transition** is its change near an anchor; **full trajectory organization** is movement, direction, and path over a segment. An anchor-based negative result therefore does not logically test every trajectory-level hypothesis: an effect might be distributed, event-aligned, or kinematic without producing broad anchor separation. This preserves rather than weakens or rewrites the O0c negative result.

For matched controls \(C\), the hypothesis promoted for testing—not the conclusion—is:

\[
P(T_{<T}\mid \text{confident-wrong}, C) \ne P(T_{<T}\mid \text{confident-correct}, C),
\]

where \(T_{<T}\) is a prespecified family of preterminal native recurrent-state trajectory observables.

## 2. Native recurrent-state kinematics

For layer \(l\) and token/model step \(t\), native recurrent state is \(S_t^{(l)}\). Unless a later separately authorized design changes capture semantics, this denotes the same kind of native selective-SSM recurrent state validated in O0c: post-consumption recurrent state \(s_t\), after token \(t\) has been incorporated, not a generic Transformer-like hidden-state proxy. This binding does not reinterpret O0c evidence. Define:

```text
V_t^(l) = S_t^(l) - S_(t-1)^(l)
speed_t^(l) = ||V_t^(l)||_F
DeltaV_t^(l) = V_t^(l) - V_(t-1)^(l)
turn_t^(l) = 1 - cos(V_t^(l), V_(t-1)^(l))
L_[u:v]^(l) = sum_{t=u+1}^v ||V_t^(l)||_F
D_[u:v]^(l) = ||S_v^(l) - S_u^(l)||_F
eta_[u:v]^(l) = D_[u:v]^(l) / (L_[u:v]^(l) + epsilon)
```

Use `DeltaV`, not \(A\), for discrete delta-velocity/acceleration because \(A\) is reserved for authorization in decision-space notation. High trajectory efficiency \(\eta\) is not inherently epistemically good: a confidently wrong lock-in trajectory could also be highly efficient.

## 3. Time, events, and geometry

\(t\) is discrete sequence/model time, not physical time. Preserve three coordinates: absolute token index \(t\), normalized progress \(r=t/T\), and semantic-event-relative time \(\tau=t-t_e\). When critical-evidence timing is known, event-relative interpretation is scientifically primary.

Candidate event anchors include critical supporting evidence, critical refuting evidence, contradiction/negation, critical evidence removal or insufficiency divergence, and other preregistered evidence-changing events. A signal first appearing after critical evidence can be a **pre-answer precursor**, but must not be called a **pre-evidence predictor**. Because semantic events may span multiple tokens, future execution authority must preregister an event-index policy—such as event onset, event end, final conclusion-critical token, or another deterministic annotation rule. \(t_e\) must not be selected after observing state dynamics; event-relative windows for multi-token spans must use the frozen anchor rule.

Initial geometry is same frozen model, same layer, same native coordinate system. Do not compare raw norms across layers as if they share one physical metric scale. Vector flattening with Euclidean L2 is mathematically equivalent to tensor Frobenius norm; the issue is coordinate/channel scaling and concentration, not flattening itself. Prespecify normalization. Start with simple native/Frobenius geometry; optional layer-specific RMS/scale normalization is a prespecified robustness check. Do not start with full high-dimensional Mahalanobis/whitening unless later evidence requires it, and do not call this geometry physical.

Euclidean norms and cosine angles are invariant under a common orthogonal rotation, so random orthogonal rotation alone is not a useful empirical robustness test. Meaningful robustness questions include coordinate rescaling, diagonal normalization, channel concentration, whitening choice, and functional state reparameterization sensitivity.

## 4. Minimal primary metric families

| Family | Purpose | Observables |
|---|---|---|
| M | movement / evidence response | \(\lVert V\rVert\), \(\lVert\Delta V\rVert\) |
| D | directional correction / persistence | turning; prespecified pre/post-event directional alignment |
| P | path organization | path length; displacement/path-length ratio |

Onset, peak timing, latency, and persistence may summarize these families, but must not become a large post-hoc feature zoo. K0 primary endpoints exclude jerk, dozens of spectral metrics, learned trajectory embeddings, large generic detectors, and broad exploratory feature mining.

Turning/cosine quantities are undefined when either adjacent velocity has zero or near-zero norm; the same issue applies to pre/post-event directional alignment. Before execution, K1 authority/design must prespecify one fixed policy: either exclude directional-angle evaluation for stationary transitions while retaining a separate stationary indicator, or use one explicitly frozen epsilon-regularized definition. It must not select observations post hoc.

## 5. Competing hypotheses and comparison design

Preregister separately:

- **H-lock:** confident errors enter an incorrect trajectory prematurely: excessive directional persistence, reduced corrective turning after contradictory evidence, or premature high trajectory efficiency \(\eta\).
- **H-wander:** confident errors fail to stabilize appropriately: excess movement/path length, turning, or delayed stabilization.
- **H-null:** after proper controls, no reproducible preterminal native-state kinematic difference exists.

Either opposite pattern must not automatically count as success. Preserve the preregistered directional hypotheses and null outcome. If a future result shows a phase-dependent mixture, it may be reported descriptively, but a new composite mechanism must not be invented post hoc and counted as confirmatory evidence unless separately tested.

The intended comparison is not naive pooled correct-versus-wrong and must not imply that correct and incorrect examples can be exactly matched simultaneously on both gold label and predicted class. Use complementary controlled views:

- **A. Gold-matched/source-semantic control:** hold gold label or semantic source condition fixed where feasible; compare correct versus incorrect outcomes while controlling confidence, length, intervention family, structure, evidence position, and other available confounds.
- **B. Prediction-matched/commitment-direction control:** hold final predicted class or commitment direction and confidence fixed where feasible; compare correct versus incorrect outcomes while controlling the remaining available confounds.

These answer different questions, and neither alone eliminates every class-related confound. A confusion-pair-stratified analysis may also be preregistered, for example REFUTE-to-SUPPORT errors rather than pooling error destinations. Confidence matching is essential because otherwise kinematics may simply reflect generic confidence or decision-margin differences. Final confidence may be used retrospectively for matching, stratification, or confound control, but it must not be an input to the native-state precursor observable or a deployment-time precursor rule; any deployable precursor must remain computable from prefix-accessible information alone. If “confident” uses a threshold, margin, percentile, or calibration criterion, freeze that rule before inspecting K-series kinematic outcomes or obtain it from an independent calibration source; no post-hoc confidence-threshold sweep may manufacture separation.

## 6. Strict precursor requirements

A promoted precursor claim must meet all applicable requirements:

1. **F1 Prefix/preterminal:** computable without terminal state or final emitted answer.
2. **F2 Confidence-controlled:** survives appropriate final-confidence or margin control.
3. **F3 Semantic/confound matched:** not explained by length, class, intervention family, or trivial structural differences.
4. **F4 Multi-seed:** reproducible beyond one seed/run.
5. **F5 Multiplicity controlled:** layer × time × metric search is prespecified and statistically/procedurally controlled; no best-layer or best-time cherry-picking.
6. **F6 Geometry robustness:** survives prespecified reasonable normalization alternatives.
7. **F7 Event specificity:** effects organize relative to semantic evidence events where relevant.
8. **F8 Non-lexicality:** cannot be explained by simple lexical markers.
9. **F9 Terminal distinction:** terminal-only separation is insufficient for a precursor claim.
10. **F10 Mechanism boundary:** observational separation supports predictive/associational language only; causal/mechanistic claims require later perturbation.
11. **F11 Limited transfer:** broader promotion requires at least some held-out intervention, evidence-order, or OOD replication.

## 7. Event-order falsification and kill condition

Candidate strong control: if the same critical evidence is moved earlier/later while preserving content as much as possible, a genuinely evidence-responsive kinematic phenomenon should shift in event-relative time rather than remaining fixed only to absolute token position. Failure of temporal tracking weakens the event-responsive interpretation.

The raw-native kinematics precursor hypothesis should be **CLOSED or REFORMULATED**, not rescued by metric fishing, if preregistered preterminal families show no reproducible matched effect, or apparent effects are explained by terminal-only localization, confidence, class imbalance, sequence length, lexical cues, evidence-position artifact, seed instability, uncontrolled layer/time selection, normalization fragility, or failure to track semantic-event timing where predicted.

K0 need not choose numeric statistical thresholds. Before K1 execution, its authority/design must freeze exact primary family-level statistics, layer aggregation/selection policy, time-window policy, multiplicity procedure, directional hypotheses, and the rule constituting K1 positive, null, or inconclusive. No post-hoc “whichever metric worked” promotion is allowed.

## 8. Exact relationship to O0c

O0c tested native selective-SSM recurrent state through predefined anchors, pair/layer/anchor comparisons, normalized state distance, paired transition delta, transition direction, and broad preregistered consistency. Its conclusion remains:

```text
BROAD_NATIVE_PRECURSOR_NOT_SUPPORTED;
TERMINAL_LOCALIZED_RECURRENT_STATE_SEPARATION_OBSERVED
```

K-series tests an untested narrower class: full trajectory segments, event alignment, velocity, delta-velocity, turning, path organization, response onset/latency/persistence, matched confident-correct versus confident-wrong, and evidence-order controls. K-series does not retroactively reinterpret O0c as positive evidence.

## 9. Separate epistemic decision geometry

Do not conflate raw recurrent-state geometry with decision geometry. Contrast a flat REFUTE / NOT_ENTITLED / SUPPORT simplex with entitlement/authorization plus conditional signed polarity. Candidate coordinates are \(A\) = entitlement/authorization and \(Q\) = signed polarity:

```text
low A             -> NOT_ENTITLED
high A and Q < 0  -> REFUTE
high A and Q > 0  -> SUPPORT
```

NOT_ENTITLED is not the midpoint between REFUTE and SUPPORT. \(Q\) may remain large in magnitude while \(A\) is low, permitting a candidate “unsupported conviction” configuration without claiming it is validated.

Given existing logits \(z_N\) = NOT_ENTITLED, \(z_R\) = REFUTE, and \(z_S\) = SUPPORT, define:

\[
A = \log(\exp(z_R) + \exp(z_S)) - z_N, \qquad Q = z_S-z_R.
\]

For a standard 3-class softmax over \(z_N,z_R,z_S\), this reparameterization is exact:

\[
\operatorname{sigmoid}(A)=P(REFUTE)+P(SUPPORT),
\]
\[
\operatorname{sigmoid}(Q)=P(SUPPORT\mid REFUTE\text{ or }SUPPORT),\quad
\operatorname{sigmoid}(-Q)=P(REFUTE\mid REFUTE\text{ or }SUPPORT).
\]

Thus the probability factorization is algebraically exact. What remains scientifically unvalidated is the semantic/epistemic interpretation of \(A\) as genuine entitlement/authorization and \(Q\) as semantically clean signed polarity. As a pure coordinate transformation, A/Q does not define a more expressive classifier hypothesis class than the original 3-way softmax. Its scientific value must therefore come from temporal organization, intervention selectivity, causal behavior, or related epistemic structure, not reparameterization alone.

An optional conceptual hierarchical probability interpretation is:

\[
p_E=\operatorname{sigmoid}(A),\quad P(NOT\_ENTITLED)=1-p_E,
\]
\[
P(SUPPORT\mid entitled)=\operatorname{sigmoid}(Q),\quad P(REFUTE\mid entitled)=\operatorname{sigmoid}(-Q).
\]

This is an interpretive candidate, not established ontology.

## 10. Commitment-before-entitlement secondary hypothesis

Signed polarity commitment may emerge before adequate authorization. Record only the candidate quantity:

\[
U_t=(1-\operatorname{sigmoid}(A_t))\,|\tanh(Q_t/2)|.
\]

It is not a validated detector, not a K1 primary native-state endpoint, and authorizes neither threshold tuning, lambda-style sweeps, nor detector optimization.

## 11. Projection hierarchy

Preserve this order:

1. raw native recurrent-state trajectory;
2. existing frozen task decision function traced over prefixes, only if technically/semantically valid;
3. tiny fixed diagnostic linear readout only if needed;
4. Jacobian or intervention-derived geometry later for mechanism analysis.

Do not start with a learned trajectory detector, large MLP, or new semantic state architecture. If a diagnostic polarity readout is later used, learn/interpret polarity only on entitled REFUTE/SUPPORT cases, not by assigning artificial \(Q=0\) targets to NOT_ENTITLED.

## 12. Parked Mamba-native mechanism decomposition

PARK this for later. A selective recurrent update can conceptually be analyzed as a mixture of retained/decayed prior state and input-driven write. A later positive-result study may ask whether confident error reflects excessive persistence/failed overwrite versus deficient evidence-driven write. Do not implement this decomposition in K0.

## 13. Result implications

A validated preterminal kinematic result would justify sequentially: richer event-relative characterization, evidence-order tests, selective-update decomposition, causal perturbation, and only then a minimal native-state epistemic mechanism/interface. It would **not** automatically justify F/P/S/Q recurrent owners, semantic multi-stream Mamba, a generic detector, or architecture complexity.

A strong negative result supports only: “A simple raw native-state kinematic precursor is not readily observable under the tested geometry and controls.” It would **not** automatically authorize semantic-state construction. A later separately authorized question could distinguish genuine absence from coordinate/representation entanglement.

## 14. Claim ladder

A K1 positive may support only language such as `PRETERMINAL_NATIVE_KINEMATIC_ASSOCIATION_OBSERVED` or `CANDIDATE_PRECURSOR`; it must not by itself establish an event-responsive precursor. K2 requires semantic-event alignment and, where applicable, evidence-order movement tests before `EVENT_ALIGNED_PRECURSOR_SUPPORTED` is allowed. K3 requires perturbational/interventional evidence before mechanistic or causal language is allowed. Exact promotion labels for future executions must be frozen before those executions.

## 15. Non-authoritative K-series progression

```text
K0 — mathematical/falsification specification
K1 — raw native-state kinematics
K2 — event-aligned temporal dynamics
K3 — selective-SSM mechanistic characterization
K4 — epistemic decision-space linkage

Measure -> Temporalize -> Causally characterize -> Structure only if justified
```

Only K0 is currently authorized.

## 16. Final K0 verdict

**PROMOTE AS LONG-TERM CORE HYPOTHESIS CANDIDATE.** This is hypothesis promotion, not empirical support. No claim that confident-error trajectories actually differ has yet been established.
