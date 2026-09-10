# K2 Exact-Prefix Corrective-Evidence Dynamics Experiment Specification Candidate

**Status:** K2 THEORETICAL PREREGISTRATION / HYPOTHESIS AND DESIGN ONLY
**Authority:** current controller instruction; frozen K0 `7bd1cf824cd53c7f6cf6215346b42cabf351b70a`; corrected frozen K1 `432ddbb1117d4b1aa1b0d89dcf66947d690ecb03`; authenticated A0 handoff/evidence and completed state-blind overlap diagnostic; validated O0c native recurrent-state semantics `ff2fb076f6e66a34a632515bb8502d8b1c90ad7f`.

## 1. Identification question and boundary

K2 is not K1 with looser matching. It changes identification from `existing population -> search for matched prestate comparisons` to `construct identical pre-event sequence -> intervene only on continuation -> observe native recurrent-state response`.

**Primary question:** Given an exactly identical misleading/insufficient prefix that has already elicited a reproducible erroneous factual commitment, how does native Mamba recurrent state respond to the predefined designed corrective continuation relative to its predefined matched non-corrective control continuation?

K2 is an input/evidence-sequence intervention, not a recurrent-state intervention. Direct recurrent-state patching, overwrite, or steering remains K3.

Exact-prefix equality controls pre-divergence history and state; it does not isolate a unique microscopic semantic feature of "correction." (C_i) and (N_i) necessarily differ in semantic content as well as correction status. Accordingly, the maximum causal interpretation available to K2 is the effect of the predefined designed corrective continuation relative to its predefined matched non-corrective control continuation on subsequent native-state dynamics. K2 does not identify an abstract causal effect of "corrective evidence" independently of all semantic differences between (C_i) and (N_i). Mechanistic necessity or sufficiency of a recurrent-state component remains K3.

## 2. Scientific unit and exact-prefix construction

For each base item \(i\), construct common prefix \(P_i\), corrective continuation \(C_i\), and matched non-corrective continuation \(N_i\):

```text
X_i^corr = P_i || C_i
X_i^ctrl = P_i || N_i
```

At the final prefix token \(tau_i\), require:

```text
input_ids_corr[0:tau_i] == input_ids_ctrl[0:tau_i]
```

Also require identical masks, separator placement, positional indexing, tokenizer configuration, model parameters, and runtime/capture semantics. Therefore, before branch divergence, for every \(t < tau_i\) and every required layer \(l\):

\[
S_{corr,t}^{(l)} = S_{ctrl,t}^{(l)}
\]

up to the exact deterministic equality contract. This equality is a design/integrity property, not a scientific finding. No matching algorithm approximates pre-event equality.

## 3. Wrong-commitment eligibility

Operationalize commitment before any K2 state analysis. For prefix-only \(P_i\) at \(tau_i^-\), require all of:

1. prefix-only gold label `NOT_ENTITLED`;
2. all three frozen A0 decision heads predict `SUPPORT`;
3. unanimous prediction across seed180/181/182.

The frozen heads are fixed decision observers only. Native encoder parameters are already proven identical across the three A0 checkpoints. This 3-of-3 rule is the primary commitment-strength eligibility criterion. Do not add a tunable probability or margin threshold; retain confidence and margin only as continuous diagnostics. Thus eligibility records reproducible false entitlement on already-insufficient evidence before correction. Native-state kinematics must not select items.

## 4. Continuation semantics

\(C_i\) must add decisive evidence making the correct final label `REFUTE`, directly resolving the same claim/entity/event relation implicated by \(P_i\). \(N_i\) must contain no corrective/refuting information and leave final evidence `NOT_ENTITLED`.

Prospectively match \(N_i\) to \(C_i\), without native-state measurements, as closely as construction permits on exact canonical-tokenizer continuation length, comparable sentence/template structure, punctuation count where practical, no branch-specific padding difference before continuation end, and the same insertion/event boundary \(tau_i\). Lexical identity after \(tau_i\) is not required because the continuations necessarily differ in semantic content and correction status; exact-prefix equality controls only the pre-divergence history/state.

## 5. Fixed model and tokenizer binding

K2 is prospective and does not carry historical A0 tokenizer-revision ambiguity forward as an unresolved question. The reproducibility environment must be frozen before execution:

```text
model family = state-spaces/mamba-130m-hf
canonical config/tokenizer revision candidate = 5708daa364c50b880e7bd92eab456e0d34492ee9
preferred common A0 encoder digest = 67bfc8cb253fef88b2b8936d442468b9ddcbffa8b79582ba3e2432cb271a937b
```

The later implementation must bind exact tokenizer/config/model file identities and hashes before execution. One canonical full-model realization may capture native state because encoder equality is proven; downstream heads are not claimed equal. The three A0 heads are limited to fixed 3-of-3 eligibility and decision-outcome diagnostics.

## 6. Event-aligned state definitions

Use only native selective-SSM recurrent state \(S_t^{(l)}\), post-consumption after token \(t\), with validated O0c semantics. Reuse K0 definitions:

\[
V_t^{(l)}=S_t^{(l)}-S_{t-1}^{(l)},\qquad
\Delta V_t^{(l)}=V_t^{(l)}-V_{t-1}^{(l)},\qquad
turn_t^{(l)}=1-\cos(V_t^{(l)},V_{t-1}^{(l)}).
\]

For predefined post-event positions \(k\), paired branch divergence is:

\[
\Delta S_{i,k}^{(l)}=S_{corr,(tau_i+k)}^{(l)}-S_{ctrl,(tau_i+k)}^{(l)}.
\]

Terminal-only separation must not be called a precursor claim.

## 7. Minimal native-state observables

K2 keeps three conceptual, M/D/P-derived families:

| Family | Question |
| --- | --- |
| R — corrective response magnitude | Is post-event native movement different for the designed corrective continuation relative to the predefined matched non-corrective control continuation over a fixed interval? |
| D — corrective reorientation | Is turning/directional change different for the designed corrective continuation relative to the predefined matched non-corrective control continuation? |
| P — stabilization versus inertia | Is post-event path organization/persistence different for the designed corrective continuation relative to the predefined matched non-corrective control continuation, using K0 path length and displacement/path ratio concepts? |

The exact implementation-stage authority must freeze continuation length, post-event index set, epsilon, layer aggregation, and multiplicity procedure. This specification creates no metric zoo and does not authorize choosing these from outcomes.

## 8. Decision recovery is distinct from kinematics

After the full corrective branch:

```text
RECOVERY = all three frozen A0 heads leave pre-event erroneous SUPPORT and predict REFUTE
INERTIA = all three heads retain erroneous SUPPORT after decisive correction
DECISION_DISAGREEMENT = a 1-of-3 or 2-of-3 mixed outcome
```

Do not merge disagreement with recovery or inertia. There is no threshold tuning. Any recovery/inertia comparison of native dynamics is secondary, stratified, and requires prospective authorization; the primary causal contrast is the within-item predefined designed corrective continuation relative to the predefined matched non-corrective control continuation.

## 9. Competing hypotheses

- **H_RESPONSE:** the predefined designed corrective continuation causes a reproducible post-event native-state response relative to its predefined matched non-corrective control continuation.
- **H_INERTIA:** persistent erroneous commitment is associated with reduced reorientation under the designed corrective continuation or stronger continuation of the pre-event trajectory.
- **H_RECOVERY:** recovery is associated with greater reorientation under the designed corrective continuation and subsequent restabilization.
- **H_NULL:** under frozen K2 observables, no reproducible native-state response attributable to the predefined designed corrective continuation relative to its predefined matched non-corrective control continuation is detectable after exact-prefix control.

Opposite directional patterns may not both count post hoc as the same successful hypothesis.

## 10. Falsification, invalidation, and interpretation

Fail closed with these labels:

```text
INVALID_EXACT_PREFIX = branches differ anywhere before tau
INVALID_PRESTATE = recurrent states differ before tau beyond deterministic equality contract
INELIGIBLE_NO_WRONG_COMMITMENT = prefix is not NOT_ENTITLED gold with unanimous SUPPORT
INVALID_CORRECTION = C_i lacks decisive REFUTE semantics
INVALID_CONTROL = N_i contains correction or changes required gold semantics
INCONCLUSIVE_NO_DECISION_RECOVERY_SUPPORT = too few recovery/inertia outcomes for a proposed secondary stratification
```

Only valid, prospectively constructed paired evidence can weaken or refute the K2 raw-state response hypothesis. Infrastructure or construction failure is not a scientific null.

## 11. Claim ladder and causal limit

| Level | Permitted conclusion |
| ---: | --- |
| 0 | exact-prefix construction/instrumentation valid |
| 1 | `POST_EVENT_NATIVE_STATE_RESPONSE_OBSERVED` |
| 2 | `DESIGNED_CORRECTIVE_CONTINUATION_RELATIVE_TO_CONTROL_NATIVE_DYNAMICS_OBSERVED` |
| 3 | `RECOVERY_VS_INERTIA_KINEMATIC_ASSOCIATION_OBSERVED` |
| 4 | not permitted in K2: recurrent-state causal mechanism, necessity, or sufficiency claim |

When construction, randomization, and controls warrant it, the input intervention may support only the causal interpretation of the predefined designed corrective continuation relative to its predefined matched non-corrective control continuation on subsequent native-state dynamics. Exact-prefix equality controls pre-divergence history/state but does not identify a unique microscopic semantic feature of "correction" or an abstract causal effect of corrective evidence independent of the semantic differences between the continuations. It cannot establish that a particular recurrent-state component is necessary or sufficient for recovery; that requires K3 recurrent-state intervention.

## 12. K-series relationship and present boundary

```text
K0 = native-state kinematics hypothesis/design foundation
K1 = existing-A0 matched observational precursor attempt; closed INCONCLUSIVE due to zero prestate matching support
K2 = prospective exact-prefix sequential designed corrective-continuation relative to predefined matched non-corrective-control intervention; pre-event equality by construction
K3 = direct recurrent-state intervention for necessity, sufficiency, and mechanism
```

K1 support failure motivates the K2 identification strategy; it is not empirical support for K2.

This theoretical preregistration does not authorize dataset generation, implementation, checkpoint/model execution, native-state capture, training, evaluation, Kaggle, event-window search, hyperparameter sweep, learned detector/probe, or recurrent-state intervention.
