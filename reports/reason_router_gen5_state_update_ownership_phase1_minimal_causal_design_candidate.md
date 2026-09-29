# ContraMamba Gen5 — Phase 1 Minimal State-Update Ownership Causal Design Candidate

## 0. Status

PHASE = `GEN5_PHASE1_MINIMAL_OWNERSHIP_CAUSAL_DESIGN`

STATUS = `FROZEN_ON_COMMIT`

SELECTED_OWNERSHIP_DIMENSION = `STATE_UPDATE_AUTHORITY`

IMPLEMENTATION = `NOT_AUTHORIZED`

TRAINING = `NOT_AUTHORIZED`

EVALUATION = `NOT_AUTHORIZED`

SCIENTIFIC_EXECUTION = `NOT_AUTHORIZED`

KAGGLE = `NOT_AUTHORIZED`

This document selects the first Gen5 ownership dimension and defines the minimal
scientific semantics that a later implementation must satisfy.

It does not authorize implementation or execution.

---

## 1. Frozen scientific parent

GEN5_PHASE0_FREEZE =
`25a4206ad64ae64cc942bacdac92b9a300d4f8c3`

GEN5_PHASE0_DOC_CORRECTION =
`62975c6434fd0e7b9e383959ee6cfb10e25e6d60`

DEVELOPMENT_SCALE =
`state-spaces/mamba-130m-hf`

The Phase 0 inheritance contract remains binding.

In particular:

- causal role is not coordinate identity;
- causal role is not semantic-owner identity;
- a dominant plane is not the complete mechanism;
- exact owner independence is not assumed;
- mechanism is not steering utility;
- 130M success is not cross-scale generality.

---

## 2. Phase 1 decision

The first Gen5 ownership dimension is:

`STATE_UPDATE_AUTHORITY`

The first Gen5 experiment will not start from information-flow ownership or
gradient ownership.

### Why state-update authority is selected first

The strongest new mechanistic evidence inherited from K-series is inside native
recurrent-state construction:

- post-state contrast is write dominated;
- the relevant write is strongly current-token / lag-0 structured;
- the U-side dominates write magnitude while U-D interaction cancels part of it;
- strong-side directional alignment makes a bounded prospective causal
  contribution to downstream native write/post-state excess.

This gives a direct mechanistic location at which "who is allowed to modify a
state component" can be made falsifiable.

By contrast, the historical `explicit_local` ownership intervention primarily
changes downstream head-to-head gradient paths through detach operations. It does
not constitute recurrent-state ownership.

Repeating gradient detach as the first Gen5 manipulation would therefore fail to
use the strongest new native-state evidence.

Information-flow ownership is also deferred because changing what representations
may be read can change forward information and update semantics simultaneously,
making the first causal attribution less identifiable.

---

## 3. Critical non-equivalence boundary

Two inherited causal objects must remain distinct.

### 3.1 Gen4 × K / five-scale role object

The strongest 130M causal-role program localizes a transportable,
geometry-specific, locally necessary, and locally restoration-sufficient
contributor at the frozen layer-17 / target-token site.

Its historical local realization is PP3-centered, with structured causal residual
outside PP3.

### 3.2 K-series native-update object

The K-series causal falsification acts at the layer-22 mixer x-branch,
pre-depthwise-convolution, current `k=2` token and establishes a bounded causal
contribution of strong-side directional alignment to downstream layer-22
write/post-state excess.

### 3.3 Forbidden shortcut

The following identity is NOT established:

`LAYER17_PP3_CAUSAL_ROLE = LAYER22_STRONG_ALIGNMENT_NATIVE_UPDATE_OWNER`

The following is also NOT established:

`PP3 = UNIVERSAL_STATE_OWNER`

Therefore Gen5 may not directly hard-code PP3, the strong-240 mask, or the K
alignment direction as a permanent semantic owner.

Before a trainable state-update ownership architecture is allowed, Gen5 must
establish a role-to-native-update bridge.

---

## 4. Gen5 ownership semantics

Gen5 Phase 1 defines ownership as **write authority**, not read access and not
semantic naming.

At a native recurrent update site, write:

`S_t = Carry_t + Write_t`

A later Gen5 correction mechanism may propose an additional update:

`DeltaW_theta`

Let `R` denote a locally authenticated causal-role realization at that native
update site and let `P_R` be its projector.

Let `C` denote a response-blind, rank-matched local control realization with
projector `P_C`.

The first Gen5 ownership interpretation is:

> The frozen native mechanism retains exclusive write authority over the
> authenticated causal-role subspace. A newly introduced correction mechanism may
> update the complementary state but may not overwrite the protected role
> subspace unless a later experiment separately establishes such authority.

This is intentionally conservative.

Gen5 does not begin by inventing a new semantic state vector or by allowing a new
module to write freely into an already validated causal mechanism.

---

## 5. Minimal future comparison family

This section defines scientific semantics only. No implementation is authorized.

### G5-C0 — shared-write reference

`DeltaW_eff = DeltaW_theta`

The correction mechanism has unrestricted write access.

This condition is a contextual baseline, not the primary capacity-matched control.

### G5-C1 — matched ownership-null control

`DeltaW_eff = (I - P_C) DeltaW_theta`

The correction mechanism is denied write access to a rank-matched,
response-blind control subspace.

### G5-M1 — causal-role-owned state update

`DeltaW_eff = (I - P_R) DeltaW_theta`

The correction mechanism is denied write access to the authenticated local
causal-role subspace.

The frozen native update remains intact in all three conditions.

### Primary comparison

The primary ownership comparison is:

`G5-M1 vs G5-C1`

because both remove the same-dimensional write authority from the correction
mechanism.

`G5-C0` is secondary context because it has more effective unconstrained update
degrees of freedom.

---

## 6. Capacity and information matching requirements

A valid later comparison must keep constant across G5-C1 and G5-M1:

- backbone identity;
- correction-module architecture;
- parameter count;
- correction-module inputs;
- initialization rule;
- optimizer;
- learning-rate schedule;
- number of update steps;
- data and split;
- seed family;
- loss functions;
- forward information available to the correction module;
- projector rank;
- ownership operation count;
- ownership application site.

Only the identity of the protected subspace may differ.

The first Gen5 experiment must not compare a smaller role-owned module against a
larger unrestricted module and call the difference an ownership effect.

---

## 7. Initial backbone policy

For the first identifiable Gen5 ownership test:

`BASE_MAMBA_BACKBONE = FROZEN`

The trainable object, if later authorized, is the minimal correction mechanism
only.

Reason:

If the native backbone is simultaneously allowed to relearn the protected role,
then a successful result cannot distinguish:

- ownership preservation;
- compensatory backbone relearning;
- geometry migration;
- ordinary fine-tuning.

Backbone unfreezing is therefore a later research question.

This restriction is for causal identification, not because a final architecture
must keep Mamba frozen.

---

## 8. Mandatory role-to-native-update bridge

A trainable G5-M1 implementation is BLOCKED until a local native-update
realization `R` is prospectively authenticated.

The bridge must connect:

`FROZEN_OPERATIONAL_CAUSAL_ROLE`
to
`NATIVE_STATE_UPDATE_REALIZATION`

without assuming coordinate identity.

### 8.1 Evidence-guided source and target

The evidence-guided source is the frozen 130M layer-17 / target-token causal-role
program.

The evidence-guided downstream native-update site is the already mechanistically
motivated layer-22 recurrent update pathway.

This is a pre-specified bridge candidate, not a layer search.

### 8.2 Bridge requirements

A later bridge design must:

1. use a prospectively frozen, non-overlapping holdout population;
2. preserve tokenizer/anchor/provenance rules;
3. use the existing layer-17 role intervention semantics as upstream causal
   manipulation rather than re-discovering a favorable plane;
4. capture layer-22 native write and post-state responses under the same model
   forward;
5. include a response-blind matched geometric control;
6. include a restoration counterpart;
7. construct the eventual native-update role realization without using
   confirmation outcomes;
8. confirm that realization on held-out examples;
9. prohibit post-hoc layer, token, channel, plane, rank, or cohort search.

### 8.3 Minimum bridge logic

The bridge must support both of the following bounded statements before `P_R`
may be used for Gen5 ownership:

**Necessity-like bridge**

Removing or neutralizing the frozen upstream causal-role component changes the
pre-specified native write/post-state phenotype more than the matched
response-blind control.

**Restoration-like bridge**

Restoring the frozen role component from the same neutralized background
recovers the pre-specified native write/post-state phenotype more strongly than
the matched replacement control.

A one-sided association, correlation, or geometry match is insufficient.

### 8.4 Bridge failure

If this bridge is not established, Gen5 must not silently:

- call PP3 a native state owner;
- copy the layer-22 K direction and call it the same causal role;
- search other layers until a bridge appears;
- switch to a favorable channel subset;
- redefine the endpoint after seeing outcomes.

A bridge failure means the current evidence does not justify this specific
state-update ownership realization.

It does not falsify the entire ownership research program.

---

## 9. Role realization requirements

The eventual `R` used in G5-M1 must satisfy all of the following:

- native-update-local;
- rank fixed before confirmation outcomes;
- construction fixed before confirmation outcomes;
- compatible with the frozen role-to-update bridge;
- distinct from its matched control `C`;
- not selected by downstream task accuracy;
- not selected by Gen5 training response;
- not semantically named without separate evidence.

Historical PP3 may seed the bridge design, but `R` is not allowed to be defined
as "PP3 because PP3 worked before" unless the bridge prospectively establishes
that exact realization at the native update site.

---

## 10. Primary Gen5 Phase 1 scientific hypothesis

Conditional on a valid role-to-native-update bridge:

`H_G5_1_UPDATE`

> Protecting a prospectively authenticated causal-role-native-update realization
> from modification by an otherwise matched learned correction preserves causal
> role integrity more strongly than protecting a response-blind rank-matched
> control realization.

This is an ownership hypothesis.

It is not a task-performance hypothesis.

---

## 11. Primary measurement ordering

The later ownership experiment must preserve the Phase 0 hierarchy.

### Primary

`CAUSAL_ROLE_INTEGRITY`

This must be measured using a prospective causal-role assay derived from the
validated transport / specificity / necessity / restoration framework.

### Secondary

`NATIVE_STATE_ORGANIZATION`

Including whether the protected native-update realization remains structurally
intact and whether compensatory residual reorganization appears.

### Tertiary

`TASK_FUNCTIONAL_READOUT`

Only after the primary mechanistic endpoint is interpreted.

### Later only

- behavioral effect;
- robustness;
- cross-scale transfer;
- objective-conditioned functional comparison.

Task accuracy cannot rescue failure of the primary role-integrity endpoint.

---

## 12. Falsification logic

The first Gen5 state-update ownership hypothesis is unsupported if:

1. the mandatory role-to-native-update bridge fails;
2. G5-M1 does not preserve causal-role integrity better than G5-C1 under a
   valid matched comparison;
3. an apparent effect is explained by unequal capacity, unequal information,
   training exposure, or projector rank;
4. the result requires post-hoc choice of layer, plane, channel subset, owner
   rank, cohort, or endpoint;
5. task performance improves while causal-role integrity does not.

The following are not valid rescue operations:

- owner-rank sweep;
- alternate layer search;
- alternate plane search;
- alternate channel search;
- changing the control after seeing results;
- adding information restrictions simultaneously;
- adding gradient restrictions simultaneously;
- unfreezing the backbone;
- optimizing task score and then redefining the mechanism.

---

## 13. What Phase 1 does not claim

Phase 1 does not claim:

- PP3 is a semantic owner;
- the K strong-240 channels are semantic owners;
- layer 17 and layer 22 carry the same coordinates;
- one causal plane is the complete mechanism;
- state ownership improves task accuracy;
- state ownership improves calibration;
- state ownership gives useful steering;
- state ownership generalizes across Mamba scales;
- Frame / Predicate / Sufficiency / Polarity / Authorization are localized
  recurrent owners;
- the parked confident-error precursor has been recovered.

---

## 14. Why gradient ownership is deferred

Historical `explicit_local` already tested a hard downstream gradient-isolation
regime and was descriptively harmful.

That result does not falsify controlled gradient ownership, but it establishes
that gradient ownership is not a neutral first Gen5 choice.

Gen5 first tests whether ownership can be defined at the native update mechanism
itself.

Only after a state-update ownership effect is established should a later phase
ask whether backward modification rights should also follow the same role
boundary.

---

## 15. Why information ownership is deferred

Information ownership changes what downstream computations may read.

That can conflate:

- state construction;
- representation availability;
- update authority;
- decision access.

The first Gen5 causal test therefore keeps read access shared and manipulates
only write authority.

Initial rule:

`READ_ACCESS = SHARED`

`WRITE_AUTHORITY = EXPERIMENTAL_VARIABLE`

`GRADIENT_AUTHORITY = UNCHANGED`

This isolates one ownership dimension.

---

## 16. Current authorization boundary

AUTHORIZED_NOW =
`STATIC_PHASE1_SCIENTIFIC_DESIGN_ONLY`

NOT_AUTHORIZED_NOW:

- implementation;
- source-code modification;
- native-state hook implementation;
- adapter/correction-module implementation;
- training;
- evaluation;
- causal execution;
- Kaggle;
- GPU execution;
- layer search;
- plane search;
- channel search;
- owner-rank search;
- hyperparameter sweep.

---

## 17. Next stage after Phase 1 freeze

If this design is reviewed and frozen without conflict:

NEXT_STAGE =
`GEN5_PHASE1B_NATIVE_UPDATE_ROLE_BRIDGE_SPECIFICATION`

Phase 1B must define the exact prospective bridge:

`layer-17 validated causal role -> layer-22 native write/post-state realization`

including:

- exact frozen parent artifacts;
- exact intervention site;
- exact capture site;
- exact population;
- matched geometric control;
- restoration control;
- construction/confirmation split;
- artifact schema;
- statistical decision rule;
- provenance gates.

Only a successful bridge may authorize later implementation design for
G5-C0 / G5-C1 / G5-M1.

---

## 18. Phase 1 decision summary

SELECTED_OWNERSHIP_DIMENSION =
`STATE_UPDATE_AUTHORITY`

FIRST_OWNERSHIP_SEMANTIC =
`NATIVE_ROLE_SUBSPACE_HAS_EXCLUSIVE_WRITE_PROTECTION_FROM_LEARNED_CORRECTION`

PRIMARY_COMPARISON =
`G5-M1_VS_G5-C1`

BACKBONE_POLICY =
`FROZEN_FOR_FIRST_IDENTIFIABLE_TEST`

READ_ACCESS =
`SHARED`

GRADIENT_AUTHORITY =
`UNCHANGED`

DIRECT_PP3_TO_OWNER_PROMOTION =
`FORBIDDEN`

DIRECT_K_ALIGNMENT_TO_OWNER_PROMOTION =
`FORBIDDEN`

ROLE_TO_NATIVE_UPDATE_BRIDGE_REQUIRED =
`YES`

NEXT_STAGE =
`GEN5_PHASE1B_NATIVE_UPDATE_ROLE_BRIDGE_SPECIFICATION`

STATUS =
`FROZEN_ON_COMMIT`
