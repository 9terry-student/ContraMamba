# ContraMamba Gen4 Six-Cell Native Mamba State Mechanistic Bridge
# Scientific Specification Candidate

STATUS =
CANDIDATE

PHASE =
GEN4_NATIVE_MAMBA_STATE_MECHANISTIC_BRIDGE_SPECIFICATION

THIS_DOCUMENT_CREATES_NEW_SCIENTIFIC_EVIDENCE =
NO

IMPLEMENTATION_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_LOADING_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO


## 1. Purpose

This specification defines the first native-Mamba-state follow-up to the
validated Gen4 six-cell behavioral result.

The frozen Gen4 result established, within its exact scope, that q_authorized
responds systematically and asymmetrically to controlled semantic-axis
substitutions.

The next question is narrower:

DO THE SAME FROZEN WITHIN-MECHANISM SEMANTIC PERTURBATIONS PRODUCE
PRESPECIFIED LOCAL CHANGES IN NATIVE MAMBA RECURRENT-STATE KINEMATICS?

This is a representational-mechanistic bridge question.

It is not yet a causal intervention on internal state.


## 2. Gen4 evidence authority

R6_VALIDATED_EVIDENCE_ANALYSIS =
e616d405f03ba55d9acd1c2244ac14907d306602

R6_VALIDATED_STATISTICAL_RESULTS =
5896d4740cd390a56ff5e2a3459c68ce65e2bc84

R6_EXECUTION_AUTHORITY =
f437ec356e448a1240a8f9c895e779fccaff1a95

R5_VALIDATED_EVALUATOR_ARTIFACTS =
a3b5bcf2ded8dc0e86e859bbba12b5601a2fdea0

GEN4_SIX_CELL_STRUCTURAL_ARTIFACT_FREEZE =
5b79d8585b20cf6fa4cfe52bbdbdce52374653aa

GEN4_SIX_CELL_STRUCTURAL_SPECIFICATION =
0a0da5354782e542520fb5bba146ab1a599d17ef

CANONICAL_STRUCTURAL_ARTIFACT_SHA256 =
b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

CANONICAL_STRUCTURAL_ROWS =
1800

SOURCE_PAIR_COUNT =
300

CELLS_PER_SOURCE_PAIR =
6


## 3. Relationship to the earlier Native State Kinematics lineage

PARENT_NATIVE_STATE_HYPOTHESIS =
12c86088f68482870dd53cbcf6c363499b248f81

EARLIER_CONFIDENT_ERROR_DESIGN_AUTHORITY =
c4286a4d8af9ae31b7e44de2a3e79b560efa2355

EARLIER_TAU_E_ANNOTATION_FREEZE =
4757a4f9baffc1eeaab7d8c9d5e9f6ababd19031

EARLIER_CONFIDENT_ERROR_COHORT_FEASIBILITY_RESULT =
373002c8382605efb835c665ff88740f0aaa744e

The earlier confident-correct versus confident-wrong design remains a valid
historical design object.

Its Phase C feasibility result established that its frozen A0 population did
not contain an adequate confident-wrong cohort for the prespecified matching
and power requirements.

This specification does not repair, overwrite, or reinterpret that result.

CONFIDENT_ERROR_BRANCH_STATUS =
PRESERVED_AS_FEASIBILITY_BLOCKED

THIS_SPEC_SUPERSEDES_CONFIDENT_ERROR_DESIGN =
NO

The present Gen4 experiment asks a different question and uses a different
scientific population:

complete within-source-pair semantic interventions,

not correctness-defined cohorts.


## 4. Methodological principles inherited from Native State Kinematics

The following principles are retained:

- inspect the native selective-SSM recurrent state rather than only a
  downstream representation;
- use simple Euclidean geometry first;
- freeze one primary layer before outcome inspection;
- use event-relative prefix-only windows;
- prohibit best-layer scans;
- prohibit arbitrary token scans;
- prohibit learned trajectory embeddings;
- prohibit nonlinear detectors as the first analysis;
- prohibit a learned probe as a substitute for native-state measurement;
- keep model seeds distinct from genuine native-state replications;
- require exact provenance before scientific execution.

These inherited principles do not transfer the earlier confident-error
population or its statistical estimands.


## 5. Frozen Gen4 structural cells

The exact cells are:

C0_SHAM =
(0,0,0,0)

C1_TITLE =
(1,0,0,0)

C2_NAME =
(0,1,0,0)

C3_ROLE =
(0,0,1,0)

C4_PREDICATE =
(0,0,0,1)

C5_TITLE_NAME =
(1,1,0,0)

Axis order is:

title
name
role
predicate

All six cells are produced under the same held-constant mechanism:

masked_slot_substitution

The native-state experiment must preserve the exact complete six-cell
source-pair block.

No historical intervention family may replace one of these cells.


## 6. Frozen behavioral context

R6 established the following q_authorized dispositions:

TITLE =
SYSTEMATIC_EFFECT_NOT_ESTABLISHED

NAME =
SUPPORTED_NEGATIVE_EFFECT

ROLE =
SUPPORTED_POSITIVE_EFFECT

PREDICATE =
SUPPORTED_POSITIVE_EFFECT

TITLE_NAME_INTERACTION =
SUPPORTED_NONADDITIVITY

TITLE_MINUS_NAME =
SUPPORTED_EFFECT_DIFFERENCE

These behavioral findings motivate the state analysis.

They do not prescribe the sign of a native-state kinematic endpoint.

For example:

a negative q_authorized effect does not imply negative speed,
negative turning, or negative path efficiency.

NATIVE_STATE_DIRECTION_MUST_NOT_BE_INFERRED_FROM_OUTPUT_SIGN =
REQUIRED


## 7. Native state object

For recurrent layer l and consumed token t, define:

s_t^(l)

as the vectorized native selective-SSM recurrent state after token t.

STATE_REPRESENTATION =
VECTORIZED_NATIVE_SELECTIVE_SSM_RECURRENT_STATE

The following are not substitutes:

- final q_authorized;
- router logits;
- final task logits;
- entitlement probability;
- Mamba hidden output alone;
- residual stream alone;
- cached encoder last_hidden_state alone;
- learned probe representation.

The exact implementation-level tensor source and coordinate must be proven by
a later static/instrumentation feasibility authority before execution.

NATIVE_STATE_TENSOR_SOURCE =
MUST_BE_EXACTLY_BOUND_BEFORE_EXECUTION


## 8. Primary layer

For a backbone with L recurrent layers indexed:

0 ... L-1

define:

L_PRIMARY =
floor((L - 1) / 2)

PRIMARY_LAYER_RULE =
ARCHITECTURE_MIDPOINT

Only L_PRIMARY may determine the first confirmatory state verdict.

No layer may be selected using:

- R6 effect size;
- native-state separation;
- p-value;
- visualization;
- downstream prediction behavior.

Q1 and Q3 layers may be considered later only under separate secondary
authority.

BEST_LAYER_SCAN =
PROHIBITED


## 9. Exact input coordinate

The state experiment must use the same active input semantics already frozen
for Gen4 Tier2 evaluation:

claim tokens
+
EOS separator
+
evidence tokens

with:

claim budget = 63
evidence budget = 64
maximum serialized length = 128
add_special_tokens = false

The canonical tokenizer active encoding is already frozen by the Gen4 R2
lineage.

GEN4_R2_TOKENIZER_VALIDATION =
17f1ddfc8286796f27c4a61716a21e14126bb836

No alternate tokenizer or rendered-text retokenization scheme may be
introduced.


## 10. Semantic event anchors

The bridge uses generator-declared semantic spans.

Anchors may not be selected from native-state behavior.

For each cell and source pair, define the relevant evidence-token event end as
the final consumed token overlapping the complete generator-declared realized
semantic span.

Four axis-local anchors are required:

A_TITLE =
end of complete realized title span

A_NAME =
end of complete realized name span

A_ROLE =
end of complete realized role span

A_PREDICATE =
end of complete realized predicate span

For the title-name interaction, define one common semantic anchor:

A_IDENTITY =
end of the complete realized title + name identity block

Because the frozen statement order places title before name, A_IDENTITY is the
end of the realized name span after the complete title-name identity has been
consumed.

The anchor must be derived separately for each rendered cell from
generator-declared source structure and the frozen tokenizer coordinate.

This permits alternate strings with different token lengths without pretending
that their absolute token indices must be identical.


## 11. Anchor pairing rules

The title main-effect state contrast uses:

C1_TITLE versus C0_SHAM

at each cell's A_TITLE.

The name main-effect state contrast uses:

C2_NAME versus C0_SHAM

at each cell's A_NAME.

The role main-effect state contrast uses:

C3_ROLE versus C0_SHAM

at each cell's A_ROLE.

The predicate main-effect state contrast uses:

C4_PREDICATE versus C0_SHAM

at each cell's A_PREDICATE.

The title-name interaction state contrast uses:

C5_TITLE_NAME
C1_TITLE
C2_NAME
C0_SHAM

all evaluated relative to each cell's A_IDENTITY.

No state-level title-name interaction may mix title-end and name-end anchors.


## 12. Why title-minus-name is not a primary native-state estimand

The frozen behavioral family contains:

Delta_title - Delta_name.

At the output level this is well-defined because both quantities are changes
in the same scalar q_authorized outcome.

At the local native-state level, the title and name main effects use different
semantic event anchors.

Subtracting their raw kinematic contrasts would therefore combine:

semantic-axis difference
and
event-position/trajectory-coordinate difference.

Accordingly:

STATE_LEVEL_TITLE_MINUS_NAME_PRIMARY_ESTIMAND =
NOT_AUTHORIZED

The behavioral title-minus-name result remains valid contextual evidence.

It must not be copied mechanically into the native-state confirmatory family.


## 13. Prefix-only window

For anchor a, the canonical local interval is:

[a, a+4]

and must end strictly before the terminal consumed token state.

Eligibility requires:

a + 4 <= terminal_index - 1

The window may not be shortened to rescue a row.

SHORTENED_POST_WINDOW =
PROHIBITED

For a source pair to enter the canonical native-state analysis, all cells and
all anchors needed for the complete primary family must satisfy the prefix
rule.

CANONICAL_TARGET_PAIR_COUNT =
300

If complete prefix eligibility is not 300 of 300 source pairs:

CANONICAL_EXECUTION =
BLOCKED_PENDING_NEW_AUTHORITY

This prevents outcome-dependent attrition or contrast-specific pair sets.


## 14. Local kinematic definitions

For each cell, source pair, primary layer, and relevant anchor:

v_t =
s_t - s_(t-1)

speed:

nu_t =
||v_t||_2

directional turning:

kappa_t =
1 - cos(v_t, v_(t-1))

POST4_SPEED =
mean(
    nu_(a+1),
    nu_(a+2),
    nu_(a+3),
    nu_(a+4)
)

POST4_TURNING =
mean(
    kappa_(a+1),
    kappa_(a+2),
    kappa_(a+3),
    kappa_(a+4)
)

POST4_PATH_EFFICIENCY =
||s_(a+4) - s_a||_2
/
sum_(t=a+1 to a+4) ||v_t||_2

when the denominator is nonzero.

Zero-denominator behavior must be explicitly frozen by a later implementation
specification.

No epsilon denominator may be invented silently.


## 15. Primary state estimands

For any one kinematic endpoint K from:

POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

define pair-level state contrasts.

Title:

Delta_title_K(p) =
K(p,C1_TITLE,A_TITLE)
-
K(p,C0_SHAM,A_TITLE)

Name:

Delta_name_K(p) =
K(p,C2_NAME,A_NAME)
-
K(p,C0_SHAM,A_NAME)

Role:

Delta_role_K(p) =
K(p,C3_ROLE,A_ROLE)
-
K(p,C0_SHAM,A_ROLE)

Predicate:

Delta_predicate_K(p) =
K(p,C4_PREDICATE,A_PREDICATE)
-
K(p,C0_SHAM,A_PREDICATE)

Title-name interaction:

Interaction_title_name_K(p) =
K(p,C5_TITLE_NAME,A_IDENTITY)
-
K(p,C1_TITLE,A_IDENTITY)
-
K(p,C2_NAME,A_IDENTITY)
+
K(p,C0_SHAM,A_IDENTITY)


## 16. Confirmatory state family

The first native-state bridge contains exactly:

5 structural estimands
x
3 kinematic endpoints

for:

PRIMARY_NATIVE_STATE_HYPOTHESIS_COUNT =
15

The five structural estimands are:

Delta_title
Delta_name
Delta_role
Delta_predicate
Interaction_title_name

The three endpoints are:

POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

No additional primary endpoint may be introduced after state inspection.

No token-wise family is permitted.

No layer-wise family is permitted.

No title-minus-name state hypothesis is included.


## 17. Statistical boundary

This specification freezes the scientific estimands but does not authorize
their computation or choose an implementation.

A later statistical-analysis specification must freeze, before native-state
outcome inspection:

- exact one-sample paired-contrast test;
- uncertainty interval;
- effect-size definition;
- handling of zero path length;
- global multiplicity control over the 15 primary hypotheses;
- deterministic serialization;
- support and non-support decision rules.

GLOBAL_PRIMARY_MULTIPLICITY_MUST_COVER_ALL_15 =
YES

Separate correction within each endpoint family is not sufficient unless a
later authority explicitly justifies a hierarchical procedure before outcome
inspection.

DEFAULT_DIRECTIONALITY =
TWO_SIDED

No direction is inferred from R6 output sign.


## 18. Relationship between state results and R6 output results

R6 provides frozen behavioral context.

The state experiment does not test whether the R6 p-values replicate.

Instead it tests whether controlled semantic perturbations create local
native-state kinematic changes under the same structural factorial design.

Possible scientifically distinct outcomes include:

A.
name affects q_authorized and native-state kinematics.

B.
name affects q_authorized but no prespecified local native-state endpoint.

C.
title affects native-state kinematics despite no established title effect on
q_authorized.

D.
title has neither an established q_authorized effect nor a supported local
state-kinematic effect.

E.
title-name nonadditivity appears at both output and native-state levels.

These outcomes have different interpretations.

Therefore:

STATE_EFFECT_MUST_NOT_BE_DEFINED_AS_OUTPUT_SIGN_MATCH =
YES


## 19. Bounded bridge interpretation

A significant native-state contrast supports only:

PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_RESPONSE

for that exact semantic estimand and endpoint.

It does not by itself establish:

- causal mediation;
- necessity;
- sufficiency;
- downstream use of that state information;
- arbitrary-model generalization;
- arbitrary-dataset generalization.

If an R6-supported semantic estimand also has a supported prespecified
native-state kinematic contrast, the report may state:

OUTPUT_EFFECT_HAS_A_PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_CORRELATE

It may not state:

NATIVE_STATE_EFFECT_CAUSES_OUTPUT_EFFECT.


## 20. Title control interpretation

Title is retained in the primary state family despite its non-established R6
main effect.

This is deliberate.

If title shows a native-state response but no q_authorized response, that is
scientifically informative:

the recurrent state may encode or react to the title perturbation without the
validated downstream output showing a systematic title-only displacement.

If title shows no state effect either, that provides a different pattern of
alignment.

TITLE_IS_NOT_A_NULL_STATE_ASSUMPTION =
YES


## 21. Native backbone identity requirement

The 18 Gen4 R5 evaluators must not automatically be treated as 18 independent
native-state models.

The Gen3 grouped evaluators were trained with the Mamba encoder frozen.

Before native-state execution, a dedicated provenance audit must determine
whether the native Mamba parameter tensors are exactly identical across all
18 evaluator checkpoints.

If exact native-backbone identity is established:

NATIVE_STATE_MODEL_REPLICATION_COUNT =
1

and one canonical authenticated native backbone may be used.

The 18 downstream evaluator heads remain relevant to the frozen R6 behavioral
average but do not become 18 native-state replications.

If native Mamba tensors differ across evaluator checkpoints:

EXECUTION =
BLOCKED

until a new multi-backbone scientific design is frozen.


## 22. No downstream-router contamination

The native-state measurement path must terminate at the frozen native
selective-SSM recurrent state.

It must not use:

- q router activations;
- entitlement head activations;
- final logits;
- downstream edge-specific representations;
- gradients from q_authorized;
- learned projection toward the output.

R6 outputs may be joined only after native-state measurements and their exact
artifact identities have been frozen if a later bridge-association analysis is
authorized.

The primary 15 state hypotheses themselves do not require R6 row-level outcome
values during state extraction.


## 23. Outcome-blind measurement ordering

The required ordering is:

PHASE A =
STATIC_NATIVE_BACKBONE_AND_INSTRUMENTATION_PROVENANCE

PHASE B =
GEN4_EVENT_ANCHOR_AND_PREFIX_FEASIBILITY_WITHOUT_NATIVE_STATES

PHASE C =
FREEZE_MEASUREMENT_IMPLEMENTATION_AND_SYNTHETIC_VALIDATION

PHASE D =
SEPARATELY_AUTHORIZED_NATIVE_STATE_EXTRACTION

PHASE E =
FREEZE_NATIVE_STATE_MEASUREMENT_ARTIFACTS

PHASE F =
SEPARATELY_AUTHORIZED_PRIMARY_15_STATISTICAL_ANALYSIS

Native-state outcomes must not be inspected while changing:

- anchor rules;
- layer;
- window;
- endpoint formulas;
- pair admission;
- tensor source.


## 24. Required Phase A feasibility questions

Before implementation authority, establish statically:

1. exact native recurrent-state tensor source in the frozen Mamba
   implementation;
2. token-time indexing semantics;
3. layer indexing semantics;
4. whether exact native Mamba parameter identity holds across all 18
   evaluator checkpoints;
5. whether the required native-state tensor can be serialized without using
   downstream router representations;
6. whether prior O0c instrumentation can be reused exactly or requires a new
   bounded implementation.

No model forward is authorized by this specification.


## 25. Required Phase B feasibility questions

Before native-state extraction, establish without state inspection:

1. deterministic A_TITLE for all cells requiring it;
2. deterministic A_NAME for all cells requiring it;
3. deterministic A_ROLE for all cells requiring it;
4. deterministic A_PREDICATE for all cells requiring it;
5. deterministic A_IDENTITY for C0/C1/C2/C5;
6. exact tokenizer coordinate binding;
7. exact 300-of-300 complete prefix eligibility.

If any required semantic anchor is ambiguous:

EXECUTION =
BLOCKED

If complete 300-of-300 prefix eligibility fails:

EXECUTION =
BLOCKED_PENDING_NEW_AUTHORITY


## 26. Falsification conditions

The bounded native-state bridge is not supported for a structural estimand if
none of its three prespecified kinematic endpoints survives the later frozen
global primary multiplicity procedure.

A failure must not be rescued by:

- another layer;
- another token window;
- another anchor;
- another distance metric;
- whitening;
- PCA;
- learned probes;
- nonlinear classifiers;
- trajectory embeddings;
- threshold tuning.

A fully negative 15-hypothesis primary family is a valid negative result.


## 27. Explicitly forbidden first-stage analyses

The following are outside this first bridge:

- PCA chosen after viewing state separation;
- UMAP or t-SNE as evidence;
- learned semantic probes;
- classifier accuracy on state vectors;
- nonlinear state detectors;
- causal state editing;
- state patching;
- activation steering;
- decision-space Jacobian projection;
- arbitrary cross-layer pooling;
- hyperparameter sweeps;
- training a new model.

These may become later research objects only if independently justified.


## 28. Provenance requirements

Any later scientific artifact must bind at minimum:

- Gen4 structural artifact SHA256;
- source-pair identities;
- cell identities;
- tokenizer identity;
- exact serialized token IDs or their canonical hash;
- event-anchor manifest SHA256;
- backbone identity;
- checkpoint/native-backbone tensor identity;
- runtime identity;
- native-state instrumentation implementation identity;
- primary layer;
- state tensor shape;
- token coordinates;
- endpoint implementation identity;
- output artifact SHA256.

Provenance mismatch is a blocker.


## 29. Current scientific status

GEN4_SEMANTIC_AXIS_DEPENDENT_AUTHORIZATION_RESPONSE =
ESTABLISHED_WITHIN_FROZEN_SCOPE

GEN4_TITLE_NAME_NONADDITIVE_OUTPUT_RESPONSE =
ESTABLISHED_WITHIN_FROZEN_SCOPE

EARLIER_CONFIDENT_ERROR_NATIVE_STATE_DESIGN =
FEASIBILITY_BLOCKED

GEN4_NATIVE_STATE_KINEMATIC_RESPONSE =
NOT_TESTED

GEN4_NATIVE_STATE_MECHANISTIC_BRIDGE =
NOT_ESTABLISHED

NATIVE_STATE_CAUSALITY =
NOT_ESTABLISHED

TRAINING =
NOT_AUTHORIZED


## 30. Next authorized research object after this specification freeze

NEXT_RESEARCH_OBJECT =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_FEASIBILITY_AUDIT_AUTHORITY

That authority must be static/outcome-blind first.

It may inspect repository source, checkpoint tensor identities under an
explicitly bounded CPU provenance operation if separately authorized, and
tokenizer/structural metadata.

It must not perform scientific native-state extraction.

It must not compute the 15 primary endpoints.

It must not perform the 15 primary statistical tests.


## 31. Stop condition

Stop after this specification candidate is created and reviewed.

Do not load checkpoints.

Do not run model inference.

Do not extract native states.

Do not compute kinematic endpoints.

Do not run new statistical tests.

Do not train.

Do not use Kaggle.

END_OF_GEN4_NATIVE_MAMBA_STATE_MECHANISTIC_BRIDGE_SPECIFICATION
