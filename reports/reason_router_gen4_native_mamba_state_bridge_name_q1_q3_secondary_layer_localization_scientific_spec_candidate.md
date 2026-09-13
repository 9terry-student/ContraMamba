# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Secondary-Layer Localization
# Scientific Specification Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_SECONDARY_LAYER_LOCALIZATION_SCIENTIFIC_SPECIFICATION

THIS_DOCUMENT_CREATES_NEW_SCIENTIFIC_EVIDENCE =
NO

IMPLEMENTATION_ALLOWED =
NO

STATISTICAL_EXECUTION_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_LOADING_ALLOWED =
NO

TOKENIZER_EXECUTION_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_ALLOWED =
NO

## 1. Frozen lineage

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

PHASE_F_VALIDATED_STATISTICAL_RESULT_FREEZE =
ab3428e7be08af26fa1fdafd1483a34e48fbcf8c

POST_PHASE_F_MECHANISTIC_INTERPRETATION =
029182257c3f9d89d41c0f37dd508eb3d5f405cd

DEPTH_SELECTIVITY_INTERPRETATION_CORRECTION =
2e076cbd8e9633c3ab7abb222a05e409366539a7

Q1_Q3_ARCHITECTURE_DEPTH_INDEX_SPECIFICATION =
ecbf6ba720c0e173ac7089a00bb5b783aa16fa6c

PHASE_D_CORRECTED_IMPLEMENTATION =
c7ae7fac4c64e9bd64adcae819a3da3dd46f17f7

PHASE_F_STATUS =
CLOSED

## 2. Scientific motivation

Frozen R6 behavioral evidence establishes a NAME effect.

Frozen Phase F primary-layer analysis does not establish a NAME response for
any of the three prespecified local kinematic endpoints at layer 11.

The next bounded question is therefore:

DOES THE FROZEN R6 NAME EFFECT HAVE A LOCAL NATIVE-MAMBA-STATE KINEMATIC
CORRELATE AT EITHER OF TWO PREDEFINED ARCHITECTURE-DEFINED SECONDARY DEPTHS?

This is a secondary localization question.

It is not a causal mediation test.

It is not an all-layer localization scan.

It is not a test that layer 5 or layer 17 differs significantly from layer 11.

## 3. Selection provenance

This follow-up was selected after observing the frozen Phase F primary-layer
result.

FOLLOWUP_SELECTION_USED_PHASE_F_OUTCOME =
YES

FOLLOWUP_STATUS =
POST_PRIMARY_OUTCOME_SECONDARY_FOLLOWUP

The selection rationale is:

- R6 establishes a behavioral DELTA_NAME effect;
- Phase F does not establish DELTA_NAME for any of the three primary-layer
  local kinematic endpoints;
- the original mechanistic bridge specification explicitly preserved Q1 and
  Q3 as possible later secondary layers;
- the post-Phase-F interpretation selected NAME as the simplest bounded
  unresolved bridge question;
- the exact Q1/Q3 indices were then frozen by architecture only before any
  Q1/Q3 state outcome was inspected.

This follow-up must not be described as if NAME and the secondary layers had
been selected before viewing Phase F.

SECONDARY_FAMILY_MULTIPLICITY_SCOPE =
WITHIN_THE_NEW_SIX_HYPOTHESIS_SECONDARY_FAMILY

OVERALL_ADAPTIVE_PROGRAM_FWER_ACROSS_PHASE_F_AND_THIS_FOLLOWUP =
NOT_CLAIMED

## 4. Frozen structural population

CANONICAL_STRUCTURAL_ARTIFACT =
reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl

CANONICAL_STRUCTURAL_ARTIFACT_SHA256 =
b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

CANONICAL_STRUCTURAL_ROWS =
1800

SOURCE_PAIR_COUNT =
300

CELLS_PER_SOURCE_PAIR =
6

No source pair may be selected using Q1/Q3 native-state behavior.

No source pair may be dropped to improve a secondary-layer result.

The frozen six-cell structural artifact remains the population authority.

The scientific contrast in this follow-up uses only the two cells required by
DELTA_NAME:

C0_SHAM =
(0,0,0,0)

C2_NAME =
(0,1,0,0)

CELL_CONTRAST =
C2_NAME_MINUS_C0_SHAM

No historical or newly generated intervention family may replace either cell.

## 5. Structural estimand

The only structural estimand is:

DELTA_NAME

TITLE =
OUT_OF_SCOPE

ROLE =
OUT_OF_SCOPE

PREDICATE =
OUT_OF_SCOPE

TITLE_NAME_INTERACTION =
OUT_OF_SCOPE

TITLE_MINUS_NAME =
OUT_OF_SCOPE

No additional structural estimand may be added after Q1/Q3 outcome
inspection.

## 6. Frozen secondary depth locations

NATIVE_MAMBA_LAYER_COUNT =
24

Q1_LAYER_INDEX =
5

PRIMARY_MIDPOINT_LAYER_INDEX =
11

Q3_LAYER_INDEX =
17

SECONDARY_LAYER_SET =
{5,17}

The layer set is frozen by:

ecbf6ba720c0e173ac7089a00bb5b783aa16fa6c

No replacement layer is allowed based on:

- effect size;
- p-value;
- visualization;
- trajectory appearance;
- runtime success at another layer;
- proximity to a supported layer.

BEST_LAYER_SCAN =
PROHIBITED

ALL_LAYER_SCAN =
PROHIBITED

FALLBACK_LAYER_SELECTION =
PROHIBITED

## 7. Semantic coordinate

SEMANTIC_ANCHOR =
A_NAME

A_NAME is the end of the complete realized generator-declared name span in the
frozen active token coordinate.

The follow-up must preserve the same active input semantics used by the
mechanistic bridge lineage:

claim tokens
+
EOS separator
+
evidence tokens

with:

CLAIM_BUDGET =
63

EVIDENCE_BUDGET =
64

MAXIMUM_SERIALIZED_LENGTH =
128

ADD_SPECIAL_TOKENS =
FALSE

No rendered-text retokenization is allowed.

No new token coordinate may be introduced.

## 8. Prefix-only local interval

For each required cell and secondary layer, preserve the frozen local interval:

[a, a+4]

where:

a =
A_NAME

Eligibility requires:

a + 4 <= terminal_index - 1

SHORTENED_POST_WINDOW =
PROHIBITED

TARGET_SOURCE_PAIR_COUNT =
300

If the later Q1/Q3 feasibility/provenance stage cannot establish the required
complete eligible population under the frozen coordinate:

SCIENTIFIC_EXECUTION =
BLOCKED_PENDING_NEW_AUTHORITY

No outcome-dependent attrition is permitted.

## 9. Native-state object

The scientific object remains:

VECTORIZED_NATIVE_SELECTIVE_SSM_RECURRENT_STATE

after consumed token t:

s_t^(l)

The follow-up does not substitute:

- Mamba hidden output;
- residual stream;
- router logits;
- q_authorized;
- task logits;
- cached final hidden state;
- learned probe representation.

The exact native recurrent-state implementation provenance remains subject to
a later bounded implementation/extraction authority.

## 10. Preserved kinematic endpoints

Exactly three local endpoints are allowed:

POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

For a fixed layer l and NAME anchor a:

v_t^(l) =
s_t^(l) - s_(t-1)^(l)

speed:

nu_t^(l) =
||v_t^(l)||_2

turning:

kappa_t^(l) =
1 - cos(v_t^(l), v_(t-1)^(l))

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

No new kinematic endpoint may be added after inspecting Q1/Q3 outcomes.

NEW_ENDPOINT_AFTER_Q1_Q3_OUTCOME_INSPECTION =
PROHIBITED

## 11. Pair-level secondary estimand

For:

l in {5,17}

and:

K in {
POST4_SPEED,
POST4_TURNING,
POST4_PATH_EFFICIENCY
}

define for source pair p:

DELTA_NAME_K_L(p) =
K(p, C2_NAME, A_NAME, layer=l)
-
K(p, C0_SHAM, A_NAME, layer=l)

The source pair remains the inferential unit.

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

TARGET_N_PER_HYPOTHESIS =
300

No row-level pseudo-replication is permitted.

## 12. Exact secondary hypothesis family

The scientific family contains exactly:

2 secondary layers
x
3 frozen endpoints
x
1 structural estimand

therefore:

SECONDARY_HYPOTHESIS_COUNT =
6

The six hypotheses are exactly:

H1 =
LAYER_5 / POST4_SPEED / DELTA_NAME

H2 =
LAYER_5 / POST4_TURNING / DELTA_NAME

H3 =
LAYER_5 / POST4_PATH_EFFICIENCY / DELTA_NAME

H4 =
LAYER_17 / POST4_SPEED / DELTA_NAME

H5 =
LAYER_17 / POST4_TURNING / DELTA_NAME

H6 =
LAYER_17 / POST4_PATH_EFFICIENCY / DELTA_NAME

No midpoint-layer hypothesis is added to this secondary family.

The frozen midpoint Phase F results remain historical primary evidence and are
not recomputed.

## 13. Statistical boundary

DEFAULT_DIRECTIONALITY =
TWO_SIDED

No sign is inferred from the negative R6 behavioral NAME effect.

A later statistical-analysis specification must freeze before Q1/Q3 outcome
inspection:

- exact test statistic;
- uncertainty interval;
- effect-size definition;
- numeric dtype;
- zero-transition behavior;
- zero-path behavior;
- deterministic hypothesis ordering;
- deterministic serialization;
- support/non-support decision labels;
- exact multiplicity procedure.

GLOBAL_SECONDARY_MULTIPLICITY_MUST_COVER_ALL_6 =
YES

Separate correction by layer is not sufficient unless a later pre-outcome
authority explicitly provides a justified alternative.

The later statistical specification must not alter the six scientific
hypotheses.

## 14. Relationship to Phase F

PHASE_F_PRIMARY_FAMILY =
CLOSED

PHASE_F_PRIMARY_HYPOTHESIS_COUNT =
15

PHASE_F_RESULTS_RECOMPUTED =
NO

PHASE_F_DECISIONS_REPLACED =
NO

PHASE_F_NAME_NON_SUPPORT_RECLASSIFIED =
NO

A future Q1/Q3 result cannot convert the layer-11 Phase F NAME hypotheses from
non-supported to supported.

Primary and secondary evidence must remain separately labeled.

## 15. Existing Phase D state tensor boundary

The frozen Phase D support-state tensor represents the primary-layer
measurement lineage.

It is not valid Q1/Q3 state evidence merely because it is already available.

PHASE_D_PRIMARY_LAYER_STATE_TENSOR_CAN_ESTABLISH_LAYER_5_RESULT =
NO

PHASE_D_PRIMARY_LAYER_STATE_TENSOR_CAN_ESTABLISH_LAYER_17_RESULT =
NO

Any scientific Q1/Q3 analysis requires separately validated native-state
evidence for the exact frozen secondary layers.

This document does not authorize producing that evidence.

## 16. Family-level decision semantics

If no member of the future six-hypothesis family is supported under its frozen
multiplicity rule:

SECONDARY_LAYER_NAME_LOCALIZATION =
NOT_ESTABLISHED

This means only that no prespecified local NAME kinematic correlate is
established at layers 5 or 17 under the tested endpoints and decision rule.

It does not establish:

NAME_NATIVE_STATE_EFFECT_EQUALS_ZERO

NAME_NATIVE_STATE_EFFECT_ABSENT_AT_ALL_LAYERS

If at least one member of the future six-hypothesis family is supported:

SECONDARY_LAYER_NAME_LOCALIZATION =
SUPPORTED

This label means only:

AT_LEAST_ONE_PRESPECIFIED_SECONDARY_ARCHITECTURE_DEFINED_LAYER_HAS_A_SUPPORTED_LOCAL_NAME_KINEMATIC_CORRELATE

The exact supported layer/endpoint combinations must be reported.

## 17. Cross-layer inference boundary

No hypothesis in this specification is a direct contrast between layers.

CROSS_LAYER_DIFFERENCE_ESTIMAND =
NOT_DEFINED

CROSS_LAYER_DIFFERENCE_TEST =
NOT_AUTHORIZED

MIDPOINT_VERSUS_Q1_TEST =
NOT_AUTHORIZED

MIDPOINT_VERSUS_Q3_TEST =
NOT_AUTHORIZED

Q1_VERSUS_Q3_TEST =
NOT_AUTHORIZED

A pattern of:

secondary-layer support
+
midpoint-layer non-support

does not establish a statistically significant between-layer difference.

DEPTH_SELECTIVITY =
NOT_ESTABLISHED_BY_THIS_FAMILY

The correction authority remains:

2e076cbd8e9633c3ab7abb222a05e409366539a7

## 18. Bounded interpretation

A supported member of the secondary family may establish only:

PRESPECIFIED_SECONDARY_LAYER_LOCAL_NAME_KINEMATIC_CORRELATE

Given the already frozen R6 behavioral NAME effect, it may be stated that the
behavioral effect has a prespecified local native-state kinematic correlate at
the exact supported secondary layer/endpoint.

It may not establish:

- causal mediation;
- necessity;
- sufficiency;
- state-to-output causation;
- significant depth selectivity;
- arbitrary-model generalization;
- arbitrary-dataset generalization.

OUTPUT_SIGN_MATCHING =
NOT_REQUIRED

## 19. Explicitly prohibited rescue analyses

Do not:

- scan layers other than 5 and 17;
- add midpoint layer 11 to rescue the secondary family;
- choose the stronger of Q1 and Q3 without multiplicity control;
- scan token offsets;
- extend the post4 window;
- shorten the post4 window;
- invent new trajectory features after seeing outcomes;
- train a probe;
- fit a nonlinear detector;
- redefine Phase F;
- pool layer 5 and layer 17 as if they were replicate observations;
- treat layer observations as independent inferential units;
- infer direction from R6 output sign.

## 20. Required future phase ordering

Before scientific Q1/Q3 execution, separate authority must establish:

1. Q1/Q3 measurement/extraction feasibility and exact runtime provenance.
2. Q1/Q3 measurement/extraction implementation and synthetic validation.
3. Q1/Q3 extraction execution authority.
4. Frozen Q1/Q3 native-state measurement artifacts.
5. Secondary six-family statistical-analysis specification.
6. Statistical implementation/validation authority.
7. One bounded canonical statistical execution.
8. Result validation and freeze.

This document alone authorizes none of those execution steps.

## 21. Current authority boundary

SCIENTIFIC_QUESTION =
FROZEN

SECONDARY_LAYERS =
FROZEN_5_AND_17

STRUCTURAL_ESTIMAND =
FROZEN_DELTA_NAME_ONLY

KINEMATIC_ENDPOINTS =
FROZEN_THREE

SECONDARY_FAMILY =
FROZEN_SIX_HYPOTHESES

MODEL_EXECUTION =
NOT_AUTHORIZED

NATIVE_STATE_EXTRACTION =
NOT_AUTHORIZED

STATISTICAL_EXECUTION =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

KAGGLE =
NOT_AUTHORIZED

GPU =
NOT_AUTHORIZED

SCIENTIFIC_CONCLUSION =
NONE_NEW

NEXT_PHASE =
NAME_Q1_Q3_MEASUREMENT_EXTRACTION_FEASIBILITY_AND_PROVENANCE
