# ContraMamba Gen4 Native Mamba State Bridge
# NAME Direct Cross-Layer Paired-Difference Scientific Specification
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_DIRECT_CROSS_LAYER_PAIRED_DIFFERENCE_SCIENTIFIC_SPECIFICATION

PHASE =
ADAPTIVE_DIRECT_CROSS_LAYER_INFERENCE_SPECIFICATION

THIS_DOCUMENT_CREATES_NEW_SCIENTIFIC_EVIDENCE =
NO

THIS_DOCUMENT_AUTHORIZES_IMPLEMENTATION =
NO

THIS_DOCUMENT_AUTHORIZES_STATISTICAL_EXECUTION =
NO

THIS_DOCUMENT_AUTHORIZES_MODEL_EXECUTION =
NO

THIS_DOCUMENT_AUTHORIZES_NATIVE_STATE_EXTRACTION =
NO

THIS_DOCUMENT_AUTHORIZES_TRAINING =
NO

THIS_DOCUMENT_AUTHORIZES_KAGGLE =
NO

THIS_DOCUMENT_AUTHORIZES_GPU =
NO


## 1. Scientific motivation

The validated NAME evidence currently establishes:

1. a frozen negative behavioral DELTA_NAME effect;

2. no established prespecified midpoint-layer NAME local kinematic correlate
   at layer 11;

3. no established prespecified secondary-layer NAME local kinematic correlate
   at layer 5;

4. a supported prespecified NAME POST4_PATH_EFFICIENCY local kinematic
   correlate at layer 17.

The frozen static interpretation correctly states that:

SUPPORTED_AT_LAYER_17

plus

NOT_ESTABLISHED_AT_LAYER_5_OR_LAYER_11

does not itself establish a between-layer difference.

The next falsifiable question is therefore a direct paired cross-layer
comparison.


## 2. Frozen evidence lineage

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

PHASE_F_VALIDATED_STATISTICAL_RESULT_FREEZE =
ab3428e7be08af26fa1fdafd1483a34e48fbcf8c

POST_PHASE_F_DEPTH_SELECTIVITY_CORRECTION =
2e076cbd8e9633c3ab7abb222a05e409366539a7

Q1_Q3_SCIENTIFIC_SPECIFICATION =
01801ad1617b2ebc3ffa859ba440636d4755a55c

Q1_Q3_SECONDARY_SIX_STATISTICAL_SPECIFICATION =
2030de52922c251caeafceb36dd5f2f57b5b39c7

Q1_Q3_VALIDATED_RETRY1_RESULT_FREEZE =
fe9066628d92b2812012463969749111471035cb

NAME_CROSS_PHASE_STATIC_INTERPRETATION_AUTHORITY =
47eb461f8143fc7d1bf72955946661d4b0bc3b48

NAME_CROSS_PHASE_STATIC_INTERPRETATION =
d8bff70b2ea58c2aea716e3cea62715fa156db1d


## 3. Structural feasibility evidence

A read-only structural feasibility check was completed before any direct
cross-layer numeric difference was computed.

FROZEN_INPUT_BLOB_IDENTITY =
PASS

SCHEMA_AND_CARDINALITY =
PASS

ESTIMAND_ENDPOINT_SEMANTICS =
PASS

EXACT_300_SOURCE_PAIR_ALIGNMENT =
PASS

STRUCTURAL_KEY_UNIQUENESS =
PASS

CROSS_LAYER_PAIRED_FEASIBILITY =
PASS

CROSS_LAYER_NUMERIC_DIFFERENCES_COMPUTED_BEFORE_THIS_SPEC =
NO

STATISTICAL_TESTING_PERFORMED_BEFORE_THIS_SPEC =
NO


## 4. Frozen pair-level inputs

### 4.1 Midpoint layer 11

PHASE_F_PAIR_LEVEL_PATH =
reports/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis_e917e4c_v1/phase_f_pair_level_contrasts.csv

PHASE_F_PAIR_LEVEL_SHA256 =
abb7e837395138d095e285e40ec8782ea863061087b58844bec6fad5d4ef5e73

PHASE_F_PAIR_LEVEL_BYTES =
542251

PHASE_F_TOTAL_ROWS =
4500

PHASE_F_DELTA_NAME_ROWS =
900

MIDPOINT_LAYER_INDEX =
11


### 4.2 Secondary layers 5 and 17

Q1_Q3_PAIR_LEVEL_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_retry1_v1/name_q1_q3_pair_level_contrasts.csv

Q1_Q3_PAIR_LEVEL_SHA256 =
c0c917560b5a37c5df82ad4e203a87441370c129622afc668b77712b98698f82

Q1_Q3_PAIR_LEVEL_BYTES =
196542

Q1_Q3_TOTAL_ROWS =
1800

SECONDARY_LAYER_INDICES =
5
17


## 5. Common inferential population

SOURCE_PAIR_COUNT =
300

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

SOURCE_PAIR_ALIGNMENT =
EXACT

SOURCE_PAIR_ORDER =
LEXICOGRAPHIC

The same 300 source_pair_id values must be present for every endpoint at:

layer 5
layer 11
layer 17

Any mismatch is a hard blocker.


## 6. Frozen structural estimand

STRUCTURAL_ESTIMAND =
DELTA_NAME

For each source pair i, layer L, and endpoint e, define:

X_i_L_e =
the already frozen DELTA_NAME pair-level contrast for that layer and endpoint

No original condition-level endpoint is recomputed.

No state tensor is reopened.

Only already frozen pair-level DELTA_NAME contrasts are used.


## 7. Prespecified direct cross-layer estimands

Exactly two layer contrasts are included.

### Contrast A

LAYER_CONTRAST_A =
5_MINUS_11

For source pair i and endpoint e:

D_i_5_MINUS_11_e =
X_i_5_e - X_i_11_e


### Contrast B

LAYER_CONTRAST_B =
17_MINUS_11

For source pair i and endpoint e:

D_i_17_MINUS_11_e =
X_i_17_e - X_i_11_e


## 8. Explicitly excluded layer contrast

LAYER_17_MINUS_5 =
NOT_INCLUDED

Reason:

The scientific question motivating this phase is whether either prespecified
secondary architecture-defined location differs directly from the original
Phase F midpoint readout.

Adding 17_MINUS_5 after inspecting the Q1/Q3 outcomes would expand the
adaptive family unnecessarily.

A future 17-versus-5 comparison would require separate authority.


## 9. Prespecified endpoints

ENDPOINT_ORDER =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

No endpoint may be added or removed after this specification is frozen.


## 10. Exact hypothesis family

The direct cross-layer family contains exactly six hypotheses.

H1 =
5_MINUS_11 / POST4_SPEED

H2 =
5_MINUS_11 / POST4_TURNING

H3 =
5_MINUS_11 / POST4_PATH_EFFICIENCY

H4 =
17_MINUS_11 / POST4_SPEED

H5 =
17_MINUS_11 / POST4_TURNING

H6 =
17_MINUS_11 / POST4_PATH_EFFICIENCY

HYPOTHESIS_COUNT =
6

DETERMINISTIC_HYPOTHESIS_ORDER =
LAYER_CONTRAST_THEN_ENDPOINT


## 11. Null hypotheses

For each layer contrast c and endpoint e:

H0_c_e =
MEAN_PAIRWISE_CROSS_LAYER_DIFFERENCE_EQUALS_ZERO

That is:

H0 =
E[D_i_c_e] = 0

ALTERNATIVE =
E[D_i_c_e] != 0

TEST_SIDEDNESS =
TWO_SIDED


## 12. Statistical procedure

For each of the six hypotheses:

N =
300

DF =
299

NUMERIC_ANALYSIS_DTYPE =
FLOAT64

TEST =
ONE_SAMPLE_STUDENT_T_ON_PAIRED_CROSS_LAYER_DIFFERENCES

CONFIDENCE_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T

STANDARDIZED_EFFECT =
D_Z_OF_PAIRED_CROSS_LAYER_DIFFERENCES

where:

D_Z =
mean(D) / sample_sd(D)

ZERO_VARIANCE_PAIRWISE_DIFFERENCE =
HARD_BLOCK


## 13. Multiplicity

MULTIPLICITY_FAMILY =
ALL_6_DIRECT_CROSS_LAYER_HYPOTHESES

METHOD =
HOLM_BONFERRONI

FAMILYWISE_ALPHA =
0.05

Separate correction by layer contrast is forbidden.

Separate correction by endpoint is forbidden.

Selection of only the observed layer-17 path-efficiency result for
confirmatory testing is forbidden.


## 14. Adaptive-program boundary

This direct cross-layer phase is adaptive.

It is motivated by results observed in:

Phase F

and

the Q1/Q3 secondary follow-up.

Therefore:

OVERALL_ADAPTIVE_PROGRAM_FWER_ACROSS_ALL_PRIOR_AND_CURRENT_PHASES =
NOT_CLAIMED

The Holm correction in this phase controls only the new frozen six-member
direct cross-layer family.

No retrospective global multiplicity claim spanning Phase F, Q1/Q3, and this
phase is permitted.


## 15. Result interpretation

For an individual hypothesis:

HOLM_ADJUSTED_P < 0.05

means:

PRESPECIFIED_DIRECT_CROSS_LAYER_DIFFERENCE_SUPPORTED

Otherwise:

PRESPECIFIED_DIRECT_CROSS_LAYER_DIFFERENCE_NOT_ESTABLISHED

A non-supported result does not establish exact equality between layers.


## 16. Direction interpretation

For:

5_MINUS_11

a positive mean means the DELTA_NAME kinematic contrast is numerically larger
at layer 5 than layer 11 for the same endpoint.

A negative mean means it is numerically smaller at layer 5 than layer 11.


For:

17_MINUS_11

a positive mean means the DELTA_NAME kinematic contrast is numerically larger
at layer 17 than layer 11 for the same endpoint.

A negative mean means it is numerically smaller at layer 17 than layer 11.

These directions describe the frozen state-space estimand only.

They do not inherit the sign of the behavioral DELTA_NAME effect.


## 17. Permitted positive claim

If a prespecified direct cross-layer hypothesis survives the global six-member
Holm correction, the strongest permitted claim for that exact hypothesis is:

PRESPECIFIED_DIRECT_BETWEEN_LAYER_DIFFERENCE_ESTABLISHED_FOR_THE_SPECIFIED_NAME_KINEMATIC_ENDPOINT

For example, support for:

17_MINUS_11 / POST4_PATH_EFFICIENCY

would establish a direct layer-17 versus layer-11 difference for the frozen
DELTA_NAME POST4_PATH_EFFICIENCY response.


## 18. Depth-selectivity terminology

A supported exact between-layer contrast may be reported as an established
difference between those two prespecified layers for that endpoint.

However, the broad unqualified claim:

NAME_IS_DEPTH_SELECTIVE

is prohibited.

Reason:

three sampled architecture-defined locations do not characterize the full
depth trajectory.

Permitted terminology must remain endpoint- and layer-pair-specific.


## 19. Claims not established by this phase

Regardless of outcome, this phase cannot establish:

- a complete depth trajectory;
- monotonic depth evolution;
- non-monotonic depth evolution;
- a globally optimal layer;
- best-layer localization;
- responses at unmeasured layers;
- causal mediation;
- native-state causation;
- necessity;
- sufficiency;
- complete mechanism localization;
- behavioral-sign/state-sign equivalence;
- arbitrary-model generalization;
- arbitrary-dataset generalization.


## 20. No outcome-driven expansion

After this specification is frozen, do not:

- add layer 17_MINUS_5;
- add arbitrary layers;
- scan all layers;
- select only POST4_PATH_EFFICIENCY;
- drop SPEED or TURNING;
- add new state endpoints;
- change to one-sided testing;
- change multiplicity families;
- redefine DELTA_NAME;
- alter source-pair membership;
- inspect cross-layer differences to redesign the family.


## 21. Required implementation semantics

A future implementation must:

1. read only the two frozen pair-level CSV artifacts;

2. validate exact bytes and SHA256 identities;

3. select only DELTA_NAME rows from the Phase F artifact;

4. treat Phase F rows as layer 11;

5. require exact source_pair_id alignment across layers 5, 11, and 17;

6. require exact endpoint alignment;

7. compute pairwise layer differences only after all structural validation
   passes;

8. use FLOAT64 for statistical analysis;

9. produce deterministic row and hypothesis ordering;

10. implement exactly one global Holm correction over six hypotheses;

11. perform no model execution or native-state extraction.


## 22. Required synthetic validation

Before any canonical statistical execution, synthetic tests must establish:

- correct 5_MINUS_11 subtraction direction;
- correct 17_MINUS_11 subtraction direction;
- rejection of source-pair mismatch;
- rejection of endpoint mismatch;
- rejection of duplicate structural keys;
- rejection of missing cells;
- rejection of unexpected layers;
- rejection of estimands other than DELTA_NAME;
- zero-variance blocker behavior;
- exact six-hypothesis ordering;
- exact global Holm-6 behavior;
- deterministic serialization;
- no canonical artifact access in synthetic tests.


## 23. Canonical execution boundary

CANONICAL_STATISTICAL_EXECUTION =
NOT_AUTHORIZED

CANONICAL_INPUT_OPEN_FOR_CROSS_LAYER_NUMERIC_ANALYSIS =
NOT_AUTHORIZED

IMPLEMENTATION =
NOT_AUTHORIZED

KAGGLE =
NOT_AUTHORIZED

GPU =
NOT_AUTHORIZED

MODEL_FORWARD =
NOT_AUTHORIZED

NATIVE_STATE_EXTRACTION =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED


## 24. Required future phase ordering

The required order is:

1. freeze this scientific specification;

2. create bounded implementation authority;

3. implement cross-layer paired-difference analysis;

4. complete synthetic validation;

5. freeze validated implementation identity;

6. create one single-use canonical execution authority;

7. execute exactly one bounded local CPU canonical analysis;

8. independently validate result artifacts;

9. freeze the validated direct cross-layer result;

10. interpret only the frozen direct cross-layer evidence.


## 25. Scientific decision rule

If none of the six hypotheses survives global Holm:

DIRECT_CROSS_LAYER_NAME_DIFFERENCE =
NOT_ESTABLISHED_WITHIN_THE_PRESPECIFIED_FAMILY

This would mean the observed significance/non-significance pattern across
layers is not converted into a supported direct between-layer difference by
this test.


If at least one hypothesis survives global Holm:

DIRECT_CROSS_LAYER_NAME_DIFFERENCE =
SUPPORTED_FOR_AT_LEAST_ONE_PRESPECIFIED_LAYER_PAIR_ENDPOINT

Only the exact supported layer-pair/endpoint combinations may receive positive
between-layer claims.


## 26. Next transition

NEXT_ACTION =
BOUNDED_DIRECT_CROSS_LAYER_IMPLEMENTATION_AUTHORITY

SCIENTIFIC_EXECUTION =
NONE

CROSS_LAYER_NUMERIC_OUTCOME_INSPECTION_BEFORE_IMPLEMENTATION_FREEZE =
FORBIDDEN
