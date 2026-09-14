# ContraMamba Gen4 Native Mamba State Bridge
# NAME Direct Cross-Layer Paired-Difference Implementation Validation Freeze
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_DIRECT_CROSS_LAYER_PAIRED_DIFFERENCE_IMPLEMENTATION_VALIDATION_FREEZE

PHASE =
VALIDATED_IMPLEMENTATION_FREEZE


## 1. Frozen lineage

DIRECT_CROSS_LAYER_SCIENTIFIC_SPECIFICATION =
b3e0ade126622f244b557e1db07296c622bd7202

DIRECT_CROSS_LAYER_IMPLEMENTATION_AUTHORITY =
724c28b528b0f182bc0c79cf5ee0b3adfca76ec2

DIRECT_CROSS_LAYER_IMPLEMENTATION =
cdf9f2117bc0d5fa119fbf24b85266c1f9ced448


## 2. Exact implementation scope

IMPLEMENTATION_PATH =
scripts/reason_router_gen4_native_mamba_state_name_direct_cross_layer_statistical_analysis.py

TEST_PATH =
tests/test_reason_router_gen4_native_mamba_state_name_direct_cross_layer_statistical_analysis.py

IMPLEMENTATION_COMMIT_FILE_COUNT =
2

IMPLEMENTATION_COMMIT_SCOPE =
PASS_EXACT_TWO_AUTHORIZED_NEW_FILES

EXISTING_Q1_Q3_IMPLEMENTATION_MODIFIED =
NO

PHASE_F_IMPLEMENTATION_MODIFIED =
NO

SCIENTIFIC_SPECIFICATION_MODIFIED =
NO


## 3. Exact implementation identities

IMPLEMENTATION_SHA256 =
bccf4b91f8b566808315ce15fafa5c948a8b20bde751c75c8af0da9671c83fff

IMPLEMENTATION_GIT_BLOB_SHA1 =
03ed38e5fa1df404c9ac2bfe01c770c232022dc2

TEST_SHA256 =
0324fc613b24ced217487cc7e91a829843c55dc96e86241c607774013d89aedb

TEST_GIT_BLOB_SHA1 =
bef28db05a00ba4570dde0d9ee8c527ef9423120

REMOTE_BLOB_IDENTITY_VERIFICATION =
PASS


## 4. Frozen implemented family

STRUCTURAL_ESTIMAND =
DELTA_NAME

MIDPOINT_LAYER =
11

SECONDARY_LAYERS =
5
17

LAYER_CONTRASTS =
5_MINUS_11
17_MINUS_11

ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

HYPOTHESIS_COUNT =
6

SOURCE_PAIR_COUNT =
300

PAIRWISE_DIFFERENCE_DIRECTION =
SECONDARY_LAYER_MINUS_MIDPOINT_LAYER

LAYER_17_MINUS_5_INCLUDED =
NO


## 5. Statistical implementation semantics

TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T_ON_PAIRED_CROSS_LAYER_DIFFERENCES

N_PER_HYPOTHESIS =
300

DF_PER_HYPOTHESIS =
299

CONFIDENCE_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T

STANDARDIZED_EFFECT =
D_Z

D_Z_DEFINITION =
MEAN_PAIRWISE_DIFFERENCE_DIVIDED_BY_SAMPLE_SD_OF_PAIRWISE_DIFFERENCES

ZERO_VARIANCE =
HARD_BLOCK

MULTIPLICITY =
ONE_GLOBAL_HOLM_BONFERRONI_FAMILY_OF_6

FAMILYWISE_ALPHA =
0.05

NUMERIC_ANALYSIS_DTYPE =
FLOAT64


## 6. Provenance implementation semantics

PHASE_F_CANONICAL_LOGICAL_PATH =
reports/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis_e917e4c_v1/phase_f_pair_level_contrasts.csv

PHASE_F_CANONICAL_SHA256 =
abb7e837395138d095e285e40ec8782ea863061087b58844bec6fad5d4ef5e73

PHASE_F_CANONICAL_BYTES =
542251

Q1_Q3_CANONICAL_LOGICAL_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_retry1_v1/name_q1_q3_pair_level_contrasts.csv

Q1_Q3_CANONICAL_SHA256 =
c0c917560b5a37c5df82ad4e203a87441370c129622afc668b77712b98698f82

Q1_Q3_CANONICAL_BYTES =
196542

PROVENANCE_PATH_SERIALIZATION =
FROZEN_LOGICAL_POSIX_PATH_CONSTANTS

HOST_NATIVE_PATH_LEAKAGE =
SYNTHETICALLY_BLOCKED


## 7. Output contract

PAIR_LEVEL_OUTPUT =
name_direct_cross_layer_pair_level_differences.csv

CONFIRMATORY_OUTPUT =
name_direct_cross_layer_confirmatory_results.csv

MANIFEST_OUTPUT =
name_direct_cross_layer_statistical_analysis_manifest.json

REPORT_OUTPUT =
name_direct_cross_layer_statistical_analysis_report_candidate.md

OUTPUT_FILE_COUNT =
4

ATOMIC_STAGING_PUBLICATION =
IMPLEMENTED

OUTPUT_COLLISION_BLOCKER =
IMPLEMENTED

STAGING_COLLISION_BLOCKER =
IMPLEMENTED


## 8. Synthetic validation

PY_COMPILE =
PASS

DEDICATED_SYNTHETIC_TEST_COUNT =
34

DEDICATED_SYNTHETIC_TEST_RESULT =
34_OF_34_PASS

SYNTHETIC_VALIDATION_COVERAGE_INCLUDES =

- 5_MINUS_11 subtraction direction;
- 17_MINUS_11 subtraction direction;
- deterministic layer-contrast ordering;
- deterministic endpoint ordering;
- lexicographic source-pair ordering;
- exact pair-level schema;
- exact confirmatory schema;
- source-pair mismatch blocker;
- duplicate-key blockers;
- missing-cell blocker;
- unexpected-layer blocker;
- unexpected-endpoint blocker;
- non-DELTA_NAME blocker;
- non-finite value blockers;
- zero-variance blocker;
- Student-t numerical semantics;
- d_z semantics;
- exact global Holm-6 semantics;
- deterministic CSV serialization;
- deterministic canonical JSON;
- deterministic report serialization;
- frozen POSIX provenance paths;
- host-native provenance path leakage regression;
- peer-artifact SHA bindings;
- output collision blocker;
- staging collision blocker;
- canonical artifact access guard.


## 9. Canonical-data boundary during validation

CANONICAL_PHASE_F_PAIR_ARTIFACT_OPENED_DURING_SYNTHETIC_VALIDATION =
NO

CANONICAL_Q1_Q3_PAIR_ARTIFACT_OPENED_DURING_SYNTHETIC_VALIDATION =
NO

CANONICAL_CROSS_LAYER_NUMERIC_DIFFERENCE_COMPUTED =
NO

CANONICAL_STATISTICAL_TESTING_PERFORMED =
NO

MODEL_FORWARD =
NO

NATIVE_STATE_EXTRACTION =
NO

TRAINING =
NO

GPU =
NO

KAGGLE =
NO


## 10. Validation status

CODE_CORRECTNESS =
PASS

IMPLEMENTATION_SCOPE_VALIDITY =
PASS

IMPLEMENTATION_PROVENANCE_VALIDITY =
PASS

SYNTHETIC_VALIDATION =
PASS

CANONICAL_EXECUTION_SUCCESS =
NOT_YET_TESTED

CANONICAL_RESULT_ARTIFACT_VALIDITY =
NOT_YET_TESTED

SCIENTIFIC_RESULT =
NOT_YET_AVAILABLE


## 11. Interpretation boundary

This implementation freeze establishes only that the bounded direct
cross-layer statistical analysis code satisfies the frozen implementation
contract under synthetic validation.

It does not establish:

- any layer-5 versus layer-11 difference;
- any layer-17 versus layer-11 difference;
- any supported cross-layer endpoint;
- broad depth selectivity;
- causal mediation;
- native-state causation;
- necessity;
- sufficiency.


## 12. Execution boundary

THIS_FREEZE_AUTHORIZES_CANONICAL_EXECUTION =
NO

THIS_FREEZE_AUTHORIZES_CANONICAL_INPUT_OPEN =
NO

A separate single-use canonical execution authority is required before either
frozen pair-level input may be opened for cross-layer numeric analysis.


## 13. Implementation status

DIRECT_CROSS_LAYER_IMPLEMENTATION_STATUS =
CLOSED_VALIDATED

NEXT_ACTION =
CREATE_SINGLE_USE_DIRECT_CROSS_LAYER_CANONICAL_EXECUTION_AUTHORITY

CANONICAL_EXECUTION =
NOT_AUTHORIZED
