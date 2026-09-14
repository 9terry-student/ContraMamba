# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Statistical Analysis Provenance-Path Serialization Recovery Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_STATISTICAL_ANALYSIS_PROVENANCE_PATH_SERIALIZATION_RECOVERY_AUTHORITY

THIS_DOCUMENT_AUTHORIZES =
BOUNDED_IMPLEMENTATION_AND_SYNTHETIC_VALIDATION_CORRECTION_ONLY

CANONICAL_RETRY_EXECUTION =
NOT_AUTHORIZED

SCIENTIFIC_CONCLUSION =
NONE


## 1. Consumed execution

STATISTICAL_EXECUTION_AUTHORITY =
1fbb393bb6f2ab7194cc8d33010a8b137d6bab42

CORRECTED_IMPLEMENTATION_COMMIT =
cc0f64104828988faa99fec1356b84014c66317a

CORRECTED_IMPLEMENTATION_VALIDATION_FREEZE =
90c45b3d4905290451e4f353ea45674c279b6232

PROCESS_EXECUTION_RESULT =
PASS

PROCESS_EXIT_CODE =
0

EXECUTION_AUTHORITY_CONSUMED =
YES

SAME_AUTHORITY_RERUN =
FORBIDDEN


## 2. Failed result-artifact validation

The canonical process produced exactly four result artifacts.

Their frozen failed-execution identities are:

name_q1_q3_pair_level_contrasts.csv
c0c917560b5a37c5df82ad4e203a87441370c129622afc668b77712b98698f82

name_q1_q3_secondary_confirmatory_results.csv
88da6fa8627695d698a83c659b5a30e4fc8d853173f5ce3dbf65853ec606e8b5

name_q1_q3_statistical_analysis_manifest.json
38d7438d7b18653a4268fd4d25e9eb701b5689d93221c8b822de88d9fab386ab

name_q1_q3_statistical_analysis_report_candidate.md
dfc50bf822a227595095f683489abc75699e3f2d0a3dda396c7635b337ef5ab4

FAILED_EXECUTION_OUTPUT_DIRECTORY =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_v1

FAILED_EXECUTION_OUTPUT_DIRECTORY_MUST_BE_PRESERVED =
YES

DELETE_FAILED_OUTPUT =
FORBIDDEN

EDIT_FAILED_OUTPUT =
FORBIDDEN

RENAME_FAILED_OUTPUT =
FORBIDDEN


## 3. Validation evidence before failure

ARTIFACT_BYTE_IDENTITY =
PASS

PAIR_LEVEL_RESULT_ROWS =
1800 PASS

SECONDARY_RESULT_ROWS =
6 PASS

PAIR_ORDER_AND_POPULATION =
PASS

SECONDARY_NUMERIC_RECOMPUTATION =
PASS

GLOBAL_SIX_MEMBER_HOLM_RECOMPUTATION =
PASS

RESULT_ARTIFACT_PROVENANCE =
FAIL

SCIENTIFIC_INTERPRETATION =
BLOCKED


## 4. Exact defect

The frozen canonical input identity is:

reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_scientific_extraction_v1/kinematic_endpoints.jsonl

The generated manifest instead serialized:

reports\reason_router_gen4_native_mamba_state_bridge_name_q1_q3_scientific_extraction_v1\kinematic_endpoints.jsonl

The implementation validates the input into a pathlib.Path and later passes:

str(input_path)

to deterministic result serialization.

On Windows, that produces platform-native backslash separators.

The same serialized input_path value is supplied to both:

- the statistical-analysis manifest;
- the statistical-analysis report.

DEFECT_CLASS =
PLATFORM_DEPENDENT_PROVENANCE_PATH_SERIALIZATION

STATISTICAL_FORMULA_DEFECT =
NO

PAIR_CONTRAST_DEFECT =
NO

STUDENT_T_DEFECT =
NO

HOLM_DEFECT =
NO

NUMERIC_RESULT_DEFECT_ESTABLISHED =
NO


## 5. Exact recovery implementation scope

Exactly these two tracked files may be modified:

scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

tests/test_reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

No other tracked implementation, test, report, extraction artifact, or
scientific result file may be modified.

The correction must ensure canonical provenance serialization uses exactly:

CANONICAL_INPUT_PATH

with POSIX forward-slash separators, independent of host operating system.

The canonical execution path may still be validated and opened through
pathlib.Path.

Only its serialized provenance representation is corrected.

A permitted minimal behavior is for run_canonical() to pass the frozen
CANONICAL_INPUT_PATH constant into result serialization rather than
str(input_path).


## 6. Required regression validation

Synthetic validation must prove:

1. serialized manifest input_path equals CANONICAL_INPUT_PATH exactly;
2. serialized report input provenance equals CANONICAL_INPUT_PATH exactly;
3. no backslash appears in the canonical serialized input path;
4. behavior is independent of a platform-native Path string returned by the
   validated input-path layer;
5. canonical input artifact is not opened during synthetic regression tests;
6. the six statistical hypotheses are unchanged;
7. pair-level DELTA_NAME computation is unchanged;
8. Student-t procedure is unchanged;
9. 95-percent CI procedure is unchanged;
10. d_z is unchanged;
11. global six-member Holm procedure is unchanged;
12. output filenames and fixed field order are unchanged.


## 7. Preserved scientific contract

LAYERS =
5,17

ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

ESTIMAND =
DELTA_NAME

N_PER_HYPOTHESIS =
300

SECONDARY_HYPOTHESIS_COUNT =
6

NUMERIC_ANALYSIS_DTYPE =
FLOAT64

MULTIPLICITY =
GLOBAL_HOLM_BONFERRONI_ALL_6

FAMILYWISE_ALPHA =
0.05

CROSS_LAYER_DIFFERENCE_TEST =
NOT_AUTHORIZED

No scientific estimand, statistic, correction family, or decision rule may
change under this recovery authority.


## 8. Execution boundary

CANONICAL_INPUT_REOPEN =
NOT_AUTHORIZED

CANONICAL_STATISTICAL_REEXECUTION =
NOT_AUTHORIZED

FAILED_RESULT_ARTIFACT_REPAIR =
NOT_AUTHORIZED

FAILED_RESULT_ARTIFACT_MUTATION =
NOT_AUTHORIZED

KAGGLE =
NOT_AUTHORIZED

GPU =
NOT_AUTHORIZED

MODEL_FORWARD =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED


## 9. Required implementation validation

After the bounded correction:

python -m py_compile scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

python -m pytest -q tests/test_reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

git diff --check -- scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py tests/test_reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

The failed canonical result directory must remain byte-identical throughout
implementation and synthetic validation.


## 10. Promotion boundary

After corrected implementation validation:

- freeze the corrected implementation under a new commit;
- freeze implementation validation;
- issue a new retry execution authority;
- use a new output directory so the failed execution remains preserved.

The consumed execution authority:

1fbb393bb6f2ab7194cc8d33010a8b137d6bab42

must never be reused.

NEXT_PHASE =
PROVENANCE_PATH_SERIALIZATION_CORRECTION_IMPLEMENTATION

SCIENTIFIC_CONCLUSION =
NONE
