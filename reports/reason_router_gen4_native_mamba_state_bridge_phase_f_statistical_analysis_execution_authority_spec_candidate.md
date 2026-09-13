# ContraMamba Gen4 Native Mamba State Bridge
# Phase F Statistical Analysis Execution Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_F_STATISTICAL_ANALYSIS_EXECUTION_AUTHORITY

THIS_DOCUMENT_AUTHORIZES =
ONE_CANONICAL_CPU_STATISTICAL_EXECUTION

SCIENTIFIC_EXECUTION_ALLOWED =
YES_BOUNDED

## 1. Frozen authorities

PHASE_F_STATISTICAL_SPECIFICATION_COMMIT =
830f7ea697ce24388dddd181c9aa301ec2b442fc

PHASE_F_IMPLEMENTATION_AUTHORITY_COMMIT =
d249de57dc8dac69f49f7110c6f1a07532f939cb

PHASE_F_OUTPUT_HASH_CORRECTION_COMMIT =
c2fa746826bc397688b8f1ae8191bb96b5f1bf17

PHASE_F_IMPLEMENTATION_COMMIT =
e917e4c4fe0c94aa4ef5be33f2a69e6c64732189

MECHANISTIC_BRIDGE_SPECIFICATION =
a2617aa037d1a9834003535b62ac81770a5b96aa

PHASE_E_MEASUREMENT_ARTIFACT_FREEZE =
4a2ef821122a36c8e48ec261012519102b63d6c3

PHASE_E_RELEASE_BINDING_FREEZE =
1695430fabf4d3ebce28d2b1622c218e65b052c3

## 2. Frozen implementation identity

IMPLEMENTATION_PATH =
scripts/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis.py

IMPLEMENTATION_SHA256 =
19d90b0d5abe11ee4023c439c021a61554e42e0b7f45cbe92db341cf48d3ebcd

TEST_PATH =
tests/test_reason_router_gen4_native_mamba_state_phase_f_statistical_analysis.py

TEST_SHA256 =
9826d49643daa134a8d20e1ff0a5dbf11043351453bcf2821f51172a24b11495

SYNTHETIC_TEST_RESULT =
PASS_21_OF_21

PY_COMPILE =
PASS

GIT_DIFF_CHECK =
PASS

CANONICAL_STATISTICAL_EXECUTION_DURING_IMPLEMENTATION =
NO

## 3. Canonical input

CANONICAL_INPUT_PATH =
reports/reason_router_gen4_native_mamba_state_extraction_bff0a75_v1/kinematic_endpoints.jsonl

CANONICAL_INPUT_SHA256 =
7ff4d24b7895745efbf5a00e0361a4fe409ad585db17ece7151e4dedb289b07c

CANONICAL_INPUT_BYTES =
1862461

CANONICAL_INPUT_ROWS =
3600

SOURCE_PAIR_COUNT =
300

ROWS_PER_SOURCE_PAIR =
12

PRIMARY_LAYER =
11

RAW_SUPPORT_STATES_REQUIRED =
NO

No native-state measurement may be recomputed during Phase F execution.

## 4. Authorized scientific computation

PRIMARY_ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

PRIMARY_ESTIMANDS =
DELTA_TITLE
DELTA_NAME
DELTA_ROLE
DELTA_PREDICATE
INTERACTION_TITLE_NAME

PRIMARY_HYPOTHESIS_COUNT =
15

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

N_PER_HYPOTHESIS =
300

STATISTICAL_TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T_TEST_ON_PAIR_LEVEL_CONTRASTS

DF =
299

CONFIDENCE_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T

EFFECT_SIZE =
D_Z

MULTIPLICITY_METHOD =
HOLM_BONFERRONI

MULTIPLICITY_FAMILY =
ALL_15_PRIMARY_HYPOTHESES_TOGETHER

FAMILYWISE_ALPHA =
0.05

STATE_LEVEL_TITLE_MINUS_NAME =
FORBIDDEN

No additional endpoint, estimand, subgroup, layer, token window, or
multiplicity family is authorized.

## 5. Runtime boundary

EXECUTION_COUNT =
1

EXECUTION_MODE =
LOCAL_CPU_STATIC_STATISTICAL_ANALYSIS

EXECUTION_WORKTREE_MODE =
CLEAN_DETACHED_EXACT_AUTHORITY_COMMIT

CPU_ONLY =
YES

GPU =
FORBIDDEN

KAGGLE =
NOT_REQUIRED

MODEL_FORWARD =
FORBIDDEN

CHECKPOINT_LOADING =
FORBIDDEN

TOKENIZER_EXECUTION =
FORBIDDEN

NATIVE_STATE_EXTRACTION =
FORBIDDEN

TRAINING =
FORBIDDEN

BACKWARD =
FORBIDDEN

OPTIMIZER =
FORBIDDEN

NEW_DEPENDENCY_INSTALLATION =
FORBIDDEN

## 6. Execution-authority commit binding

EXECUTION_AUTHORITY_COMMIT =
COMMIT_CONTAINING_THIS_DOCUMENT

The canonical execution must occur from a clean detached worktree at the exact
commit containing this authority document.

The runtime value passed as:

--execution-authority-commit

must equal:

git rev-parse HEAD

inside that detached execution worktree.

## 7. Mandatory pre-execution gates

Before canonical statistical computation, verify all of the following:

1. detached execution HEAD equals the execution-authority commit;
2. git status is clean;
3. implementation Git-blob SHA256 equals the frozen implementation SHA256;
4. canonical input worktree SHA256 equals the frozen input SHA256;
5. canonical input byte count equals 1862461;
6. output directory does not already exist;
7. Python implementation compiles;
8. no GPU, model, checkpoint, tokenizer, or native-state operation is involved.

Any failure blocks execution.

No hash mismatch may be repaired by changing the frozen scientific input.

## 8. Canonical output directory

CANONICAL_OUTPUT_DIRECTORY =
reports/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis_e917e4c_v1

OUTPUT_COLLISION =
BLOCK

The execution must produce exactly four files:

phase_f_pair_level_contrasts.csv
phase_f_primary_confirmatory_results.csv
phase_f_statistical_analysis_manifest.json
phase_f_statistical_analysis_report_candidate.md

PAIR_LEVEL_RESULT_ROWS =
4500

PRIMARY_RESULT_ROWS =
15

## 9. Output provenance contract

The manifest must bind:

- statistical specification commit;
- implementation authority commit;
- output-hash correction commit;
- this execution-authority commit;
- implementation commit;
- implementation script SHA256;
- canonical input path;
- canonical input SHA256;
- canonical input byte count;
- canonical input row count;
- source-pair count;
- hypothesis count;
- familywise alpha;
- multiplicity method;
- numeric analysis dtype;
- the exact SHA256 of the three peer output artifacts.

MANIFEST_SELF_SHA256 =
EXTERNALLY_FROZEN_AFTER_EXECUTION

No manifest self-hash field is permitted.

## 10. Fail-closed scientific blockers

Execution must fail rather than continue on:

- wrong canonical input hash;
- wrong canonical input byte count;
- wrong input row count;
- incomplete 300-pair population;
- duplicate endpoint identity;
- missing anchor/cell identity;
- unexpected anchor/cell identity;
- wrong layer;
- non-finite endpoint;
- boolean endpoint;
- N other than 300;
- zero-variance primary contrast distribution;
- Holm family other than exactly 15;
- output collision.

No imputation, epsilon variance, row dropping, endpoint replacement, or
post-hoc rescue is authorized.

## 11. Interpretation boundary

A Holm-supported Phase F result establishes at most:

PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_RESPONSE

for the exact frozen endpoint and structural estimand.

Where the corresponding Gen4 R6 behavioral estimand was also supported, the
strongest permitted bridge statement is:

OUTPUT_EFFECT_HAS_A_PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_CORRELATE

The execution does not establish:

- causal mediation;
- necessity;
- sufficiency;
- state-to-output causation;
- arbitrary-model generalization;
- arbitrary-dataset generalization;
- training benefit.

## 12. Post-execution boundary

A successful process exit does not by itself establish a scientific claim.

After execution, separately verify:

1. exact four-file output set;
2. 4500 pair-level rows;
3. 15 primary-result rows;
4. manifest bindings;
5. peer artifact SHA256 identities;
6. deterministic provenance;
7. interpretation against the frozen decision rules.

The output artifacts must then be frozen under a separate Phase F
artifact/result freeze before the branch advances beyond this analysis.

SCIENTIFIC_CONCLUSION_BEFORE_EXECUTION =
NONE

NEXT_PHASE_AFTER_SUCCESSFUL_EXECUTION =
PHASE_F_STATISTICAL_RESULT_ARTIFACT_VALIDATION_AND_FREEZE
