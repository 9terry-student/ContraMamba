# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Secondary Six-Family Statistical Analysis Execution Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_STATISTICAL_ANALYSIS_EXECUTION_AUTHORITY

PHASE =
CANONICAL_STATISTICAL_EXECUTION

THIS_DOCUMENT_AUTHORIZES =
ONE_BOUNDED_LOCAL_CPU_CANONICAL_STATISTICAL_EXECUTION

SCIENTIFIC_CONCLUSION_BEFORE_EXECUTION =
NONE


## 1. Frozen authority lineage

SECONDARY_SIX_STATISTICAL_SPECIFICATION =
2030de52922c251caeafceb36dd5f2f57b5b39c7

SCIENTIFIC_SPECIFICATION =
01801ad1617b2ebc3ffa859ba440636d4755a55c

EXTRACTION_PROVENANCE_VALIDATION_FREEZE =
e500beabf8a7e1dfc0f8260586d8f5ed8e374d5c

EXTRACTION_STORAGE_FREEZE =
60b430fccec9e51e4d1f12131eabf37d03a8be66

EXTRACTION_RELEASE_BINDING =
a0ce07ad828c5ee0e6a4c36d8f9a6e972090590a

INITIAL_STATISTICAL_IMPLEMENTATION_AUTHORITY =
66a35cb469a24c881830fbf69169fb6efcd6cc0c

ROW_IDENTITY_CORRECTION_AUTHORITY =
851d9a96581e9dc124a0a70e3a2c13a902357682

CORRECTED_IMPLEMENTATION_COMMIT =
cc0f64104828988faa99fec1356b84014c66317a

CORRECTED_IMPLEMENTATION_VALIDATION_FREEZE =
90c45b3d4905290451e4f353ea45674c279b6232


## 2. Exact executable identity

SCRIPT_PATH =
scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

SCRIPT_SHA256 =
5f75c06e69cdf2655e7973505ecd0a2bbb337f381b8fb8601906a1b26e7c3761

IMPLEMENTATION_COMMIT =
cc0f64104828988faa99fec1356b84014c66317a

The execution must use exactly this script identity.

The affected pre-correction implementation:

c83b370ce920c6878bb86f6da75da9b7689cf48b

is forbidden for canonical execution.


## 3. Exact canonical input

CANONICAL_INPUT_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_scientific_extraction_v1/kinematic_endpoints.jsonl

CANONICAL_INPUT_SHA256 =
b47e32496f73f5493e8737b040b30bc4a23dd8cd22454a85244fa812273b6605

CANONICAL_INPUT_BYTES =
513396

CANONICAL_INPUT_ROWS =
1200

SOURCE_PAIR_COUNT =
300

ROWS_PER_SOURCE_PAIR =
4

The canonical input may be opened only by the authorized canonical
statistical execution after this authority itself is committed and pushed.

No alternate endpoint file, reconstructed endpoint file, filtered copy,
regenerated copy, or transformed substitute is authorized.


## 4. Exact statistical family

STRUCTURAL_ESTIMAND =
DELTA_NAME

LAYERS =
5
17

ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

SECONDARY_HYPOTHESIS_COUNT =
6

HYPOTHESIS_ORDER =
LAYER_THEN_ENDPOINT

PAIR_ORDER =
LEXICOGRAPHIC_SOURCE_PAIR_ID

PAIR_LEVEL_CONTRAST =
C2_NAME_MINUS_C0_SHAM

INFERENTIAL_UNIT =
SOURCE_PAIR

N_PER_HYPOTHESIS =
300

DF =
299


## 5. Exact statistical procedure

NUMERIC_ANALYSIS_DTYPE =
FLOAT64

TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T

NULL_MEAN =
0

CONFIDENCE_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T

EFFECT_SIZE =
D_Z

D_Z =
MEAN_DIVIDED_BY_SAMPLE_SD

ZERO_VARIANCE =
HARD_BLOCK

FAMILYWISE_ALPHA =
0.05

MULTIPLICITY =
GLOBAL_HOLM_BONFERRONI_ALL_6

SEPARATE_HOLM_BY_LAYER =
FORBIDDEN

SEPARATE_HOLM_BY_ENDPOINT =
FORBIDDEN

CROSS_LAYER_DIFFERENCE_ESTIMAND =
NOT_DEFINED

CROSS_LAYER_DIFFERENCE_TEST =
NOT_AUTHORIZED

DEPTH_SELECTIVITY_CLAIM =
NOT_AUTHORIZED


## 6. Exact output identity

OUTPUT_DIRECTORY =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_v1

The output directory must not exist before execution.

The corresponding staging directory:

reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_v1.staging

must also not exist before execution.

Exactly four canonical result files are authorized:

name_q1_q3_pair_level_contrasts.csv

name_q1_q3_secondary_confirmatory_results.csv

name_q1_q3_statistical_analysis_manifest.json

name_q1_q3_statistical_analysis_report_candidate.md

PAIR_LEVEL_RESULT_ROWS =
1800

SECONDARY_RESULT_ROWS =
6

PARTIAL_PUBLICATION =
FORBIDDEN

OUTPUT_COLLISION =
HARD_BLOCK


## 7. Runtime boundary

EXECUTION_COUNT =
ONE

EXECUTION_MODE =
LOCAL_CPU_ONLY

KAGGLE =
NOT_REQUIRED

GPU =
NOT_AUTHORIZED

MODEL_FORWARD =
NOT_AUTHORIZED

CHECKPOINT_LOADING =
NOT_AUTHORIZED

TOKENIZER_EXECUTION =
NOT_AUTHORIZED

NATIVE_STATE_EXTRACTION =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

BACKWARD =
NOT_AUTHORIZED

OPTIMIZER =
NOT_AUTHORIZED

NETWORK_ACCESS =
NOT_REQUIRED

NEW_DEPENDENCIES =
FORBIDDEN


## 8. One-shot identity

This authority is single-use for canonical statistical execution.

The execution identity is bound to:

- this execution-authority commit;
- corrected implementation commit
  cc0f64104828988faa99fec1356b84014c66317a;
- script SHA256
  5f75c06e69cdf2655e7973505ecd0a2bbb337f381b8fb8601906a1b26e7c3761;
- canonical input SHA256
  b47e32496f73f5493e8737b040b30bc4a23dd8cd22454a85244fa812273b6605;
- exact output directory.

If execution fails after canonical input is opened or statistical execution
begins, the same execution authority must not be silently reused.

A failure must be classified before any retry.

Any retry requiring changed code, changed command semantics, changed input,
or a second canonical execution requires a separately frozen recovery or
retry authority.


## 9. Required pre-execution gates

Before execution, all of the following must hold:

1. HEAD equals the committed execution-authority commit.
2. Branch is gen4-phase-d-cache-validator-correction.
3. no staged tracked changes exist.
4. no unstaged tracked changes exist.
5. the only pre-existing untracked paths are the two intentionally preserved
   support_states.npy files.
6. script SHA256 equals the frozen script SHA256.
7. canonical input path exists.
8. canonical input byte count and SHA256 equal the frozen identity.
9. output directory does not exist.
10. output staging directory does not exist.

Any mismatch blocks execution.


## 10. Authorized command semantics

After this authority is committed and pushed, the controller may issue one
exact command equivalent to:

python scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py run-canonical \
  --input-jsonl reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_scientific_extraction_v1/kinematic_endpoints.jsonl \
  --output-dir reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_v1 \
  --implementation-commit cc0f64104828988faa99fec1356b84014c66317a \
  --script-sha256 5f75c06e69cdf2655e7973505ecd0a2bbb337f381b8fb8601906a1b26e7c3761 \
  --execution-authority-commit <THIS_AUTHORITY_FULL_COMMIT_SHA>

The placeholder must be replaced by the full commit SHA that freezes this
authority.

No other CLI operation is authorized.


## 11. Post-execution validation

A process exit code of zero establishes execution success only.

It does not by itself establish scientific validity.

After execution, validate separately:

1. exactly four output files exist;
2. no staging directory remains;
3. pair-level CSV has exactly 1800 result rows;
4. secondary result CSV has exactly six result rows;
5. manifest provenance binds the exact input, implementation commit, script
   SHA256, and this execution-authority commit;
6. peer SHA256 values match the generated peer artifacts;
7. result ordering matches the frozen hypothesis order;
8. all numeric values are finite;
9. Holm correction is one global family of six;
10. no cross-layer difference result is present.

Only after these checks may scientific interpretation begin.


## 12. Scientific interpretation boundary

A successful execution may establish only the prespecified secondary
layer-local NAME kinematic correlate decisions under the frozen six-family
procedure.

It does not establish:

- causal mediation;
- state-to-output causation;
- necessity;
- sufficiency;
- significant differences between layers;
- depth selectivity;
- an overall adaptive-program FWER guarantee across Phase F and this
  follow-up;
- arbitrary-model generalization;
- arbitrary-dataset generalization.

A non-significant result means NOT_ESTABLISHED, not exact zero.


## 13. Next transition

ON_AUTHORITY_FREEZE =
ONE_LOCAL_CPU_CANONICAL_STATISTICAL_EXECUTION_ALLOWED

ON_EXECUTION_SUCCESS =
RESULT_ARTIFACT_VALIDATION_REQUIRED

ON_EXECUTION_FAILURE =
FAILURE_CLASSIFICATION_REQUIRED

SCIENTIFIC_INTERPRETATION_BEFORE_RESULT_VALIDATION =
FORBIDDEN
