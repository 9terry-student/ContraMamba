# ContraMamba Gen4 Native Mamba State Bridge
# NAME Direct Cross-Layer Paired-Difference Implementation Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_DIRECT_CROSS_LAYER_PAIRED_DIFFERENCE_IMPLEMENTATION_AUTHORITY

PHASE =
BOUNDED_STATISTICAL_IMPLEMENTATION

THIS_DOCUMENT_AUTHORIZES_IMPLEMENTATION =
YES

THIS_DOCUMENT_AUTHORIZES_SYNTHETIC_VALIDATION =
YES

THIS_DOCUMENT_AUTHORIZES_CANONICAL_STATISTICAL_EXECUTION =
NO

THIS_DOCUMENT_AUTHORIZES_CANONICAL_NUMERIC_OUTCOME_INSPECTION =
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


## 1. Frozen scientific authority

DIRECT_CROSS_LAYER_SCIENTIFIC_SPECIFICATION =
b3e0ade126622f244b557e1db07296c622bd7202

The implementation must realize that specification exactly.

It may not alter:

- the two layer contrasts;
- the three endpoints;
- DELTA_NAME;
- the 300-pair inferential unit;
- subtraction direction;
- two-sided testing;
- Student-t inference;
- d_z definition;
- the exact six-member Holm family;
- the adaptive-program interpretation boundary.


## 2. Exact permitted tracked delta

Exactly two implementation files may be created:

IMPLEMENTATION_PATH =
scripts/reason_router_gen4_native_mamba_state_name_direct_cross_layer_statistical_analysis.py

TEST_PATH =
tests/test_reason_router_gen4_native_mamba_state_name_direct_cross_layer_statistical_analysis.py

No other tracked file may be created, modified, deleted, or renamed during
implementation.

PERMITTED_TRACKED_FILE_COUNT =
2

EXISTING_Q1_Q3_IMPLEMENTATION_MODIFICATION =
FORBIDDEN

PHASE_F_IMPLEMENTATION_MODIFICATION =
FORBIDDEN

SCIENTIFIC_SPECIFICATION_MODIFICATION =
FORBIDDEN

REPORT_MODIFICATION_DURING_IMPLEMENTATION =
FORBIDDEN


## 3. Frozen canonical input identities

### Phase F midpoint pair-level artifact

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


### Q1/Q3 pair-level artifact

Q1_Q3_PAIR_LEVEL_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_retry1_v1/name_q1_q3_pair_level_contrasts.csv

Q1_Q3_PAIR_LEVEL_SHA256 =
c0c917560b5a37c5df82ad4e203a87441370c129622afc668b77712b98698f82

Q1_Q3_PAIR_LEVEL_BYTES =
196542

Q1_Q3_TOTAL_ROWS =
1800

These canonical files must not be opened during implementation or synthetic
validation.

Their paths, byte counts, and SHA256 identities may be represented only as
constants until a separate execution authority exists.


## 4. Frozen cross-layer family

STRUCTURAL_ESTIMAND =
DELTA_NAME

MIDPOINT_LAYER =
11

SECONDARY_LAYERS =
5
17

LAYER_CONTRAST_ORDER =
5_MINUS_11
17_MINUS_11

ENDPOINT_ORDER =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

HYPOTHESIS_COUNT =
6

SOURCE_PAIR_COUNT =
300

DF =
299

The implementation must construct:

D_i_5_MINUS_11_e =
X_i_5_e - X_i_11_e

and:

D_i_17_MINUS_11_e =
X_i_17_e - X_i_11_e

No 17_MINUS_5 quantity may be created as an inferential result.


## 5. Input schema contracts

The Phase F pair-level CSV must have exactly:

schema_version
source_pair_id
endpoint
estimand
contrast_value

Expected Phase F schema version:

gen4_native_mamba_phase_f_pair_level_contrasts_v1


The Q1/Q3 pair-level CSV must have exactly:

schema_version
source_pair_id
layer_index
endpoint
estimand
contrast_value

Expected Q1/Q3 schema version:

gen4_name_q1_q3_pair_level_contrasts_v1

Unexpected fields, missing fields, duplicate structural keys, unexpected
estimands, unexpected endpoints, or unexpected layers must hard-block.


## 6. Structural validation before subtraction

Before any cross-layer difference is computed, the implementation must verify:

1. exact Phase F artifact bytes and SHA256;
2. exact Q1/Q3 artifact bytes and SHA256;
3. exact CSV field contracts;
4. exact row cardinalities;
5. exactly 900 Phase F DELTA_NAME rows;
6. exactly layers 5 and 17 in Q1/Q3;
7. exactly the three frozen endpoints;
8. exactly 300 unique source_pair_id values per cell;
9. exact source-pair set equality across all layer/endpoint cells;
10. deterministic lexicographic pair ordering;
11. uniqueness of every structural lookup key;
12. finite contrast values.

No subtraction may occur before all structural checks pass.


## 7. Pair-level cross-layer output contract

PAIR_LEVEL_SCHEMA =
gen4_name_direct_cross_layer_pair_level_differences_v1

PAIR_LEVEL_OUTPUT_FILENAME =
name_direct_cross_layer_pair_level_differences.csv

PAIR_LEVEL_FIELDS_IN_ORDER =

schema_version
source_pair_id
layer_contrast
secondary_layer_index
midpoint_layer_index
endpoint
estimand
secondary_contrast_value
midpoint_contrast_value
cross_layer_difference

PAIR_LEVEL_RESULT_ROWS =
1800

Deterministic ordering must be:

1. layer contrast:
   5_MINUS_11
   17_MINUS_11

2. endpoint:
   POST4_SPEED
   POST4_TURNING
   POST4_PATH_EFFICIENCY

3. source_pair_id:
   lexicographic

The output must retain both component frozen contrast values so every direct
cross-layer difference is auditable.


## 8. Confirmatory result output contract

CONFIRMATORY_SCHEMA =
gen4_name_direct_cross_layer_confirmatory_results_v1

CONFIRMATORY_OUTPUT_FILENAME =
name_direct_cross_layer_confirmatory_results.csv

CONFIRMATORY_FIELDS_IN_ORDER =

schema_version
layer_contrast
secondary_layer_index
midpoint_layer_index
endpoint
estimand
n
mean
sample_sd
standard_error
median
minimum
maximum
df
t_statistic
raw_p_value
holm_adjusted_p_value
reject_holm_alpha_0_05
ci95_low
ci95_high
d_z
decision

CONFIRMATORY_RESULT_ROWS =
6

Deterministic result order must equal the frozen six-hypothesis order.


## 9. Statistical semantics

NUMERIC_ANALYSIS_DTYPE =
FLOAT64

TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T

TEST_INPUT =
PAIRWISE_CROSS_LAYER_DIFFERENCES

N_PER_HYPOTHESIS =
300

DF_PER_HYPOTHESIS =
299

CONFIDENCE_INTERVAL =
TWO_SIDED_95_PERCENT_STUDENT_T

D_Z =
MEAN_PAIRWISE_DIFFERENCE_DIVIDED_BY_SAMPLE_SD_OF_PAIRWISE_DIFFERENCES

ZERO_VARIANCE =
HARD_BLOCK

MULTIPLICITY =
ONE_GLOBAL_HOLM_BONFERRONI_FAMILY_OF_6

FAMILYWISE_ALPHA =
0.05

Separate Holm families by layer contrast or endpoint are forbidden.


## 10. Numeric implementation reference

The already validated Q1/Q3 statistical implementation contains the accepted
standard-library numerical semantics for:

- finite-number validation;
- regularized incomplete beta;
- two-sided Student-t p-values;
- Student-t critical values;
- descriptive summary;
- one-sample t statistics;
- d_z;
- exact six-member Holm-Bonferroni.

REFERENCE_PATH =
scripts/reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py

REFERENCE_IMPLEMENTATION_SHA256 =
dc50ac3552182067d5dc76e4f3cecf2f674b24d045fa08352afa84c640266cf4

The new implementation may reproduce those pure numerical algorithms.

It must not reuse:

- the Q1/Q3 canonical input contract;
- the Q1/Q3 runner provenance;
- the Q1/Q3 result schemas;
- the Q1/Q3 scientific labels;
- the Q1/Q3 output directory logic.

The cross-layer implementation must have its own explicit constants, schemas,
labels, provenance, and runner.


## 11. Decision labels

SUPPORTED_DECISION =
PRESPECIFIED_DIRECT_CROSS_LAYER_DIFFERENCE_SUPPORTED

NOT_ESTABLISHED_DECISION =
PRESPECIFIED_DIRECT_CROSS_LAYER_DIFFERENCE_NOT_ESTABLISHED

FAMILY_SUPPORTED_DECISION =
DIRECT_CROSS_LAYER_NAME_DIFFERENCE_SUPPORTED_FOR_AT_LEAST_ONE_PRESPECIFIED_LAYER_PAIR_ENDPOINT

FAMILY_NOT_ESTABLISHED_DECISION =
DIRECT_CROSS_LAYER_NAME_DIFFERENCE_NOT_ESTABLISHED_WITHIN_THE_PRESPECIFIED_FAMILY

An individual result is supported iff:

HOLM_ADJUSTED_P < 0.05


## 12. Manifest contract

MANIFEST_SCHEMA =
gen4_name_direct_cross_layer_statistical_analysis_manifest_v1

MANIFEST_OUTPUT_FILENAME =
name_direct_cross_layer_statistical_analysis_manifest.json

The manifest must bind at minimum:

- this scientific specification commit;
- this implementation authority commit;
- future implementation commit;
- future implementation script SHA256;
- future execution authority commit;
- exact frozen Phase F logical input path;
- Phase F input SHA256;
- Phase F input bytes;
- exact frozen Q1/Q3 logical input path;
- Q1/Q3 input SHA256;
- Q1/Q3 input bytes;
- source-pair count;
- layer-contrast order;
- endpoint order;
- hypothesis count;
- global Holm method;
- familywise alpha;
- FLOAT64 analysis;
- primary inferential unit SOURCE_PAIR;
- overall adaptive-program FWER not claimed;
- no model inference;
- no training;
- no backward pass.

MANIFEST_SERIALIZATION =
CANONICAL_SORTED_JSON_SINGLE_TERMINAL_LF


## 13. Provenance path serialization rule

The canonical logical paths serialized into the manifest and report must be
the exact frozen POSIX-slash constants:

reports/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis_e917e4c_v1/phase_f_pair_level_contrasts.csv

and:

reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_retry1_v1/name_q1_q3_pair_level_contrasts.csv

Host-native Path string serialization is forbidden.

PROVENANCE_PATH_SERIALIZATION =
FROZEN_LOGICAL_POSIX_PATH_CONSTANTS

This rule prevents recurrence of the previously closed platform-dependent
path serialization defect.


## 14. Report contract

REPORT_OUTPUT_FILENAME =
name_direct_cross_layer_statistical_analysis_report_candidate.md

The report must contain:

- exact provenance bindings;
- the six hypotheses in deterministic order;
- means;
- t statistics;
- raw p-values;
- Holm-adjusted p-values;
- reject decisions;
- d_z values;
- family decision;
- explicit adaptive-program FWER limitation;
- explicit prohibition on broad unqualified depth-selectivity claims;
- explicit statement that exact supported pair/endpoint differences only may
  receive positive between-layer claims.


## 15. Exact output filename set

OUTPUT_FILENAMES_IN_ORDER =

name_direct_cross_layer_pair_level_differences.csv

name_direct_cross_layer_confirmatory_results.csv

name_direct_cross_layer_statistical_analysis_manifest.json

name_direct_cross_layer_statistical_analysis_report_candidate.md

No additional canonical result file is authorized.


## 16. Atomic publication requirement

A future canonical runner must use a staging directory.

It must:

1. refuse output collision;
2. refuse pre-existing staging collision;
3. build exactly four output artifacts;
4. serialize deterministic bytes;
5. write all outputs to staging;
6. atomically publish only after all writes succeed;
7. leave no staging directory on successful publication.

Partial public result publication is forbidden.


## 17. Required CLI surface

The implementation must provide a canonical execution entry point equivalent
to:

python scripts/reason_router_gen4_native_mamba_state_name_direct_cross_layer_statistical_analysis.py run-canonical \
  --phase-f-pair-csv <PATH> \
  --q1-q3-pair-csv <PATH> \
  --output-dir <PATH> \
  --implementation-commit <SHA> \
  --script-sha256 <SHA256> \
  --execution-authority-commit <SHA>

The runner must validate that the provided canonical input paths correspond
to the exact frozen logical paths and exact frozen artifact identities.

Presence of this runner does not authorize its canonical invocation.


## 18. Synthetic validation requirements

The dedicated tests must establish at minimum:

1. correct 5_MINUS_11 subtraction direction;
2. correct 17_MINUS_11 subtraction direction;
3. exact layer-contrast ordering;
4. exact endpoint ordering;
5. lexicographic source-pair ordering;
6. exact pair-level output fields;
7. exact confirmatory output fields;
8. rejection of source-pair population mismatch;
9. rejection of duplicate Phase F keys;
10. rejection of duplicate Q1/Q3 keys;
11. rejection of missing structural cells;
12. rejection of unexpected secondary layer;
13. rejection of unexpected endpoint;
14. rejection of non-DELTA_NAME estimand;
15. rejection of non-finite contrast values;
16. zero-variance blocker;
17. exact two-sided Student-t behavior;
18. exact d_z behavior;
19. exact six-member global Holm behavior;
20. deterministic canonical JSON;
21. deterministic CSV line endings;
22. deterministic report serialization;
23. manifest uses exact frozen POSIX Phase F path;
24. manifest uses exact frozen POSIX Q1/Q3 path;
25. report uses exact frozen POSIX paths;
26. host-native path representation does not leak into provenance;
27. exact peer-artifact SHA bindings;
28. output collision blocker;
29. staging collision blocker;
30. no canonical Phase F artifact is opened by synthetic tests;
31. no canonical Q1/Q3 artifact is opened by synthetic tests.


## 19. Canonical-data access boundary

During implementation and synthetic validation:

CANONICAL_PHASE_F_PAIR_ARTIFACT_OPEN =
FORBIDDEN

CANONICAL_Q1_Q3_PAIR_ARTIFACT_OPEN =
FORBIDDEN

CROSS_LAYER_CANONICAL_NUMERIC_DIFFERENCE_COMPUTATION =
FORBIDDEN

CANONICAL_STATISTICAL_TESTING =
FORBIDDEN

Synthetic records only may be used to exercise the implementation.


## 20. Prohibited implementation changes

Do not:

- change scientific estimands;
- add 17_MINUS_5;
- add layers;
- remove endpoints;
- inspect only the previously supported path-efficiency endpoint;
- introduce one-sided tests;
- alter Holm family size;
- add learned components;
- add NumPy/SciPy dependencies;
- open state tensors;
- open condition-level endpoint artifacts;
- execute the model;
- load checkpoints;
- run tokenizers;
- train;
- use GPU;
- use Kaggle;
- modify existing statistical scripts.


## 21. Required validation before implementation freeze

Before the implementation may be frozen:

PY_COMPILE =
REQUIRED_PASS

DEDICATED_SYNTHETIC_TEST_SUITE =
REQUIRED_PASS

CANONICAL_ARTIFACT_ACCESS_DURING_TESTS =
REQUIRED_NO

GIT_DIFF_CHECK =
REQUIRED_PASS

TRACKED_DELTA_SCOPE =
EXACT_TWO_FILES

The implementation and test file SHA256 identities must be recorded before
their implementation freeze.


## 22. Execution boundary

THIS_AUTHORITY_AUTHORIZES_CANONICAL_RUN =
NO

THIS_AUTHORITY_AUTHORIZES_RESULT_INTERPRETATION =
NO

A successful synthetic test suite establishes code correctness only.

It does not establish:

- execution success;
- artifact provenance validity;
- any direct between-layer scientific result.


## 23. Required next phase ordering

After this authority is committed and pushed:

1. create exactly the two authorized implementation files;
2. run only synthetic validation;
3. independently inspect the implementation delta;
4. freeze implementation/test identities;
5. create a separate single-use execution authority;
6. only then open the canonical pair-level artifacts for numeric analysis.


## 24. Next transition

NEXT_ACTION =
IMPLEMENT_BOUNDED_DIRECT_CROSS_LAYER_ANALYSIS_AND_SYNTHETIC_TESTS

CANONICAL_EXECUTION =
NOT_AUTHORIZED

SCIENTIFIC_OUTCOME_INSPECTION =
NOT_AUTHORIZED
