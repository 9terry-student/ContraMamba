# ContraMamba Gen4 Native Mamba State Bridge
# NAME Direct Cross-Layer Paired-Difference Canonical Execution Authority
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_DIRECT_CROSS_LAYER_PAIRED_DIFFERENCE_CANONICAL_EXECUTION_AUTHORITY

PHASE =
SINGLE_USE_CANONICAL_STATISTICAL_EXECUTION


## 1. Frozen lineage

DIRECT_CROSS_LAYER_SCIENTIFIC_SPECIFICATION =
b3e0ade126622f244b557e1db07296c622bd7202

DIRECT_CROSS_LAYER_IMPLEMENTATION_AUTHORITY =
724c28b528b0f182bc0c79cf5ee0b3adfca76ec2

DIRECT_CROSS_LAYER_IMPLEMENTATION =
cdf9f2117bc0d5fa119fbf24b85266c1f9ced448

DIRECT_CROSS_LAYER_IMPLEMENTATION_VALIDATION_FREEZE =
82bd2cd2fa90d15051b245dbf8d9c484a30b6e51


## 2. Implementation identity

IMPLEMENTATION_PATH =
scripts/reason_router_gen4_native_mamba_state_name_direct_cross_layer_statistical_analysis.py

IMPLEMENTATION_SHA256 =
bccf4b91f8b566808315ce15fafa5c948a8b20bde751c75c8af0da9671c83fff

IMPLEMENTATION_STATUS =
CLOSED_VALIDATED

DEDICATED_SYNTHETIC_TEST_RESULT =
34_OF_34_PASS


## 3. Exact canonical inputs

PHASE_F_PAIR_LEVEL_PATH =
reports/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis_e917e4c_v1/phase_f_pair_level_contrasts.csv

PHASE_F_PAIR_LEVEL_SHA256 =
abb7e837395138d095e285e40ec8782ea863061087b58844bec6fad5d4ef5e73

PHASE_F_PAIR_LEVEL_BYTES =
542251

PHASE_F_PAIR_LEVEL_ROWS =
4500


Q1_Q3_PAIR_LEVEL_PATH =
reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_retry1_v1/name_q1_q3_pair_level_contrasts.csv

Q1_Q3_PAIR_LEVEL_SHA256 =
c0c917560b5a37c5df82ad4e203a87441370c129622afc668b77712b98698f82

Q1_Q3_PAIR_LEVEL_BYTES =
196542

Q1_Q3_PAIR_LEVEL_ROWS =
1800


## 4. Authorized inferential family

STRUCTURAL_ESTIMAND =
DELTA_NAME

SOURCE_PAIR_COUNT =
300

PRIMARY_INFERENTIAL_UNIT =
SOURCE_PAIR

LAYER_CONTRASTS =
5_MINUS_11
17_MINUS_11

ENDPOINTS =
POST4_SPEED
POST4_TURNING
POST4_PATH_EFFICIENCY

HYPOTHESIS_COUNT =
6

PAIRWISE_DIFFERENCE_DIRECTION =
SECONDARY_LAYER_MINUS_MIDPOINT_LAYER

TEST =
TWO_SIDED_ONE_SAMPLE_STUDENT_T_ON_PAIRED_CROSS_LAYER_DIFFERENCES

N_PER_HYPOTHESIS =
300

DF_PER_HYPOTHESIS =
299

MULTIPLICITY =
ONE_GLOBAL_HOLM_BONFERRONI_FAMILY_OF_6

FAMILYWISE_ALPHA =
0.05

NUMERIC_ANALYSIS_DTYPE =
FLOAT64


## 5. Exact authorized output location

OUTPUT_DIRECTORY =
reports/reason_router_gen4_native_mamba_state_bridge_name_direct_cross_layer_statistical_analysis_v1

STAGING_DIRECTORY =
reports/reason_router_gen4_native_mamba_state_bridge_name_direct_cross_layer_statistical_analysis_v1.staging

The output directory and staging directory must both be absent before
execution begins.


## 6. Exact authorized output file set

The canonical execution may produce exactly four files:

1. name_direct_cross_layer_pair_level_differences.csv

2. name_direct_cross_layer_confirmatory_results.csv

3. name_direct_cross_layer_statistical_analysis_manifest.json

4. name_direct_cross_layer_statistical_analysis_report_candidate.md

No other canonical result artifact is authorized.


## 7. Execution mode

EXECUTION_ENVIRONMENT =
LOCAL

PROCESSOR =
CPU

GPU =
OFF_NOT_USED

KAGGLE =
NO

MODEL_FORWARD =
NO

CHECKPOINT_LOADING =
NO

TOKENIZER_EXECUTION =
NO

NATIVE_STATE_EXTRACTION =
NO

TRAINING =
NO

BACKWARD =
NO

Only already frozen pair-level statistical artifacts may be read.


## 8. Single-use rule

AUTHORIZED_CANONICAL_EXECUTION_COUNT =
1

This authority is single-use.

The authority is consumed when canonical execution begins.

If the process:

- succeeds;
- exits nonzero;
- raises an exception;
- partially creates staging output;
- fails artifact publication;

the authority is consumed and must not be reused.

A retry requires a new explicit recovery/retry authority.


## 9. Required pre-execution gates

Immediately before execution, all of the following must pass:

1. branch is:
   gen4-phase-d-cache-validator-correction

2. HEAD is exactly the committed execution-authority commit;

3. no tracked changes exist;

4. the only pre-existing untracked files are the six intentionally preserved
   files already known before this authority;

5. implementation SHA256 equals:
   bccf4b91f8b566808315ce15fafa5c948a8b20bde751c75c8af0da9671c83fff

6. Phase F canonical input byte count and SHA256 match this authority;

7. Q1/Q3 canonical input byte count and SHA256 match this authority;

8. canonical output directory is absent;

9. canonical staging directory is absent.

Any failed gate blocks execution without consuming this authority.


## 10. Authorized canonical command semantics

After this authority is committed, the single authorized invocation is
semantically:

python scripts/reason_router_gen4_native_mamba_state_name_direct_cross_layer_statistical_analysis.py run-canonical \
  --phase-f-pair-csv reports/reason_router_gen4_native_mamba_state_phase_f_statistical_analysis_e917e4c_v1/phase_f_pair_level_contrasts.csv \
  --q1-q3-pair-csv reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_statistical_analysis_retry1_v1/name_q1_q3_pair_level_contrasts.csv \
  --output-dir reports/reason_router_gen4_native_mamba_state_bridge_name_direct_cross_layer_statistical_analysis_v1 \
  --implementation-commit cdf9f2117bc0d5fa119fbf24b85266c1f9ced448 \
  --script-sha256 bccf4b91f8b566808315ce15fafa5c948a8b20bde751c75c8af0da9671c83fff \
  --execution-authority-commit <THIS_EXECUTION_AUTHORITY_COMMIT>

The exact committed authority SHA must replace the final placeholder.


## 11. Process success versus scientific validity

A zero process exit code establishes only:

EXECUTION_SUCCESS =
PASS

It does not independently establish:

ARTIFACT_PROVENANCE_VALIDITY =
PASS

or:

SCIENTIFIC_RESULT_VALIDATION =
PASS

or:

SCIENTIFIC_CONCLUSION =
SUPPORTED

The four produced artifacts require independent validation after execution.


## 12. Required post-execution validation

Before scientific interpretation, independently validate:

- exact output file set;
- absence of staging directory;
- output byte identities;
- 1800 pair-level rows;
- 6 confirmatory rows;
- exact deterministic ordering;
- exact 300-pair population;
- pairwise subtraction direction;
- component contrast auditability;
- descriptive-statistic recomputation;
- Student-t recomputation;
- 95% CI with df=299;
- d_z recomputation;
- exact global Holm-6 recomputation;
- manifest provenance;
- exact POSIX canonical input paths;
- peer artifact SHA256 bindings;
- report provenance;
- adaptive-program FWER limitation;
- endpoint- and layer-pair-specific interpretation boundary.

No result may be scientifically interpreted before this validation passes.


## 13. Adaptive inference boundary

This is an adaptive follow-up selected after observing prior Phase F and
Q1/Q3 results.

OVERALL_ADAPTIVE_PROGRAM_FWER =
NOT_CLAIMED

The new global Holm correction controls only this six-member direct
cross-layer family.


## 14. Positive claim boundary

If an exact hypothesis survives the frozen Holm-6 procedure, the strongest
permitted claim is:

PRESPECIFIED_DIRECT_BETWEEN_LAYER_DIFFERENCE_ESTABLISHED_FOR_THE_SPECIFIED_NAME_KINEMATIC_ENDPOINT

Only the exact supported layer-pair/endpoint combination receives that claim.

The broad statement:

NAME_IS_DEPTH_SELECTIVE

remains prohibited.


## 15. Explicitly prohibited execution

Do not:

- execute more than once;
- add 17_MINUS_5;
- scan other layers;
- inspect only POST4_PATH_EFFICIENCY;
- change endpoint scope;
- alter multiplicity;
- alter sidedness;
- alter source-pair population;
- open native-state tensors;
- recompute condition-level endpoints;
- run model inference;
- use Kaggle;
- use GPU;
- train;
- rerun under this authority after any execution start.


## 16. Authority consumption states

Before execution:

AUTHORITY_STATUS =
READY_SINGLE_USE_AFTER_COMMIT

Once execution begins:

AUTHORITY_STATUS =
CONSUMED

On successful process completion:

PROCESS_STATUS =
PASS_PENDING_ARTIFACT_VALIDATION

On process failure:

PROCESS_STATUS =
FAILED_AUTHORITY_CONSUMED_RETRY_REQUIRES_NEW_AUTHORITY


## 17. Next transition

NEXT_ACTION =
COMMIT_THIS_EXECUTION_AUTHORITY_THEN_RUN_EXACTLY_ONE_BOUNDED_LOCAL_CPU_CANONICAL_ANALYSIS

RESULT_INTERPRETATION =
NOT_AUTHORIZED_UNTIL_INDEPENDENT_ARTIFACT_VALIDATION_PASSES
