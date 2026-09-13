# ContraMamba Gen4 Native Mamba State Bridge
# Phase F Statistical Result Artifact Validation and Freeze
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_F_STATISTICAL_RESULT_ARTIFACT_VALIDATION_FREEZE

## 1. Frozen lineage

PHASE_F_STATISTICAL_SPECIFICATION =
830f7ea697ce24388dddd181c9aa301ec2b442fc

PHASE_F_IMPLEMENTATION_AUTHORITY =
d249de57dc8dac69f49f7110c6f1a07532f939cb

PHASE_F_OUTPUT_HASH_CORRECTION =
c2fa746826bc397688b8f1ae8191bb96b5f1bf17

PHASE_F_IMPLEMENTATION_COMMIT =
e917e4c4fe0c94aa4ef5be33f2a69e6c64732189

PHASE_F_EXECUTION_AUTHORITY =
1e28e8144163e3d2fe5faa5ff2ec01b3d077ca68

R6_VALIDATED_STATISTICAL_RESULTS =
5896d4740cd390a56ff5e2a3459c68ce65e2bc84

CANONICAL_INPUT_SHA256 =
7ff4d24b7895745efbf5a00e0361a4fe409ad585db17ece7151e4dedb289b07c

## 2. Execution status

CANONICAL_EXECUTION_COUNT =
1

EXECUTION_RESULT =
PASS_STATISTICAL_ANALYSIS_PRODUCED

MODEL_FORWARD =
NO

TOKENIZER_EXECUTION =
NO

NATIVE_STATE_EXTRACTION =
NO

TRAINING =
NO

GPU =
NO

KAGGLE =
NO

## 3. Exact frozen output artifacts

phase_f_pair_level_contrasts.csv
BYTES = 542251
SHA256 = abb7e837395138d095e285e40ec8782ea863061087b58844bec6fad5d4ef5e73

phase_f_primary_confirmatory_results.csv
BYTES = 6347
SHA256 = b298783934724698671d3fe56c9805a1b73afca1bef4bfaac50d273258deeda7

phase_f_statistical_analysis_manifest.json
BYTES = 1581
SHA256 = 8540003a23d9ffcb0842ac1535491d0b22d1b75cc1393a095c89e3390fdb6608

phase_f_statistical_analysis_report_candidate.md
BYTES = 3543
SHA256 = c298c4fec96b66748ccf10440ca9885d8ae14de5d31f5f76b552b4211e50a938

## 4. Independent post-execution validation

OUTPUT_FILE_SET =
PASS_EXACT_FOUR

OUTPUT_BYTE_IDENTITIES =
PASS

PAIR_LEVEL_ROWS =
PASS_4500

PRIMARY_RESULT_ROWS =
PASS_15

PAIR_POPULATION =
PASS_300_SHARED

PAIR_ORDER =
PASS_LEXICOGRAPHIC

HYPOTHESIS_ORDER =
PASS_ENDPOINT_THEN_ESTIMAND

INDEPENDENT_STATISTICAL_RECOMPUTATION =
PASS_15_OF_15

GLOBAL_HOLM_FAMILY =
PASS_EXACT_15

N_PER_HYPOTHESIS =
PASS_300

DF =
PASS_299

ZERO_VARIANCE_BLOCKER =
PASS_NONE_PRESENT

MANIFEST_BINDINGS =
PASS

PEER_ARTIFACT_HASH_BINDINGS =
PASS

MANIFEST_SELF_HASH =
PASS_ABSENT

## 5. Primary 15 results

| Endpoint | Estimand | Mean | Holm p | Reject | d_z |
|---|---|---:|---:|:---:|---:|
| POST4_SPEED | DELTA_TITLE | 0.012976405223210653 | 1.0 | False | 0.04038516839741582 |
| POST4_SPEED | DELTA_NAME | -0.0017506202061971028 | 1.0 | False | -0.032949077414079396 |
| POST4_SPEED | DELTA_ROLE | -0.01269685705502828 | 0.0003043308642735479 | True | -0.248107583331849 |
| POST4_SPEED | DELTA_PREDICATE | 0.02669889489809672 | 4.4464068870193605e-05 | True | 0.27418601873648807 |
| POST4_SPEED | INTERACTION_TITLE_NAME | -0.005837910572687785 | 0.6196721302470742 | False | -0.10229047444833596 |
| POST4_TURNING | DELTA_TITLE | -0.00030808647473653156 | 1.0 | False | -0.0089921879230302 |
| POST4_TURNING | DELTA_NAME | -0.001766198476155599 | 1.0 | False | -0.0641825439533307 |
| POST4_TURNING | DELTA_ROLE | -0.0025326824188232422 | 0.09994044617226422 | False | -0.14968276289451 |
| POST4_TURNING | DELTA_PREDICATE | 0.0007807246843973796 | 1.0 | False | 0.04165009935937225 |
| POST4_TURNING | INTERACTION_TITLE_NAME | -0.0026908330122629802 | 0.004578539750836566 | True | -0.207465365101926 |
| POST4_PATH_EFFICIENCY | DELTA_TITLE | -0.005520589101864759 | 1.0 | False | -0.08198642663429895 |
| POST4_PATH_EFFICIENCY | DELTA_NAME | 0.0018628185970151687 | 0.10181988893338632 | False | 0.14714727900839558 |
| POST4_PATH_EFFICIENCY | DELTA_ROLE | 0.0006059926820386793 | 1.0 | False | 0.07992484911511936 |
| POST4_PATH_EFFICIENCY | DELTA_PREDICATE | 0.012613116527876384 | 1.7827235514026965e-33 | True | 0.8091836537131226 |
| POST4_PATH_EFFICIENCY | INTERACTION_TITLE_NAME | 0.0017680603082027553 | 0.021856932869681503 | True | 0.18011583142119403 |

HOLM_SUPPORTED_HYPOTHESES =
5_OF_15

The five supported prespecified local native-state kinematic responses are:

1. POST4_SPEED / DELTA_ROLE: negative response.
2. POST4_SPEED / DELTA_PREDICATE: positive response.
3. POST4_TURNING / INTERACTION_TITLE_NAME: negative response.
4. POST4_PATH_EFFICIENCY / DELTA_PREDICATE: positive response.
5. POST4_PATH_EFFICIENCY / INTERACTION_TITLE_NAME: positive response.

All remaining ten prespecified endpoint/estimand hypotheses are not
established under the global Holm-adjusted alpha=0.05 decision rule.

Non-rejection does not establish exact zero.

The largest absolute standardized Phase F response is:

POST4_PATH_EFFICIENCY / DELTA_PREDICATE

with d_z approximately 0.809.

## 6. Behavioral-to-state bridge interpretation

The frozen R6 validated result establishes behavioral support for:

DELTA_NAME
DELTA_ROLE
DELTA_PREDICATE
INTERACTION_TITLE_NAME
TITLE_MINUS_NAME

and does not establish the DELTA_TITLE main effect.

Phase F independently establishes a prespecified local native-state kinematic
response for at least one endpoint for:

DELTA_ROLE
DELTA_PREDICATE
INTERACTION_TITLE_NAME

Therefore the strongest authorized bridge statement for those three semantic
structures is:

OUTPUT_EFFECT_HAS_A_PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_CORRELATE

More specifically:

- DELTA_ROLE has a supported POST4_SPEED correlate.
- DELTA_PREDICATE has supported POST4_SPEED and POST4_PATH_EFFICIENCY
  correlates.
- INTERACTION_TITLE_NAME has supported POST4_TURNING and
  POST4_PATH_EFFICIENCY correlates.

DELTA_NAME has a frozen supported R6 behavioral effect but no supported Phase F
response among the three prespecified local kinematic endpoints.

TITLE_MINUS_NAME is not an authorized Phase F native-state estimand, so no
native-state bridge claim is made for it.

No output-sign matching criterion applies. A state-response direction opposite
to an output-effect direction is not a contradiction under the frozen
mechanistic bridge specification.

## 7. Scientific limitations

These results do not establish:

- causal mediation;
- necessity;
- sufficiency;
- state-to-output causation;
- arbitrary-model generalization;
- arbitrary-dataset generalization;
- training benefit.

The evidence is limited to the frozen model, frozen 300 source-pair
population, primary layer 11, and three prespecified local kinematic endpoints.

## 8. Phase conclusion

PHASE_F_RESULT =
VALIDATED

PRIMARY_CONFIRMATORY_RESULT =
5_OF_15_HOLM_SUPPORTED

BEHAVIORAL_TO_NATIVE_STATE_BRIDGE =
SUPPORTED_FOR_ROLE_PREDICATE_AND_TITLE_NAME_INTERACTION_WITHIN_PRESPECIFIED_ENDPOINTS

SCIENTIFIC_CONCLUSION =
PRESPECIFIED_LOCAL_NATIVE_STATE_KINEMATIC_CORRELATES_ESTABLISHED_FOR_THREE_R6_SUPPORTED_STRUCTURAL_EFFECTS

Phase F does not establish a causal mechanism.

NEXT_PHASE =
POST_PHASE_F_MECHANISTIC_INTERPRETATION_AFTER_RESULT_FREEZE
