# Native State Kinematics
# Feasibility Tokenizer-Gate Supersession Authority Specification

STATUS = CANDIDATE

PHASE =
TOKENIZER_GATE_SUPERSESSION_AND_FEASIBILITY_RESUMPTION

PARENT_DESIGN_AUTHORITY =
c4286a4d8af9ae31b7e44de2a3e79b560efa2355

PARENT_FEASIBILITY_AUTHORITY =
cc30cf2b2b92df2f7cf6c18cbe3124f109ed8cc0

FROZEN_BLOCKED_AUDIT_RESULT =
148dbd1105069d0313065246c344fac9b80a28a0

TOKENIZER_CORRECTION_AUTHORITY =
daa5d479f1a040cac308358db711871b0af27020

TOKENIZER_CORRECTION_VALIDATION =
c9b2d16c6b14ddc0a14a0c0e55963640c86ffee2


## 1. Purpose

This authority supersedes exactly one prerequisite in the frozen
cohort-and-measurement feasibility authority:

EXACT_HISTORICAL_TOKENIZER_SNAPSHOT_REQUIRED

is replaced, for A0 token-coordinate reproduction only, by:

EXACT_ACTIVE_ENCODING_SEMANTICS_REQUIRED

All other requirements of:

cc30cf2b2b92df2f7cf6c18cbe3124f109ed8cc0

remain in force unless this document explicitly changes them.


## 2. Historical blocked result remains valid

The frozen blocked audit result:

148dbd1105069d0313065246c344fac9b80a28a0

remains historically correct under its then-active authority.

This supersession does not rewrite that result.

HISTORICAL_TOKENIZER_EXACT_SNAPSHOT =
NOT_RECOVERED

HISTORICAL_WRAPPER_RUNTIME_EQUIVALENCE =
UNRESOLVED_TRANSFORMERS_5_0_0_NOT_LOCALLY_AVAILABLE

These uncertainties remain permanent provenance fields unless separately
resolved by future evidence.


## 3. Correction evidence admitted

The tokenizer active-encoding correction validation is frozen at:

TOKENIZER_CORRECTION_VALIDATION_COMMIT =
c9b2d16c6b14ddc0a14a0c0e55963640c86ffee2

The following exact artifact identities are admitted:

TOKENIZER_FAMILY_IDENTITY_MANIFEST_SHA256 =
34fd4ea20f10a4d16edf38ac9db155c888f38a12b69a96f98f88e109511568d6

STATIC_ACTIVE_COMPONENT_COMPARISON_SHA256 =
b312d1bc7461e8e12ea4e4e7c543d813a781f52101b6583ea8354a732f973e04

DYNAMIC_CONFORMANCE_SUMMARY_SHA256 =
511ef7a94cba5281c162477581f6040b14353c0cd5c5f3b4ba78536dbd509657

MISMATCH_MANIFEST_SHA256 =
c8b5379f8a90070f7aad9f7af9b1a7026c970adecb214bdabf1e317308422385

TOKENIZER_CORRECTION_VALIDATION_REPORT_SHA256 =
fc55198b7a61e5114bbfb3353599c2483af145d5420e2049a1ce0036219c3930

CORRECTION_ARTIFACT_SHA256_MANIFEST_SHA256 =
923cc433cb6b36c6edf5ed68d23862b97b22b99dec7aa6500df28f0268999e0a


## 4. Validated active-encoding result

The admitted validation establishes:

DATASET_ROWS =
3600

DATASET_SHA256 =
eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

VOCAB_DIFF_COUNT =
0

MERGE_DIFF_COUNT =
0

ADDED_TOKEN_DIFF_COUNT =
0

CLAIM_RAW_TOKEN_IDS_EQUAL =
PASS_3600_OF_3600

EVIDENCE_RAW_TOKEN_IDS_EQUAL =
PASS_3600_OF_3600

CLAIM_A0_TOKEN_IDS_EQUAL =
PASS_3600_OF_3600

EVIDENCE_A0_TOKEN_IDS_EQUAL =
PASS_3600_OF_3600

EOS_TOKEN_ID_EQUAL =
PASS

PAD_TOKEN_ID_AFTER_A0_NORMALIZATION_EQUAL =
PASS

SERIALIZED_INPUT_IDS_EQUAL =
PASS_3600_OF_3600

A0_INPUT_COORDINATE_EQUAL =
PASS_3600_OF_3600

MISMATCH_COUNT =
0

ACTIVE_ENCODING_CONTENT_EQUIVALENCE =
PASS

TOKENIZER_PROVENANCE_CRITERION_CORRECTION =
PASS


## 5. Superseded tokenizer criterion

The following parent-authority rule is superseded:

exact historical Hugging Face snapshot revision must be recovered before
token-level tau_e work can proceed.

The replacement rule is:

TOKENIZER_COORDINATE_PROVENANCE_CRITERION =
EXACT_ACTIVE_ENCODING_SEMANTICS_BOUND_TO_VALIDATED_CONTENT

TOKENIZER_PROVENANCE_FEASIBILITY =
PASS_FOR_A0_TOKEN_COORDINATE_WITH_HISTORICAL_REVISION_UNCERTAINTY

TOKEN_COORDINATE_PROVENANCE =
PASS_FOR_FROZEN_A0_ACTIVE_ENCODING_CONTRACT

This PASS applies only to reproducing the frozen A0 token coordinate.

It is not a claim that the exact historical Hugging Face revision is known.


## 6. Why the supersession is admissible

The correction audit established that the two locally observed tokenizer
families that created revision ambiguity are indistinguishable for every
one of the 3600 frozen source rows under the exact A0 active encoding
contract.

The active contract fixes:

A0_MAX_LENGTH =
128

A0_CLAIM_BUDGET =
63

A0_EVIDENCE_BUDGET =
64

A0_ADD_SPECIAL_TOKENS =
FALSE

A0_SEPARATOR =
EOS_TOKEN_ID

A0_PAD_RULE =
IF_MISSING_SET_PAD_TO_EOS

Both families produced identical:

- raw claim token IDs;
- raw evidence token IDs;
- A0-truncated claim token IDs;
- A0-truncated evidence token IDs;
- EOS identity;
- normalized PAD identity;
- serialized input IDs;
- claim length;
- evidence length;
- evidence_start;
- terminal index;
- attention mask;
- claim mask;
- evidence mask.

Therefore historical revision ambiguity does not induce ambiguity in the
frozen dataset's A0 token coordinate.


## 7. Canonical downstream tokenizer reference

For all subsequent feasibility-audit token-coordinate work, one exact
analysis reference is frozen to eliminate procedural ambiguity.

CANONICAL_ANALYSIS_TOKENIZER_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

CANONICAL_ANALYSIS_TOKENIZER_ROLE =
REPRODUCIBLE_ANALYSIS_REFERENCE_ONLY

CANONICAL_ANALYSIS_TOKENIZER_IS_CLAIMED_HISTORICAL_A0_SNAPSHOT =
NO

CANONICAL_TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

CANONICAL_TOKENIZER_CONFIG_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

CANONICAL_SPECIAL_TOKENS_MAP_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

The canonical reference may be used because its frozen A0 token-coordinate
output was validated as identical to the alternate observed family on all
3600 source rows.


## 8. Runtime rule for resumed feasibility work

TOKENIZER_ONLY_CPU_EXECUTION =
ALLOWED

NETWORK_DOWNLOAD =
NO

MODEL_LOAD =
NO

MODEL_FORWARD =
NO

CHECKPOINT_INSTANTIATION =
NO

GPU =
NO

KAGGLE =
NO

The resumed audit may use the frozen local canonical tokenizer content.

It must not claim wrapper-level Transformers 5.0.0 equivalence.

For token-coordinate construction, the validated serialized active encoding
content is authoritative.


## 9. Resumed feasibility phase

After this authority is frozen, the original feasibility audit resumes at:

RESUMED_PHASE =
TAU_E_PHASE_A_SEMANTIC_ANNOTATION_FEASIBILITY

Phase A may read only the inputs allowed by the parent feasibility authority:

- frozen 3600-row dataset;
- id / stable identity;
- pair_id;
- claim;
- evidence;
- intervention_type;
- gold semantic labels;
- primary failure type;
- generator/source code;
- validated regeneration metadata;
- canonical tokenizer content.

Phase A must remain outcome-blind.


## 10. Prediction embargo remains unchanged

Before Phase B tau_e artifacts are byte-identified, the following remain
prohibited:

A0_PREDICTION_ARTIFACT_ACCESS =
NO

PRED_LABEL_ACCESS =
NO

FINAL_PROBS_ACCESS =
NO

FINAL_LOGITS_ACCESS =
NO

CONFIDENCE_DISTRIBUTION_ACCESS =
NO

CORRECT_WRONG_COHORT_COUNT_ACCESS =
NO

The existing historical knowledge that the prediction schema exists does
not authorize inspection of outcome distributions.


## 11. Tau-e requirements remain unchanged

TAU_E =
NOT_YET_RESOLVED

TAU_E_REQUIRES_DETERMINISTIC_SEMANTIC_ANNOTATION =
YES

A valid tau_e must remain:

- conclusion-critical;
- deterministic from allowed semantic inputs;
- mapped to the frozen A0 consumed-token coordinate;
- independent of prediction correctness;
- independent of confidence;
- independent of native-state values;
- auditable;
- actually consumed by the frozen A0 input.

The following remain prohibited as automatic tau_e substitutes:

- evidence_start;
- evidence_end;
- arbitrary midpoint;
- final token;
- O0b absolute token index;
- O0b divergence index;
- O0c anchor index;
- native-state change point.


## 12. Phase ordering remains unchanged

The resumed audit must preserve:

PHASE_A =
SEMANTIC_RULE_AND_ANNOTATION_FEASIBILITY_WITHOUT_PREDICTIONS

then:

PHASE_B =
FREEZE_TAU_E_RULE_CANDIDATE_AND_ANNOTATION_MANIFEST_CANDIDATE

then and only then:

PHASE_C =
JOIN_FROZEN_A0_PREDICTIONS_AND_COMPUTE_COHORT_FEASIBILITY

No Phase C field may influence Phase A or Phase B.


## 13. Required Phase B identities

Before any A0 prediction artifact is opened, the audit must produce and
byte-identify:

TAU_E_RULE_CANDIDATE

and:

TAU_E_ANNOTATION_MANIFEST_CANDIDATE

The annotation manifest must preserve the parent-authority minimum fields:

stable_id
tau_e_status
tau_e_absolute_token_index_or_null
event_type
annotation_basis
evidence_start
terminal_index
post4_prefix_eligible
exclusion_code


## 14. Prefix rule remains unchanged

For each annotated example:

T =
last non-padding consumed token index

POST4_PREFIX_ELIGIBLE =
TRUE

iff:

tau_e + 4 <= T - 1

No shortened POST window is authorized.


## 15. Confidence and cohort rules remain unchanged

CONFIDENCE_STATISTIC =
PREDICTED_CLASS_FINAL_PROBABILITY

CONFIDENCE_THRESHOLD =
0.5

CONFIDENCE_THRESHOLD_SWEEP =
PROHIBITED

DECISIVE_PREDICTION =
pred_label in {SUPPORT, REFUTE}

MINIMUM_MATCHED_PAIR_COUNT =
45

MINIMUM_POTENTIAL_MATCHED_PAIRS_PER_REPRESENTED_PREDICTED_CLASS =
10

These rules may be applied only after Phase B has frozen tau_e artifacts.


## 16. Native-state boundary remains unchanged

SCIENTIFIC_TRAJECTORY_OUTCOMES_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

PRIMARY_KINEMATIC_ENDPOINT_COMPUTATION_ALLOWED =
NO

POST4_SPEED_ALLOWED =
NO

POST4_TURNING_ALLOWED =
NO

POST4_PATH_EFFICIENCY_ALLOWED =
NO

CORRECT_VS_WRONG_NATIVE_STATE_COMPARISON_ALLOWED =
NO

The resumed feasibility audit remains a prerequisite audit, not scientific
execution.


## 17. A0 native-state binding remains separately unresolved

This tokenizer supersession does not itself establish:

A0_NATIVE_STATE_BINDING_FEASIBILITY =
PASS

That feasibility dimension remains to be audited under the parent authority.

Tokenizer coordinate recovery and native recurrent-state checkpoint binding
must remain distinct provenance questions.


## 18. Supersession scope

SUPERSEDED_PARENT_REQUIREMENT =
EXACT_HISTORICAL_TOKENIZER_SNAPSHOT_REQUIRED

REPLACEMENT_REQUIREMENT =
EXACT_ACTIVE_ENCODING_SEMANTICS_REQUIRED

ALL_OTHER_PARENT_FEASIBILITY_REQUIREMENTS =
UNCHANGED

BLOCKED_AUDIT_RESULT_148DBD1 =
PRESERVED_AS_HISTORICAL_RESULT

CORRECTION_VALIDATION_C9B2D16 =
ADMITTED_AS_SUPERSESSION_EVIDENCE


## 19. Authority verdict

AUTHORITY_VERDICT =
READY_FOR_FREEZE_REVIEW

TOKENIZER_GATE =
READY_TO_SUPERSEDE_AFTER_FREEZE

RESUMED_NEXT_PHASE =
TAU_E_PHASE_A_SEMANTIC_ANNOTATION_FEASIBILITY

TRAINING_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

PREDICTION_ACCESS_BEFORE_PHASE_B_FREEZE =
NO

KAGGLE_ALLOWED =
NO

END_OF_NATIVE_STATE_KINEMATICS_FEASIBILITY_TOKENIZER_GATE_SUPERSESSION_AUTHORITY
