# Native State Kinematics
# Tokenizer Active-Encoding Provenance Correction Authority Specification

STATUS = CANDIDATE

PHASE =
TOKENIZER_ACTIVE_ENCODING_PROVENANCE_CORRECTION

PARENT_BLOCKED_AUDIT_RESULT =
148dbd1105069d0313065246c344fac9b80a28a0

PARENT_FEASIBILITY_AUTHORITY =
cc30cf2b2b92df2f7cf6c18cbe3124f109ed8cc0

PARENT_DESIGN_AUTHORITY =
c4286a4d8af9ae31b7e44de2a3e79b560efa2355


## 1. Purpose

The frozen feasibility audit is blocked because the exact historical
Hugging Face tokenizer snapshot revision used by the A0 execution was not
recorded and cannot presently be recovered.

Static inspection nevertheless found two locally cached tokenizer byte
families whose active encoding core is identical.

This authority permits one bounded correction audit to determine whether
the provenance requirement may be changed from:

EXACT_HISTORICAL_SNAPSHOT_REQUIRED

to:

EXACT_ACTIVE_ENCODING_SEMANTICS_REQUIRED

for the sole purpose of reproducing the frozen A0 token coordinate.


## 2. Scientific boundary

THIS_AUTHORITY_CREATES_SCIENTIFIC_TRAJECTORY_EVIDENCE =
NO

TRAINING_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_INFERENCE_ALLOWED =
NO

CHECKPOINT_INSTANTIATION_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

P1_P2_P3_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_ALLOWED =
NO

PREDICTION_ARTIFACT_ACCESS_ALLOWED =
NO

CONFIDENCE_DISTRIBUTION_ACCESS_ALLOWED =
NO

CORRECT_WRONG_COUNT_ACCESS_ALLOWED =
NO

TAU_E_ANNOTATION_ALLOWED =
NO

MATCHING_ALLOWED =
NO

Only tokenizer/provenance validation is authorized.


## 3. Frozen A0 contract

A0_SOURCE_COMMIT =
55debe94f0d19d16a334395e8561901fed6b52fa

A0_TRAINER_SHA256 =
9792f95df934b8b78cffe07bb7613a35984dba79d56ee7ea719d533dd7117d87

A0_MODEL_NAME =
state-spaces/mamba-130m-hf

A0_RECORDED_TRANSFORMERS_VERSION =
5.0.0

A0_MAX_LENGTH =
128

A0_CLAIM_BUDGET =
63

A0_EVIDENCE_BUDGET =
64

A0_ADD_SPECIAL_TOKENS =
FALSE

A0_SEPARATOR =
TOKENIZER_EOS_TOKEN_ID

A0_PAD_NORMALIZATION =
IF_PAD_MISSING_SET_PAD_TO_EOS

A0 serialization is:

claim_ids[:63]
+
[eos_token_id]
+
evidence_ids[:64]

with claim and evidence tokenized separately.


## 4. Frozen dataset identity

A0_MAIN_DATASET_PATH =
reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl

A0_MAIN_DATASET_SHA256 =
eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

A0_MAIN_DATASET_ROWS =
3600

The correction audit may read:

- id;
- pair_id;
- claim;
- evidence;
- intervention_type;
- semantic dataset metadata needed only for provenance.

It must not read any A0 prediction artifact.


## 5. Historical revision status

HISTORICAL_TOKENIZER_EXACT_SNAPSHOT =
NOT_RECOVERED

This fact must remain explicitly recorded.

The correction must not claim that the historical snapshot revision has
been recovered.


## 6. Observed tokenizer family A

FAMILY_A_REFERENCE_SNAPSHOT =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

FAMILY_A_TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

FAMILY_A_TOKENIZER_CONFIG_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

FAMILY_A_SPECIAL_TOKENS_MAP_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8


## 7. Observed tokenizer family B

FAMILY_B_REFERENCE_SNAPSHOT =
5708daa364c50b880e7bd92eab456e0d34492ee9

FAMILY_B_TOKENIZER_JSON_SHA256 =
3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8

FAMILY_B_TOKENIZER_CONFIG_SHA256 =
fcd5669efe1150240c13ee4bd863316de4f2abd14cb1806a8cdbcbea6577bc99

FAMILY_B_SPECIAL_TOKENS_MAP_SHA256 =
10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0


## 8. Static equivalence already observed

The correction audit must independently reproduce and record the following
static facts.

VOCAB_SIZE =
50254

VOCAB_DIFF_COUNT =
0

MERGE_COUNT =
50009

MERGE_DIFF_COUNT =
0

ADDED_TOKEN_DIFF_COUNT =
0

Required equality:

- model vocabulary;
- merge ordering;
- normalizer;
- pre-tokenizer;
- added tokens;
- decoder;
- post-processor.

Observed non-identical metadata:

1. serialized tokenizer truncation state;
2. pad-token declaration.

These differences may be classified as inactive only if the frozen A0
trainer contract proves that it overrides them.


## 9. Inactive-difference proof obligations

The correction audit must prove both:

### 9.1 Truncation

The historical A0 code explicitly supplies:

truncation=True

and an explicit max_length separately for claim and evidence.

Therefore a serialized tokenizer-side default truncation state must not be
treated as controlling A0 tokenization.

### 9.2 Padding

The historical A0 code performs:

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

before encoding.

The controlled encoder manually creates padded tensors after active token
IDs have been constructed.

Therefore absent versus predeclared PAD metadata may be classified as
inactive only when both resolve to the same EOS/PAD ID.


## 10. Runtime compatibility prerequisite

The preferred dynamic conformance environment is:

transformers == 5.0.0

matching the recorded A0 runtime.

The audit must first inspect the locally available runtime.

NETWORK_INSTALLATION_ALLOWED =
NO

If transformers 5.0.0 is locally available, it must be used.

If it is not locally available, the correction audit must not silently use
a different Transformers version as the sole proof of equivalence.

In that case it may still perform a version-independent serialized-tokenizer
conformance check, but the final correction verdict must separately state
whether wrapper-level historical runtime equivalence remains unresolved.


## 11. Dynamic tokenization conformance

Tokenizer-only CPU execution is authorized for this correction audit.

TOKENIZER_ONLY_CPU_EXECUTION_ALLOWED =
YES

NETWORK_ACCESS_ALLOWED =
NO

MODEL_LOAD_ALLOWED =
NO

Only the two frozen local tokenizer family references may be loaded.

For every one of the 3600 frozen dataset rows, the audit must compare
Family A and Family B for both claim and evidence.


### 11.1 Raw active token sequence

With:

add_special_tokens=False

the full active token ID sequence must match exactly before A0 truncation.

Required:

CLAIM_RAW_TOKEN_IDS_EQUAL =
PASS_3600_OF_3600

EVIDENCE_RAW_TOKEN_IDS_EQUAL =
PASS_3600_OF_3600


### 11.2 A0 explicit truncation

Using the historical explicit budgets:

claim max_length = 63
evidence max_length = 64

the final claim/evidence IDs must match exactly.

Required:

CLAIM_A0_TOKEN_IDS_EQUAL =
PASS_3600_OF_3600

EVIDENCE_A0_TOKEN_IDS_EQUAL =
PASS_3600_OF_3600


### 11.3 Special-token identity

Required equality after historical pad normalization:

EOS_TOKEN_ID_EQUAL =
PASS

PAD_TOKEN_ID_AFTER_A0_NORMALIZATION_EQUAL =
PASS

The EOS token ID must be the same ID used as the A0 separator.


### 11.4 Serialized controlled input

For each row construct:

claim_ids
+
[eos_token_id]
+
evidence_ids

using the frozen budgets.

Required:

SERIALIZED_INPUT_IDS_EQUAL =
PASS_3600_OF_3600


### 11.5 Coordinate-derived metadata

The following must also be identical for every row:

- claim length;
- evidence length;
- evidence_start;
- terminal non-padding index;
- attention mask;
- claim mask;
- evidence mask.

Required:

A0_INPUT_COORDINATE_EQUAL =
PASS_3600_OF_3600


## 12. No semantic-event annotation

This correction audit may compute only token-coordinate infrastructure.

It must not identify:

- decisive evidence;
- conclusion-critical token;
- tau_e;
- POST4 eligibility relative to tau_e.

Evidence start is not tau_e.


## 13. No prediction join

The following files remain prohibited during this correction audit:

clean_dev_predictions.json
training_report_predictions.jsonl

The correction audit may not inspect:

pred_label
final_probs
final_logits
is_correct
confidence
correct/wrong cohort counts


## 14. Correction PASS criterion

TOKENIZER_PROVENANCE_CRITERION_CORRECTION =
PASS

requires all of:

1. Family A and B exact frozen byte identities pass;
2. vocabulary equality passes;
3. merge equality passes;
4. normalizer equality passes;
5. pre-tokenizer equality passes;
6. added-token equality passes;
7. EOS identity passes;
8. historical PAD normalization removes the observed PAD metadata difference;
9. historical explicit truncation removes the observed default-truncation difference;
10. raw claim token IDs match for all 3600 rows;
11. raw evidence token IDs match for all 3600 rows;
12. A0-truncated claim IDs match for all 3600 rows;
13. A0-truncated evidence IDs match for all 3600 rows;
14. serialized A0 input IDs match for all 3600 rows;
15. all A0 coordinate metadata matches for all 3600 rows;
16. no prediction artifact or native-state result has been accessed.

If wrapper-level Transformers 5.0.0 validation cannot be executed,
the final report must explicitly distinguish:

ACTIVE_ENCODING_CONTENT_EQUIVALENCE

from:

HISTORICAL_WRAPPER_RUNTIME_EQUIVALENCE

and may not overstate the latter.


## 15. Meaning of a correction PASS

A PASS does not recover the historical Hugging Face revision.

Instead it establishes a narrower provenance claim:

the observed locally available tokenizer families that could otherwise
create revision ambiguity are indistinguishable under the frozen A0 active
encoding contract for the entire frozen 3600-row source dataset.

After a validated PASS, a new superseding feasibility authority may replace
the exact-snapshot gate with:

EXACT_ACTIVE_ENCODING_SEMANTICS_REQUIRED

while preserving the historical-revision uncertainty in provenance.


## 16. Correction FAIL / BLOCKED criterion

The correction must FAIL or BLOCK if any of the following occurs:

- vocabulary differs;
- merge ordering differs;
- normalizer differs;
- pre-tokenizer differs;
- added tokens differ;
- EOS ID differs;
- pad normalization resolves differently;
- any raw claim token IDs differ;
- any raw evidence token IDs differ;
- any A0-truncated token IDs differ;
- any serialized input differs;
- any input coordinate or mask differs;
- a required tokenizer file identity changes;
- a network download would be required for the chosen proof;
- prediction artifacts are accessed.

No mismatch may be repaired by selecting the better-looking family.


## 17. Required correction artifacts

A completed correction audit should produce:

1. tokenizer family identity manifest;
2. static active-component comparison manifest;
3. 3600-row dynamic conformance summary;
4. mismatch manifest, empty on PASS;
5. tokenizer provenance correction validation report candidate;
6. SHA256 identities for all generated artifacts.

No artifact from this correction audit is native-state scientific evidence.


## 18. Relationship to the blocked feasibility audit

The frozen blocked result at:

148dbd1105069d0313065246c344fac9b80a28a0

remains historically correct.

This authority does not edit or reinterpret that result in place.

If the correction passes, a new authority must explicitly supersede only
the tokenizer provenance criterion before the feasibility audit resumes.

All other frozen feasibility requirements remain unchanged, including:

- confidence threshold 0.5;
- tau_e anti-leakage ordering;
- POST4 requirement;
- matching requirements;
- 45-pair prospective power gate;
- native-state scientific outcome prohibition.


## 19. Current authority verdict

AUTHORITY_VERDICT =
READY_FOR_FREEZE_REVIEW

CORRECTION_QUESTION =
CAN_EXACT_ACTIVE_ENCODING_SEMANTICS_REPLACE_EXACT_HISTORICAL_SNAPSHOT_FOR_A0_TOKEN_COORDINATE_PROVENANCE

TOKENIZER_ONLY_CPU_EXECUTION_ALLOWED =
YES

PREDICTION_ARTIFACT_ACCESS_ALLOWED =
NO

TAU_E_ANNOTATION_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

END_OF_NATIVE_STATE_KINEMATICS_TOKENIZER_ACTIVE_ENCODING_PROVENANCE_CORRECTION_AUTHORITY
