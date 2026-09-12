# Native State Kinematics
# Cohort and Measurement Feasibility Audit — Bounded Blocked Report

STATUS = CANDIDATE

AUTHORITY_COMMIT =
cc30cf2b2b92df2f7cf6c18cbe3124f109ed8cc0

PARENT_DESIGN_AUTHORITY =
c4286a4d8af9ae31b7e44de2a3e79b560efa2355

AUDIT_PHASE_REACHED =
TOKENIZER_PROVENANCE_GATE

OVERALL_FEASIBILITY =
BLOCKED


## 1. Boundary

SCIENTIFIC_TRAJECTORY_OUTCOMES_ACCESSED =
NO

NATIVE_STATE_EXTRACTION_PERFORMED =
NO

MODEL_FORWARD_PERFORMED =
NO

CHECKPOINT_INFERENCE_PERFORMED =
NO

TRAINING_PERFORMED =
NO

KAGGLE_USED =
NO

CONFIDENCE_DISTRIBUTION_ACCESSED =
NO

CONFIDENT_CORRECT_WRONG_COUNTS_ACCESSED =
NO

TAU_E_TOKEN_ANNOTATION_PERFORMED =
NO

P1_P2_P3_COMPUTED =
NO


## 2. Frozen A0 source binding

SOURCE_PROVENANCE_FEASIBILITY =
PASS

A0_SOURCE_POPULATION =
A0_REPLACEMENT_R1_SEED180_CLEAN_DEV

A0_SPLIT_SEED =
8192

A0_CLEAN_DEV_ROWS =
720

A0_MAIN_DATASET_ROWS =
3600

A0_MAIN_DATASET_SHA256 =
eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3

A0_PREDICTIONS_SHA256 =
5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d

A0_RUN_PROVENANCE_SHA256 =
a758538e93e6e52ca261cb593285c298344808a3626eed7d9b9664e29a6c1a3d

A0_TRAINING_REPORT_SHA256 =
2cdf0925e3a0ef1b925f6b00ac4b2095d18a113896a437ded77622f5134b2013

A0_SELECTED_CHECKPOINT_SHA256 =
4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c

A0_PREDICTION_SOURCE_ID_BINDING =
PASS_720_OF_720


## 3. Confidence availability

CONFIDENCE_FEASIBILITY =
PASS_STATIC_AVAILABILITY_ONLY

CONFIDENCE_STATISTIC =
PREDICTED_CLASS_FINAL_PROBABILITY

FROZEN_CONFIDENCE_THRESHOLD =
0.5

A0_FINAL_OUTPUT_SCHEMA =
PASS_720_OF_720

The frozen prediction export stores:

- stable_id
- gold_label
- pred_label
- external_class_order
- final_logits
- final_probs

No confidence distribution or confident correct/wrong count was inspected
during this blocked audit stage.


## 4. Historical A0 tokenizer loading contract

A0_MODEL_NAME =
state-spaces/mamba-130m-hf

A0_SOURCE_COMMIT =
55debe94f0d19d16a334395e8561901fed6b52fa

A0_TRANSFORMERS_VERSION =
5.0.0

A0_TRAINER_SHA256 =
9792f95df934b8b78cffe07bb7613a35984dba79d56ee7ea719d533dd7117d87

The historical trainer loads:

AutoTokenizer.from_pretrained(args.model_name)

without an explicit tokenizer revision.

The historical trainer then normalizes a missing pad token by assigning
the EOS token as the pad token.

The A0 controlled input encoder tokenizes claim and evidence separately,
with add_special_tokens=False and explicit per-segment truncation budgets,
then inserts EOS as the separator.


## 5. Local tokenizer provenance search

LOCAL_HF_MODEL_CACHE_PRESENT =
YES

LOCAL_SNAPSHOT_COUNT =
6

LOCAL_REFS_PRESENT =
NO

No exact historical snapshot revision was recorded in the A0 run provenance.

At least two distinct tokenizer.json byte families are present locally.


### Family A

TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

OBSERVED_SNAPSHOTS =
1e76775f628fbf1350fbe4dbb3d971ba64af25a1
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37
6e836dba97a56128484e7de8015b4352a1625ec1


### Family B

TOKENIZER_JSON_SHA256 =
3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8

OBSERVED_SNAPSHOTS =
5708daa364c50b880e7bd92eab456e0d34492ee9
dfd4f103217b8f803ed654f65de0e950d5e9660e


## 6. Static content-equivalence finding

TOKENIZER_BYTE_IDENTITY_ACROSS_CACHED_FAMILIES =
NO

TOKENIZER_ACTIVE_ENCODING_CORE_EQUIVALENCE =
PASS_STATIC_COMPARISON

The two observed tokenizer families have identical:

- vocabulary;
- merge table;
- normalizer;
- pre-tokenizer;
- added tokens;
- decoder;
- post-processor.

Observed vocabulary:

VOCAB_SIZE =
50254

VOCAB_DIFF_COUNT =
0

Observed merge table:

MERGE_COUNT =
50009

MERGE_DIFF_COUNT =
0

ADDED_TOKEN_DIFF_COUNT =
0

The observed tokenizer.json differences are:

1. serialized truncation metadata:
   - Family A: Right / LongestFirst / max_length 1024
   - Family B: null

2. pad-token metadata:
   - Family A: pad token explicitly present as <|endoftext|>
   - Family B: pad token absent

The historical A0 trainer explicitly supplies truncation=True and the
per-segment max_length during encode(), so the serialized 1024 truncation
metadata is not the frozen A0 truncation parameter.

The historical trainer also maps a missing pad token to EOS before
controlled encoding.

Therefore the static evidence supports active-encoding semantic
equivalence for the inspected differences.

However, this equivalence is not accepted as a substitute for exact
historical tokenizer identity under the currently frozen authority.


## 7. Tokenizer provenance verdict

HISTORICAL_TOKENIZER_EXACT_SNAPSHOT =
NOT_RECOVERED

TOKENIZER_PROVENANCE_FEASIBILITY =
BLOCKED

TOKEN_COORDINATE_PROVENANCE =
BLOCKED

Reason:

the frozen feasibility authority requires exact tokenizer
identity/revision recovery before token-level tau_e annotation or
prefix eligibility is accepted.

Neither the frozen A0 run provenance nor the local HF cache binds the
historical A0 execution to exactly one snapshot revision.

The existence of multiple locally cached revisions must not be used to
guess the historical revision.


## 8. Downstream gates

TAU_E_ANNOTATION_FEASIBILITY =
NOT_REACHED_DUE_TO_TOKENIZER_PROVENANCE_BLOCK

POST4_PREFIX_FEASIBILITY =
NOT_REACHED_DUE_TO_TOKENIZER_PROVENANCE_BLOCK

CONFIDENT_ERROR_COHORT_FEASIBILITY =
NOT_REACHED_DUE_TO_TOKENIZER_PROVENANCE_BLOCK

MATCHING_FEASIBILITY =
NOT_REACHED_DUE_TO_TOKENIZER_PROVENANCE_BLOCK

PROSPECTIVE_POWER_FEASIBILITY =
NOT_REACHED_DUE_TO_TOKENIZER_PROVENANCE_BLOCK

A0_NATIVE_STATE_BINDING_FEASIBILITY =
NOT_REACHED_DUE_TO_TOKENIZER_PROVENANCE_BLOCK

No later gate is interpreted as PASS or FAIL.


## 9. Scientific interpretation boundary

This blocked result does not test the native-state kinematics hypothesis.

It does not support:

NATIVE_KINEMATICS_PRECURSOR_SUPPORTED

and it does not support:

NATIVE_KINEMATICS_PRECURSOR_NOT_SUPPORTED

The blocker is provenance/design infrastructure only.


## 10. No silent repair

This audit must not proceed by silently treating tokenizer
content-equivalence as exact historical revision recovery.

It must not:

- choose one cached snapshot by convenience;
- choose the newest or oldest cached snapshot;
- infer the historical revision from local cache ordering;
- run tau_e annotation despite the frozen exact-revision requirement;
- inspect confident correct/wrong counts before Phase B;
- relax the frozen authority in place.


## 11. Bounded next research action

A new narrow authority may evaluate whether the frozen tokenizer
provenance criterion should be corrected from:

EXACT_HISTORICAL_SNAPSHOT_REQUIRED

to a content-bound criterion such as:

EXACT_ACTIVE_ENCODING_SEMANTICS_REQUIRED

Such a correction must:

1. remain independent of prediction correctness and confidence;
2. remain independent of native-state outcomes;
3. bind exact tokenizer-defining content hashes;
4. prove that all differences outside the bound content are inactive
   under the historical A0 encoding contract;
5. preserve the A0 claim/evidence truncation and EOS-separator contract;
6. remain fail-closed if active encoding semantics differ;
7. be frozen before tokenizer execution, tau_e annotation, or cohort join.

This blocked audit itself does not authorize that correction.


## 12. Final verdict

SOURCE_PROVENANCE_FEASIBILITY =
PASS

CONFIDENCE_FEASIBILITY =
PASS_STATIC_AVAILABILITY_ONLY

TOKENIZER_PROVENANCE_FEASIBILITY =
BLOCKED

TAU_E_ANNOTATION_FEASIBILITY =
NOT_REACHED

POST4_PREFIX_FEASIBILITY =
NOT_REACHED

CONFIDENT_ERROR_COHORT_FEASIBILITY =
NOT_REACHED

MATCHING_FEASIBILITY =
NOT_REACHED

PROSPECTIVE_POWER_FEASIBILITY =
NOT_REACHED

A0_NATIVE_STATE_BINDING_FEASIBILITY =
NOT_REACHED

OVERALL_FEASIBILITY =
BLOCKED

BLOCK_REASON =
EXACT_HISTORICAL_TOKENIZER_REVISION_NOT_RECOVERED

END_OF_NATIVE_STATE_KINEMATICS_FEASIBILITY_BLOCKED_REPORT
