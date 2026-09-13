# ContraMamba Gen4 Native Mamba State Bridge
# NAME Q1/Q3 Frozen Input-Coordinate Reconstruction
# Provenance Correction Specification
# Candidate

STATUS =
CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_NAME_Q1_Q3_INPUT_COORDINATE_RECONSTRUCTION_PROVENANCE_CORRECTION

THIS_DOCUMENT_CREATES_NEW_SCIENTIFIC_EVIDENCE =
NO

IMPLEMENTATION_ALLOWED =
NO

MODEL_FORWARD_ALLOWED =
NO

CHECKPOINT_LOADING_ALLOWED =
NO

NATIVE_STATE_EXTRACTION_ALLOWED =
NO

STATISTICAL_TESTING_ALLOWED =
NO

TRAINING_ALLOWED =
NO

KAGGLE_ALLOWED =
NO

GPU_ALLOWED =
NO

## 1. Frozen lineage

R2_TOKENIZER_CONFORMANCE_VALIDATED_RESULT =
17f1ddfc8286796f27c4a61716a21e14126bb836

Q1_Q3_SECONDARY_LOCALIZATION_SCIENTIFIC_SPECIFICATION =
01801ad1617b2ebc3ffa859ba440636d4755a55c

Q1_Q3_EXTRACTION_FEASIBILITY_PROVENANCE =
c7d841a920a0a6d075f7f3804da9212ce4706673

PHASE_F_STATUS =
CLOSED

## 2. Scope of correction

The frozen Q1/Q3 feasibility document correctly requires preservation of the
existing active input coordinate and correctly prohibits an alternate
retokenization scheme.

However, its literal execution boundary:

TOKENIZER_EXECUTION_ALLOWED =
NO

together with:

No retokenization is permitted.

is too restrictive for the currently frozen repository provisioning.

The validated R2 lineage froze exact active-encoding identities, but did not
commit the full serialized 1800-row model-input payload as a permanent Git
artifact.

Therefore a future Q1/Q3 runner cannot obtain model input tensors solely by
loading a tracked pre-serialized input artifact.

This document corrects only that provisioning boundary.

SCIENTIFIC_QUESTION_CHANGE =
NO

STRUCTURAL_POPULATION_CHANGE =
NO

SEMANTIC_ANCHOR_CHANGE =
NO

TOKEN_COORDINATE_CHANGE =
NO

Q1_Q3_LAYER_CHANGE =
NO

KINEMATIC_ENDPOINT_CHANGE =
NO

SECONDARY_HYPOTHESIS_FAMILY_CHANGE =
NO

## 3. R2 frozen coordinate identity

R2 established exact active-encoding content equivalence for the canonical
1800-row Gen4 population.

R2_CANONICAL_ROW_COUNT =
1800

R2_CANONICAL_SOURCE_PAIR_COUNT =
300

R2_CANONICAL_TOKENIZER_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

R2_ENCODED_COORDINATE_SHA256 =
d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a

The hash covers the deterministic rowwise coordinate containing:

- row_id;
- source_pair_id;
- contrast_cell_id;
- input_ids;
- attention_mask;
- claim_mask;
- evidence_mask.

This frozen coordinate identity remains authoritative.

## 4. Canonical tokenizer content

The only tokenizer content permitted for reconstruction is the frozen R2
canonical Family A snapshot.

TOKENIZER_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

TOKENIZER_JSON_SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

TOKENIZER_CONFIG_JSON_SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

SPECIAL_TOKENS_MAP_JSON_SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

EOS_TOKEN_ID =
0

EFFECTIVE_PAD_TOKEN_ID =
0

No network lookup, revision substitution, tokenizer fallback, or alternate
tokenizer family is permitted.

## 5. Exact reconstruction procedure

The canonical active encoding remains:

claim token IDs truncated to at most 63
+
one explicit EOS token
+
evidence token IDs truncated to at most 64

with right padding to:

MAX_LENGTH =
128

and:

ADD_SPECIAL_TOKENS =
FALSE

The tokenizer must first encode claim and evidence independently with
automatic special-token insertion disabled.

The existing frozen coordinate construction semantics must not be changed.

CLAIM_BUDGET =
63

EVIDENCE_BUDGET =
64

EOS_SEPARATOR_COUNT =
1

MAX_LENGTH =
128

AUTOMATIC_SPECIAL_TOKENS =
PROHIBITED

## 6. Narrow tokenizer-execution correction

The following bounded operation is permitted in a later implementation and
pre-execution provenance gate:

TOKENIZER_EXECUTION_FOR_FROZEN_COORDINATE_RECONSTRUCTION =
YES_BOUNDED

Its sole purpose is to reconstruct the already-frozen R2 active coordinate.

This is not authority to define a new token coordinate.

This is not authority to inspect alternative tokenizations.

This is not authority to select among tokenizer revisions.

The future implementation must:

1. authenticate all three frozen tokenizer files by exact SHA256;
2. authenticate the canonical 1800-row structural artifact;
3. deterministically encode all 1800 canonical rows under the frozen
   63/1/64 procedure;
4. compute the exact aggregate encoded-coordinate SHA256;
5. require exact equality to:

d3cf61e55bb04e6dbebe44be433597cce958cb15989412dcf6efc15a329c576a

6. fail closed before model construction or scientific forward if the identity
   differs.

COORDINATE_HASH_MISMATCH_BEHAVIOR =
BLOCK_BEFORE_MODEL_FORWARD

## 7. Meaning of retokenization prohibition after correction

The earlier phrase:

No retokenization is permitted.

is superseded only in its literal interpretation as "no tokenizer function may
ever execute."

The preserved scientific prohibition is:

ALTERNATE_RETOKENIZATION =
PROHIBITED

where alternate retokenization includes any change to:

- tokenizer bytes;
- tokenizer revision;
- tokenizer family;
- EOS handling;
- padding semantics;
- claim budget;
- evidence budget;
- max length;
- special-token behavior;
- claim/evidence serialization;
- row ordering;
- row identity;
- active-coordinate semantics.

Exact deterministic regeneration that reproduces the frozen R2 coordinate hash
is:

FROZEN_COORDINATE_RECONSTRUCTION

not a new scientific tokenization.

## 8. Q1/Q3 subset boundary

The coordinate identity must first be validated over the complete canonical
1800-row R2 population.

Only after the full aggregate hash passes may the future Q1/Q3 extraction path
select the frozen secondary scientific subset:

C0_SHAM
C2_NAME

for every source pair.

Q1_Q3_MODEL_INPUT_ROWS =
600

Q1_Q3_SOURCE_PAIRS =
300

Q1_Q3_CELL_SET =
C0_SHAM
C2_NAME

SUBSET_SELECTION_BEFORE_FULL_1800_COORDINATE_VALIDATION =
PROHIBITED

The subset must preserve the canonical row identities and encoded rows exactly.

## 9. Event-coordinate binding

The frozen event manifest remains:

reports/reason_router_gen4_six_cell_native_mamba_state_bridge_feasibility_audit_a2617aa/event_anchor_prefix_manifest_candidate.jsonl

EVENT_MANIFEST_SHA256 =
70c84c68b36751bb7c7145b33ccb71ab91bc8ee9e6cc5f2c7a0d4e925f36581f

EVENT_MANIFEST_BYTES =
2268260

A_NAME / C0_SHAM and A_NAME / C2_NAME event coordinates remain frozen.

The future implementation must cross-check event row identity and anchor token
identity against the reconstructed frozen active coordinate before scientific
forward.

A_NAME_SEMANTIC_COORDINATE_CHANGE =
NO

## 10. Serialized input artifact requirement

A new permanently tracked 1800-row serialized-input artifact is not required
solely to solve this provisioning issue.

NEW_TRACKED_SERIALIZED_INPUT_ARTIFACT_REQUIRED =
NO

The provenance requirement is satisfied only if deterministic reconstruction
reproduces the frozen full-population coordinate SHA exactly.

This does not prohibit a later separately authorized immutable input artifact,
but such an artifact is not required by this correction.

## 11. Execution boundary

This correction does not itself authorize tokenizer execution now.

TOKENIZER_EXECUTION_NOW =
NOT_AUTHORIZED

It permits a later bounded implementation authority to implement and
synthetically/staticly validate the reconstruction path.

It also permits a later scientific extraction execution authority to allow the
exact reconstruction preflight before any model forward, provided all required
implementation and provenance gates have already passed.

MODEL_FORWARD_NOW =
NOT_AUTHORIZED

CHECKPOINT_LOAD_NOW =
NOT_AUTHORIZED

NATIVE_STATE_EXTRACTION_NOW =
NOT_AUTHORIZED

STATISTICAL_TESTING_NOW =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

KAGGLE =
NOT_AUTHORIZED

GPU =
NOT_AUTHORIZED

## 12. Preserved scientific boundary

Q1_LAYER_INDEX =
5

Q3_LAYER_INDEX =
17

STRUCTURAL_ESTIMAND =
DELTA_NAME_ONLY

SEMANTIC_ANCHOR =
A_NAME

KINEMATIC_ENDPOINT_COUNT =
3

SECONDARY_HYPOTHESIS_COUNT =
6

DEPTH_SELECTIVITY =
NOT_ESTABLISHED

OVERALL_ADAPTIVE_PROGRAM_FWER =
NOT_CLAIMED

## 13. Corrected feasibility conclusion

NEW_TOKEN_COORDINATE_REQUIRED =
NO

NEW_TOKENIZER_CONTENT_REQUIRED =
NO

DETERMINISTIC_FROZEN_COORDINATE_RECONSTRUCTION_REQUIRED =
YES

BOUNDED_TOKENIZER_EXECUTION_IN_FUTURE_PROVENANCE_PATH =
ALLOWED_ONLY_IF_FULL_R2_COORDINATE_HASH_MATCHES

ALTERNATE_RETOKENIZATION =
PROHIBITED

SCIENTIFIC_EVIDENCE_CREATED =
NO

NEXT_PHASE =
NAME_Q1_Q3_MEASUREMENT_EXTRACTION_IMPLEMENTATION_AUTHORITY

NEXT_EXECUTION =
NOT_AUTHORIZED
