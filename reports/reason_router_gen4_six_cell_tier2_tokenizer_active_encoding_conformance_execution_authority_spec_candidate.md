# ContraMamba Gen4 Six-Cell Tier-2 Tokenizer Active-Encoding Conformance Execution Authority - Candidate

## 1. Status

STATUS =
CANDIDATE

PHASE =
R2_TOKENIZER_ACTIVE_ENCODING_GEN4_CONFORMANCE_PREFLIGHT

PARENT_R1_AUTHORITY =
c15c6302df0f4715152ea93b142630adbcff72a7

R1_REPORT_SHA256 =
47054fab5f9c74bd811348cc0ebef6e88f70bfb240aaf86863e0ae5c2484e432

EXECUTION_CLASS =
CPU_ONLY_TOKENIZER_CONTENT_CONFORMANCE

TRAINING =
FORBIDDEN

MODEL_IMPORT =
FORBIDDEN

MODEL_INSTANTIATION =
FORBIDDEN

CHECKPOINT_LOAD =
FORBIDDEN

MODEL_FORWARD =
FORBIDDEN

STATISTICAL_TESTING =
FORBIDDEN

KAGGLE =
NOT_REQUIRED

## 2. Scientific objective

The sole objective of R2 is to determine whether the two already-frozen
tokenizer content families produce identical active model-input coordinates
for every row of the canonical Gen4 six-cell artifact under the exact
historical 128-token input procedure recovered in R1.

R2 does not evaluate any model.

R2 does not estimate any scientific outcome.

## 3. Canonical Gen4 structural input

CANONICAL_GEN4_ARTIFACT_SHA256 =
b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7

CANONICAL_GEN4_ARTIFACT_BYTES =
1465573

CANONICAL_GEN4_ROW_COUNT =
1800

CANONICAL_GEN4_SOURCE_PAIR_COUNT =
300

Required carried identifiers:

row_id
source_pair_id
contrast_cell_id

No text-derived identity reconstruction is permitted.

## 4. Historical input contract from R1

MAX_LENGTH =
128

CLAIM_TOKEN_BUDGET =
63

SEPARATOR_TOKEN_COUNT =
1

EVIDENCE_TOKEN_BUDGET =
64

ADD_SPECIAL_TOKENS =
false

CLAIM_TRUNCATION =
right_to_63

EVIDENCE_TRUNCATION =
right_to_64

SEQUENCE_CONSTRUCTION =
claim_ids + [eos_token_id] + evidence_ids

PADDING_LENGTH =
128

SEPARATOR_IN_CLAIM_MASK =
false

SEPARATOR_IN_EVIDENCE_MASK =
false

ATTENTION_MASK =
true_only_over_nonpadding_sequence

## 5. Tokenizer family A canonical provisioning

FAMILY_A_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

FAMILY_A_SNAPSHOT_SELECTION =
CANONICAL_EXACT_REVISION_REFERENCE

Expected exact files:

tokenizer.json SHA256 =
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

tokenizer_config.json SHA256 =
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

special_tokens_map.json SHA256 =
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8

The execution command must resolve this snapshot under the local Hugging Face
cache and authenticate all three files before tokenizer construction.

## 6. Tokenizer family B canonical provisioning

FAMILY_B_REVISION_REFERENCE =
5708daa364c50b880e7bd92eab456e0d34492ee9

FAMILY_B_SNAPSHOT_SELECTION =
CANONICAL_EXACT_REVISION_REFERENCE

Expected exact files:

tokenizer.json SHA256 =
3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8

tokenizer_config.json SHA256 =
fcd5669efe1150240c13ee4bd863316de4f2abd14cb1806a8cdbcbea6577bc99

special_tokens_map.json SHA256 =
10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c

The execution command must resolve this snapshot under the local Hugging Face
cache and authenticate all three files before tokenizer construction.

## 7. Family B duplicate snapshot resolution

The local provisioning audit also found:

dfd4f103217b8f803ed654f65de0e950d5e9660e

with the same expected Family B active-file SHA256 values.

This is not treated as a second scientific tokenizer family.

It is an exact-content duplicate for the active-file contract.

Therefore:

FAMILY_B_DUPLICATE_SNAPSHOT =
CONTENT_EQUIVALENT_NONCANONICAL

FAMILY_B_EXECUTION_SELECTION =
5708daa364c50b880e7bd92eab456e0d34492ee9_ONLY

No filesystem "newest" or arbitrary directory selection is permitted.

## 8. Runtime mode

Historical run provenance records:

HISTORICAL_TRANSFORMERS_VERSION =
5.0.0

The previous tokenizer provenance correction established that the historical
Transformers 5.0.0 wrapper runtime was not locally available.

Therefore R2 does not claim wrapper-runtime equivalence.

R2 execution mode is:

R2_RUNTIME_MODE =
SERIALIZED_TOKENIZER_CONTENT_FALLBACK

The execution must use the exact tokenizer.json serialized backend content.

It may parse tokenizer_config.json and special_tokens_map.json only for
configuration/provenance checks.

## 9. Permitted tokenizer execution

After this authority is frozen, R2 may:

- import the Python tokenizers package;
- record its runtime version;
- load each exact tokenizer.json locally;
- encode claim and evidence text with special tokens disabled;
- obtain raw token IDs;
- apply the frozen historical right-truncation budgets 63 and 64;
- resolve the EOS token id from exact serialized tokenizer content;
- normalize PAD to EOS where required by the historical procedure;
- construct deterministic 128-position input_ids;
- construct attention_mask;
- construct claim_mask;
- construct evidence_mask;
- compare Family A and Family B results.

No network resolution is permitted.

## 10. Forbidden tokenizer behavior

The R2 execution must not call:

AutoTokenizer.from_pretrained(model_name)

with a mutable model identifier.

It must not download tokenizer files.

It must not call any model configuration or model class.

It must not infer tokenizer identity from model name alone.

It must not modify either local tokenizer snapshot.

## 11. Raw-token comparison

For every one of the 1800 rows R2 must compare:

claim raw token IDs before truncation

evidence raw token IDs before truncation

Required:

CLAIM_RAW_TOKEN_IDS_EQUAL =
PASS_1800_OF_1800

EVIDENCE_RAW_TOKEN_IDS_EQUAL =
PASS_1800_OF_1800

Any mismatch fails R2.

## 12. Truncated-token comparison

For every row R2 must compare:

claim first 63 active IDs

evidence first 64 active IDs

Required:

CLAIM_TRUNCATED_TOKEN_IDS_EQUAL =
PASS_1800_OF_1800

EVIDENCE_TRUNCATED_TOKEN_IDS_EQUAL =
PASS_1800_OF_1800

Any mismatch fails R2.

## 13. Special token contract

Required comparisons:

eos_token_id A versus B

effective pad token id A versus B

Required:

EOS_TOKEN_ID_EQUAL =
PASS

EFFECTIVE_PAD_TOKEN_ID_EQUAL =
PASS

The historical sequence separator is the EOS token ID.

## 14. Final model-coordinate comparison

For every row R2 must construct and compare:

input_ids[128]

attention_mask[128]

claim_mask[128]

evidence_mask[128]

Required:

INPUT_IDS_EQUAL =
PASS_1800_OF_1800

ATTENTION_MASK_EQUAL =
PASS_1800_OF_1800

CLAIM_MASK_EQUAL =
PASS_1800_OF_1800

EVIDENCE_MASK_EQUAL =
PASS_1800_OF_1800

## 15. Aggregate deterministic identity

Each family must serialize the complete 1800-row model-feature coordinate set
in canonical Gen4 input order.

The execution must compute:

FAMILY_A_ALL_SERIALIZED_INPUTS_SHA256

FAMILY_B_ALL_SERIALIZED_INPUTS_SHA256

Required:

FAMILY_SERIALIZED_INPUT_SHA256_EQUAL =
PASS

The exact digest value is not prespecified.

Equality is prespecified.

## 16. Identity preservation

Each comparison row must retain directly:

row_id
source_pair_id
contrast_cell_id

The mismatch manifest must identify rows only through those frozen identifiers.

No fuzzy, semantic, or rendered-text join is permitted.

## 17. Truncation diagnostics

R2 must report:

claim truncation count

evidence truncation count

either-span truncation count

maximum untruncated claim token length

maximum untruncated evidence token length

These diagnostics are descriptive only.

They do not change the PASS criterion.

## 18. Output artifacts

The execution must create a dedicated untracked output directory whose name
contains the eventual frozen R2 execution-authority commit SHA.

Required files:

gen4_tokenizer_conformance_summary.json

gen4_tokenizer_mismatch_manifest.jsonl

gen4_tokenizer_serialized_input_identity.json

The mismatch manifest must exist even when empty.

No output file may be staged automatically.

## 19. PASS criterion

R2 passes only if all are true:

canonical Gen4 input SHA and byte count match

canonical row count is exactly 1800

Family A exact three-file authentication passes

Family B exact three-file authentication passes

canonical snapshot IDs are exactly the frozen revision references

raw claim IDs equal 1800/1800

raw evidence IDs equal 1800/1800

truncated claim IDs equal 1800/1800

truncated evidence IDs equal 1800/1800

EOS token IDs equal

effective PAD token IDs equal

input_ids equal 1800/1800

attention_mask equal 1800/1800

claim_mask equal 1800/1800

evidence_mask equal 1800/1800

serialized aggregate SHA256 values equal

mismatch count is zero

## 20. Fail-closed conditions

R2 must stop without PASS if:

a tokenizer file is missing

a tokenizer file SHA differs

a canonical snapshot revision path is absent

the Gen4 artifact identity differs

the Gen4 row count differs

the tokenizers package is unavailable

serialized tokenizer loading fails

EOS cannot be resolved

raw active IDs differ

final model coordinates differ

any Gen4 identity field is missing or duplicated

network access would be required

## 21. Scope of a PASS

An R2 PASS establishes only:

GEN4_ACTIVE_ENCODING_CONTENT_EQUIVALENCE =
PASS

between the two frozen tokenizer content families under the recovered
historical input procedure.

An R2 PASS does not establish:

HISTORICAL_TRANSFORMERS_5_0_0_WRAPPER_EQUIVALENCE

Therefore:

HISTORICAL_WRAPPER_RUNTIME_EQUIVALENCE =
UNRESOLVED

must remain true after R2 unless separately proven.

## 22. Canonical tokenizer consequence

If and only if R2 passes, the already preferred analysis reference becomes the
canonical procedural tokenizer content for later Gen4 evaluator execution:

GEN4_CANONICAL_TOKENIZER_CONTENT =
FAMILY_A

GEN4_CANONICAL_TOKENIZER_REVISION_REFERENCE =
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

This selection is provenance-based and occurs before model outcome inspection.

## 23. Post-R2 boundary

Even after R2 PASS:

CHECKPOINT_LOAD =
NOT_AUTHORIZED

MODEL_INSTANTIATION =
NOT_AUTHORIZED

MODEL_FORWARD =
NOT_AUTHORIZED

SCIENTIFIC_INFERENCE =
NOT_AUTHORIZED

STATISTICAL_TESTING =
NOT_AUTHORIZED

The next recovery phase remains the dedicated inference-adapter implementation
authority path, followed separately by checkpoint-loadability preflight.

## 24. Current authorization

Before this candidate is frozen:

TOKENIZER_EXECUTION =
NOT_AUTHORIZED

After this exact authority is frozen:

TOKENIZER_EXECUTION =
AUTHORIZED_ONLY_FOR_R2_CPU_CONTENT_CONFORMANCE

NETWORK_ACCESS =
FORBIDDEN

MODEL_EXECUTION =
FORBIDDEN

CHECKPOINT_ACCESS =
FORBIDDEN

TRAINING =
FORBIDDEN

## 25. Result

R2_EXECUTION_AUTHORITY_RESULT =
READY_FOR_FREEZE_REVIEW

NEXT_ACTION_AFTER_FREEZE =
RUN_CPU_ONLY_GEN4_TOKENIZER_ACTIVE_ENCODING_CONFORMANCE_PREFLIGHT
