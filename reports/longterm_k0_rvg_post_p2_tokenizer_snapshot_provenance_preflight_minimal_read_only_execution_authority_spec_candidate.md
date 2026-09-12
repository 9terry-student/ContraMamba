# K0-RVG Post-P2 Tokenizer Snapshot Provenance Preflight Minimal Read-Only Execution Authority

**Status:** one-time execution-authority candidate only.

**Date:** 2026-09-12

## Scope

Authorize exactly one bounded read-only provenance preflight against the local Hugging Face cache for the frozen tokenizer identity:

- model: `state-spaces/mamba-130m-hf`
- revision: `5708daa364c50b880e7bd92eab456e0d34492ee9`
- expected Transformers version identity: `5.12.1`

The execution may:

- inspect the exact local model-cache namespace;
- inspect the exact frozen-revision snapshot directory;
- enumerate only the frozen tokenizer/config file family;
- read those selected local file bytes;
- compute per-file raw SHA256;
- inspect local symlink/content-addressed cache structure;
- compute the canonical tokenizer snapshot manifest SHA256;
- write one canonical provenance JSON artifact.

This authority does not authorize tokenizer construction, tokenization, scientific artifact reads, model execution, network access, download, training, evaluation, Kaggle, or scientific interpretation.

Missing snapshot, namespace mismatch, revision mismatch, content-address mismatch, or any required provenance failure must fail closed. No network fallback is permitted.

Historical P1 byte identity remains a separate proposition and must not be inferred merely from a locally valid current snapshot.

## Frozen bindings

`PARENT_VALIDATION_COMMIT = ae9f8a5ab7d8a942dd3be6e1f95e9a54db8555c2`

`PARENT_VALIDATION_GIT_BLOB = 4a18f0efb55e3c29231e7319f71467c2b856f98b`

`TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_IMPLEMENTATION_COMMIT = d54c57f286a8147808fd0c950ea8dd4896b33b5c`

`TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_IMPLEMENTATION_SHA256 = 25029fa2127ea892595ec09be6078765aef70029659ee2c42c7388b90eb8aef6`

`TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_IMPLEMENTATION_GIT_BLOB = 7f91d9415f07947476ca59814c086c8aac8ed9cc`

`TOKENIZER_MODEL_ID = state-spaces/mamba-130m-hf`

`TOKENIZER_REVISION = 5708daa364c50b880e7bd92eab456e0d34492ee9`

`TOKENIZER_TRANSFORMERS_VERSION = 5.12.1`

## Authorization markers

`TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_REAL_CACHE_EXECUTION_AUTHORIZED = YES`

`LOCAL_REAL_TOKENIZER_CACHE_METADATA_READ_AUTHORIZED = YES`

`LOCAL_REAL_TOKENIZER_FILE_BYTE_READ_AUTHORIZED = YES`

`REAL_TOKENIZER_FILE_SHA256_AUTHORIZED = YES`

`ONE_TIME_READ_ONLY_EXECUTION_AUTHORIZED = YES`

`REAL_HF_TOKENIZER_LOAD_AUTHORIZED = NO`

`TOKENIZER_REEXECUTION_AUTHORIZED = NO`

`REAL_P0_SCIENTIFIC_TOKEN_AUDIT_AUTHORIZED = NO`

`REAL_P2_CROSS_LEVEL_AUDIT_AUTHORIZED = NO`

`NETWORK_ACCESS_AUTHORIZED = NO`

`SNAPSHOT_DOWNLOAD_AUTHORIZED = NO`

`MODEL_CONSTRUCTION_AUTHORIZED = NO`

`CHECKPOINT_LOADING_AUTHORIZED = NO`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`EVALUATION_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`KAGGLE_EXECUTION_AUTHORIZED = NO`

## Output

`AUTHORIZED_OUTPUT_RELATIVE_PATH = reports/longterm_k0_rvg_post_p2_tokenizer_snapshot_provenance_preflight_ae9f8a5_v1/tokenizer_snapshot_provenance_preflight.json`

`AUTHORIZED_OUTPUT_SCHEMA = k0-rvg-post-p2-tokenizer-snapshot-provenance-preflight-v1`

No other real-cache artifact is authorized.

## Interpretation boundary

Permitted provenance verdicts remain:

- `PASS_REVISION_BOUND_LOCAL_SNAPSHOT_MANIFEST`
- `PARTIAL_LOCAL_SNAPSHOT_PROVENANCE`
- `BLOCKED_LOCAL_TOKENIZER_SNAPSHOT_PROVENANCE`

These are provenance verdicts only.

They do not establish matched/swapped token relations or any Mamba-state scientific claim.

`HISTORICAL_P1_BYTE_IDENTITY_STATUS = HISTORICAL_P1_BYTE_IDENTITY_NOT_ESTABLISHED`

`NEXT_BOUNDARY = K0_RVG_POST_P2_ONE_TIME_READ_ONLY_TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_EXECUTION`