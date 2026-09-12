# K0-RVG Post-P2 Active-Encoding Token-Coordinate Equivalence Minimal Read-Only Execution Authority

**Status:** one-time real equivalence audit authority candidate only.

**Date:** 2026-09-12

## Purpose

Authorize exactly one bounded real 336-item audit whose sole question is:

Does the frozen active encoding reproduce every persisted P0 token-coordinate
contract invariant exactly, while deterministically freezing the resulting
active token evidence?

This is a provenance/measurement-coordinate audit.

It is not the post-P2 matched-versus-swapped scientific token-relation audit.

## Frozen implementation

`ACTIVE_ENCODING_EQUIVALENCE_IMPLEMENTATION_COMMIT = ebdecf9f2cc833e5e5845f296cdf2685040e28d0`

`ACTIVE_ENCODING_EQUIVALENCE_IMPLEMENTATION_SHA256 = 7bd5a30c502a2e1f462256ebe76e8001d36e8e94a389579a1c80d0a84413c1dc`

`ACTIVE_ENCODING_EQUIVALENCE_IMPLEMENTATION_GIT_BLOB = 5826b174aa6d71d2be3238450b31c90ae37d70f9`

`ACTIVE_ENCODING_EQUIVALENCE_TEST_SHA256 = cf378c9a3f14d743511c485f6d099a92a3c537cd26de84681775fa29a021b331`

`ACTIVE_ENCODING_EQUIVALENCE_TEST_GIT_BLOB = fa49820d8ea20ac2177e204f13d0842e8693f65a`

Validated before freeze:

`PY_COMPILE = PASS`

`FOCUSED_TEST_RESULT = 27_PASSED`

`SYNTHETIC_SELF_CHECK = PASS_SYNTHETIC_ACTIVE_ENCODING_TOKEN_COORDINATE_EQUIVALENCE_CORE`

## Historical provenance boundary

`HISTORICAL_P1_BYTE_IDENTITY_STATUS = HISTORICAL_P1_BYTE_IDENTITY_NOT_ESTABLISHED`

`HISTORICAL_P1_BYTE_IDENTITY_RECOVERY = NOT_RECOVERABLE_FROM_FROZEN_REPOSITORY_EVIDENCE`

This execution must not alter either marker.

A PASS establishes only active-encoding/token-coordinate equivalence against
persisted frozen P0 invariants.

It does not recover historical tokenizer bytes or unavailable historical
full branch token sequences.

## Active encoding identity

`ACTIVE_TOKENIZER_MODEL_ID = state-spaces/mamba-130m-hf`

`ACTIVE_TOKENIZER_REVISION_LABEL = 5708daa364c50b880e7bd92eab456e0d34492ee9`

`ACTIVE_TOKENIZER_TRANSFORMERS_VERSION = 5.12.1`

`ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256 = f45af0fad1ae940487eb6461c65f13cf1634b6141f4e4e03b6c88bc38fa6db9c`

The active manifest's previously frozen provenance status remains
`PARTIAL_LOCAL_SNAPSHOT_PROVENANCE`.

That status is not upgraded by this authority.

The manifest is used here only to freeze the exact active encoding bytes
whose behavioral equivalence to persisted P0 token-coordinate invariants is
being tested.

## Exact success criterion

PASS is permitted only if all 336 items satisfy every exact persisted
token-coordinate invariant enforced by the frozen implementation.

`REQUIRED_EXACT_ITEM_PASS_COUNT = 336`

`REQUIRED_EXACT_ITEM_FAILURE_COUNT = 0`

`PERMITTED_PASS_VERDICT = PASS_ACTIVE_ENCODING_TOKEN_COORDINATE_EQUIVALENCE`

Any exact mismatch must result in:

`PERMITTED_BLOCKED_VERDICT = BLOCKED_ACTIVE_ENCODING_TOKEN_COORDINATE_EQUIVALENCE`

No tolerance, repair, majority rule, or post-hoc exclusion is authorized.

## Authorized real reads/actions

`REAL_ACTIVE_ENCODING_EQUIVALENCE_AUDIT_AUTHORIZED = YES`

`REAL_P0_SCIENTIFIC_INPUT_READ_AUTHORIZED = YES`

`TOKENIZER_REEXECUTION_AUTHORIZED = YES`

`REAL_HF_TOKENIZER_LOAD_AUTHORIZED = YES`

The tokenizer must be loaded only from the already-present authenticated
local snapshot bytes, with the frozen manifest and frozen Transformers
version enforced by the implementation.

`NETWORK_ACCESS_AUTHORIZED = NO`

`SNAPSHOT_DOWNLOAD_AUTHORIZED = NO`

## Scientific noninterference

`REAL_P2_ARTIFACT_READ_AUTHORIZED = NO`

`POST_P2_TOKEN_WINDOW_REAL_ARTIFACT_EXECUTION_AUTHORIZED = NO`

`SCIENTIFIC_TOKEN_RELATION_CLASSIFICATION_AUTHORIZED = NO`

`MODEL_CONSTRUCTION_AUTHORIZED = NO`

`CHECKPOINT_LOADING_AUTHORIZED = NO`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`EVALUATION_AUTHORIZED = NO`

`KAGGLE_EXECUTION_AUTHORIZED = NO`

The output may contain active token IDs, their canonical hashes, token
counts, event anchors, and event-relative coordinates only as provenance
evidence.

It must not interpret matched-versus-swapped token difference, token-window
identity, state relation, collapse, convergence, or transition mechanism.

## One-time execution

`ONE_TIME_REAL_EQUIVALENCE_EXECUTION_AUTHORIZED = YES`

This authority is consumed by one attempted real execution after successful
authority authentication.

No retry under this authority is authorized if execution reaches real P0
input reading.

A pre-input authority/authentication failure does not consume the scientific
read authorization.

## Output

`AUTHORIZED_OUTPUT_DIRECTORY = reports/longterm_k0_rvg_post_p2_active_encoding_token_coordinate_equivalence_ebdecf9_v1`

`AUTHORIZED_OUTPUT_NAME = active_encoding_token_coordinate_equivalence.json`

Exactly one canonical JSON artifact is authorized.

## Next boundary

If and only if the artifact validates as:

`PASS_ACTIVE_ENCODING_TOKEN_COORDINATE_EQUIVALENCE`

then:

`NEXT_TOKEN_WINDOW_SCIENCE_PROVENANCE_PREREQUISITE = SATISFIED_BY_ACTIVE_ENCODING_EQUIVALENCE`

and the next action is the already-defined matched/swapped token-window
relation science using the frozen active token evidence and frozen raw
state evidence.

No further historical tokenizer-byte archaeology is authorized.

`NEXT_BOUNDARY = K0_RVG_POST_P2_ACTIVE_ENCODING_TOKEN_COORDINATE_EQUIVALENCE_REAL_EXECUTION`