# K0-RVG Post-P2 Tokenizer Snapshot Provenance Preflight Implementation Authority Specification Candidate

**Status:** implementation-authority specification candidate only.

**Date:** 2026-09-12

**Immediate parent / frozen implementation-validation commit:**

`50e06ff4bcc4f176696565f0e91d8b15cd731bbb`

**Frozen implementation-validation report:**

`reports/longterm_k0_rvg_post_p2_token_window_static_audit_implementation_validation_readiness_report_candidate.md`

**Frozen implementation-validation Git blob:**

`0dd64aa0d6f2f875672fea6aca96ff39a68d7322`

This document authorizes only a bounded implementation and synthetic validation of a tokenizer-snapshot provenance preflight.

It does not authorize reading the user's real Hugging Face tokenizer cache, hashing real tokenizer snapshot files, loading a tokenizer, tokenizing scientific text, reading real P0/P2 scientific artifacts, model construction, checkpoint loading, model forward, recurrent-state read, logits read, training, evaluation, network access, snapshot download, Kaggle, or scientific interpretation.

The purpose of this stage is to prepare a deterministic and fail-closed mechanism for a later separately authorized read-only provenance preflight.

## 1. Active scientific line and phase

`ACTIVE_SCIENTIFIC_LINE = K0_RVG_RAW_NATIVE_VECTOR_GEOMETRY`

`ACTIVE_STAGE = K0_RVG_POST_P2_TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_IMPLEMENTATION`

`ACTIVE_PHASE = IMPLEMENTATION_AND_SYNTHETIC_VALIDATION_ONLY`

The unresolved scientific quantity remains the post-P2 matched/swapped token-window relation.

No token relation is measured in this phase.

## 2. Authority source and immediate boundary

The immediate authority source is the frozen implementation-validation/readiness commit:

`50e06ff4bcc4f176696565f0e91d8b15cd731bbb`

That report establishes:

- the post-P2 token-window implementation is code-correct for the frozen static contract;
- no real tokenizer execution has occurred;
- no real post-P2 token audit has occurred;
- a future execution authority requires an exact tokenizer snapshot manifest digest;
- the next boundary is tokenizer snapshot provenance preflight.

This document narrows that boundary further to implementation and synthetic validation only.

Real local-cache provenance execution requires a later separately frozen authority.

## 3. Frozen post-P2 implementation binding

Effective implementation commit:

`5cfc9a61d8fe534e7bb6a037bb075bbe10092b74`

Runner:

`scripts/longterm_k0_rvg_post_p2_token_window_static_audit.py`

Runner SHA256:

`83390903028682b4d802ffab118fd16e5c8125305ce2a27a78cf829c504e0f19`

Runner Git blob:

`561bbcf8d34970cee22e4cbf40c1c814dfc31f50`

The existing runner must not be modified by this stage.

Its tokenizer snapshot manifest semantics are frozen reference semantics for the preflight implementation.

## 4. Frozen P1 source binding

Frozen P1 branch/tokenizer source commit:

`50a1daa781e47d1c0f1ba158beb445878e049a65`

File:

`scripts/longterm_k0_rvg_p1_raw_vector_execution.py`

Git blob:

`1f70bfe36ed0efa9014d0a47471241febcfadcf4`

The preflight implementation may reproduce only state-blind tokenizer snapshot provenance semantics from this frozen source.

It must not import or execute P1 scientific model-running logic.

## 5. Frozen tokenizer identity

The exact frozen tokenizer identity is:

`HF_MODEL = state-spaces/mamba-130m-hf`

`HF_REVISION = 5708daa364c50b880e7bd92eab456e0d34492ee9`

`TRANSFORMERS_VERSION = 5.12.1`

These identifiers constrain the future provenance target.

They do not, by themselves, establish tokenizer byte identity.

## 6. Existing frozen manifest semantics

The effective post-P2 runner defines the tokenizer/config file family as:

- `config.json`
- `tokenizer*`
- `special_tokens_map.json`
- `vocab.*`
- `merges.txt`

For an authorized snapshot directory, the frozen runner:

1. recursively enumerates regular files;
2. retains only files matching the frozen file-family patterns;
3. hashes file contents with SHA256;
4. keys hashes by snapshot-relative POSIX path;
5. sorts paths deterministically;
6. serializes the resulting mapping as canonical UTF-8 JSON with sorted keys and compact separators;
7. computes SHA256 over those canonical JSON bytes.

The preflight implementation must produce an exactly equivalent manifest digest on synthetic fixtures.

It must not define a competing manifest algorithm.

## 7. Why a dedicated preflight implementation is required

The production token-window runner performs provenance authentication as part of a later real audit path.

That real path is intentionally gated and also contains real P0/P2 scientific-input reads.

This provenance stage must not invoke that production path.

A dedicated preflight implementation is therefore authorized so that tokenizer snapshot provenance can later be inspected independently of:

- P0 scientific artifact reads;
- P2 scientific artifact reads;
- tokenizer construction;
- tokenizer execution;
- model execution.

## 8. Authorized implementation files

Exactly two new files are authorized:

`scripts/longterm_k0_rvg_post_p2_tokenizer_snapshot_provenance_preflight.py`

`tests/test_longterm_k0_rvg_post_p2_tokenizer_snapshot_provenance_preflight.py`

No existing tracked file may be modified.

In particular, do not modify:

- the frozen post-P2 token-window runner;
- its frozen test file;
- P0 artifacts;
- P1 source;
- P2 artifacts;
- K1 files;
- any model or training code.

The historical untracked K1 files must remain untouched.

## 9. Implementation dependency boundary

The preflight implementation must be Python standard-library only.

It may use modules such as:

- `argparse`
- `hashlib`
- `json`
- `os`
- `pathlib`
- `subprocess`
- `tempfile`

It must not import:

- `transformers`
- `huggingface_hub`
- `torch`
- `mamba_ssm`
- any HTTP client for network access.

The implementation must not rely on package import side effects to establish provenance.

## 10. Future local snapshot path contract

The later real preflight must accept an explicitly resolved local snapshot path.

It must not discover a snapshot by contacting Hugging Face.

It must not select `main`, `latest`, or another revision.

The expected structural identity is the cache namespace for:

`state-spaces/mamba-130m-hf`

and the exact immutable revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

A directory name equal to that revision is not, by itself, sufficient evidence for a provenance PASS.

## 11. Required provenance observations

The implementation must be capable, under later execution authority, of recording without mutation:

- whether the exact expected snapshot directory exists;
- the normalized snapshot-relative identity;
- the tokenizer/config files matching the frozen file family;
- exact raw-byte SHA256 for each selected file;
- deterministic canonical snapshot-manifest SHA256;
- symlink status where available;
- resolved content target where safely representable relative to the model cache;
- content-addressed cache evidence where locally verifiable;
- whether every selected file remains within the intended model-cache namespace.

No absolute home-directory path is required in a committed provenance artifact.

Machine-specific path prefixes should be omitted or normalized where possible.

## 12. Content-addressed cache evidence

The implementation may inspect synthetic cache structures and must be capable of later recording local cache linkage without treating it as stronger evidence than it is.

If a selected snapshot file resolves to a cache blob whose basename is digest-shaped, the implementation may verify the corresponding digest from raw bytes.

For a 64-hex blob identifier, raw SHA256 equality may be checked.

For a 40-hex Git-blob identifier, Git blob SHA1 may be checked using the standard:

`sha1("blob " + decimal_byte_length + NUL + raw_bytes)`

Such checks are structural provenance evidence.

They do not independently establish historical P1 execution byte identity.

Unknown or unsupported blob naming must produce partial provenance rather than an invented PASS.

## 13. Required evidence separation

The implementation must keep these propositions separate:

1. exact local snapshot path exists;
2. tokenizer/config files can be enumerated;
3. current local file bytes have a deterministic manifest digest;
4. local cache structure provides content-addressed evidence for those bytes;
5. current bytes are proven identical to the bytes used by historical P1 scientific execution.

The fifth proposition must not be inferred from the first four.

Repository inspection to date has not located a frozen historical P1 tokenizer file-SHA256 reference set.

Therefore the implementation must support the status:

`HISTORICAL_P1_BYTE_IDENTITY_NOT_ESTABLISHED`

without treating that status as an implementation failure.

## 14. Required future verdict vocabulary

A later real preflight must be able to distinguish at least:

`PASS_REVISION_BOUND_LOCAL_SNAPSHOT_MANIFEST`

`PARTIAL_LOCAL_SNAPSHOT_PROVENANCE`

`BLOCKED_LOCAL_TOKENIZER_SNAPSHOT_PROVENANCE`

These are provenance/precondition verdicts only.

They are not scientific outcomes.

The exact historical P1-byte-identity claim must remain independently represented.

## 15. Future output contract

This implementation phase freezes the intended future output name and schema but does not authorize creation from the real local tokenizer cache.

Future artifact:

`tokenizer_snapshot_provenance_preflight.json`

Schema:

`k0-rvg-post-p2-tokenizer-snapshot-provenance-preflight-v1`

A future real artifact must contain at minimum:

- schema version;
- frozen model ID;
- frozen revision;
- frozen expected Transformers version;
- normalized snapshot identity;
- selected file-family patterns;
- deterministic selected-file list;
- per-file raw SHA256;
- canonical manifest SHA256;
- local structural provenance observations;
- historical-P1-byte-identity status;
- no-network assertion;
- no-tokenizer-load assertion;
- no-tokenization assertion;
- no-model assertion;
- provenance verdict.

## 16. Required fail-closed production gate

Any future real-cache entry point in the preflight implementation must authenticate a separately frozen execution authority before:

- enumerating the real tokenizer snapshot;
- reading real tokenizer/config file bytes;
- computing real tokenizer/config file hashes.

The implementation-authority document itself is not sufficient to open those reads.

Synthetic temporary-directory fixtures are exempt because they contain no real tokenizer data.

A later execution authority must provide separate future-requirement markers rather than redefining current canonical authorization markers inside this document.

## 17. Mandatory synthetic validation

Synthetic tests must cover at minimum:

1. deterministic tokenizer/config file selection;
2. exclusion of unrelated files;
3. recursive relative-path handling;
4. raw SHA256 correctness;
5. deterministic sorted manifest generation;
6. manifest digest repeat identity;
7. exact equivalence with the frozen runner's manifest semantics on synthetic fixtures;
8. empty matching file set fails closed;
9. missing snapshot directory fails closed;
10. duplicate relative-path ambiguity fails closed where representable;
11. 64-hex content-addressed blob verification;
12. 64-hex blob mismatch detection;
13. 40-hex Git-blob verification;
14. 40-hex Git-blob mismatch detection;
15. unsupported blob identifier produces partial evidence rather than false PASS;
16. snapshot-path revision mismatch fails closed;
17. model-cache namespace mismatch fails closed;
18. future execution authority missing fails before real-cache byte read;
19. duplicate authority-marker rejection;
20. canonical JSON serialization repeat identity.

Tests may add further bounded synthetic cases.

## 18. Required synthetic self-check

A synthetic self-check must explicitly report:

`real_tokenizer_cache_metadata_read = false`

`real_tokenizer_file_byte_read = false`

`real_hf_tokenizer_loaded = false`

`scientific_text_tokenized = false`

`real_p0_artifact_read = false`

`real_p2_artifact_read = false`

`network_access = false`

`model_constructed = false`

`checkpoint_loaded = false`

`scientific_model_forward_executed = false`

`scientific_recurrent_state_read = false`

A suitable success status is:

`PASS_SYNTHETIC_TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_CORE`

## 19. Scientific non-claims

Passing this implementation stage establishes only code correctness for a provenance preflight mechanism.

It does not establish:

- that the real local snapshot exists;
- that the real local snapshot has any particular manifest digest;
- historical P1 tokenizer byte identity;
- tokenizer runtime compatibility;
- tokenization equivalence;
- matched/swapped token identity or difference;
- the distinct-to-identity transition location;
- any Mamba state mechanism;
- any raw-native-vector organization claim.

The frozen P1 scientific conclusion remains unchanged.

## 20. Explicitly forbidden work

Under this authority, do not:

- read or hash the real Hugging Face tokenizer cache;
- construct or load a real tokenizer;
- call `AutoTokenizer.from_pretrained`;
- tokenize any real P0/P1 scientific text;
- read real P0 scientific artifacts through the preflight;
- read real P2 scientific artifacts through the preflight;
- rerun P1;
- rerun P2;
- download a tokenizer snapshot;
- access Hugging Face over the network;
- construct a Mamba model;
- load a checkpoint;
- execute a model forward pass;
- read recurrent state;
- read logits;
- train or evaluate;
- perform causal intervention;
- run Kaggle;
- run K4;
- modify the frozen post-P2 token-window runner;
- modify either historical K1 untracked file.

## 21. Stop conditions

Implementation must stop and report `BLOCKED` if:

- frozen parent identity does not match;
- existing frozen runner manifest semantics cannot be reproduced exactly;
- implementation requires `transformers` or `huggingface_hub`;
- implementation requires real tokenizer-cache reads for its tests;
- implementation would require modifying an existing scientific file;
- a provenance status cannot distinguish current-local-byte evidence from historical-P1-byte identity;
- production gate ordering cannot be proven to precede real-cache byte reads.

## 22. Required implementation-validation report

After implementation, validation must separately report:

1. code correctness;
2. synthetic validation;
3. exact manifest-semantic equivalence to the frozen runner;
4. future real-cache gate ordering;
5. implementation file SHA256;
6. test file SHA256;
7. Git blobs after implementation freeze;
8. explicit confirmation that no real tokenizer cache was read;
9. explicit confirmation that no tokenizer was loaded;
10. explicit confirmation that no scientific tokenization occurred.

Only after that implementation validation is frozen may a one-time read-only real-cache provenance preflight authority be drafted.

## 23. Authority markers

`PARENT_VALIDATION_COMMIT = 50e06ff4bcc4f176696565f0e91d8b15cd731bbb`

`PARENT_VALIDATION_GIT_BLOB = 0dd64aa0d6f2f875672fea6aca96ff39a68d7322`

`POST_P2_TOKEN_WINDOW_IMPLEMENTATION_COMMIT = 5cfc9a61d8fe534e7bb6a037bb075bbe10092b74`

`POST_P2_TOKEN_WINDOW_RUNNER_SHA256 = 83390903028682b4d802ffab118fd16e5c8125305ce2a27a78cf829c504e0f19`

`POST_P2_TOKEN_WINDOW_RUNNER_GIT_BLOB = 561bbcf8d34970cee22e4cbf40c1c814dfc31f50`

`P1_BRANCH_CONSTRUCTION_COMMIT = 50a1daa781e47d1c0f1ba158beb445878e049a65`

`P1_BRANCH_CONSTRUCTION_GIT_BLOB = 1f70bfe36ed0efa9014d0a47471241febcfadcf4`

`TOKENIZER_MODEL_ID = state-spaces/mamba-130m-hf`

`TOKENIZER_REVISION = 5708daa364c50b880e7bd92eab456e0d34492ee9`

`TOKENIZER_TRANSFORMERS_VERSION = 5.12.1`

`TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_IMPLEMENTATION_AUTHORIZED = YES`

`TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_SYNTHETIC_VALIDATION_AUTHORIZED = YES`

`TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_REAL_CACHE_EXECUTION_AUTHORIZED = NO`

`LOCAL_REAL_TOKENIZER_CACHE_METADATA_READ_AUTHORIZED = NO`

`LOCAL_REAL_TOKENIZER_FILE_BYTE_READ_AUTHORIZED = NO`

`REAL_TOKENIZER_FILE_SHA256_AUTHORIZED = NO`

`REAL_HF_TOKENIZER_LOAD_AUTHORIZED = NO`

`TOKENIZER_REEXECUTION_AUTHORIZED = NO`

`REAL_P0_SCIENTIFIC_TOKEN_AUDIT_AUTHORIZED = NO`

`REAL_P2_CROSS_LEVEL_AUDIT_AUTHORIZED = NO`

`POST_P2_TOKEN_WINDOW_REAL_ARTIFACT_EXECUTION_AUTHORIZED = NO`

`NETWORK_ACCESS_AUTHORIZED = NO`

`SNAPSHOT_DOWNLOAD_AUTHORIZED = NO`

`MODEL_CONSTRUCTION_AUTHORIZED = NO`

`CHECKPOINT_LOADING_AUTHORIZED = NO`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`P1_SCIENTIFIC_RERUN_AUTHORIZED = NO`

`P2_REAL_ARTIFACT_DIAGNOSTIC_RERUN_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`KAGGLE_EXECUTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`AUTHORIZED_IMPLEMENTATION_FILE = scripts/longterm_k0_rvg_post_p2_tokenizer_snapshot_provenance_preflight.py`

`AUTHORIZED_TEST_FILE = tests/test_longterm_k0_rvg_post_p2_tokenizer_snapshot_provenance_preflight.py`

`FUTURE_OUTPUT_ARTIFACT = tokenizer_snapshot_provenance_preflight.json`

`FUTURE_OUTPUT_SCHEMA = k0-rvg-post-p2-tokenizer-snapshot-provenance-preflight-v1`

`NEXT_BOUNDARY = K0_RVG_POST_P2_TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_IMPLEMENTATION_VALIDATION`

This authority becomes frozen only after this exact document is reviewed, committed, and pushed as the immediate one-file child of:

`50e06ff4bcc4f176696565f0e91d8b15cd731bbb`