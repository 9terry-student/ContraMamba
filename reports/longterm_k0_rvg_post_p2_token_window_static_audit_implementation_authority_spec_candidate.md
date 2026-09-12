# K0-RVG Post-P2 Token-Window Static Audit Implementation Authority Specification Candidate

**Status:** implementation-authority specification candidate only.

**Date:** 2026-09-12

**Immediate parent / frozen hypothesis commit:**

`3b1a3deb177bbb1a73e1cb0803c5a206feabc7bf`

**Frozen hypothesis report:**

`reports/longterm_k0_rvg_post_p2_distinct_to_identity_transition_hypothesis_spec_candidate.md`

**Frozen hypothesis report SHA256:**

`f2d39522b362c7d0672c985b40780a43e62a823692feaa1a29a7d1873077b530`

**Frozen hypothesis report Git blob:**

`eb13623cce0979ede5febe75636dd5f8867bd914`

This document authorizes only a bounded implementation and fabricated/synthetic validation of the post-P2 token-window static audit.

It does not authorize reading the real P0/P2 scientific artifacts through the production audit path, retokenizing the real 336-item population, loading the frozen Hugging Face tokenizer, network access, model construction, checkpoint loading, model forward, recurrent-state read, logits read, training, evaluation, causal intervention, learned geometry, Kaggle, or K4.

The historical K-series A/B fork remains context only and is not reopened.

## 1. Active authority and phase

`ACTIVE_SCIENTIFIC_LINE = K0_RVG_RAW_NATIVE_VECTOR_GEOMETRY`

`ACTIVE_STAGE = K0_RVG_POST_P2_TOKEN_WINDOW_STATIC_AUDIT_IMPLEMENTATION`

`ACTIVE_PHASE = IMPLEMENTATION_AND_SYNTHETIC_VALIDATION_ONLY`

The frozen scientific parent remains the distinct-to-identity transition hypothesis at:

`3b1a3deb177bbb1a73e1cb0803c5a206feabc7bf`

The implementation must answer only the token-relation contract needed to discriminate:

- `H_TOKEN_WINDOW_IDENTITY`;
- `H_PREWINDOW_TOKEN_DIFFERENCE`;
- `H_INWINDOW_TOKEN_DIFFERENCE_WITH_STATE_IDENTITY`;
- `H_STATIC_EVIDENCE_INSUFFICIENT`.

No positive native-vector claim is authorized.

## 2. Scientific target

The frozen unresolved quantity is:

`POST_P2_PRIMARY_UNKNOWN = MATCHED_SWAPPED_TOKEN_WINDOW_RELATION`

The implementation must be capable, under a later separately frozen execution authority, of determining exact token-ID relations between:

- `matched_corr` and `swapped_corr`;
- `matched_ctrl` and `swapped_ctrl`;

for all 336 items under the exact P1 branch reconstruction and event-anchor semantics.

The P1 event anchor remains:

`CORRECTION_VERSUS_CONTROL_WITHIN_PAIR`

and is not a matched-versus-swapped divergence anchor.

## 3. Frozen provenance bindings

### 3.1 Hypothesis

Commit:

`3b1a3deb177bbb1a73e1cb0803c5a206feabc7bf`

Report SHA256:

`f2d39522b362c7d0672c985b40780a43e62a823692feaa1a29a7d1873077b530`

Report Git blob:

`eb13623cce0979ede5febe75636dd5f8867bd914`

### 3.2 Frozen P2 validated interpretation

Commit:

`6a342810857d55da8e9d23b5910da1d27b8ea96a`

Report:

`reports/longterm_k0_rvg_p2_validated_scientific_interpretation_report_candidate.md`

Report SHA256:

`f3e3aa4df9d9ff92a61f02baf5cb1b008e513afcba5d1c041fa6359ba4733c28`

### 3.3 Frozen P2 diagnostic artifact identity

Future real audit binding only:

`p2_degeneracy_diagnostic.json`

SHA256:

`058d00adb99cdbfad1893593a3e8917322c5ca2f186f7305561460f868ddddc3`

Schema:

`k0-rvg-p2-item-level-degeneracy-static-diagnostic-v1`

This implementation phase must not read that real artifact through the production path.

### 3.4 Frozen P1 branch-construction implementation

Commit:

`50a1daa781e47d1c0f1ba158beb445878e049a65`

File:

`scripts/longterm_k0_rvg_p1_raw_vector_execution.py`

Git blob:

`1f70bfe36ed0efa9014d0a47471241febcfadcf4`

The implementation may reproduce only the state-blind branch-text and token-contract semantics from this frozen source.

It must not import or execute model-running code from that runner.

### 3.5 Frozen P0 archive

Archive:

`reports/longterm_k0_rvg_p0_state_blind_provisioning_421d798_v1`

Exact future real-audit SHA256 identities:

- `candidate_pool.jsonl`:
  `743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246`
- `generated_source.jsonl`:
  `8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf`
- `phase_pair_mapping.json`:
  `c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda`
- `token_contracts.jsonl`:
  `6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998`
- `provisioning_manifest.json`:
  `feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52`
- `validation_report_candidate.md`:
  `ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a`

The implementation phase may encode and test these identities as constants but must not use the real archive to obtain scientific token results.

## 4. Frozen tokenizer identity for a future real audit

The frozen P1 execution used:

`HF_MODEL = state-spaces/mamba-130m-hf`

`HF_REVISION = 5708daa364c50b880e7bd92eab456e0d34492ee9`

`TRANSFORMERS_VERSION = 5.12.1`

The historical P1 path resolved tokenizer files from that exact model/revision.

A future real audit, if separately authorized, must use the exact same tokenizer identity.

The future production implementation must support fail-closed local-only resolution.

It must not silently fall back to a network download.

The future production path must require:

`local_files_only = true`

or an equivalent authenticated pre-resolved local snapshot contract.

If the exact tokenizer snapshot is unavailable locally, the correct result is a blocker requiring separate provisioning authority.

`NETWORK_FALLBACK_POLICY = FORBIDDEN`

## 5. Exact branch reconstruction contract

For local item `i`, define phase mate `m` using the frozen P0 168-pair mapping.

The exact four branch strings are:

`prefix = item[i].prefix_text`

`matched_corr = prefix + item[i].correction_text`

`matched_ctrl = prefix + item[i].control_text`

`swapped_corr = prefix + item[m].correction_text`

`swapped_ctrl = prefix + item[m].control_text`

No whitespace normalization, Unicode normalization, trimming, template editing, punctuation editing, or alternate prompt formatting is allowed.

The implementation must compare exact UTF-8 source strings and exact tokenizer-produced integer token IDs.

## 6. Exact event-anchor contract

The frozen P1 event anchor for each pair is the first correction-versus-control token divergence within the first eight continuation-token positions after the exact prefix.

For the matched pair:

`matched_te = first_divergence(matched_corr_ids, matched_ctrl_ids)`

For the swapped pair:

`swapped_te = first_divergence(swapped_corr_ids, swapped_ctrl_ids)`

The implementation must validate these values against the frozen archived P0 token contract in a future real execution.

The implementation must never substitute a matched-versus-swapped divergence for `t_e`.

`MATCHED_SWAPPED_DIVERGENCE_MAY_DEFINE_EVENT_ANCHOR = NO`

## 7. Exact token relation questions

For each of the two branch roles, the implementation must compute exact relations.

### 7.1 Full branch relation

For correction:

`matched_corr_ids == swapped_corr_ids`

For control:

`matched_ctrl_ids == swapped_ctrl_ids`

### 7.2 First matched-versus-swapped token difference

For each role, determine the first absolute token index at which matched and swapped differ.

If one sequence is a strict prefix of the other, the first difference is the shorter sequence length.

If sequences are identical, the first difference is null.

### 7.3 Event-relative first difference

Because the frozen P2 construction audit established exact matched/swapped event-anchor equality for all 336 items, the future production path must revalidate:

`matched_te == swapped_te`

before assigning one event-relative first-difference coordinate.

If that equality fails, production execution must fail closed.

When equal:

`first_difference_relative_to_te = first_difference_absolute_index - matched_te`

### 7.4 Exact endpoint-window relation

For each role and every:

`k in {-1,0,1,2,3,4,5,6,7}`

compare:

`matched_ids[matched_te + k]`

against:

`swapped_ids[swapped_te + k]`

using exact integer equality.

The implementation must report all nine booleans and the exact matched/swapped token IDs for each coordinate in a future real artifact.

No tolerance, embedding similarity, decoded-text similarity, or semantic comparison is allowed.

### 7.5 Pre-window relation

The implementation must determine whether any token difference occurs at an index before:

`matched_te - 1`

and, if so, report the first such difference.

## 8. Required per-role classification

Each item-role pair must receive exactly one classification:

`TOKEN_SEQUENCES_EXACTLY_IDENTICAL_FULL_BRANCH`

`TOKEN_SEQUENCES_DIFFER_BEFORE_ENDPOINT_WINDOW`

`TOKEN_SEQUENCES_FIRST_DIFFER_AT_WINDOW_K_MINUS_1`

`TOKEN_SEQUENCES_FIRST_DIFFER_WITHIN_WINDOW_K_0_TO_PLUS_7`

`TOKEN_SEQUENCES_FIRST_DIFFER_AFTER_ENDPOINT_WINDOW`

`TOKEN_RELATION_NOT_IDENTIFIABLE_FROM_AUTHORIZED_EVIDENCE`

Classification order must be deterministic and mutually exclusive.

If full sequences are exactly identical, the first classification wins.

Otherwise classify from the exact first-difference coordinate relative to the common event anchor.

## 9. Required cross-level classification

A future real audit may read the already validated P2 diagnostic only under separate execution authority.

For items with:

`ENDPOINT_SUPPORTING_STATE_HASH_IDENTITY`

the audit may assign exactly one of:

`TOKEN_WINDOW_IDENTITY_SUPPORTS_STATE_IDENTITY`

`PREWINDOW_TOKEN_DIFFERENCE_PRECEDES_STATE_IDENTITY`

`INWINDOW_TOKEN_DIFFERENCE_WITH_STATE_IDENTITY`

`TOKEN_BOUNDARY_UNRESOLVED`

These are descriptive localization labels.

They are not causal-mechanism labels.

The implementation must not emit labels such as:

`MODEL_COLLAPSE`

`STATE_ERASURE`

`GATE_COLLAPSE`

or any equivalent causal conclusion.

## 10. Special-item policy

Item:

`local_template_index = 163`

is the unique frozen P2 common exception.

The six frozen token-count mismatch indices are:

`63, 95, 163, 231, 263, 331`

The implementation may emit their ordinary deterministic records and descriptive summary counts.

It must not branch logic, thresholds, output schema, or scientific endpoints based on those indices.

No post-hoc subgroup-specific metric is allowed.

## 11. Authorized implementation files

Exactly two implementation files are authorized:

`scripts/longterm_k0_rvg_post_p2_token_window_static_audit.py`

`tests/test_longterm_k0_rvg_post_p2_token_window_static_audit.py`

No existing scientific implementation file may be modified.

No P0, P1, P2, K1, K2S, model, head, checkpoint, or run artifact may be modified.

## 12. Required implementation architecture

The runner must separate pure/static logic from future production I/O.

At minimum it must provide testable functions for:

- exact canonical JSON / JSONL parsing;
- SHA256 verification;
- deterministic phase-mate lookup;
- exact branch-text reconstruction;
- pure token-array relation comparison;
- exact first-difference calculation including strict-prefix cases;
- event-relative coordinate conversion;
- exact `k=-1..7` token-window comparison;
- per-role classification;
- cross-level classification from a synthetic P2-like state label;
- canonical output serialization;
- future execution-authority authentication;
- fail-closed output-directory handling.

The production entry point must authenticate a separately frozen real-execution authority before reading real P0/P2 artifacts or loading the real tokenizer.

## 13. Required fail-closed execution gate

The future production CLI may exist in the implementation, but it must be impossible to use on real artifacts under this authority.

Before any real P0 artifact read, real P2 artifact read, or real HF tokenizer load, production execution must require a separately frozen authority file containing at minimum:

`REQUIRED_FUTURE_REAL_ARTIFACT_EXECUTION_VALUE = YES`

`REQUIRED_FUTURE_TOKENIZER_REEXECUTION_VALUE = YES`

`MODEL_CONSTRUCTION_AUTHORIZED = NO`

`CHECKPOINT_LOADING_AUTHORIZED = NO`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`NETWORK_ACCESS_AUTHORIZED = NO`

`POST_P2_TOKEN_WINDOW_IMPLEMENTATION_COMMIT = <exact 40-char commit>`

The implementation must verify that the implementation commit marker equals the latest commit touching the runner.

If any required marker is absent or mismatched, execution must fail before real scientific inputs are read.

## 14. Synthetic validation authority

Synthetic/fabricated validation is authorized.

Synthetic validation may use:

- fabricated candidate rows;
- fabricated phase-mate mappings;
- fabricated archived token contracts;
- fabricated integer token arrays;
- a fake deterministic tokenizer object;
- fabricated P2-like state classifications;
- temporary directories.

Synthetic validation must not use:

- the real 336 P0 candidate pool to derive scientific token relations;
- the real P2 diagnostic to derive cross-level conclusions;
- the real frozen HF tokenizer;
- any network access;
- model or checkpoint loading.

## 15. Mandatory synthetic test cases

At minimum, tests must cover:

1. exact full-branch token identity;
2. first difference before `k=-1`;
3. first difference exactly at `k=-1`;
4. first difference at `k=0`;
5. first difference at `k=+7`;
6. first difference after `k=+7`;
7. strict-prefix matched sequence;
8. strict-prefix swapped sequence;
9. unequal event anchors fail closed;
10. missing window coordinate fail closed;
11. exact nine-coordinate window equality;
12. one-coordinate window difference;
13. deterministic phase-mate reconstruction;
14. branch-text concatenation with no normalization;
15. cross-level `TOKEN_WINDOW_IDENTITY_SUPPORTS_STATE_IDENTITY`;
16. cross-level `PREWINDOW_TOKEN_DIFFERENCE_PRECEDES_STATE_IDENTITY`;
17. cross-level `INWINDOW_TOKEN_DIFFERENCE_WITH_STATE_IDENTITY`;
18. unsupported/missing evidence -> `TOKEN_BOUNDARY_UNRESOLVED`;
19. execution authority missing -> fail before real input read;
20. wrong implementation commit marker -> fail before real input read;
21. output directory already exists -> fail closed;
22. canonical serialization repeat identity.

Tests may add further bounded cases.

## 16. Required synthetic self-check

The runner must expose a synthetic self-check mode that proves:

`real_p0_artifact_read = false`

`real_p2_artifact_read = false`

`real_hf_tokenizer_loaded = false`

`network_access = false`

`model_constructed = false`

`checkpoint_loaded = false`

`scientific_model_forward_executed = false`

`scientific_recurrent_state_read = false`

A suitable successful status marker is:

`PASS_SYNTHETIC_POST_P2_TOKEN_WINDOW_AUDIT_CORE`

## 17. Future real-audit output contract

This section freezes the intended schema but does not authorize its creation.

A future separately authorized real audit must produce exactly one scientific artifact:

`post_p2_token_window_audit.json`

Schema:

`k0-rvg-post-p2-token-window-static-audit-v1`

The artifact must contain at minimum:

- provenance block;
- exact P0/P2 input SHA256 identities;
- tokenizer identity and local snapshot identity;
- item count `336`;
- per-item stable ID and local index;
- matched/swapped event anchors;
- correction-role token relation;
- control-role token relation;
- exact first-difference information;
- exact nine-coordinate window comparison records;
- per-role classifications;
- frozen P2 state-hash classification;
- cross-level classification where defined;
- deterministic aggregate counts.

It must not contain model logits, recurrent tensors, embeddings, hidden states, or learned features.

## 18. Future real-audit tokenizer policy

A future execution may use the exact tokenizer only if a separate authority explicitly enables it.

The production implementation must support:

`state-spaces/mamba-130m-hf`

revision:

`5708daa364c50b880e7bd92eab456e0d34492ee9`

and must verify the expected tokenizer/runtime contract before scientific tokenization.

The implementation must forbid:

- unpinned revision;
- latest/main resolution;
- network fallback;
- alternate tokenizer family;
- normalization changes;
- special-token insertion changes;
- `add_special_tokens=True`.

The exact P1 tokenization semantic is:

`add_special_tokens = false`

## 19. Validation boundary

Passing implementation tests establishes only code correctness for the frozen token-audit contract.

It does not establish:

- real token relations;
- execution success;
- real artifact validity;
- the location of the distinct-to-identity transition;
- any causal mechanism;
- any native-vector organization claim.

Those evidence layers remain separate.

## 20. Explicitly forbidden work

Under this authority, do not:

- execute the production audit on the real 336 items;
- load the real frozen HF tokenizer;
- download tokenizer files;
- access Hugging Face over the network;
- construct any Mamba model;
- load any checkpoint;
- perform model forward;
- read logits;
- capture or reread recurrent tensors;
- rerun P1;
- rerun P2;
- modify P0/P1/P2 artifacts;
- introduce a new layer or window;
- perform PCA/SVD/whitening/probes;
- train or tune anything;
- select subgroups after results;
- run Kaggle;
- run K4.

## 21. Stop conditions

Implementation work must stop and report `BLOCKED` if any of the following occurs:

- frozen parent/provenance identities do not match;
- required P1 branch reconstruction semantics cannot be reproduced statically;
- exact phase-mate mapping cannot be represented deterministically;
- production gate cannot be proven to precede all real scientific input reads;
- real HF tokenizer is required for synthetic tests;
- model/checkpoint code becomes necessary;
- existing scientific files would need modification;
- output classification semantics become ambiguous.

## 22. Required implementation-validation report

After implementation, validation must report separately:

1. code correctness;
2. synthetic validation;
3. production real-input gate ordering;
4. implementation file SHA256;
5. implementation Git blob after commit;
6. tests file SHA256;
7. tests Git blob after commit;
8. explicit confirmation that no real P0/P2 scientific token audit ran.

Only after that validation is frozen may a one-time real tokenizer-only execution authority be drafted.

## 23. Authority markers

`HISTORICAL_AB_FORK_AUTHORITY = CONTEXT_ONLY_SUPERSEDED_BY_K0_RVG`

`POST_P2_HYPOTHESIS_COMMIT = 3b1a3deb177bbb1a73e1cb0803c5a206feabc7bf`

`POST_P2_HYPOTHESIS_SHA256 = f2d39522b362c7d0672c985b40780a43e62a823692feaa1a29a7d1873077b530`

`POST_P2_HYPOTHESIS_GIT_BLOB = eb13623cce0979ede5febe75636dd5f8867bd914`

`P1_BRANCH_CONSTRUCTION_COMMIT = 50a1daa781e47d1c0f1ba158beb445878e049a65`

`P1_BRANCH_CONSTRUCTION_GIT_BLOB = 1f70bfe36ed0efa9014d0a47471241febcfadcf4`

`POST_P2_TOKEN_WINDOW_STATIC_AUDIT_IMPLEMENTATION_AUTHORIZED = YES`

`POST_P2_TOKEN_WINDOW_STATIC_AUDIT_SYNTHETIC_VALIDATION_AUTHORIZED = YES`

`POST_P2_TOKEN_WINDOW_REAL_ARTIFACT_EXECUTION_AUTHORIZED = NO`

`REAL_P0_SCIENTIFIC_TOKEN_AUDIT_AUTHORIZED = NO`

`REAL_P2_CROSS_LEVEL_AUDIT_AUTHORIZED = NO`

`TOKENIZER_REEXECUTION_AUTHORIZED = NO`

`REAL_HF_TOKENIZER_LOAD_AUTHORIZED = NO`

`NETWORK_ACCESS_AUTHORIZED = NO`

`MODEL_CONSTRUCTION_AUTHORIZED = NO`

`CHECKPOINT_LOADING_AUTHORIZED = NO`

`SCIENTIFIC_MODEL_FORWARD_AUTHORIZED = NO`

`SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED = NO`

`LOGITS_READ_AUTHORIZED = NO`

`P1_SCIENTIFIC_RERUN_AUTHORIZED = NO`

`P2_REAL_ARTIFACT_DIAGNOSTIC_RERUN_AUTHORIZED = NO`

`TRAINING_AUTHORIZED = NO`

`CAUSAL_INTERVENTION_AUTHORIZED = NO`

`LEARNED_OR_TUNED_GEOMETRY_AUTHORIZED = NO`

`INFERENTIAL_TESTING_AUTHORIZED = NO`

`KAGGLE_EXECUTION_AUTHORIZED = NO`

`K4_EXECUTION_AUTHORIZED = NO`

`AUTHORIZED_IMPLEMENTATION_FILE = scripts/longterm_k0_rvg_post_p2_token_window_static_audit.py`

`AUTHORIZED_TEST_FILE = tests/test_longterm_k0_rvg_post_p2_token_window_static_audit.py`

`FUTURE_OUTPUT_ARTIFACT = post_p2_token_window_audit.json`

`FUTURE_OUTPUT_SCHEMA = k0-rvg-post-p2-token-window-static-audit-v1`

`NEXT_BOUNDARY = K0_RVG_POST_P2_TOKEN_WINDOW_STATIC_AUDIT_IMPLEMENTATION_VALIDATION`

This authority becomes frozen only after this exact document is committed and pushed as the immediate one-file child of:

`3b1a3deb177bbb1a73e1cb0803c5a206feabc7bf`
