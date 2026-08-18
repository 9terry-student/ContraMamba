# P3-W6-F2-P4-D Controlled-Data Integrity Gate Specification

Decision scope: P4-B R1 Gate 5 controlled-data integrity validator.

## A. Active Authority And Phase Boundary

Active authority is the frozen P4-B R1 regeneration specification:

- `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_spec.md`
- frozen authority commit: `fcc3b9ccaf2bbee33ac18dcef10d50acff54aab4`

Current repository HEAD for this specification authoring pass is:

- `a780a076644a16dfdb6c2bb2d89584daf0d4f1e7`

Committed P4-B R1 regeneration artifacts 1-7 and Stage185 compatibility artifacts 8-10 are under:

- `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/`

The completed predecessor gate status is:

- Gate 1 code correctness: PASS
- Gate 2 deterministic regeneration execution: PASS
- Gate 3 artifact/provenance validity: PASS
- Gate 4 semantic/grammar equivalence: PASS
- Stage185 compatibility: PASS

The independent Gate-5 audit decision is:

`P3W6F2P4B_R1_CONTROLLED_DATA_INTEGRITY_GATE_BLOCKED`

The blocker is absence of an authorized Gate-5-capable validator, not a demonstrated regenerated-data defect.

This document is static specification only. It authorizes no data mutation, no regeneration, no training, no evaluation, no model loading, no checkpoint access, no Kaggle execution, and no changes to historical Stage185-v1 authority. It specifies the smallest future read-only validator that may constitute Gate 5 after independent specification verification and future implementation authorization.

## B. Validator Purpose And Non-Purpose

The future validator purpose is to determine whether the P4-B R1 regenerated full controlled dataset preserves controlled-data integrity relative to the historical controlled-v5 dataset and historical Stage185-v1 split/linkage contract, while recognizing P4-B predicate-realization compatibility only as separate effective evidence in new P4-B artifacts.

The validator must be read-only with respect to all inputs. It may only materialize a new Gate-5 report artifact set under a new P4-D report directory if future implementation authority permits report output.

The validator is not a dataset builder, not a regeneration script, not a Stage185-v1 replacement, not a training-admission gate, not a result-review gate, not a grammar repair tool, not an evaluation runner, and not a mechanism for treating Gate 4 PASS as Gate 5 PASS.

The validator must not modify, import-and-repurpose as output authority, weaken, or repoint `scripts/build_stage185a_controlled_train_integrity_sidecar.py`. The historical Stage185-v1 builder remains immutable historical authority.

## C. Exact Input Authorities

Historical controlled dataset:

- path: `data/controlled_v5_v3_without_time_swap.jsonl`
- frozen SHA256: `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`

Regenerated full dataset artifact:

- path: `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`
- SHA256: `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- semantic SHA256 from P4-B summary: `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

P4-B artifacts 1-10 are mandatory Gate-5 inputs:

1. `controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl` SHA256 `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
2. `p3w6f2_p4b_r1_regenerated_members.jsonl` SHA256 `3dd91d6e2888d50ccd45acb8243d2b8d47bd3476e2a76f5c2b9e7cd93b82bbf3`
3. `p3w6f2_p4b_r1_regeneration_audit.jsonl` SHA256 `17eaae3e20779fc6bbfe730222bd4e410d5bd03b108aaf7ea98214eaeb8d77a1`
4. `p3w6f2_p4b_r1_regeneration_summary.json` SHA256 `e09bcd09207d78a211a4b63a94af8db2a93b6c4c6b1d618e2a673168618f1157`
5. `p3w6f2_p4b_r1_full_output_isolation.json` SHA256 `c4b342d200757b4e330fd7b4bfb1b5550b3c74933bca5c99323eeac9c87ebb7e`
6. `p3w6f2_p4b_r1_deterministic_generator_invocation.json` SHA256 `2cbd5057cf89c3ba0a01bbaa1d6168b3bd595a1704bbcec50902de44066494f0`
7. `p3w6f2_p4b_r1_base_form_coverage.json` SHA256 `5d130c529e0ebcca9fa3f7137620222488a9eb8d2db137e5a7283c345a277bb3`
8. `p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl` SHA256 `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`
9. `p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json` SHA256 `ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`
10. `p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json` SHA256 `09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6`

Historical Stage185-v1 source authority:

- path: `scripts/build_stage185a_controlled_train_integrity_sidecar.py`
- SHA256: `11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc`
- split contract: `split_by_pair(rows, seed=174, ratio=0.2)` using sorted pair IDs, `random.Random(seed).shuffle(pair_ids)`, and `round(len(pair_ids) * ratio)` bounded to at least one and at most `len(pair_ids) - 1`

Relevant structured generator identity:

- path/symbol: `scripts/build_controlled_v5.py::fact_templates_for_count`
- source SHA256: `9fbd94a151c4d83a5e824412d7c0837062fedd20628f4f198116b2d08b679410`
- predicate mapping symbol: `_BASE_PREDICATE_BY_INFLECTED`
- mapping SHA256 from artifact 4: `617ce712753bc09282b4bb6792154cbe1daa7713021895160c01ce1a839cc309`

The P4-B deterministic invocation artifact's `random_seed_policy` is not Stage185 split authority. Gate 5 must use Stage185 `seed=174` and `dev_ratio=0.2` for split replay, and must treat any unrelated P4-B helper seed as distinct non-authority for Stage185 split identity. A replay using `seed=17`, or any seed other than `174`, cannot satisfy Gate 5 even if all row counts, pair counts, and other hashes happen to match. Any validator that accepts `seed=17` as Stage185 Gate-5 split authority is itself invalid.

## D. Read-Only Validation Contract

The validator must compute all checks from input bytes and deterministic in-memory replay only.

Required checks:

- Regenerated dataset physical identity: exact path and SHA256 must match section C.
- Historical dataset physical identity: exact path and SHA256 must match section C.
- Exact full dataset row count: historical and regenerated JSONL must each contain 3600 rows.
- Exact schema: every row in both datasets must have exactly these fields in this order: `id`, `pair_id`, `claim`, `evidence`, `final_label`, `frame_compatible_label`, `predicate_covered_label`, `sufficiency_label`, `polarity_label`, `primary_failure_type`, `intervention_type`.
- Exact row ID uniqueness: every `id` must be nonempty and unique within each dataset.
- Exact row-order preservation: historical and regenerated datasets must have the same ordered `id` sequence.
- Exact controlled-data topology: historical and regenerated datasets must each contain exactly 300 unique `pair_id` values, exactly 12 admitted intervention families, exactly 12 rows/members per pair, exactly one member for each `pair_id` + `intervention_type` combination, and the exact same complete 12-family intervention set in every pair. This is an exact rectangular `300 pair x 12 family` topology and is the only admitted full controlled-data topology for Gate 5.
- Exact admitted 12-family set: `entity_swap`, `event_swap`, `evidence_deletion`, `evidence_truncation`, `irrelevant_evidence`, `location_swap`, `none`, `paraphrase`, `polarity_flip`, `predicate_swap`, `role_swap`, `title_name_swap`.
- Exact pair/family topology preservation: each pair must retain its ordered members, pair IDs, intervention types, family membership, and rectangular membership; authorized F2 pairs must have exactly one `none`, one `paraphrase`, and one `polarity_flip` member inside the complete 12-family set.
- No `time_swap`: both datasets and P4-B artifacts must have zero `intervention_type == "time_swap"` rows.
- Deterministic Stage185 pair split replay: replay the historical Stage185 pair split over the regenerated dataset using `seed=174` and `dev_ratio=0.2`; do not use any P4-B split helper as the authority for this check. Replay with `seed=17`, or any seed other than `174`, must be rejected and cannot be treated as an alternate valid split replay.
- Split identity preservation: the replayed regenerated split assignment for every row must equal the replayed historical Stage185 split assignment for the same `id` and `pair_id`.
- Label integrity: all label fields and `primary_failure_type` must be byte-identical between historical and regenerated rows except no label field may ever differ; authorized F2 expectations must remain canonical/paraphrase `final_label == REFUTE`, polarity_flip `final_label == SUPPORT`, frame/predicate/sufficiency all `1`, and polarity labels matching the P4-B specification.
- Canonical linkage integrity: canonical row is `intervention_type == "none"`; derivative rows share `pair_id` and `claim`; Stage185-style `canonical_row_id` must resolve to the canonical member `id`.
- Exactly 238 authorized F2 negative evidence deltas: only authorized F2 `none` and `paraphrase` rows may differ, and only in `evidence`.
- Exactly 119 authorized F2 pairs and exactly 357 authorized F2 members.
- `polarity_flip` unchanged: all 119 authorized F2 polarity_flip rows must be byte-identical across every dataset field.
- All non-F2 rows byte-identical: every non-authorized-F2 row must have byte-identical canonical JSON row representation when compared field-by-field in schema order.
- Compatibility artifacts 8-10 required: absence, hash mismatch, schema mismatch, or unresolved status in artifacts 8-10 is Gate-5 failure.
- Compatibility artifact hash/schema/count validation: artifact 8 must contain 357 rows using schema `P3W6F2P4B_R1_STAGE185_COMPATIBILITY_ROW_V1`; artifact 9 must use schema `P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1`; artifact 10 must use schema `P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1` and must bind artifacts 7-9 and Stage185 source hashes.
- Exact Stage185 compatibility summary validation for artifact 9: require `authorized_pair_count == 119`, `authorized_member_count == 357`, `row_count == 357`, `compatibility_pass_count == 357`, `compatibility_fail_count == 0`, `compatibility_unresolved_count == 0`, `compatibility_gate_status == "PASS"`, `stage185_v1_mutated == false`, `historical_authority_weakened == false`, and `training_admission_released == false`. Any mismatch fails closed, including spoofed PASS claims with altered pass/fail/unresolved counts.
- Historical Stage185 raw predicate observation retained: artifact 9 must report `raw_stage185_predicate_axis_observation_count == 238`; artifact 8 must retain raw Stage185 changed/expected/status evidence per member.
- P4-B compatibility used only as separate effective evidence: effective compatibility may pass only through P4-B artifacts 8-10 and must never be represented as a mutation or weakening of Stage185-v1 output.
- Historical Stage185-v1 source/hash not modified, weakened, or repointed: artifact 9 `stage185_v1_mutated` must be `false`, artifact 9 `historical_authority_weakened` must be `false`, and artifact 10 Stage185 source hash must match section C.
- `training_admission_released` remains `false` in compatibility artifact 9 and any Gate-5 report.

## E. Fail-Closed Conditions

The validator must return BLOCKED or FAIL, never PASS, for any of these conditions:

- missing required input path;
- SHA256 mismatch for any required input;
- malformed JSON/JSONL, duplicate keys where detectable, non-UTF-8 bytes, non-LF line endings, NaN, or nondeterministic serialization where a schema requires deterministic bytes;
- row count not exactly 3600 for either full dataset;
- schema names, field order, field types, or enumerated labels differ from contract;
- duplicate, missing, empty, or reordered row IDs;
- pair count other than 300, family count other than 12, missing family in any pair, extra family, duplicate `pair_id` + `intervention_type`, nonrectangular pair/family topology, pair topology drift, duplicate member, missing member, extra member, or claim drift within a pair;
- any `time_swap` row;
- Stage185 split replay cannot be performed exactly with `seed=174` and `dev_ratio=0.2`;
- Stage185 split replay uses `seed=17` or any seed other than `174`, or the validator accepts `seed=17` as satisfying Gate 5;
- split drift for any row;
- label drift, canonical linkage drift, row-ID drift, pair-ID drift, or intervention-type drift;
- changed row count other than 238, changed pair count other than 119, authorized member count other than 357, authorized pair count other than 119;
- any changed field other than `evidence` in authorized F2 `none`/`paraphrase` rows;
- any changed field in authorized F2 `polarity_flip` rows;
- any changed field in non-F2 rows;
- artifact 8, 9, or 10 missing, optionalized, schema-invalid, hash-invalid, count-invalid, unresolved, or failed;
- artifact 9 has `authorized_pair_count != 119`, `authorized_member_count != 357`, `row_count != 357`, `compatibility_pass_count != 357`, `compatibility_fail_count != 0`, `compatibility_unresolved_count != 0`, `compatibility_gate_status != "PASS"`, `stage185_v1_mutated != false`, `historical_authority_weakened != false`, or `training_admission_released != false`;
- raw Stage185 predicate-axis observation is absent, hidden, zeroed, or overwritten by effective compatibility;
- Stage185-v1 source hash differs from `11e6ba89b8131c76eac4504b4273867eaa99a131abe23d3238eb65ecda207bbc`;
- any implementation path mutates or reuses `scripts/build_stage185a_controlled_train_integrity_sidecar.py` as the Gate-5 validator;
- `training_admission_released` is anything other than `false`;
- a check requires model loading, training, evaluation, external labels, threshold fitting, checkpoint access, or dataset regeneration;
- report output is partial, non-atomic, or claims PASS without all checks passing.

## F. Validator Output Schema And Decision Tokens

If future implementation materializes a report, it must write:

- `reports/reason_router_p2_p3w6f2_p4d_controlled_data_integrity_gate_<validator_commit>/p3w6f2_p4d_controlled_data_integrity_gate_report.json`
- `reports/reason_router_p2_p3w6f2_p4d_controlled_data_integrity_gate_<validator_commit>/p3w6f2_p4d_controlled_data_integrity_gate_report.md`

JSON schema version:

`P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_REPORT_V1`

Required JSON fields:

- `schema_version`
- `decision_token`
- `validator_contract_version`
- `validator_commit`
- `authority_commit`
- `current_head`
- `phase`
- `training_admission_released`
- `historical_dataset_path`
- `historical_dataset_sha256`
- `regenerated_dataset_path`
- `regenerated_dataset_sha256`
- `regenerated_dataset_semantic_sha256`
- `p4b_artifact_directory`
- `p4b_artifact_hashes`
- `stage185_source_script`
- `stage185_source_script_sha256`
- `structured_source_producer`
- `structured_source_producer_sha256`
- `stage185_split_seed`
- `stage185_dev_ratio`
- `row_count_historical`
- `row_count_regenerated`
- `schema_status`
- `row_id_status`
- `row_order_status`
- `pair_topology_status`
- `time_swap_status`
- `split_replay_status`
- `split_identity_status`
- `label_integrity_status`
- `canonical_linkage_status`
- `delta_isolation_status`
- `polarity_flip_status`
- `non_f2_identity_status`
- `compatibility_artifact_status`
- `raw_stage185_observation_status`
- `historical_stage185_immutability_status`
- `determinism_status`
- `provenance_status`
- `failure_reasons`
- `created_at_utc`

Allowed decision tokens:

- `P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_PASS`
- `P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_FAIL`
- `P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_BLOCKED`

PASS requires every status field to be PASS, all expected counts/hashes to match, `failure_reasons == []`, and `training_admission_released == false`.

## G. Deterministic, Hash, And Provenance Requirements

The validator must compute physical SHA256 over exact bytes. Semantic comparisons must use deterministic JSON canonicalization with schema field order, UTF-8, LF line endings, no NaN, and stable separators. Any timestamp must be excluded from semantic hashes and must never appear in dataset rows.

The report must record all input paths and hashes, the validator commit, authority commit, current HEAD, and the exact Stage185 split parameters. A dirty worktree may not invalidate read-only validation by itself, but any materialized report must record dirty tracked state if present and must not claim a clean execution state unless verified.

No P4-D hash namespace may replace, rename, or overwrite P3-W4, Level-1 F2, P4-B, or Stage185-v1 hashes. Historical hashes remain historical controls.

## H. Atomicity And Output Rules

If report materialization is implemented, write to a temporary staging directory, validate the complete JSON and markdown reports in staging, compute report hashes, then atomically promote to the final directory. Never overwrite an existing final directory. Existing output may be accepted only as read-only idempotent replay when every byte and hash matches the deterministic expected report.

Validation failure must not leave a final-looking complete PASS report. Partial output must be marked BLOCKED or removed before final promotion.

## I. Future Implementation Whitelist

Future implementation may add only:

- `scripts/validate_reason_router_p3w6f2_p4d_controlled_data_integrity_gate.py`
- `tests/test_reason_router_p3w6f2_p4d_controlled_data_integrity_gate.py`
- optional future P4-D report directory described in section F, only as validator output

Future implementation must not modify:

- `scripts/build_stage185a_controlled_train_integrity_sidecar.py`
- `scripts/build_controlled_v5.py`
- any P4-B artifact 1-10
- historical Stage182/184/185 artifacts or scripts
- `data/controlled_v5_v3_without_time_swap.jsonl`
- `reports/stage180a_pass2_annotations_completed.csv`
- F1 behavior/APIs
- unrelated patch files
- any training, evaluation, model, checkpoint, or dataset-generation code

## J. Exact Test Contract

Future tests must prove:

- the validator is read-only for all authority inputs;
- all required inputs and hashes are enforced;
- exact 3600-row count is required;
- exact positive rectangular topology is required: 300 unique pairs, 12 admitted families, 12 rows per pair, the same complete 12-family set in every pair, one member per `pair_id` + `intervention_type`, and no `time_swap` in the admitted family set;
- exact schema and field order are required;
- duplicate/missing/reordered row IDs fail;
- adversarial topology drift fails for pair count not 300, family count not 12, missing family in any pair, extra family, duplicate `pair_id` + `intervention_type`, and any nonrectangular pair/family topology;
- any `time_swap` row fails;
- Stage185 split replay uses `seed=174` and `dev_ratio=0.2`;
- Stage185 split replay with `seed=174` succeeds when all other authority matches;
- Stage185 split replay with `seed=17` is rejected;
- validator acceptance of `seed=17` is itself a validator failure;
- split drift fails;
- label drift fails;
- canonical linkage drift fails;
- counts of 119 authorized F2 pairs, 357 members, and 238 evidence deltas are required;
- polarity_flip mutations fail;
- non-F2 mutations fail;
- compatibility artifacts 8-10 are mandatory and hash/schema/count validated;
- artifact 9 exact summary contract is enforced: authorized pair/member/row counts 119/357/357, pass/fail/unresolved counts 357/0/0, `compatibility_gate_status == "PASS"`, `stage185_v1_mutated == false`, `historical_authority_weakened == false`, and `training_admission_released == false`;
- adversarial compatibility summary spoofing fails for `compatibility_pass_count != 357`, `compatibility_fail_count != 0`, and `compatibility_unresolved_count != 0`, including when `compatibility_gate_status` is spoofed as `"PASS"`;
- raw Stage185 predicate-axis observation count 238 is retained;
- effective P4-B compatibility is accepted only through artifacts 8-10;
- historical Stage185 source hash mutation/repointing fails;
- `training_admission_released != false` fails;
- model loading, training, evaluation, external-label use, and dataset regeneration are not invoked;
- PASS, FAIL, and BLOCKED decision tokens are emitted only under their specified conditions;
- partial or conflicting report output is rejected.

Tests must use temporary copied fixtures for negative cases and must not mutate frozen repository artifacts.

## K. CPU-Only Validation Command Expectations

Future narrow validation commands are CPU-only and file-read-only except for temporary test/report output:

```bash
python scripts/validate_reason_router_p3w6f2_p4d_controlled_data_integrity_gate.py --p4b-artifact-dir reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458 --historical-dataset data/controlled_v5_v3_without_time_swap.jsonl --stage185-split-seed 174 --stage185-dev-ratio 0.2
python -m pytest tests/test_reason_router_p3w6f2_p4d_controlled_data_integrity_gate.py
git diff --check
```

These commands must not load Torch models, checkpoints, trainers, evaluators, external datasets, or network resources. They must not run Stage185-v1 as a mutating builder. They may parse Stage185-v1 source only to verify the frozen hash and replay the specified split algorithm in the new validator.

## L. Gate Boundary

Gate 5 PASS authorizes only Gate 6 Level-2 result review over the P4-B R1 emitted artifacts and the Gate-5 validation report.

Gate 5 PASS does not authorize Level-3 training, Level-3 evaluation, model loading, checkpoint mutation, dataset replacement, Kaggle execution, promotion criteria changes, or training admission. Level-3 requires a separate explicit authority after Gate 6.

## M. Final Specification-Readiness Token

P3W6F2P4D_CONTROLLED_DATA_INTEGRITY_GATE_SPEC_READY_FOR_INDEPENDENT_VERIFICATION
