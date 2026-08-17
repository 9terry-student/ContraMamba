# P3-W6-F2-P4-B Level-2 R1 Regeneration Specification

Title: P3-W6-F2-P4-B R1 CLEAN STRUCTURED REGENERATION CONTRACT

## A. Active Authority And Phase Boundary

The active frozen Level-1 authority commit is:

`acc078f8ddb5ba362d0c6861e23de21aad09cb8b`

The parent implementation/runtime authority identified by the Level-1 task context is:

`cf80d52c222450cf84622a4f830b7331355bee07`

The frozen Level-1 result authority is the immutable P3-W6-F2 hybrid human review artifact set:

- `reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_cf80d52c/p3w6f2_hybrid_review_completed.csv`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_cf80d52c/p3w6f2_hybrid_review_decisions.jsonl`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_execution_cf80d52c/p3w6f2_hybrid_review_summary.json`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_final_result_review.json`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_final_result_review.md`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_final_review_wip.jsonl`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_structural_cohort_audit_v1.json`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_structural_cohort_confirmation_v1.json`
- `reports/reason_router_p2_p3w6f2_hybrid_human_review_result_review_cf80d52c/p3w6f2_reviewer_alias_evidence_v1.json`

Additional context authority, for precedent only, is:

- `reports/reason_router_p2_p3w5_separate_remediation_spec.md`
- `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/*`
- P3-W6-F1 final regeneration and materialization specifications, manifests, compatibility reports, and result reports where explicitly cited as precedent.

The P3-W6-F2-P4-A source-of-truth audit supplied in the Codex task context reached:

`P3W6F2P4A_REGENERATION_SOURCE_AUDIT_PASS_R1`

This P4-B specification therefore authorizes only the R1 architecture:

`R1 CLEAN_STRUCTURED_REGENERATION`

This document is a Level-2 regeneration specification. It authorizes future implementation only after this specification itself receives independent specification verification and is committed/frozen at a full 40-character commit identity. It does not authorize regeneration execution, data mutation, training, evaluation, Kaggle execution, or Level-3 controlled-data/training admission.

Level-1 artifacts remain historical authority. Level-2 remediation is not complete until future implementation, execution, validation, and result review pass their own gates. Level-3 admission remains not released.

## B. Exact Source-Of-Truth Contract

The structured source producer for R1 regeneration is `scripts/build_controlled_v5.py`.

The authoritative symbols are:

- `_GENERATED_PREDICATES`
- `_BASE_PREDICATE_BY_INFLECTED`
- `_generated_fact_template(index: int)`
- `fact_templates_for_count(num_pairs: int)`
- `_FACT_FIELDS`
- `_statement(...)`
- `_paraphrase(...)`
- `_record(...)`
- `_build_records(...)`

The earliest clean regeneration source is the structured fact dictionary emitted by `_generated_fact_template(index)` and collected by `fact_templates_for_count(num_pairs)` before any surface evidence text is rendered by `_statement` or `_paraphrase`.

An authorized F2 `pair_id` maps to exactly one structured fact as follows:

1. Reconstruct the controlled-v5 template sequence with `fact_templates_for_count(...)` under the same source code and deterministic order as the historical controlled-v5 generation authority.
2. Select the structured fact whose `pair_id` exactly equals the authorized F2 `pair_id`.
3. Require exactly one match. Zero matches or multiple matches fail closed.
4. Require the matched structured fact slots to replay the historical row-level semantic slots implied by the frozen Level-1 F2 review source record. Any semantic-slot drift fails closed.

The permitted structured input fields are exactly the fields listed by `_FACT_FIELDS` in `scripts/build_controlled_v5.py`:

- `pair_id`
- `title`
- `name`
- `alternate_title`
- `alternate_name`
- `role`
- `alternate_role`
- `predicate`
- `alternate_predicate`
- `object`
- `alternate_object`
- `time`
- `alternate_time`
- `location`
- `alternate_location`

For P4-B R1 regeneration, the generation root may also consume these deterministic control identities from the historical dataset row and frozen Level-1 authority:

- authorized `pair_id`
- authorized row `id`
- `intervention_type`
- dataset row order
- split assignment identity
- frozen source artifact identities and hashes
- historical row values only for invariant comparison, delta validation, and provenance evidence.

The following are explicitly forbidden as generation inputs:

- historical malformed canonical evidence surface text
- historical malformed paraphrase evidence surface text
- Level-1 `canonical_evidence` as a root for generating new canonical evidence
- Level-1 `paraphrase_evidence` as a root for generating new paraphrase evidence
- historical source-record hashes as sources for new text construction

Historical malformed text may be read only for comparison, audit, delta proof, historical hash isolation, and fail-closed verification. It must never be the authoritative regeneration root.

## C. Explicit F2 Predicate Realization Authority

This specification creates the F2-specific predicate realization contract:

`P3W6F2P4B_R1_BASE_PREDICATE_REALIZATION_V1`

Under this contract, F2 Level-2 R1 regeneration may use `_BASE_PREDICATE_BY_INFLECTED` from `scripts/build_controlled_v5.py` to realize grammatical negative auxiliary evidence as:

`did not <base predicate>`

This is not inherited from the F1 textual-repair authority. F1 authority remains historical precedent only. P3-W6-F2 uses `_BASE_PREDICATE_BY_INFLECTED` because P4-A established that the earliest clean F2 regeneration source is structured and because the mapping exists before F2 surface regeneration as a repository source for base predicate realization.

The semantic predicate identity is the structured fact field:

`predicate`

The semantic predicate identity is distinct from both:

1. the historical inflected positive predicate surface, for example `restored`
2. the regenerated negative auxiliary base-form surface, for example `did not restore`

The seven authorized F2 predicate identities and required base realizations are:

| Structured semantic predicate | Required negative auxiliary base predicate |
| --- | --- |
| `approved` | `approve` |
| `delivered` | `deliver` |
| `launched` | `launch` |
| `opened` | `open` |
| `published` | `publish` |
| `restored` | `restore` |
| `selected` | `select` |

Future implementation must prove total coverage for every authorized F2 structured predicate. It must also prove that no extra F2 predicate appears outside this set. Missing mapping, duplicate mapping, extra mapping consumption, or ambiguous mapping fails closed.

The implementation must not rewrite the semantic predicate slot from the inflected historical identity to the base-form surface. The base form is a surface-realization value used only inside negative auxiliary rendering.

## D. R1 Member Generation Contract

For every authorized F2 pair, future implementation must independently regenerate exactly three members from the structured fact dictionary:

1. canonical `none` member with `final_label == REFUTE`
2. `paraphrase` member with `final_label == REFUTE`
3. `polarity_flip` member with `final_label == SUPPORT`

The canonical member must be generated from the structured fact and the canonical statement template. For F2 REFUTE canonical evidence, negative auxiliary realization must use:

`did not <base predicate>`

The paraphrase member must be generated from the structured fact and the paraphrase template. For F2 REFUTE paraphrase evidence, negative auxiliary realization must use:

`did not <base predicate>`

The polarity-flip member must be generated independently from the same structured fact, with affirmative evidence corresponding to the claim-side proposition. For authorized F2 rows this means the regenerated polarity-flip output is expected to be byte-identical to the historical affirmative polarity-flip dataset row, but that byte identity does not permit deriving the member from historical canonical or paraphrase text.

Regenerated paraphrase must not consume regenerated canonical evidence text.

Regenerated polarity_flip must not consume regenerated canonical evidence text.

All three members must be regenerated from structured slots and fixed templates. A future implementation must make this independence observable in per-member provenance.

## E. Exact Identity Preservation

The authorized universe is exactly:

- 119 F2 pairs
- 357 F2 members
- exactly one `none` member per pair
- exactly one `paraphrase` member per pair
- exactly one `polarity_flip` member per pair

The dataset row schema in `scripts/build_controlled_v5.py` is exactly:

- `id`
- `pair_id`
- `claim`
- `evidence`
- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`
- `intervention_type`

For every authorized F2 member, these fields must remain byte-identical to the historical controlled-v5 row:

- `id`
- `pair_id`
- `claim`
- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`
- `intervention_type`

The `evidence` field is the only dataset row field permitted to change, and only for authorized F2 `none` and `paraphrase` members.

For authorized F2 `polarity_flip` members, no dataset row field is permitted to change. The member must still be regenerated from the structured source and included in provenance and deterministic replay evidence.

For all non-authorized rows in a full dataset materialization, every dataset field must remain byte-identical to the historical controlled-v5 dataset row.

Pair IDs remain exactly stable. Member row IDs remain exactly stable. Row order remains exactly stable relative to the historical full controlled-v5 dataset. Split assignment identity must remain exactly stable under the existing deterministic pair split contract used by Stage185 sidecar construction.

Canonical linkage fields must remain exact:

- canonical row is the member with `intervention_type == none`
- derivative rows link to the same `pair_id`
- all three members in a pair preserve the same `claim`
- Stage185-style sidecars must preserve `canonical_row_id` as the canonical member `id`

Labels must remain exact for each authorized F2 triple:

- canonical `none`: `final_label == REFUTE`
- paraphrase: `final_label == REFUTE`
- polarity_flip: `final_label == SUPPORT`

Polarity labels must remain exact:

- canonical `none`: `polarity_label == REFUTE`
- paraphrase: `polarity_label == REFUTE`
- polarity_flip: `polarity_label == SUPPORT`

Frame, predicate, and sufficiency labels must remain:

- `frame_compatible_label == 1`
- `predicate_covered_label == 1`
- `sufficiency_label == 1`

Primary failure type must remain exact:

- canonical `none`: `primary_failure_type == none`
- paraphrase: `primary_failure_type == none`
- polarity_flip: `primary_failure_type == polarity`

Source fact identity is the exact `pair_id` and complete `_FACT_FIELDS` dictionary used for regeneration. It must be recorded in Level-2 provenance and must be independently replayable.

Historical Level-1 review schemas remain historical and must not be mutated. In particular, the Level-1 source-record schema fields from `scripts/reason_router_p3w6f2_manual_review.py` are historical review evidence, not a Level-2 row schema. Their `source_record_sha256` values must remain historical.

## F. Exact Text-Delta Contract

The semantic invariants are:

- same pair identity
- same member identity
- same claim proposition
- same structured fact slots
- same semantic predicate identity
- same entity/person slots
- same role slot
- same object slot
- same location slot
- same time/month slot
- same final labels
- same polarity labels
- same frame, predicate, and sufficiency labels
- same primary failure type
- same split identity
- same row order

The permitted surface-realization changes are limited to authorized F2 canonical and paraphrase evidence:

- replace the defective negative auxiliary predicate surface `did not <inflected predicate>` with grammatical `did not <base predicate>`
- preserve all other structured slots and template text generated from the structured fact
- preserve the negative polarity of the evidence

This is not a surface patch architecture. The future implementation must regenerate the evidence from structured slots and then compare the resulting text against historical rows to prove that the only dataset row delta is the negative auxiliary predicate surface.

Expected row-field deltas:

| Member type | Field | Expected delta |
| --- | --- | --- |
| `none` | `evidence` | changed from malformed `did not <inflected predicate>` to grammatical `did not <base predicate>` |
| `paraphrase` | `evidence` | changed from malformed `did not <inflected predicate>` to grammatical `did not <base predicate>` |
| `polarity_flip` | all dataset fields | no change |

No other dataset row fields may change for any authorized F2 member.

Provenance changes are required in new Level-2 artifacts. Provenance changes must use a new P4-B namespace and must not overwrite historical P3-W4, P3-W5, P3-W6-F1, or Level-1 F2 provenance.

Historical fields that must never be rewritten include:

- P3-W4 source dataset hashes
- P3-W4 grammar authority hashes
- Level-1 F2 `source_record_sha256`
- structural-audit hashes
- cohort-confirmation hashes
- reviewer identifiers, including the non-blocking same-human alias provenance
- Level-1 review decisions
- Level-1 result review summaries

## G. Full-Dataset Versus F2-Subset Materialization Contract

Future P3-W6-F2-P4-B Level-2 implementation must emit a full controlled dataset artifact containing regenerated F2 rows.

The selected materialization architecture is:

`FULL_CONTROLLED_DATASET_WITH_F2_R1_REGENERATED_ROWS`

This is the smallest repository-supported architecture that preserves immutable historical data, provides exact downstream hash checking, supports eventual Level-3 admission, and follows the successful P3-W6-F1 precedent of emitting a new full dataset rather than mutating `data/controlled_v5_v3_without_time_swap.jsonl`.

The historical malformed dataset must not be modified. The regenerated full dataset must be emitted under a new Level-2 execution report directory. F2-only regenerated-row and audit artifacts are required as evidence, but they are not sufficient by themselves for eventual controlled-data admission because downstream consumers require a complete dataset identity.

The future implementation must prove:

- full output row count equals the historical full controlled-v5 row count
- authorized F2 row count equals 357
- authorized F2 pair count equals 119
- unchanged non-F2 rows are byte-identical to the historical full dataset
- row order is byte-stable except for permitted `evidence` field changes in F2 canonical and paraphrase rows
- output hashes are computed in the new P4-B namespace

## H. Exact Level-2 Artifact Set And Schemas

Future implementation must write all artifacts into a new directory named:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_<execution_commit>/`

`<execution_commit>` must be the full 40-character commit SHA of the future implementation execution state. If the future execution protocol permits a separate frozen implementation commit and execution commit, both must be recorded, but the directory name must still contain the execution commit.

Required artifact 1:

`controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Purpose: full controlled dataset with regenerated F2 rows.

Schema version token:

`P3W6F2P4B_R1_FULL_DATASET_V1`

Each row must use exactly the controlled-v5 dataset row fields, in this order:

- `id`
- `pair_id`
- `claim`
- `evidence`
- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`
- `intervention_type`

Required artifact 2:

`p3w6f2_p4b_r1_regenerated_members.jsonl`

Purpose: one regenerated-member record for each authorized F2 member.

Schema version token:

`P3W6F2P4B_R1_REGENERATED_MEMBER_V1`

Required fields:

- `schema_version`
- `generation_contract_version`
- `pair_id`
- `member_id`
- `intervention_type`
- `member_role`
- `structured_fact`
- `structured_fact_sha256`
- `semantic_predicate`
- `base_predicate`
- `negative_auxiliary_realization`
- `claim`
- `regenerated_evidence`
- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`
- `generation_root`
- `generation_template`
- `historical_member_id`
- `historical_row_sha256`
- `regenerated_row_sha256`
- `row_field_delta_keys`
- `old_text_used_for_generation`
- `source_authority_commit`
- `execution_commit`

`old_text_used_for_generation` must be `false`.

Required artifact 3:

`p3w6f2_p4b_r1_regeneration_audit.jsonl`

Purpose: per-member before/after and invariant audit for all 357 authorized F2 members.

Schema version token:

`P3W6F2P4B_R1_MEMBER_AUDIT_V1`

Required fields:

- `schema_version`
- `pair_id`
- `member_id`
- `intervention_type`
- `historical_row`
- `regenerated_row`
- `field_delta`
- `permitted_delta`
- `semantic_slot_preservation`
- `label_preservation`
- `identity_preservation`
- `row_order_preservation`
- `split_preservation`
- `structured_source_replay_status`
- `predicate_base_mapping_status`
- `old_text_isolation_status`
- `member_audit_status`
- `failure_reasons`

Required artifact 4:

`p3w6f2_p4b_r1_regeneration_summary.json`

Purpose: execution-level summary and hash manifest.

Schema version token:

`P3W6F2P4B_R1_REGENERATION_SUMMARY_V1`

Required fields:

- `schema_version`
- `generation_contract_version`
- `level1_freeze_commit`
- `parent_runtime_authority_commit`
- `implementation_commit`
- `execution_commit`
- `head_clean_required`
- `authority_artifacts`
- `authority_artifact_sha256`
- `historical_dataset_path`
- `historical_dataset_sha256`
- `regenerated_dataset_path`
- `regenerated_dataset_sha256`
- `regenerated_dataset_semantic_sha256`
- `authorized_pair_count`
- `authorized_member_count`
- `changed_pair_count`
- `changed_member_count`
- `canonical_changed_member_count`
- `paraphrase_changed_member_count`
- `polarity_flip_changed_member_count`
- `unchanged_non_f2_row_count`
- `predicate_base_mapping_version`
- `predicate_base_mapping_sha256`
- `structured_source_producer`
- `structured_source_producer_sha256`
- `deterministic_invocation_sha256`
- `artifact_set_complete`
- `fail_closed_status`
- `created_at_utc`

Required artifact 5:

`p3w6f2_p4b_r1_full_output_isolation.json`

Purpose: proof that only authorized F2 canonical and paraphrase `evidence` fields changed in the full output.

Schema version token:

`P3W6F2P4B_R1_FULL_OUTPUT_ISOLATION_V1`

Required fields:

- `schema_version`
- `historical_dataset_path`
- `historical_dataset_sha256`
- `regenerated_dataset_path`
- `regenerated_dataset_sha256`
- `row_count_historical`
- `row_count_regenerated`
- `authorized_pair_count`
- `authorized_member_count`
- `authorized_changed_row_ids`
- `authorized_unchanged_row_ids`
- `unauthorized_changed_row_ids`
- `field_delta_counts`
- `row_order_identical`
- `non_f2_rows_byte_identical`
- `isolation_status`
- `failure_reasons`

Required artifact 6:

`p3w6f2_p4b_r1_deterministic_generator_invocation.json`

Purpose: deterministic invocation and configuration identity.

Schema version token:

`P3W6F2P4B_R1_INVOCATION_V1`

Required fields:

- `schema_version`
- `command`
- `arguments`
- `environment_policy`
- `python_version`
- `locale_policy`
- `timezone_policy`
- `random_seed_policy`
- `input_paths`
- `output_directory`
- `source_authority_commit`
- `implementation_commit`
- `execution_commit`
- `dirty_tracked_worktree_allowed`
- `deterministic_invocation_sha256`

Required artifact 7:

`p3w6f2_p4b_r1_base_form_coverage.json`

Purpose: total predicate/base-form coverage proof.

Schema version token:

`P3W6F2P4B_R1_BASE_FORM_COVERAGE_V1`

Required fields:

- `schema_version`
- `predicate_realization_contract_version`
- `mapping_source_symbol`
- `mapping_source_sha256`
- `authorized_predicates`
- `observed_authorized_predicates`
- `required_base_forms`
- `missing_mappings`
- `extra_observed_predicates`
- `ambiguous_mappings`
- `pair_coverage`
- `coverage_status`

Required artifact 8:

`p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl`

Purpose: per-member Stage185 compatibility/equivalence evidence.

Schema version token:

`P3W6F2P4B_R1_STAGE185_COMPATIBILITY_ROW_V1`

Required fields:

- `schema_version`
- `compatibility_rule_version`
- `pair_id`
- `member_id`
- `intervention_type`
- `raw_stage185_changed_axes`
- `raw_stage185_expected_axes`
- `raw_stage185_statuses`
- `historical_semantic_predicate`
- `regenerated_negative_base_surface`
- `structured_fact`
- `semantic_slot_preservation`
- `permitted_predicate_realization_delta`
- `effective_compatibility_status`
- `effective_reason_codes`
- `training_admission_effect`

Required artifact 9:

`p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json`

Purpose: aggregate Stage185 compatibility gate result.

Schema version token:

`P3W6F2P4B_R1_STAGE185_COMPATIBILITY_SUMMARY_V1`

Required fields:

- `schema_version`
- `compatibility_rule_version`
- `row_count`
- `authorized_pair_count`
- `authorized_member_count`
- `raw_stage185_predicate_axis_observation_count`
- `permitted_predicate_realization_delta_count`
- `compatibility_pass_count`
- `compatibility_fail_count`
- `compatibility_unresolved_count`
- `stage185_v1_mutated`
- `historical_authority_weakened`
- `training_admission_released`
- `compatibility_gate_status`
- `failure_reasons`

Required artifact 10:

`p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json`

Purpose: provenance and source hash manifest for the compatibility gate.

Schema version token:

`P3W6F2P4B_R1_STAGE185_COMPATIBILITY_PROVENANCE_V1`

Required fields:

- `schema_version`
- `compatibility_rule_version`
- `stage185_source_script`
- `stage185_source_script_sha256`
- `historical_stage185_authority`
- `historical_stage185_authority_sha256`
- `regenerated_dataset_path`
- `regenerated_dataset_sha256`
- `structured_source_producer`
- `structured_source_producer_sha256`
- `base_form_coverage_path`
- `base_form_coverage_sha256`
- `compatibility_rows_path`
- `compatibility_rows_sha256`
- `compatibility_summary_path`
- `compatibility_summary_sha256`
- `created_at_utc`

Timestamp policy:

- Dataset JSONL rows must contain no wall-clock timestamp.
- Per-member deterministic records must avoid wall-clock timestamps unless the timestamp is explicitly excluded from semantic hashes.
- Manifest timestamps may appear as `created_at_utc`.
- Physical SHA256 hashes must be computed over exact artifact bytes.
- Semantic/canonical SHA256 hashes must use deterministic JSON canonicalization with documented field ordering and must exclude explicitly declared non-semantic runtime timestamp fields.

All JSON and JSONL records must use deterministic ordering, UTF-8 encoding, LF line endings, `ensure_ascii=false`, no NaN values, and compact or explicitly documented separators. Any deviation must be reported as a fail-closed validation failure.

## I. Historical Hash Isolation

P3-W4 hashes remain historical.

P3-W6-F2 Level-1 `source_record_sha256` values remain historical.

Structural-audit and cohort-confirmation hashes remain historical.

No Level-2 P4-B artifact may masquerade as, overwrite, rename, or reassign those authorities. No Level-2 artifact may replace the meaning of `F2_SOURCE_RECORD_HASH_V1`.

New regenerated data must receive a new provenance/hash namespace:

`P3W6F2P4B_R1_REGENERATION_HASH_V1`

The new namespace must distinguish:

- historical malformed source rows
- regenerated Level-2 rows
- per-member structured fact hashes
- full dataset physical hash
- full dataset semantic hash
- compatibility evidence hashes

Historical malformed text may appear inside audit fields only when clearly labeled as historical comparison evidence.

## J. Stage185 / Semantic-Equivalence Compatibility

The current Stage185 sidecar authority in `scripts/build_stage185a_controlled_train_integrity_sidecar.py` must not be mutated or weakened. Stage185-v1 observes semantic state by checking literal predicate surface containment. Therefore, a row changed from:

`did not restored`

to:

`did not restore`

may be observed by Stage185-v1 as a predicate-axis surface change even though the structured semantic predicate identity remains `restored`.

Future Level-2 implementation must not silently suppress that observation. It must create and pass the dedicated compatibility rule:

`P3W6F2P4B_R1_STAGE185_PREDICATE_REALIZATION_COMPATIBILITY_V1`

Scope of this compatibility rule:

- authorized P3-W6-F2 pairs only
- authorized P3-W6-F2 canonical `none` and `paraphrase` members only for negative auxiliary base-form realization
- authorized P3-W6-F2 `polarity_flip` members only for proving independent affirmative regeneration and no row-field mutation
- no F1 behavior
- no non-F2 behavior
- no time-swap, frame, object, location, role, name, title, or label changes

The compatibility rule must prove that historical semantic predicate identity and regenerated negative auxiliary realization are equivalent under P4-B R1 only when all of the following hold:

- structured `predicate` is unchanged
- `_BASE_PREDICATE_BY_INFLECTED[predicate]` equals the regenerated base predicate
- regenerated evidence contains exactly `did not <base predicate>`
- historical malformed evidence contains exactly the corresponding `did not <inflected predicate>`
- all non-predicate structured slots remain preserved
- final, polarity, frame, predicate-covered, sufficiency, and primary-failure labels remain unchanged
- the raw Stage185-v1 predicate-axis observation is recorded
- the effective compatibility status is derived in a separate P4-B artifact and never by editing Stage185-v1 output

A dedicated F2 compatibility sidecar/materializer is required. It must consume the regenerated full dataset and produce the Stage185 compatibility artifacts listed in section H. The future gate that consumes it is:

`P3W6F2P4B_R1_STAGE185_COMPATIBILITY_GATE`

Passing this gate does not authorize training admission. It only removes the P4-A Stage185 compatibility blocker for Level-2 result review. Level-3 admission remains blocked until a separate post-remediation validation and admission authority is created and passed.

## K. Fail-Closed Implementation Requirements

Future implementation must reject at least the following conditions:

- current HEAD does not match the declared execution commit
- execution commit is not a full 40-character SHA
- required authority commit or artifact identity is missing
- Level-1 freeze commit mismatch
- dirty tracked worktree when execution identity requires a clean state
- missing authority artifact
- authority artifact SHA mismatch
- historical dataset SHA mismatch
- F2 universe not exactly 119 pairs
- member universe not exactly 357 members
- unauthorized pair
- missing pair
- duplicate pair
- unauthorized member
- missing member
- duplicate member
- missing `none`, `paraphrase`, or `polarity_flip` member in any authorized pair
- extra F2 member in any authorized pair
- structured fact replay mismatch
- missing predicate base mapping
- extra observed F2 predicate outside the authorized seven-predicate set
- ambiguous predicate base mapping
- semantic-slot drift
- label drift
- polarity-label drift
- frame/predicate/sufficiency drift
- row-ID drift
- pair-ID drift
- row-order drift
- split drift
- canonical linkage drift
- any generation path reading old malformed canonical or paraphrase evidence text as input
- nondeterministic replay
- existing conflicting output
- partial output set
- source SHA/provenance mismatch
- malformed JSON or JSONL
- non-LF line endings in generated artifacts
- output hash mismatch
- Stage185 compatibility unresolved or failed

Atomicity expectations:

- write to a temporary staging directory outside the final artifact directory
- validate the complete artifact set in staging
- compute all physical and semantic hashes before promotion
- promote to the final execution directory only after all required artifacts are complete
- never overwrite an existing final directory
- if an existing final directory is present, accept it only as an explicitly requested idempotent read-only replay and only when every required artifact hash matches the expected deterministic output
- otherwise fail with an existing-output error

Idempotence expectations:

- repeated execution at the same code, authority, input, and invocation identity must produce byte-identical deterministic artifacts, excluding declared runtime timestamp fields from semantic hashes
- no partial artifact set may be accepted as success
- no validation failure may leave a final-looking complete directory

## L. Post-Regeneration Validation Gates

Future validation must be ordered and phase-separated:

1. Code correctness gate

   Static and unit tests for the new F2-specific implementation only. This gate verifies code behavior and does not execute official regeneration authority.

2. Deterministic regeneration execution gate

   Run the authorized P4-B R1 regeneration command at a specific full execution commit. This gate emits the Level-2 artifact set only.

3. Artifact/provenance validity gate

   Verify artifact presence, schemas, counts, source hashes, output hashes, row order, full-output isolation, structured fact replay, and deterministic replay.

4. Semantic/grammar equivalence gate

   Verify grammatical canonical and paraphrase negative construction, polarity_flip correctness, old-text isolation, semantic-slot preservation, base-form coverage, and the P4-B Stage185 predicate-realization compatibility artifacts.

5. Controlled-data integrity gate

   Verify the regenerated full dataset under controlled-data integrity rules without mutating historical Stage185-v1 authority. Any successor effective compatibility must be recorded as P4-B evidence.

6. Level-2 result review gate

   Produce an independent Level-2 result review over the emitted artifacts. This review determines whether Level-2 remediation is complete.

7. Separate Level-3 training-admission decision gate

   A new authority must explicitly admit the regenerated controlled dataset to Level-3 training or evaluation. No previous gate implicitly authorizes this phase.

## M. Future Implementation Scope

Future implementation should be additive and F2-specific. The minimal likely whitelist is:

- `scripts/build_controlled_v5.py`
  - may add `build_controlled_records_with_f2_p4b_r1_regeneration(...)`
  - may add `build_controlled_records_with_f2_p4b_r1_regeneration_audit(...)`
  - may add narrowly scoped F2 helper symbols for negative auxiliary base-form realization
  - must not alter existing historical `build_controlled_records(...)` behavior
  - must not alter existing F1 API behavior

- `scripts/regenerate_reason_router_p3w6f2_p4b_r1_structured.py`
  - may be added as the deterministic Level-2 artifact materializer

- `scripts/analyze_reason_router_p3w6f2_p4b_r1_regeneration.py`
  - may be added as the artifact/provenance/full-output isolation analyzer

- `scripts/materialize_reason_router_p3w6f2_p4b_r1_stage185_compatibility.py`
  - may be added as the dedicated F2 Stage185 compatibility sidecar/materializer

- `tests/test_reason_router_p3w6f2_p4b_r1_regeneration.py`
  - may be added for the R1 regeneration contract

- `tests/test_reason_router_p3w6f2_p4b_r1_stage185_compatibility.py`
  - may be added for the Stage185 compatibility contract

Existing historical F1 APIs and behavior must remain byte/behavior compatible. This includes `build_controlled_records_with_f1_polarity_repair(...)`, `build_controlled_records_with_f1_polarity_repair_audit(...)`, and all frozen P3-W6-F1 result artifacts.

No unrelated refactor is authorized.

## N. Exact Test Contract

Future tests must cover:

- exact 119 authorized F2 pairs
- exact 357 authorized F2 members
- exactly one `none`, one `paraphrase`, and one `polarity_flip` member per authorized pair
- all seven F2 predicates and base mappings:
  - `approved -> approve`
  - `delivered -> deliver`
  - `launched -> launch`
  - `opened -> open`
  - `published -> publish`
  - `restored -> restore`
  - `selected -> select`
- structured-source regeneration from `_generated_fact_template` / `fact_templates_for_count`
- canonical negative grammar uses `did not <base predicate>`
- paraphrase negative grammar uses `did not <base predicate>`
- polarity_flip is regenerated independently from structured source
- old malformed canonical/paraphrase text is not consumed as generation input
- semantic-slot preservation
- exact label preservation
- exact polarity-label preservation
- exact frame/predicate/sufficiency preservation
- exact row ID preservation
- exact pair ID preservation
- exact row order preservation
- exact split preservation
- unchanged non-F2 rows when the full dataset is emitted
- deterministic replay and byte-identical output under identical invocation
- provenance/hash spoof rejection
- source SHA mismatch rejection
- partial output rejection
- conflicting output rejection
- historical artifact immutability
- Stage185 raw predicate-axis observation is retained
- P4-B effective predicate-realization compatibility is derived only in new artifacts
- Level-3 training admission remains blocked

Tests must not mutate frozen Level-1 artifacts. Tests must not normalize reviewer IDs. Tests must not execute training or evaluation.

## O. Execution Boundary

This P4-B task itself does not execute regeneration.

Future implementation validation is not regeneration authority.

Future commit or push does not itself authorize Kaggle execution.

Kaggle execution must be separately authorized at a specific full 40-character commit after implementation and validation. A Kaggle sync or checkout step must verify the exact commit before any future authorized execution. No Stage185 run, training run, evaluation run, or Level-3 admission may be inferred from this specification.

## P. Final Decision Token

P3W6F2P4B_R1_REGENERATION_SPEC_READY_FOR_INDEPENDENT_VERIFICATION
