# P3-W5 Separate Remediation Specification

## Static Status

Decision: `P3W5_SEPARATE_REMEDIATION_SPEC_READY_FOR_STATIC_REVIEW`.

Revision history:

- v1 static review: `P3W5_SPEC_STATIC_REVIEW_BLOCKED_REVISION_REQUIRED`
- v2 static review: `P3W5_SPEC_V2_STATIC_REVIEW_BLOCKED_REVISION_REQUIRED`
- v3 static review: `P3W5_SPEC_V3_STATIC_REVIEW_BLOCKED_REVISION_REQUIRED`
- v4 static review: `P3W5_SPEC_V4_STATIC_REVIEW_BLOCKED_SURGICAL_REVISION_REQUIRED`

This is a static remediation specification only. It does not implement code, modify data, regenerate rows, perform human annotation, run Python, run pytest, run py_compile, run analyzers, load models or tokenizers, train, evaluate, release A1/A2/A3, release polarity supervision, or perform Git actions.

## Authority

Final authority commit: `f0a9afddc5b93c54aa72b0335c5a1a2f517cf934`.

P3-W4 implementation and execution commit: `ca99038d812696467a4330cffc1c4c5b5f72cfe2`.

P3-W4 final result decision: `P3W4_RESULT_REVIEW_PASS_F1_REGENERATION_F2_MANUAL_REVIEW_REQUIRED`.

Authority artifacts:

- `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_summary.json`
- `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_canonical_grammar_authority_pairs.jsonl`
- `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_f2_manual_review.csv`

Authority artifact SHA-256 identities:

- `p3w4_canonical_grammar_authority_summary.json`: `7c0cc383dde38a1c564dae445a78eaf9171b8648d0720de3a2acc0ba68e68e80`
- `p3w4_canonical_grammar_authority_pairs.jsonl`: `850ac6e8924fe334fa7f18659d204f6e0546381b1c3d3eb601f893f3eb00a493`
- `p3w4_f2_manual_review.csv` Git canonical LF SHA: `ccc539e743d1a4226391cdca1422bb0a1054c53fd7c53a4210a54271d1e9e8a5`
- `p3w4_f2_manual_review.csv` original execution CRLF SHA: `0c3c0f85bed08cedc9d664a3c685d1c40560dbe3f9bd5b8bb88543ed9e528515`

P3-W3 authority and existing P2 contracts remain in force:

- Conditional First-Blocker Reason Router
- Reason-Specific Supervision
- Explicit Gradient Ownership
- `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`
- final 3-way CE = router-only
- F/P/S/polarity inputs detached from final CE
- secondary reasons = diagnostic only
- EMA = observer/baseline only
- A0-A3 and E0 preserved

## Partition Authority

P3-W5 uses the P3-W4 pair JSONL partition unchanged.

Affected pair universe:

- total affected pairs: 240
- F1 supporting pairs: 121
- F2 blocking pairs: 119
- F1 intersection F2: empty
- F1 union F2: complete affected-pair universe

Row namespaces:

- F1 remediation target: 121 `polarity_flip` rows
- F2 review target: 119 pair families, 357 member rows, consisting of canonical `none`, `paraphrase`, and `polarity_flip`
- P3-W3 exported REFUTE rows: 359
- P3-W4 affected member rows: 478

F1 and F2 do not share pair ID authority, row ID authority, candidate output authority, decision authority, or release-gate authority.

## F1 Regeneration Specification

### Purpose

F1 defines the future execution contract for regenerating the 121 `polarity_flip` rows whose defects were reproduced by the P3-W4 production grammar validator.

F1 is not an immediate training-authority recovery of existing rows. The required future sequence is:

1. generator/template repair
2. deterministic regeneration
3. full-output generator impact isolation
4. grammar and integrity audit
5. semantic and lineage validation
6. result review
7. separate release decision

### Input Authority

The F1 target set is exactly the P3-W4 pair records satisfying all of:

- `family == F1`
- `automatic_root_cause_class == F1_TRUE_POLARITY_GENERATION_DEFECT`
- `remediation_state == REGENERATION_REQUIRED`
- `pair_id` is in `decision_supporting_pair_ids`

The target cardinality must be exactly 121 pairs and 121 `polarity_flip` rows.

### Candidate Scope

The row-level F1 candidate scope permits regeneration of only the F1 `polarity_flip` member.

The following must not be modified as accepted candidate rows:

- canonical `none` member
- optional `paraphrase` member
- any F2 family member
- unaffected dataset rows

Each regenerated candidate must preserve at least:

- `pair_id`
- `intervention_type == polarity_flip`
- `final_label == REFUTE`
- `polarity_label == REFUTE`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `primary_failure_type`
- canonical linkage
- split membership
- fact/template identity

Row ID decision: existing `row_id` is preserved as the stable logical identity. The future regeneration artifact must record old text and regenerated text in separate provenance fields. A future stage that permits row ID replacement must first define a complete old-to-new identity mapping and prove a complete bijection.

### Generator Impact Isolation

The F1 generator/template code impact scope is distinct from the row-level candidate scope. A shared generator or template repair must prove full-output isolation, not merely export 121 candidate rows.

`F1_source_identity_contract` fixes complete generator output identity:

- `dataset_identity_field = "id"`
- `normalized_artifact_identity_field = "row_id"`
- `normalized_row_id_derivation = source_row["id"]`

The full-output baseline/repaired comparison key is the dataset source row `id`:

- `comparison_key = "id"`

`authorized_F1_row_ids` means the exact 121 IDs extracted from the P3-W4 pair artifact polarity member `source_row.id` values.

Future artifacts must satisfy:

- `original_row_id == source_original["id"]`
- `regenerated_row_id == source_regenerated["id"]`
- `original_row_id == regenerated_row_id`

`F1_generator_impact_isolation_contract` requires:

- `baseline_generator_commit`
- `baseline_generator_source_path`
- `baseline_generator_source_sha256`
- `repaired_generator_commit`
- `repaired_generator_source_path`
- `repaired_generator_source_sha256`
- `deterministic_generator_invocation`
- `generator_configuration_identity`
- `baseline_complete_output_sha256`
- `repaired_complete_output_sha256`
- `baseline_row_count`
- `repaired_row_count`
- `baseline_id_sequence`
- `repaired_id_sequence`
- `baseline_id_sequence_sha256`
- `repaired_id_sequence_sha256`

The complete baseline output and complete repaired output must be compared by source row `id`. The only allowed changed source-row ID set is exactly the 121 authorized F1 `polarity_flip` IDs. `baseline_id_sequence` and `repaired_id_sequence` are the source-row `id` arrays extracted from the complete generator outputs in row order.

Required full-output comparison results:

- `changed_ids == authorized_F1_row_ids`
- `missing_ids == []`
- `added_ids == []`
- `duplicate_ids == []`
- `row_order_changed == false`
- `unauthorized_changed_row_ids == []`
- `F2_changed_row_ids == []`
- `unaffected_changed_row_ids == []`
- `canonical_changed_row_ids == []`
- `paraphrase_changed_row_ids == []`
- `baseline_id_sequence == repaired_id_sequence`
- `baseline_id_sequence_sha256 == repaired_id_sequence_sha256`

For the 121 authorized F1 rows, the only allowed source-row field change is exactly:

- `authorized_changed_source_fields == ["evidence"]`

Required changed-field comparison results:

- `evidence_changed_row_ids == authorized_F1_row_ids`
- `claim_changed_row_ids == []`
- `non_text_field_changed_row_ids == []`

Each authorized F1 row must preserve byte/semantic identity for:

- `claim`
- `id`
- `row_id` as the normalized artifact identity derived from `id`
- `pair_id`
- `intervention_type`
- `final_label`
- `polarity_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `primary_failure_type`
- canonical linkage
- split
- fact identity
- all other source-row fields

Generator code or template identity may differ in provenance, but if any complete dataset row output field other than `evidence` differs, isolation fails.

Set equality, changed-ID equality, and per-row field equality do not substitute for row-order equality.

If a shared generator/template change fails this full-output contract, F1 execution fails. Candidate-only export of 121 rows does not satisfy generator impact isolation.
### Candidate Acceptance Contract

Each F1 candidate must satisfy all of:

- exactly one candidate per authorized F1 pair
- no F2 pair included
- no duplicate `pair_id`
- no duplicate `row_id`
- nonempty textual change from the defective source
- same intended polarity semantics established through the semantic validation derivation contract
- same canonical relationship
- production grammar validator no longer reports anomaly
- Stage185 integrity sidecar contract passes
- no new frame, predicate, or sufficiency defect
- no label mutation
- no split movement
- no `time_swap` admission
- full-output generator impact isolation passes

Simple string replacement is not automatically accepted. For example, a repair such as `did not approved` to `did not approve` must still be generated under production generator/template authority or an explicit deterministic repair contract.

### Semantic Validation Derivation

F1 semantic validation status enum:

- `DETERMINISTIC_POLARITY_REPAIR_PASS`
- `MANUAL_REVIEW_REQUIRED`
- `REJECTED`

`semantic_polarity_preserved` must not be an arbitrary boolean. Future F1 execution must derive it from Stage185 state, full-output impact isolation, and explicit text-transformation evidence.

Required Stage185 baseline-to-accepted transition token:

- `F1_integrity_transition: INELIGIBLE_TO_ELIGIBLE`

Required baseline `polarity_flip` state:

- `grammar_status == FAIL`
- `integrity_status == INELIGIBLE`
- `canonical_status == PASS`

Required accepted regenerated `polarity_flip` state:

- `dataset_source_status == PASS`
- `schema_status == PASS`
- `intervention_contract_status == PASS`
- `grammar_status == PASS`
- `integrity_status == ELIGIBLE`
- `canonical_status == PASS`
- `polarity_contamination_status == PASS`
- `time_swap_status == PASS`
- `audit_expected_axes == ["polarity"]`
- `audit_changed_axes == ["polarity"]`
- `audit_pair_failure_scope == "none"`

`PASS` is not an allowed `integrity_status` value in this contract.

Required preserved-axis identity:

- frame
- predicate
- sufficiency
- name
- role
- title
- location
- object
- time

Each F1 audit record must include canonical-to-regenerated text transformation evidence:

- `canonical_text`
- `original_defective_text`
- `regenerated_text`
- `normalized_changed_span`
- `negation_markers_added`
- `negation_markers_removed`
- `auxiliary_verb_changes`
- `predicate_inflection_changes`
- `duplicate_or_missing_tokens`
- `semantic_validation_method`
- `semantic_validation_evidence`

Positive F1 semantic authority requires all of the following:

- original defective evidence contains the authority-reproduced `did not <inflected predicate>` span
- regenerated evidence preserves exactly one `did` auxiliary and exactly one `not` negation marker
- the predicate governed by `did not` is converted to the generator-authorized base form
- claim is byte-identical
- all evidence tokens outside the authorized predicate-inflection replacement span are byte-identical
- no negation marker is added or removed relative to the defective polarity evidence
- no entity, role, title, object, location, or time token changes
- `final_label` remains `REFUTE`
- `polarity_label` remains `REFUTE`
- Stage185 accepted post-state passes
- full-output impact isolation passes

Base-form derivation provenance must be recorded:

- `inflected_predicate_surface`
- `expected_base_predicate`
- `base_form_derivation_method`
- `base_form_derivation_source_path`
- `base_form_derivation_source_sha256`
- `authorized_replacement_span`
- `outside_span_byte_identity`

`base_form_derivation_method` must be an explicitly approved deterministic generator rule. Arbitrary LLM judgment and the label itself must not be used as semantic proof.

A candidate can be accepted only when:

- `semantic_validation_status == DETERMINISTIC_POLARITY_REPAIR_PASS`
- `semantic_polarity_preserved == true`
- `candidate_accepted == true`

If any positive semantic authority requirement is not proven:

- `semantic_validation_status = MANUAL_REVIEW_REQUIRED`
- `semantic_polarity_preserved = null`
- `candidate_accepted = false`
- `ordered_rejection_codes` includes `SEMANTIC_AUTHORITY_UNRESOLVED`

If an explicit semantic contradiction is confirmed:

- `semantic_validation_status = REJECTED`
- `semantic_polarity_preserved = false`
- `candidate_accepted = false`
### Future Output Contract

A future F1 execution stage must produce:

- `p3w5_f1_regeneration_summary.json`
- `p3w5_f1_regenerated_rows.jsonl`
- `p3w5_f1_regeneration_audit.jsonl`

F1 execution status and decision namespace:

- `F1_execution_status`: `P3W5_F1_REGENERATION_COMPLETE_ALL_CANDIDATES_ACCEPTED_PENDING_RESULT_REVIEW` or `P3W5_F1_REGENERATION_COMPLETE_WITH_BLOCKERS_PENDING_RESULT_REVIEW`
- `F1_execution_decision`
- `F1_artifact_paths`
- `F1_input_sha256`
- `F1_execution_commit`
- `F1_output_sha256`

F1 summary must include count and pair-ID array symmetry at least:

- `F1_target_pair_count`
- `F1_target_pair_ids`
- `F1_generated_candidate_count`
- `F1_generated_candidate_pair_ids`
- `F1_accepted_candidate_count`
- `F1_accepted_candidate_pair_ids`
- `F1_manual_review_required_count`
- `F1_manual_review_required_pair_ids`
- `F1_rejected_candidate_count`
- `F1_rejected_candidate_pair_ids`
- `F1_missing_candidate_count`
- `F1_missing_candidate_pair_ids`
- `F1_unauthorized_candidate_count`
- `F1_unauthorized_candidate_pair_ids`

Each F1 count must equal the length of its corresponding pair-ID array. The `accepted`, `manual_review_required`, and `rejected` arrays must be pairwise disjoint, and their union must equal generated authorized candidate pair IDs.

`P3W5_F1_REGENERATION_COMPLETE_ALL_CANDIDATES_ACCEPTED_PENDING_RESULT_REVIEW` requires:

- target = 121
- generated = 121
- accepted = 121
- manual_review_required = 0
- rejected = 0
- missing = 0
- unauthorized = 0

Any other normally terminated F1 execution must use `P3W5_F1_REGENERATION_COMPLETE_WITH_BLOCKERS_PENDING_RESULT_REVIEW`.

Machine-readable F1 result-review decision enum:

- `P3W5_F1_RESULT_REVIEW_PASS_ALL_ACCEPTED`
- `P3W5_F1_AUDIT_ARTIFACT_REVIEW_PASS_WITH_BLOCKERS`
- `P3W5_F1_RESULT_REVIEW_BLOCKED`

Decision meanings:

- `P3W5_F1_RESULT_REVIEW_PASS_ALL_ACCEPTED`: accepted count == 121, manual-review-required count == 0, rejected count == 0, missing count == 0, unauthorized count == 0, and `accepted F1 candidate artifact authority established`.
- `P3W5_F1_AUDIT_ARTIFACT_REVIEW_PASS_WITH_BLOCKERS`: audit artifact identity and accounting may be authoritative, but `accepted complete F1 candidate-set authority not established`.
- `P3W5_F1_RESULT_REVIEW_BLOCKED`: artifact identity, accounting, isolation, or contract validation failed.

Only `P3W5_F1_RESULT_REVIEW_PASS_ALL_ACCEPTED` may assert `accepted F1 candidate artifact authority established`.

Each regenerated row or audit record must include at least:

- `pair_id`
- `original_row_id`
- `regenerated_row_id`
- `intervention_type`
- `original_text`
- `regenerated_text`
- `original_final_label`
- `regenerated_final_label`
- `canonical_row_id`
- `generator_source_path`
- `generator_source_sha256`
- `generator_commit`
- `fact_identity`
- `grammar_validator_source`
- `grammar_validator_sha256`
- `grammar_before`
- `grammar_after`
- `sidecar_before`
- `sidecar_after`
- `lineage_preserved`
- `semantic_validation_status`
- `semantic_polarity_preserved`
- `candidate_accepted`
- `ordered_rejection_codes`
- all fields required by `F1_generator_impact_isolation_contract`
- all fields required by the semantic validation derivation contract

### Release Gate

The following are insufficient for polarity supervision release:

- 121 candidates generated
- grammar validator passes
- minimum REFUTE count 50 exceeded

A separate F1 result review must verify all of:

- 121/121 exact coverage
- zero unauthorized pair
- zero label mutation
- zero lineage mutation
- zero new integrity defect
- full-output generator impact isolation pass
- production generator authority established
- candidate artifact Git preservation
- result static review pass

F1 result review pass has two possible artifact-authority outcomes:

- If all candidates are accepted, it means only: `accepted F1 candidate artifact authority established`.
- If any candidate is unresolved or rejected, artifact integrity may pass but the outcome is: `F1 regeneration audit artifact authority established`; `accepted complete F1 candidate-set authority not established`.

Both outcomes retain machine-readable non-authority wording: `production dataset repair authority not established`; `training admission authority not established`; `polarity supervision released = false`; `A1/A2/A3 released = false`.

## F2 Manual Review Specification

### Purpose

F2 separates the P3-W4 lineage observation from semantic and remediation decisions for the 119 pair families where canonical defects propagated to derivatives.

The P3-W4 class `F2_CANONICAL_DEFECT_PROPAGATED_TO_DERIVATIVES` is a lineage observation only. It does not mean:

- F2 is automatically repairable
- F2 is approved for batch regeneration
- existing F2 rows are admitted for training
- F2 can release polarity supervision

### Input Authority

The F2 review target set is exactly the P3-W4 pair records satisfying all of:

- `family == F2`
- `automatic_root_cause_class == F2_CANONICAL_DEFECT_PROPAGATED_TO_DERIVATIVES`
- `remediation_state == MANUAL_REVIEW_REQUIRED`
- `pair_id` is in `decision_blocking_pair_ids`

The target cardinality must be exactly 119 pair families. Each review unit is a complete triple:

- canonical `none` REFUTE
- `paraphrase` REFUTE
- `polarity_flip` SUPPORT

### Review Input And Immutable Source Schema

The template authority for future manual review is:

- `reports/reason_router_p2_p3w4_canonical_grammar_authority_execution_ca99038d/p3w4_f2_manual_review.csv`

The immutable source schema is exactly:

- `pair_id`
- `canonical_none_row_id`
- `paraphrase_row_id`
- `polarity_flip_row_id`
- `canonical_final_label`
- `paraphrase_final_label`
- `polarity_flip_final_label`
- `canonical_claim`
- `paraphrase_claim`
- `polarity_flip_claim`
- `canonical_evidence`
- `paraphrase_evidence`
- `polarity_flip_evidence`
- `canonical_grammar_status`
- `paraphrase_grammar_status`
- `polarity_flip_grammar_status`
- `canonical_reason_codes`
- `paraphrase_reason_codes`
- `polarity_flip_reason_codes`
- `canonical_claim_text_diff_summary`
- `paraphrase_claim_text_diff_summary`
- `polarity_flip_claim_text_diff_summary`
- `canonical_evidence_text_diff_summary`
- `paraphrase_evidence_text_diff_summary`
- `polarity_flip_evidence_text_diff_summary`
- `automatic_root_cause_class`
- `automatic_evidence`

Each completed review record must add reviewer provenance:

- `source_record_sha256`
- `reviewer_id`
- `review_protocol_version`
- `reviewed_at_utc`

Reviewer provenance contract:

- `reviewer_id`: nonempty trimmed stable identifier
- `review_protocol_version`: exactly `P3W5_F2_MANUAL_REVIEW_V1`
- `reviewed_at_utc`: RFC 3339 UTC timestamp ending in `Z`

`source_record_sha256` uses canonicalization contract `F2_SOURCE_RECORD_HASH_V1`:

1. Read the authority CSV with an RFC 4180-compatible CSV parser.
2. Use the 27 immutable source schema columns above in the exact specified order.
3. Treat each cell as the exact string returned by the CSV parser.
4. Do not trim, apply Unicode normalization, reparse JSON, or sort reason codes.
5. Preserve an empty cell as the empty string `""`.
6. Serialize each row as a JSON array: `[value_of_pair_id,value_of_canonical_none_row_id,...,value_of_automatic_evidence]`.
7. JSON serialization is fixed to UTF-8 without BOM, `ensure_ascii = false`, `allow_nan = false`, `separators = [",", ":"]`, and no trailing newline.
8. Use the SHA-256 lowercase hex of that UTF-8 byte sequence as `source_record_sha256`.

Human fields and reviewer provenance are not included in the hash input. The completion gate must verify that each pair's source-field hash in the completed review matches the original template source-field hash.

### Human Field Contract

The following human fields must begin empty:

- `human_canonical_semantics`
- `human_paraphrase_semantics`
- `human_polarity_flip_semantics`
- `human_grammar_validity`
- `human_authority_decision`
- `human_notes`

Each semantic field evaluates both:

- whether the member text is consistent with that member's final label
- whether the member is consistent with the intended family transformation relationship

Allowed semantic field values:

- `VALID`
- `INVALID`
- `UNCLEAR`

Allowed grammar field values:

- `CANONICAL_ONLY_DEFECT`
- `MULTI_MEMBER_DEFECT`
- `NO_REPRODUCIBLE_DEFECT`
- `UNCLEAR`

Allowed authority decision values:

- `CANONICAL_TEXTUAL_REPAIR_CANDIDATE`
- `CANONICAL_REGENERATION_REQUIRED`
- `SEMANTIC_CONFLICT`
- `INSUFFICIENT_EVIDENCE_KEEP_BLOCKED`
- `NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED`

Empty values and undefined values do not count as completed review.

### Decision Meaning

`CANONICAL_TEXTUAL_REPAIR_CANDIDATE` means canonical semantics are judged preserved but surface grammar/text defect exists. This is not automatic admission. A textual repair candidate, triple reconstruction or derivative revalidation, and separate audit are required.

`CANONICAL_REGENERATION_REQUIRED` means the canonical member and derivative authority must be newly generated. The existing triple is not reused as-is.

`SEMANTIC_CONFLICT` means the canonical, paraphrase, and polarity-flip relationship conflicts with the intended transformation contract. Automatic repair and regeneration are not approved; the pair moves to separate conflict resolution.

`INSUFFICIENT_EVIDENCE_KEEP_BLOCKED` means the current artifacts are insufficient for a semantic or remediation conclusion. The blocked state remains.

`NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED` means human review could not reproduce the P3-W4 automatic grammar defect. This does not prove existing triple correctness, recoverability, training admission, or polarity supervision eligibility. The entire pair remains blocked until separate authority resolution.

### Compatibility Matrix

Compatibility matrix version: `F2_REVIEW_COMPATIBILITY_V1`.

Compatibility precedence:

1. `UNCLEAR`
2. `INVALID`
3. `ALL_VALID_BY_GRAMMAR_STATUS`

The matrix is an exhaustive, mutually exclusive truth table over semantic enums `VALID`, `INVALID`, `UNCLEAR` and grammar enums `CANONICAL_ONLY_DEFECT`, `MULTI_MEMBER_DEFECT`, `NO_REPRODUCIBLE_DEFECT`, `UNCLEAR`.

Rule 1 - Unclear:

- if any semantic field is `UNCLEAR`, or `human_grammar_validity == UNCLEAR`, the required decision is `INSUFFICIENT_EVIDENCE_KEEP_BLOCKED`.

Rule 2 - Semantic invalidity:

- if Rule 1 does not apply and any semantic field is `INVALID`, the required decision is `SEMANTIC_CONFLICT`.
- semantic invalidity is not routed to an automatic regeneration candidate.

Rule 3 - All semantics valid:

- if all three semantic fields are `VALID` and `human_grammar_validity == CANONICAL_ONLY_DEFECT`, the required decision is `CANONICAL_TEXTUAL_REPAIR_CANDIDATE`.
- if all three semantic fields are `VALID` and `human_grammar_validity == MULTI_MEMBER_DEFECT`, the required decision is `CANONICAL_REGENERATION_REQUIRED`.
- if all three semantic fields are `VALID` and `human_grammar_validity == NO_REPRODUCIBLE_DEFECT`, the required decision is `NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED`.

Undefined combinations, different decisions, and empty values are invalid review records and do not count as completed review.

`human_notes` is checked after trim and must be nonempty when any semantic field is `UNCLEAR`, when `human_grammar_validity == UNCLEAR`, or when the decision is `SEMANTIC_CONFLICT`, `INSUFFICIENT_EVIDENCE_KEEP_BLOCKED`, or `NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED`.
### Review Completion Gate

F2 review execution is complete only when all of the following hold:

- 119 unique pair IDs
- 119 review rows
- zero missing pair
- zero duplicate pair
- all six human fields present
- all reviewer provenance fields present
- all categorical fields use allowed enums
- all decision combinations satisfy the compatibility matrix
- nonempty trimmed `human_notes` when required
- no source field mutation
- per-pair completed-review `source_record_sha256` matches the original template source-field hash
- no row, pair, or label mutation

### Future Output Contract

A future F2 manual review stage must produce:

- `p3w5_f2_review_completed.csv`
- `p3w5_f2_review_summary.json`
- `p3w5_f2_review_decisions.jsonl`

F2 execution status and decision namespace:

- `F2_execution_status`: `P3W5_F2_MANUAL_REVIEW_EXECUTION_COMPLETE_PENDING_RESULT_REVIEW`
- `F2_execution_decision`
- `F2_artifact_paths`
- `F2_input_sha256`
- `F2_execution_commit`
- `F2_output_sha256`

The summary must include the authoritative universe and count/pair-ID array symmetry at least:

- `authorized_F2_pair_count = 119`
- `authorized_F2_pair_ids`


- `reviewed_pair_count`
- `reviewed_pair_ids`
- `unreviewed_pair_count`
- `unreviewed_pair_ids`
- `textual_repair_candidate_count`
- `textual_repair_candidate_pair_ids`
- `regeneration_required_count`
- `regeneration_required_pair_ids`
- `semantic_conflict_count`
- `semantic_conflict_pair_ids`
- `insufficient_evidence_count`
- `insufficient_evidence_pair_ids`
- `no_reproducible_defect_keep_blocked_count`
- `no_reproducible_defect_keep_blocked_pair_ids`
- `invalid_review_count`
- `invalid_review_pair_ids`
- `invalid_combination_count`
- `invalid_combination_pair_ids`
- `missing_reviewer_provenance_count`
- `missing_reviewer_provenance_pair_ids`
- `source_hash_mismatch_count`
- `source_hash_mismatch_pair_ids`

Each count must equal the length of its corresponding pair-ID array.

F2 exact universe partition contract:

- `reviewed_pair_ids ∩ unreviewed_pair_ids = ∅`
- `reviewed_pair_ids ∪ unreviewed_pair_ids == authorized_F2_pair_ids`
- `invalid_review_pair_ids ⊆ reviewed_pair_ids`
- `completed_decision_pair_ids == reviewed_pair_ids - invalid_review_pair_ids`

`invalid_review_pair_ids` is the unique union of at least:

- `invalid_combination_pair_ids`
- `missing_reviewer_provenance_pair_ids`
- `source_hash_mismatch_pair_ids`
- `missing_human_field_pair_ids`
- `invalid_enum_pair_ids`
- `missing_required_notes_pair_ids`

The summary must also include:

- `missing_human_field_count`
- `missing_human_field_pair_ids`
- `invalid_enum_count`
- `invalid_enum_pair_ids`
- `missing_required_notes_count`
- `missing_required_notes_pair_ids`
- `completed_decision_pair_count`
- `completed_decision_pair_ids`

Completed decision category arrays `textual_repair_candidate_pair_ids`, `regeneration_required_pair_ids`, `semantic_conflict_pair_ids`, `insufficient_evidence_pair_ids`, and `no_reproducible_defect_keep_blocked_pair_ids` must be pairwise disjoint, and their union must exactly equal `completed_decision_pair_ids`. Invalid and unreviewed records are not included in completed decision categories.

F2 review execution complete requires:

- `reviewed_pair_count = 119`
- `unreviewed_pair_count = 0`
- `invalid_review_count = 0`
- `completed_decision_pair_count = 119`

F2 execution artifacts require Git-preserved artifact identity before separate result review.
## Workstream Isolation Contract

F1 and F2 do not share:

- input target set
- candidate generator
- review decision
- acceptance criterion
- output namespace
- release gate
- supporting or blocking pair IDs

Forbidden shortcuts:

- using F1 grammar reproduction as F2 semantic proof
- using F2 propagated lineage as F1 regeneration proof
- automatically propagating an F1 regenerated row into F2 derivative authority
- using an F2 human decision as F1 generator acceptance
- combining both workstream counts into one recovery rate

Summary namespace must remain separate:

- `F1_target_pair_count`
- `F1_target_row_count`
- `F1_regeneration_candidate_count`
- `F1_accepted_candidate_count`
- `F2_target_pair_count`
- `F2_target_member_count`
- `F2_reviewed_pair_count`
- `F2_decision_counts`

Each workstream must separately record artifact paths, input SHA, execution commit, output SHA, execution status, and execution decision. F1 and F2 execution artifacts both require Git-preserved artifact identity before separate result review. A combined F1/F2 recovery rate is forbidden.

## Release Matrix

| Condition | F1 execution | F2 review | Production dataset repaired | Polarity supervision | A1/A2/A3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P3-W5 spec only | blocked | blocked | false | blocked | blocked |
| F1 regeneration complete, F2 unresolved | complete candidate set only | blocked | false | blocked | blocked |
| F2 review complete, F1 unresolved | blocked | review complete only | false | blocked | blocked |
| F1 all-accepted result review pass only | accepted F1 candidate artifact authority established | blocked | false | blocked | blocked |
| F1 with-blockers audit artifact review pass only | F1 regeneration audit artifact authority established; accepted complete F1 candidate-set authority not established | blocked | false | blocked | blocked |
| F2 result review pass only | blocked | F2 human-review decision artifact authority established | false | blocked | blocked |
| F1 all-accepted result review and F2 result review both pass | accepted F1 candidate artifact authority established | F2 human-review decision artifact authority established | false | blocked | blocked |
| F1 with-blockers audit artifact review and F2 result review both pass | F1 regeneration audit artifact authority established; accepted complete F1 candidate-set authority not established | F2 human-review decision artifact authority established | false | blocked | blocked |
| Separate training-readiness release passes for explicitly approved subset | complete for explicitly approved subset only | complete for explicitly approved subset only | only if separately established | may be released explicitly for explicitly approved subset only | only then considered |

F1 result review pass means `accepted F1 candidate artifact authority established` only when all candidates are accepted; otherwise it means `F1 regeneration audit artifact authority established` and `accepted complete F1 candidate-set authority not established`.

F2 result review pass means only: `F2 human-review decision artifact authority established`.

Machine-readable non-authority wording: `production dataset repair authority not established`; `training admission authority not established`.

F2 per-pair decisions are authoritative review outcomes only. `SEMANTIC_CONFLICT`, `INSUFFICIENT_EVIDENCE_KEEP_BLOCKED`, and `NO_REPRODUCIBLE_DEFECT_KEEP_BLOCKED` pairs remain blocked. Repair or regeneration candidate decisions still require separate execution and result review.

Result reviews do not mean: production dataset repaired, training rows admitted, polarity supervision released, or A1/A2/A3 released.

Even if both reviews pass:

- production dataset repaired = false
- polarity supervision released = false
- A1/A2/A3 released = false

A separate training-readiness release audit may review only explicitly approved regenerated or repaired subsets. It must not aggregate blocked F2 categories or admit them implicitly.

## Static Blockers

P3-W5 retains these blockers:

- `P3W5_SPEC_NOT_STATICALLY_REVIEWED`
- `P3W5_F1_REGENERATION_NOT_EXECUTED`
- `P3W5_F2_MANUAL_REVIEW_NOT_EXECUTED`
- `P2_POLARITY_LOCAL_SUPERVISION_NOT_TRAINING_READY`
- `A1_A3_NOT_RELEASED`

## Forbidden Claims

This stage does not claim:

- `P3W5_EXECUTION_PASS`
- `F1_REGENERATED`
- `F1_REPAIR_COMPLETE`
- `F2_REVIEW_COMPLETE`
- `F2_ROWS_RECOVERABLE`
- `F2_REGENERATION_APPROVED`
- `POLARITY_SUPERVISION_RELEASED`
- `P2_POLARITY_SUPERVISION_RESOLVED`
- `A1_READY`
- `A2_READY`
- `A3_READY`
- `P3_PASS`

## Execution Non-Claims

For this P3-W5 spec-only step:

- implementation executed: false
- regeneration executed: false
- manual review executed: false
- training executed: false
- evaluation executed: false
- Git actions executed: false
